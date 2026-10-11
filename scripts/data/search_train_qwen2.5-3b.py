"""Training set and SFT tasks for search-based tasks with Qwen2.5-3B-Instruct.

Fills the strata in QUOTA from the Search-R1 and FlashRAG training splits and from PopQA-style questions
(trex_popqa.py), cell by cell (see the shares below), and draws SFT_TOTAL SFT tasks in proportion to strata and cells.
Questions whose normalized text is a test question are skipped, and so are near-duplicates of a test question (e5
cosine >= DUP_COS with a shared answer); PopQA-style questions are checked by subject instead.
Writes train.parquet, sft_tasks.parquet, sampled_tasks.jsonl and BUILD.json to --out.

Usage: python scripts/data/search_train_qwen2.5-3b.py --data-dir DATA --flashrag-dir FLASHRAG \
           --trex-popqa TREX/trex_popqa_relations.jsonl --out OUT
"""
import collections, json, math, os, random, re, string, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

import argparse

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--data-dir", required=True, type=Path,
                    help="directory with searchR1_processed_direct/{train,test}.parquet")
parser.add_argument("--flashrag-dir", required=True, type=Path,
                    help="FlashRAG datasets: triviaqa/, 2wikimultihopqa/ and musique/ with train.jsonl")
parser.add_argument("--out", required=True, type=Path)
parser.add_argument("--trex-popqa", required=True, type=Path, help="trex_popqa_relations.jsonl from trex_popqa.py")
parser.add_argument("--e5-model", default="intfloat/e5-base-v2")
parser.add_argument("--test-embeddings", type=Path, default=None,
                    help="cache of the e5 embeddings of the test questions (default: DATA_DIR/test_e5.npy; built if missing)")
parser.add_argument("--threads", type=int, default=64)
args = parser.parse_args()
torch.set_num_threads(args.threads)
D, F, OUT, E5 = args.data_dir, args.flashrag_dir, args.out, args.e5_model
QUOTA = {"nq": 2800, "triviaqa": 2800, "popqa_like": 2800, "hotpotqa": 5250, "2wikimultihopqa": 4200, "musique": 3150}
STRATA = tuple(QUOTA)
SFT_TOTAL = 500
DUP_COS = 0.90
KINDS = ("who", "when", "where", "what", "which", "how", "why")
YES_NO = ("is", "are", "was", "were", "do", "does", "did", "can", "could", "has", "have", "had", "will", "would", "should")
PUNCT = set(string.punctuation)
t0 = time.time()
log = lambda *a: print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)
if (OUT / "train.parquet").exists():
    raise SystemExit(f"{OUT} already built")


def norm_q(text):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", str(text).lower())).strip()


def norm_a(text):
    text = "".join(ch for ch in str(text).lower() if ch not in PUNCT)
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def kind(question):
    words = norm_q(question).split()
    if words and words[0] in YES_NO:
        return "yes_no"
    return next((w for w in words[:6] if w in KINDS), "other")


def keys(answers):
    return {k for k in (norm_a(a) for a in answers) if len(k) >= 3 and k not in ("yes", "no")}


def as_list(v):
    if v is None:
        return []
    if isinstance(v, (list, tuple, np.ndarray)):
        return [str(x) for x in list(v)]
    return [str(v)]


def allocate(shares, total, capacity):
    """Largest-remainder quotas proportional to shares, capped by capacity, deficits spread over the other cells."""
    alloc = {c: 0 for c in shares}
    active = [c for c in shares if shares[c] > 0 and capacity.get(c, 0) > 0]
    remaining = total
    while remaining > 0 and active:
        weight = sum(shares[c] for c in active)
        raw = {c: remaining * shares[c] / weight for c in active}
        given = 0
        for c in active:
            add = min(int(math.floor(raw[c])), capacity[c] - alloc[c])
            alloc[c] += add
            given += add
        remaining -= given
        for c in sorted(active, key=lambda c: raw[c] - math.floor(raw[c]), reverse=True):
            if remaining and alloc[c] < capacity[c]:
                alloc[c] += 1
                remaining -= 1
        active = [c for c in active if alloc[c] < capacity[c]]
    assert remaining == 0, (shares, capacity, alloc)
    return alloc


# ---------------------------------------------------------------- test set: texts, answers, cell shares
test = pd.read_parquet(D / "searchR1_processed_direct/test.parquet", columns=["data_source", "extra_info", "metadata", "reward_model"])
test_q = [str(i["question"]) for i in test["extra_info"]]
test_src = list(test["data_source"])
test_keys = [keys(as_list(r["ground_truth"]["target"])) for r in test["reward_model"]]
test_norm = {norm_q(q) for q in test_q}
md = lambda m: m if isinstance(m, dict) else {}
shares = {}
for src, cell_of in (("nq", lambda q, m: kind(q)), ("triviaqa", lambda q, m: kind(q)), ("popqa", lambda q, m: str(m.get("prop")))):
    c = collections.Counter(cell_of(q, md(m)) for q, s, m in zip(test_q, test_src, test["metadata"]) if s == src)
    n = sum(c.values())
    shares["popqa_like" if src == "popqa" else src] = {k: v / n for k, v in c.items()}
popqa_subjects = {norm_a(md(m).get("subj")) for s, m in zip(test_src, test["metadata"]) if s == "popqa" and md(m).get("subj")}
log(f"test: {len(test_q)} questions, {len(test_norm)} distinct; PopQA subjects {len(popqa_subjects)}")

# ---------------------------------------------------------------- candidate pools (stratum -> cell -> items)
pools = {s: collections.defaultdict(list) for s in STRATA}
counts = collections.Counter()
seen = set()


def add(stratum, cell, question, answers, source, source_id):
    question = re.sub(r"\s+", " ", str(question)).strip()
    answers = [a for a in dict.fromkeys(str(a).strip() for a in answers) if a and norm_a(a)]
    key = norm_q(question)
    if not key or not answers:
        counts[(stratum, "empty")] += 1
        return
    if key in test_norm:
        counts[(stratum, "in_test")] += 1
        return
    if key in seen:
        counts[(stratum, "duplicate")] += 1
        return
    seen.add(key)
    pools[stratum][cell].append({"q": question, "a": answers, "source": source, "id": source_id, "cell": cell, "key": key})


train = pd.read_parquet(D / "searchR1_processed_direct/train.parquet", columns=["data_source", "extra_info", "metadata", "reward_model"])
for i, (src, info, m, rew) in enumerate(zip(train["data_source"], train["extra_info"], train["metadata"], train["reward_model"])):
    answers = as_list(rew["ground_truth"]["target"])
    if src == "nq":
        add("nq", kind(info["question"]), info["question"], answers, "nq", f"searchR1-train-{i}")
    elif src == "hotpotqa":
        m = md(m)
        add("hotpotqa", f"{m.get('type')}/{m.get('level')}", info["question"], answers, "hotpotqa", f"searchR1-train-{i}")
for line in (F / "triviaqa/train.jsonl").open():
    r = json.loads(line)
    add("triviaqa", kind(r["question"]), r["question"], as_list(r["golden_answers"]), "triviaqa", r["id"])
for line in (F / "2wikimultihopqa/train.jsonl").open():
    r = json.loads(line)
    t = str(r["metadata"].get("type"))
    if t == "bridge_comparison":
        counts[("2wikimultihopqa", "bridge_comparison_skipped")] += 1
        continue
    add("2wikimultihopqa", t, r["question"], as_list(r["golden_answers"]), "2wikimultihopqa", r["id"])
for line in (F / "musique/train.jsonl").open():
    r = json.loads(line)
    hops = len(r["metadata"].get("question_decomposition") or [])
    if hops != 2:
        counts[("musique", f"{hops}hop_skipped")] += 1
        continue
    add("musique", "2", r["question"], as_list(r["golden_answers"]), "musique", r["id"])
for line in args.trex_popqa.open():
    r = json.loads(line)
    if norm_a(r["subject"]) in popqa_subjects:
        counts[("popqa_like", "popqa_test_subject")] += 1
        continue
    add("popqa_like", r["relation"], r["question"], r["golden_answers"], "trex", r["subject"] + " | " + r["relation"])
for s in ("hotpotqa", "2wikimultihopqa", "musique"):
    n = sum(len(v) for v in pools[s].values())
    shares[s] = {c: len(v) / n for c, v in pools[s].items()}
log("pools:", {s: {c: len(v) for c, v in sorted(pools[s].items())} for s in STRATA})
log("filtered:", dict(counts))

# ---------------------------------------------------------------- e5 embeddings
tok = AutoTokenizer.from_pretrained(E5)
model = AutoModel.from_pretrained(E5).eval()


def embed(texts, bs=256):
    out = []
    with torch.inference_mode():
        for i in range(0, len(texts), bs):
            batch = tok(["query: " + t for t in texts[i:i + bs]], padding=True, truncation=True, max_length=64, return_tensors="pt")
            h = model(**batch).last_hidden_state
            m = batch["attention_mask"].unsqueeze(-1).to(h.dtype)
            out.append(torch.nn.functional.normalize((h * m).sum(1) / m.sum(1), dim=-1).numpy())
    return np.concatenate(out).astype(np.float32) if out else np.zeros((0, 768), np.float32)


TE_PATH = args.test_embeddings or D / "test_e5.npy"
if not TE_PATH.exists():
    TE_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(TE_PATH, embed(test_q, bs=512))
TE = np.load(TE_PATH)
assert TE.shape == (len(test_q), 768)
log(f"loaded e5 embeddings of the {len(TE)} test questions")
examples = collections.defaultdict(list)


def near_dup(items, E):
    """Flags items with a test question at cosine >= DUP_COS that shares an answer."""
    flags = []
    for i in range(0, len(items), 1024):
        S = E[i:i + 1024] @ TE.T
        for r in range(S.shape[0]):
            item = items[i + r]
            k = keys(item["a"])
            hit = next((j for j in np.nonzero(S[r] >= DUP_COS)[0] if k & test_keys[j]), None)
            flags.append(hit is not None)
            if hit is not None and len(examples["near_dup"]) < 12:
                examples["near_dup"].append((round(float(S[r, hit]), 3), item["q"], test_q[hit], sorted(k & test_keys[hit])[:2]))
    return flags


selected = {s: [] for s in STRATA}
stats = {}
for stratum in STRATA:
    share = {c: shares[stratum].get(c, 0.0) for c in set(shares[stratum]) | set(pools[stratum])}
    cells = {}
    for c in share:
        items = list(pools[stratum].get(c, []))
        random.Random(f"2026:search6:{stratum}:{c}").shuffle(items)
        cells[c] = {"items": items, "pos": 0, "acc": [], "dropped": 0}
    target = allocate(share, QUOTA[stratum], {c: len(v["items"]) for c, v in cells.items()})
    first_quota = dict(target)
    while True:
        for c, q in target.items():
            cell = cells[c]
            while len(cell["acc"]) < q and cell["pos"] < len(cell["items"]):
                start = cell["pos"]
                chunk = cell["items"][start:start + max(64, (q - len(cell["acc"])) * 5 // 4)]
                flags = [False] * len(chunk) if stratum == "popqa_like" else near_dup(chunk, embed([it["q"] for it in chunk]))
                consumed = 0
                for it, dup in zip(chunk, flags):
                    if len(cell["acc"]) >= q:
                        break
                    consumed += 1
                    if dup:
                        cell["dropped"] += 1
                    else:
                        cell["acc"].append(it)
                cell["pos"] = start + consumed
        deficit = QUOTA[stratum] - sum(len(v["acc"]) for v in cells.values())
        if deficit == 0:
            break
        # A cell ran out after decontamination: its shortfall goes to the other cells in proportion to their shares.
        spare = {c: len(v["items"]) - v["pos"] for c, v in cells.items() if len(v["items"]) > v["pos"]}
        extra = allocate({c: share[c] for c in spare}, deficit, spare)
        target = {c: len(cells[c]["acc"]) + extra.get(c, 0) for c in cells}
    cell_stats = {}
    for c, v in sorted(cells.items()):
        selected[stratum] += v["acc"]
        cell_stats[c] = {"share": round(share[c], 4), "candidates": len(v["items"]), "quota": first_quota.get(c, 0),
                         "near_dup_dropped": v["dropped"], "selected": len(v["acc"])}
    assert len(selected[stratum]) == QUOTA[stratum]
    stats[stratum] = cell_stats
    log(stratum, {c: (v["selected"], v["near_dup_dropped"]) for c, v in cell_stats.items()})


# ---------------------------------------------------------------- rows, SFT tasks, files
def row(stratum, it):
    gt = {"target": [str(a) for a in it["a"]]}
    return {"data_source": stratum,
            "prompt": [{"content": "You are a helpful and harmless assistant.", "role": "system"}, {"content": it["q"], "role": "user"}],
            "ability": "fact-reasoning",
            "reward_model": {"ground_truth": gt, "style": "rule"},
            "extra_info": {"index": 0, "need_tools_kwargs": True, "question": it["q"], "split": "train",
                           "tools_kwargs": {"search": {"create_kwargs": {"data_source": stratum, "ground_truth": gt, "question": it["q"]}}}},
            "metadata": json.dumps({"stratum": stratum, "cell": it["cell"], "source": it["source"], "source_id": it["id"]}, ensure_ascii=False),
            "env_kwargs": {"data_source": stratum, "ground_truth": gt, "question": it["q"]}}


rows = [row(s, it) for s in STRATA for it in selected[s]]
random.Random("2026:search6:order").shuffle(rows)
for index, r in enumerate(rows):
    r["extra_info"]["index"] = index
total = sum(QUOTA.values())
sft_quota = allocate({s: QUOTA[s] / total for s in STRATA}, SFT_TOTAL, dict(QUOTA))
sft = []
for s in STRATA:
    cells = collections.defaultdict(list)
    for it in selected[s]:
        cells[it["cell"]].append(it)
    q = allocate({c: len(v) / len(selected[s]) for c, v in cells.items()}, sft_quota[s], {c: len(v) for c, v in cells.items()})
    for c in sorted(q):
        sft += [(s, it) for it in random.Random(f"2026:search6:sft:{s}:{c}").sample(cells[c], q[c])]
assert len(sft) == SFT_TOTAL and len({it["key"] for _, it in sft}) == SFT_TOTAL
OUT.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_parquet(OUT / "train.parquet", index=False)
pd.DataFrame([row(s, it) for s, it in sft]).to_parquet(OUT / "sft_tasks.parquet", index=False)
with (OUT / "sampled_tasks.jsonl").open("w") as fh:
    for i, (s, it) in enumerate(sft):
        gt = {"target": [str(a) for a in it["a"]]}
        fh.write(json.dumps({"task_id": f"search_{i:06d}", "task_type": "search_qa", "row_index": i, "sample_index": i, "split": "train",
                             "data_source": s, "task_description": it["q"], "question": it["q"], "ground_truth": gt,
                             "env_kwargs": {"question": it["q"], "ground_truth": gt, "data_source": s},
                             "stratum": f"{s}:{it['cell']}"}, ensure_ascii=False) + "\n")
build = {"description": __doc__, "quota": QUOTA, "sft_total": SFT_TOTAL, "sft_quota": sft_quota, "near_dup_cosine": DUP_COS,
         "embedder": "e5-base-v2 (query: prefix, mean pooling)",
         "filtered_before_selection": {f"{s}/{why}": n for (s, why), n in counts.items()}, "strata": stats,
         "sft_strata": dict(collections.Counter(f"{s}:{it['cell']}" for s, it in sft)),
         "examples": {k: v for k, v in examples.items()}}
(OUT / "BUILD.json").write_text(json.dumps(build, indent=1, ensure_ascii=False) + "\n")
check = pd.read_parquet(OUT / "train.parquet")
log(len(check), dict(collections.Counter(check["data_source"])), "| sft", len(sft), dict(collections.Counter(s for s, _ in sft)))
for k, v in examples.items():
    print(f"== {k}")
    for e in v[:6]:
        print("   ", e)
