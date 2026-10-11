"""Training set and SFT tasks for search-based tasks with Qwen2.5-7B-Instruct.

Draws PER_DATASET questions from each training split in SOURCES (NQ and HotpotQA from Search-R1, the others from
FlashRAG), skipping questions whose normalized text is a test question, and SFT_PER_DATASET SFT tasks per dataset.
Writes train.parquet, sft_tasks.parquet and BUILD.json to --out.

Usage: python scripts/data/search_train_qwen2.5-7b.py --data-dir DATA --flashrag-dir FLASHRAG --out OUT
"""
import collections, json, random, re, sys
from pathlib import Path

import pandas as pd

import argparse

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--data-dir", required=True, type=Path,
                    help="directory with searchR1_processed_direct/{train,test}.parquet")
parser.add_argument("--flashrag-dir", required=True, type=Path,
                    help="FlashRAG datasets: triviaqa/, 2wikimultihopqa/ and musique/ with train.jsonl")
parser.add_argument("--out", required=True, type=Path)
args = parser.parse_args()
W, F, OUT = args.data_dir, args.flashrag_dir, args.out
PER_DATASET, SFT_PER_DATASET = 10000, 100
SOURCES = ("nq", "hotpotqa", "triviaqa", "2wikimultihopqa", "musique")
KINDS = ("who", "when", "where", "what", "which", "how", "why", "yes_no", "other")
YES_NO = ("is", "are", "was", "were", "do", "does", "did", "can", "could", "has", "have", "had", "will", "would", "should")


def norm(text):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", str(text).lower())).strip()


def kind(question):
    words = norm(question).split()
    if words and words[0] in YES_NO:
        return "yes_no"
    for word in words[:6]:
        if word in KINDS:
            return word
    return "other"


def row(source, question, answers):
    gt = {"target": [str(a) for a in answers]}
    return {"data_source": source,
            "prompt": [{"content": "You are a helpful and harmless assistant.", "role": "system"}, {"content": question, "role": "user"}],
            "ability": "fact-reasoning",
            "reward_model": {"ground_truth": gt, "style": "rule"},
            "extra_info": {"index": 0, "need_tools_kwargs": True, "question": question, "split": "train",
                           "tools_kwargs": {"search": {"create_kwargs": {"data_source": source, "ground_truth": gt, "question": question}}}},
            "metadata": None,
            "env_kwargs": {"data_source": source, "ground_truth": gt, "question": question}}


if (OUT / "train.parquet").exists():
    raise SystemExit(f"{OUT} already built")
test = pd.read_parquet(W / "searchR1_processed_direct/test.parquet")
test_questions = {norm(info["question"]) for info in test["extra_info"]}
candidates = collections.defaultdict(list)
train = pd.read_parquet(W / "searchR1_processed_direct/train.parquet", columns=["data_source", "extra_info", "reward_model"])
for source, info, reward in zip(train["data_source"], train["extra_info"], train["reward_model"]):
    candidates[source].append((str(info["question"]), list(reward["ground_truth"]["target"])))
for source in ("triviaqa", "2wikimultihopqa", "musique"):
    for line in (F / f"{source}/train.jsonl").open():
        r = json.loads(line)
        answers = r["golden_answers"]
        if isinstance(answers, str):
            answers = [answers]
        candidates[source].append((str(r["question"]).strip(), list(answers)))

stats = {}
pool = []
for source in SOURCES:
    seen, kept, in_test, dup, empty = set(), [], 0, 0, 0
    for question, answers in candidates[source]:
        key = norm(question)
        if not key or not [a for a in answers if str(a).strip()]:
            empty += 1
        elif key in test_questions:
            in_test += 1
        elif key in seen:
            dup += 1
        else:
            seen.add(key)
            kept.append((question, answers))
    chosen = random.Random(f"2026:rl:{source}").sample(kept, PER_DATASET)
    pool.extend(row(source, q, a) for q, a in chosen)
    stats[source] = {"candidates": len(candidates[source]), "in_test_dropped": in_test, "duplicates_dropped": dup,
                     "empty_dropped": empty, "eligible": len(kept), "sampled": PER_DATASET}
random.Random(2026).shuffle(pool)
for index, r in enumerate(pool):
    r["extra_info"]["index"] = index

sft = []
for source in SOURCES:
    groups = collections.defaultdict(list)
    for r in pool:
        if r["data_source"] == source:
            groups[kind(r["extra_info"]["question"])].append(r)
    for k in groups:
        random.Random(f"2026:{source}:{k}").shuffle(groups[k])
    picked, turn = [], 0
    while len(picked) < SFT_PER_DATASET:
        k = KINDS[turn % len(KINDS)]
        if groups.get(k):
            picked.append(groups[k].pop())
        turn += 1
    sft.extend(picked)
    stats[source]["sft_kinds"] = dict(collections.Counter(kind(r["extra_info"]["question"]) for r in picked))

OUT.mkdir(parents=True, exist_ok=True)
pd.DataFrame(pool).to_parquet(OUT / "train.parquet", index=False)
pd.DataFrame(sft).to_parquet(OUT / "sft_tasks.parquet", index=False)
(OUT / "BUILD.json").write_text(json.dumps({"description": __doc__, "per_dataset": PER_DATASET, "sft_per_dataset": SFT_PER_DATASET,
                                            "sources": stats}, indent=1) + "\n")
check = pd.read_parquet(OUT / "train.parquet")
print(len(check), dict(collections.Counter(check["data_source"])), "| sft", len(sft))
print(json.dumps(stats, indent=1))
print("sample rows:", [(r["data_source"], r["extra_info"]["question"][:70], r["reward_model"]["ground_truth"]["target"][:2]) for r in sft[::100]])
