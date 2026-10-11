"""PopQA-style questions from the KILT T-REx training split (FlashRAG's trex/train.jsonl).

The PopQA relations that T-REx covers are asked with PopQA's templates (TEMPLATES); one question per (subject,
relation) with the answers of all its rows merged; subjects with brackets are skipped, and so are questions whose
answer occurs in the subject; 'sport' keeps person-like subjects and 'country' subjects without digits. At most CAP
questions per relation. Subjects of the PopQA test set are removed by the training-set builders.
Writes trex_popqa_relations.jsonl and STATS.json to --out.

Usage: python scripts/data/trex_popqa.py --trex FLASHRAG/trex/train.jsonl --out TREX
"""
import collections, json, random, re, string
from pathlib import Path

import argparse

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--trex", required=True, type=Path, help="FlashRAG trex/train.jsonl")
parser.add_argument("--out", required=True, type=Path)
args = parser.parse_args()
OUT = args.out
CAP = 20000
TEMPLATES = {
    "occupation": "What is {}'s occupation?",
    "place of birth": "In what city was {} born?",
    "genre": "What genre is {}?",
    "father": "Who is the father of {}?",
    "country": "In what country is {}?",
    "producer": "Who was the producer of {}?",
    "director": "Who was the director of {}?",
    "capital of": "What is {} the capital of?",
    "screenwriter": "Who was the screenwriter for {}?",
    "composer": "Who was the composer of {}?",
    "religion": "What is the religion of {}?",
    "sport": "What sport does {} play?",
    "author": "Who is the author of {}?",
    "mother": "Who is the mother of {}?",
    "capital": "What is the capital of {}?",
}
NOT_PERSON = re.compile(r"\b(fc|f\.c\.|afc|sc|cf|club|team|league|cup|championships?|tournament|season|games|olympics|national|"
                        r"association|federation|university|college|school|stadium|arena|open|grand prix|series|division|"
                        r"conference|united|city|athletic|rovers|wanderers|sports?|racing|men's|women's)\b", re.I)


def normalize(text):
    text = str(text).lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def keep_subject(relation, subject):
    if not subject or "(" in subject or ")" in subject or len(subject) > 80:
        return False
    if relation == "sport":
        return not re.search(r"\d", subject) and len(subject.split()) <= 4 and not NOT_PERSON.search(subject)
    if relation == "country":
        return not re.search(r"\d", subject)
    return True


merged = collections.OrderedDict()
seen_rows = collections.Counter()
with args.trex.open() as fh:
    for line in fh:
        r = json.loads(line)
        if " [SEP] " not in r["question"]:
            continue
        subject, relation = (x.strip() for x in r["question"].split(" [SEP] ", 1))
        if relation not in TEMPLATES:
            continue
        seen_rows[relation] += 1
        if not keep_subject(relation, subject):
            continue
        answers = r["golden_answers"] if isinstance(r["golden_answers"], list) else [r["golden_answers"]]
        slot = merged.setdefault((subject, relation), [])
        for a in answers:
            a = str(a).strip()
            if a and normalize(a) and a not in slot:
                slot.append(a)

by_relation = collections.defaultdict(list)
dropped_leak = collections.Counter()
for (subject, relation), answers in merged.items():
    if not answers:
        continue
    padded = f" {normalize(subject)} "
    if any(len(normalize(a)) >= 3 and f" {normalize(a)} " in padded for a in answers):
        dropped_leak[relation] += 1
        continue
    by_relation[relation].append({"subject": subject, "relation": relation, "question": TEMPLATES[relation].format(subject),
                                  "golden_answers": answers})

OUT.mkdir(parents=True, exist_ok=True)
stats = {}
with (OUT / "trex_popqa_relations.jsonl").open("w") as out:
    for relation in TEMPLATES:
        rows = by_relation[relation]
        random.Random(f"2026:trex:{relation}").shuffle(rows)
        kept = rows[:CAP]
        for row in kept:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
        stats[relation] = {"trex_rows": seen_rows[relation], "subject_relation_pairs": len(rows) + dropped_leak[relation],
                           "answer_in_subject_dropped": dropped_leak[relation], "eligible": len(rows), "kept": len(kept)}
(OUT / "STATS.json").write_text(json.dumps({"description": __doc__, "templates": TEMPLATES, "relations": stats}, indent=1) + "\n")
print(json.dumps(stats, indent=1))
for relation in TEMPLATES:
    for row in by_relation[relation][:2]:
        print(f"   [{relation}] {row['question']} -> {row['golden_answers'][:4]}")
