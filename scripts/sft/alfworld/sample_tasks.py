"""Write the ALFWorld SFT tasks of the paper (500 games: 84, 84, 83, 83, 83, 83 per task type in the order of
TASK_TYPES) as sampled_tasks.jsonl, with the sampler of scripts/sft/_common/pipeline.py: one generator seeded with
--seed, the solvable training games of each type shuffled in turn. prepare_data.sh with RESUME=true then collects the
rollouts of exactly these games.

Usage: python scripts/sft/alfworld/sample_tasks.py --output-dir OUT
       OUTPUT_DIR=OUT RESUME=true bash scripts/sft/alfworld/prepare_data.sh
"""
import argparse
import random
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.sft._common.pipeline import TASK_TYPES, append_jsonl, collect_train_game_files  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--alf-config", default=str(ROOT / "agent_system/environments/env_package/alfworld/configs/config_tw.yaml"))
    parser.add_argument("--counts", default="84,84,83,83,83,83", help="tasks per type, in the order of TASK_TYPES")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    counts = [int(c) for c in args.counts.split(",")]
    if len(counts) != len(TASK_TYPES):
        raise SystemExit(f"--counts needs {len(TASK_TYPES)} values for {TASK_TYPES}")
    path = args.output_dir / "sampled_tasks.jsonl"
    if path.exists():
        raise SystemExit(f"{path} already exists")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    grouped = collect_train_game_files(args.alf_config)
    rng = random.Random(args.seed)
    for task_type, count in zip(TASK_TYPES, counts):
        pool = list(grouped.get(task_type, []))
        if len(pool) < count:
            raise SystemExit(f"Need {count} {task_type} games, found {len(pool)}; check ALFWORLD_DATA")
        rng.shuffle(pool)
        for idx, item in enumerate(pool[:count]):
            record = dict(item)
            record["task_id"] = f"{task_type}_{idx:03d}_{uuid.uuid5(uuid.NAMESPACE_URL, item['game_file']).hex[:10]}"
            record["sample_index_in_type"] = idx
            record["split"] = "train"
            append_jsonl(path, record)
    print(f"wrote {sum(counts)} tasks to {path}: {dict(zip(TASK_TYPES, counts))}")


if __name__ == "__main__":
    main()
