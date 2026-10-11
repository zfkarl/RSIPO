"""Evaluate one RSIPO checkpoint on ALFWorld with SEED's evaluation protocol (examples/prompt_agent).

Merged FSDP actor, full seen (140) and unseen (134) splits, seeds 1/2/3, temperature 0.4, no skill in the prompt.
When the run root lists its nodes in ray-nodes.txt (one node, or two reachable by SSH), the games of each round are
striped across one vLLM engine per GPU on those nodes. Otherwise the three rounds run on three local GPU pairs.
"""
import argparse
import datetime
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

from rsipo.eval_shards import shard_games

CODE = Path(__file__).resolve().parents[2]
SPLITS = {"eval_in_distribution": 140, "eval_out_of_distribution": 134}
SEEDS = (1, 2, 3)
GPUS_PER_NODE = 8
BASE_PORT = 60031
CHILDREN = []
LOG_HANDLES = []
REMOTE_GROUPS = []


def chat_extra_args(model):
    """Qwen3 checkpoints are evaluated in non-thinking mode, as they are trained (scripts/rsipo/train.sh)."""
    try:
        model_type = json.loads((Path(model) / "config.json").read_text()).get("model_type", "")
    except (OSError, ValueError):
        return []
    if "qwen3" not in str(model_type).lower():
        return []
    return ["--extra-body-json", json.dumps({"chat_template_kwargs": {"enable_thinking": False}})]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def run(cmd, log, timeout, env, cwd=CODE):
    with log.open("a") as handle:
        process = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        CHILDREN.append(process)
        code = process.wait(timeout=timeout)
    if code:
        raise RuntimeError(f"{log.name}: exit {code}")


def stop_process(process):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=15)


def stop_remote_groups():
    for node, pidfile in REMOTE_GROUPS:
        if not pidfile.is_file():
            continue
        pid = pidfile.read_text().strip()
        if pid.isdigit():
            subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", node, f"kill -TERM -{pid}"],
                           timeout=30, check=False)


def local_ip(nodes):
    ips = set(subprocess.check_output(["hostname", "-I"], text=True).split())
    found = [node for node in nodes if node in ips]
    if len(found) != 1:
        raise RuntimeError(f"This host has {sorted(ips)}, expected one of {nodes}")
    return found[0]


def eval_dir(phase, step, max_steps, seeds):
    """The default rounds keep their original directory; any other seed list gets its own."""
    suffix = "" if tuple(seeds) == (1, 2, 3) else "-seeds" + "_".join(str(seed) for seed in seeds)
    return phase / f"eval-step{step}-max{max_steps}{suffix}"


def eval_nodes(phase):
    listed = phase.parent / "ray-nodes.txt"
    if not listed.exists():
        return None
    nodes = [item.strip() for item in listed.read_text().split(",") if item.strip()]
    if len(nodes) not in (1, 2):
        raise RuntimeError(f"{listed} must list one or two evaluation nodes, found {nodes}")
    return nodes


def start_logged(node, local, script, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("w")
    LOG_HANDLES.append(handle)
    if node == local:
        process = subprocess.Popen(["bash", str(script)], cwd=CODE, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        CHILDREN.append(process)
    else:
        # Keep this SSH session open. setsid would fork and make SSH exit, which looks like a server crash.
        process = subprocess.Popen(
            ["ssh", "-n", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", node, "bash", str(script)],
            stdout=handle, stderr=subprocess.STDOUT, start_new_session=True,
        )
        CHILDREN.append(process)
    return process


def server_script(path, pidfile, gpu, command):
    path.write_text(
        "#!/bin/bash\nset -euo pipefail\n"
        "trap 'kill -TERM -$$' HUP TERM INT\n"
        f"source {shlex.quote(str(CODE / 'scripts/rsipo/env.sh'))}\n"
        f"cd {shlex.quote(str(CODE))}\n"
        f"export CUDA_VISIBLE_DEVICES={gpu} NCCL_IB_DISABLE=1\n"
        f"{shlex.join(command)} &\n"
        "echo $! > " + shlex.quote(str(pidfile)) + "\n"
        "wait $!\n"
    )


def wait_until_ready(url, name, process, timeout=900):
    deadline = time.monotonic() + timeout
    while True:
        if process.poll() is not None:
            raise RuntimeError(f"Model server for {url} exited before readiness")
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if name in [item["id"] for item in json.load(response)["data"]]:
                    return
        except Exception:
            pass
        if time.monotonic() > deadline:
            raise TimeoutError(f"Model server startup for {url}")
        time.sleep(3)


def merge_shards(shard_dirs, round_dir, expected):
    pooled = {}
    lines = []
    manifest = []
    api_errors = []
    for shard_dir in shard_dirs:
        result_files = list((shard_dir / "results").glob("*.json"))
        trajectories = list((shard_dir / "trajectories").glob("*.jsonl"))
        if len(result_files) != 1 or len(trajectories) != 1 or not (shard_dir / "coverage.json").is_file():
            raise RuntimeError(f"Incomplete shard output in {shard_dir}")
        report = json.loads(result_files[0].read_text())
        api_errors.append(report["summary"]["api_error_fallback"]["mean"])
        for subset in report["rounds"][0]["subsets"]:
            item = pooled.setdefault(subset["task_key"], {"name": subset["name"], "task_key": subset["task_key"],
                                                           "success_count": 0, "total": 0})
            item["success_count"] += subset["success_count"]
            item["total"] += subset["total"]
        lines.extend(line for line in trajectories[0].read_text().splitlines() if line)
        manifest.extend(json.loads((shard_dir / "coverage.json").read_text())["manifest"])
    if any(value != 0 for value in api_errors):
        raise RuntimeError("API errors invalidate evaluation")
    subsets = []
    for item in pooled.values():
        item["success_rate"] = item["success_count"] / item["total"] if item["total"] else None
        subsets.append(item)
    present = [item["success_rate"] for item in subsets if item["total"]]
    total = sum(item["total"] for item in subsets)
    wins = sum(item["success_count"] for item in subsets)
    if total != expected or len(lines) != expected or len({item["game_file"] for item in manifest}) != expected:
        raise RuntimeError(f"Shard merge for {round_dir} covered {total} games, expected {expected}")
    (round_dir / "results").mkdir(parents=True, exist_ok=True)
    (round_dir / "trajectories").mkdir(parents=True, exist_ok=True)
    write(round_dir / "results" / "merged.json", {
        "total_envs": expected,
        "summary": {
            "overall_success": {"mean": wins / expected},
            "category_macro_success": {"mean": sum(present) / len(present)},
            "api_error_fallback": {"mean": 0.0},
        },
        "rounds": [{"subsets": subsets}],
    })
    (round_dir / "trajectories" / "merged.jsonl").write_text("\n".join(lines) + "\n")
    counts = {}
    for item in manifest:
        counts[item["task_type"]] = counts.get(item["task_type"], 0) + 1
    write(round_dir / "coverage.json", {"expected_games": expected, "manifest": manifest, "category_counts": counts})


def check_round(round_dir, count):
    files = list((round_dir / "results").glob("*.json"))
    assert len(files) == 1
    report = json.loads(files[0].read_text())
    assert report["total_envs"] == count and len(report["rounds"]) == 1
    assert report["summary"]["api_error_fallback"]["mean"] == 0, "API errors invalidate evaluation"
    trajectories = list((round_dir / "trajectories").glob("*.jsonl"))
    assert len(trajectories) == 1
    games = [
        str(Path(json.loads(line)["initial_info"]["extra.gamefile"]).resolve())
        for line in trajectories[0].read_text().splitlines() if line
    ]
    coverage = json.loads((round_dir / "coverage.json").read_text())
    assert len(games) == len(set(games)) == count
    assert set(games) == {item["game_file"] for item in coverage["manifest"]}
    return report


def evaluate_round(out, model, name, seed, gpus, port, max_steps, results, errors):
    """One seed on one local GPU pair. Used only when the run is not pinned to two nodes."""
    try:
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=gpus, NCCL_IB_DISABLE="1")
        env.pop("NCCL_NET", None)
        server_cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", str(model),
            "--served-model-name", name, "--host", "127.0.0.1", "--port", str(port),
            "--tensor-parallel-size", "2", "--dtype", "bfloat16", "--max-model-len", "4096",
            "--max-num-seqs", "140", "--gpu-memory-utilization", "0.70", "--enforce-eager",
            "--disable-custom-all-reduce", "--disable-log-requests", "--seed", str(seed),
        ]
        with (out / f"server-{seed}.log").open("w") as log:
            server = subprocess.Popen(server_cmd, cwd=CODE, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        CHILDREN.append(server)
        try:
            wait_until_ready(f"http://127.0.0.1:{port}/v1/models", name, server)
            for split, count in SPLITS.items():
                round_dir = out / split / f"round-{seed}"
                cmd = [
                    sys.executable, str(CODE / "scripts/rsipo/evaluate_full_split.py"), "--env-file", "/dev/null",
                    "--base-url", f"http://127.0.0.1:{port}/v1", "--api-key", "EMPTY", "--model-name", name,
                    "--env-num", str(count), "--test-times", "1", "--max-steps", str(max_steps), "--history-length", "5",
                    "--seed", str(seed), "--temperature", "0.4", "--max-completion-tokens", "512",
                    "--request-workers", str(count), "--eval-dataset", split, "--log-dir", str(round_dir),
                    *chat_extra_args(model),
                ]
                eval_env = os.environ.copy()
                eval_env["PYTHONPATH"] = str(CODE)
                run(cmd, out / f"eval-{split}-{seed}.log", 9000, eval_env)
                results[(split, seed)] = check_round(round_dir, count)
        finally:
            stop_process(server)
    except BaseException as exc:
        errors.append(f"seed {seed}: {type(exc).__name__}: {exc}")


def on_node(node, local, command):
    """Run argv on a node; SSH gets one quoted string so the remote shell cannot split an argument."""
    if node == local:
        return command
    return ["ssh", "-n", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", node, shlex.join(command)]


def cleanup_our_processes(nodes, model, out):
    """Stop every engine and shard client of this evaluation on both nodes, then wait for their GPUs to empty."""
    local = local_ip(nodes)
    stop = [sys.executable, str(CODE / "scripts/rsipo/kill_eval_processes.py"), str(model), str(out)]
    for node in nodes:
        result = subprocess.run(on_node(node, local, stop), capture_output=True, text=True, timeout=120, check=False)
        print(f"cleanup {node}: {result.stdout.strip()} {result.stderr.strip()[-200:]}", flush=True)
    query = ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"]
    deadline = time.monotonic() + 180
    while True:
        busy = {}
        for node in nodes:
            result = subprocess.run(on_node(node, local, query), capture_output=True, text=True, timeout=60, check=False)
            busy[node] = sum(1 for line in result.stdout.splitlines() if line.strip())
        if not any(busy.values()) or time.monotonic() > deadline:
            print(f"GPU processes after cleanup: {busy}", flush=True)
            return
        time.sleep(10)


def evaluate_on_nodes(out, model, name, nodes, max_steps, state, base_port=BASE_PORT):
    """Stripe every round across one engine per GPU on the pinned nodes."""
    local = local_ip(nodes)
    engines = [{"node": node, "gpu": gpu, "port": base_port + gpu} for node in nodes for gpu in range(GPUS_PER_NODE)]
    server_dir = out / "servers"
    if server_dir.exists():
        shutil.rmtree(server_dir)
    server_dir.mkdir(parents=True)
    processes = []
    try:
        state("starting_servers", nodes=nodes, engines=len(engines), local_node=local)
        for engine in engines:
            label = f"{engine['node'].split('.')[-1]}-gpu{engine['gpu']}"
            pidfile = server_dir / f"{label}.pid"
            script = server_dir / f"{label}.sh"
            server_script(script, pidfile, engine["gpu"], [
                sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", str(model),
                "--served-model-name", name, "--host", "0.0.0.0", "--port", str(engine["port"]),
                "--tensor-parallel-size", "1", "--dtype", "bfloat16", "--max-model-len", "4096",
                "--max-num-seqs", "16", "--gpu-memory-utilization", "0.70",
                "--disable-custom-all-reduce", "--disable-log-requests", "--seed", "0",
            ])
            if engine["node"] != local:
                REMOTE_GROUPS.append((engine["node"], pidfile))
            processes.append(start_logged(engine["node"], local, script, server_dir / f"{label}.log"))
        for engine, process in zip(engines, processes):
            wait_until_ready(f"http://{engine['node']}:{engine['port']}/v1/models", name, process)
        state("evaluating", nodes=nodes, engines=len(engines), gpus=len(engines))
        results = {}
        for seed in SEEDS:
            for split, count in SPLITS.items():
                shard_root = out / "shards" / split / f"seed-{seed}"
                if shard_root.exists():
                    shutil.rmtree(shard_root)
                errors = []
                threads = []

                def run_shard(index, engine, errors=errors):
                    try:
                        games = len(shard_games(range(count), len(engines), index))
                        shard_dir = shard_root / f"shard-{index:02d}"
                        script = shard_dir / "run.sh"
                        shard_dir.mkdir(parents=True)
                        command = [
                            sys.executable, str(CODE / "scripts/rsipo/evaluate_full_split.py"), "--env-file", "/dev/null",
                            "--base-url", f"http://127.0.0.1:{engine['port']}/v1", "--api-key", "EMPTY", "--model-name", name,
                            "--env-num", str(games), "--test-times", "1", "--max-steps", str(max_steps), "--history-length", "5",
                            "--seed", str(seed), "--temperature", "0.4", "--max-completion-tokens", "512",
                            "--request-workers", str(games), "--eval-dataset", split, "--log-dir", str(shard_dir),
                            *chat_extra_args(model),
                        ]
                        script.write_text(
                            "#!/bin/bash\nset -euo pipefail\n"
                            "trap 'kill -TERM -$$' HUP TERM INT\n"
                            f"source {shlex.quote(str(CODE / 'scripts/rsipo/env.sh'))}\n"
                            f"cd {shlex.quote(str(CODE))}\n"
                            f"export RSIPO_EVAL_SHARD_INDEX={index} RSIPO_EVAL_SHARD_COUNT={len(engines)}\n"
                            f"{shlex.join(command)} &\n"
                            "echo $! > " + shlex.quote(str(shard_dir / "client.pid")) + "\n"
                            "wait $!\n"
                        )
                        if engine["node"] != local:
                            REMOTE_GROUPS.append((engine["node"], shard_dir / "client.pid"))
                        process = start_logged(engine["node"], local, script, shard_dir / "client.log")
                        code = process.wait(timeout=9000)
                        if code:
                            raise RuntimeError(f"exit {code}")
                    except BaseException as exc:
                        errors.append(f"{split} seed {seed} shard {index}: {type(exc).__name__}: {exc}")

                for index, engine in enumerate(engines):
                    thread = threading.Thread(target=run_shard, args=(index, engine))
                    thread.start()
                    threads.append(thread)
                for thread in threads:
                    thread.join()
                if errors:
                    raise RuntimeError("; ".join(errors))
                round_dir = out / split / f"round-{seed}"
                if round_dir.exists():
                    shutil.rmtree(round_dir)
                merge_shards([shard_root / f"shard-{index:02d}" for index in range(len(engines))], round_dir, count)
                results[(split, seed)] = check_round(round_dir, count)
                state("evaluating", nodes=nodes, finished=f"{split} seed {seed}")
        return results
    finally:
        stop_remote_groups()
        for process in processes:
            if process.poll() is None:
                stop_process(process)
        cleanup_our_processes(nodes, model, out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-dir", required=True)
    parser.add_argument("--step", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=30, help="environment steps allowed per evaluation episode")
    parser.add_argument("--name", required=True)
    parser.add_argument("--base-port", type=int, default=BASE_PORT)
    parser.add_argument("--gpu-pairs", default="0,1;2,3;4,5")
    parser.add_argument("--seeds", default="1,2,3", help="comma-separated round seeds")
    parser.add_argument("--model-dir", default=None,
                        help="evaluate this Hugging Face model instead of merging the step's FSDP checkpoint")
    args = parser.parse_args()
    global SEEDS
    SEEDS = tuple(int(item) for item in args.seeds.split(",") if item.strip())

    phase = Path(args.phase_dir)
    checkpoint = phase / "checkpoints" / f"global_step_{args.step}" / "actor"
    out = eval_dir(phase, args.step, args.max_steps, SEEDS)
    out.mkdir(parents=True, exist_ok=True)
    model = Path(args.model_dir) if args.model_dir else phase / f"model-hf-step{args.step}"
    if args.model_dir:
        checkpoint = model
    summary_path = out / "summary.json"
    if summary_path.exists():
        print(summary_path.read_text())
        return
    limits_file = phase.parent / "eval-limits.json"
    if limits_file.exists():
        limits = json.loads(limits_file.read_text())
        if args.max_steps not in limits["allowed_max_steps"]:
            write(summary_path, {"skipped": True, "reason": limits.get("reason", ""), "checkpoint_step": args.step,
                                 "max_steps": args.max_steps, "splits": {}})
            print(f"Skipped: {args.max_steps}-step evaluation is not allowed for this run")
            return

    def state(stage, **extra):
        write(out / "status.json", {"stage": stage, "pid": os.getpid(), "checkpoint": str(checkpoint),
                                     "updated_at": datetime.datetime.now().astimezone().isoformat(), **extra})

    nodes = eval_nodes(phase)
    protocol = {
        "checkpoint": str(checkpoint), "checkpoint_step": args.step, "round_seeds": list(SEEDS), "splits": SPLITS,
        "coverage": "each task exactly once per round", "temperature": 0.4, "max_completion_tokens": 512,
        "max_steps": args.max_steps, "history_length": 5, "test_skill_prompt": False,
        "primary_metric": f"six-category macro success averaged over {len(SEEDS)} rounds",
        "engines": "one vLLM per GPU on ray-nodes.txt" if nodes else "three local tensor-parallel pairs",
        "nodes": nodes or ["local"],
        "reference": "examples/prompt_agent/local_vllm_alfworld.py",
    }
    write(out / "protocol.json", protocol)
    try:
        if args.model_dir is None:
            shards = sorted(checkpoint.glob("model_world_size_*_rank_0.pt"))
            found = {int(match.group(1)) for path in shards if (match := re.search(r"world_size_(\d+)_", path.name))}
            if len(found) != 1:
                raise FileNotFoundError(f"Expected one FSDP world size in {checkpoint}, found {sorted(found)}")
            world = found.pop()
            files = [checkpoint / f"{kind}_world_size_{world}_rank_{rank}.pt" for kind in ("model", "optim", "extra_state") for rank in range(world)]
            missing = [str(path) for path in files if not path.is_file() or path.stat().st_size == 0]
            if missing:
                raise FileNotFoundError(f"Incomplete checkpoint: {missing[:3]}")
            if not (model / "config.json").exists():
                state("merging_checkpoint")
                cpu = os.environ.copy()
                cpu["CUDA_VISIBLE_DEVICES"] = ""
                run([sys.executable, str(CODE / "scripts/model_merger.py"), "merge", "--backend", "fsdp",
                     "--local_dir", str(checkpoint), "--target_dir", str(model)], out / "merge.log", 3600, cpu)
        assert (model / "config.json").exists() and list(model.glob("*.safetensors"))

        if nodes:
            results = evaluate_on_nodes(out, model, args.name, nodes, args.max_steps, state, args.base_port)
        else:
            pairs = args.gpu_pairs.split(";")
            assert len(pairs) == len(SEEDS)
            state("evaluating", gpu_pairs=pairs)
            results, errors, threads = {}, [], []
            for index, seed in enumerate(SEEDS):
                thread = threading.Thread(
                    target=evaluate_round,
                    args=(out, model, args.name, seed, pairs[index], args.base_port + index, args.max_steps, results, errors),
                )
                thread.start()
                threads.append(thread)
            for thread in threads:
                thread.join()
            if errors:
                raise RuntimeError("; ".join(errors))

        summary = {"checkpoint": str(checkpoint), "checkpoint_step": args.step, "max_steps": args.max_steps, "splits": {}}
        for split, count in SPLITS.items():
            reports = [results[(split, seed)] for seed in SEEDS]
            macros = [report["summary"]["category_macro_success"]["mean"] * 100 for report in reports]
            overall = [report["summary"]["overall_success"]["mean"] * 100 for report in reports]
            summary["splits"][split] = {
                "unique_tasks": count, "rounds": len(SEEDS), "episodes": len(SEEDS) * count,
                "macro_pct_by_round": macros, "macro_mean_pct": sum(macros) / len(macros),
                "overall_pct_by_round": overall, "overall_mean_pct": sum(overall) / len(overall),
                "category_results_by_round": [report["rounds"][0]["subsets"] for report in reports],
                "api_error_fallbacks": 0,
            }
        write(summary_path, summary)
        state("complete", summary=str(summary_path))
    except BaseException as exc:
        state("failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        stop_remote_groups()
        for process in CHILDREN:
            if process.poll() is None:
                try:
                    stop_process(process)
                except ProcessLookupError:
                    pass


if __name__ == "__main__":
    main()
