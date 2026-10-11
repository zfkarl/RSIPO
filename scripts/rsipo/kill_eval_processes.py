"""Stop this node's evaluation engines and shard clients by process group.

Arguments are path fragments (the merged model and the evaluation directory). A process is stopped only if its
command line is a vLLM API server or a shard client and contains one of them; its whole process group goes, so
the engine subprocess that holds the GPU goes with it.
"""
import os
import signal
import sys
import time
from pathlib import Path

KINDS = ("vllm.entrypoints.openai.api_server", "evaluate_full_split.py")


def targets(needles):
    own = os.getpgid(0)
    groups = set()
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            cmd = Path(f"/proc/{entry}/cmdline").read_bytes().replace(b"\x00", b" ").decode("utf-8", "replace")
            group = os.getpgid(int(entry))
        except (OSError, ProcessLookupError):
            continue
        if group != own and any(kind in cmd for kind in KINDS) and any(needle in cmd for needle in needles):
            groups.add(group)
    return groups


def main():
    needles = [arg for arg in sys.argv[1:] if arg]
    if not needles:
        raise SystemExit("usage: kill_eval_processes.py PATH_FRAGMENT [...]")
    groups = targets(needles)
    for group in groups:
        try:
            os.killpg(group, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
    deadline = time.monotonic() + 30
    while targets(needles) and time.monotonic() < deadline:
        time.sleep(2)
    for group in targets(needles):
        try:
            os.killpg(group, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    time.sleep(2)
    print(f"stopped {len(groups)} process groups; remaining {len(targets(needles))}", flush=True)


if __name__ == "__main__":
    main()
