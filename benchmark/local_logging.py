import csv
import json
import os
from datetime import datetime


def make_run_dir(args) -> str:
    run_name = args.name or f"{args.data}_{args.model}_{args.train_pct}_seed{args.seed}"
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(args.results_dir, "runs", f"{timestamp}_{run_name}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def append_jsonl(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(payload, default=str) + "\n")


def write_metrics_csv(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        return

    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
