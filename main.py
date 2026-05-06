import argparse
import subprocess
import sys


DEFAULT_STAGES = ["preprocess", "format", "embed", "eval", "compare"]


def run_command(cmd, dry_run=False):
    print("\n" + " ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def add_config(cmd, config):
    if config:
        cmd.extend(["--config", config])
    return cmd


def parse_args():
    parser = argparse.ArgumentParser(description="Run the full ECG benchmark workflow.")
    parser.add_argument("--config", type=str, default=None, help="Path to local JSON config.")
    parser.add_argument(
        "--stages",
        nargs="+",
        default=DEFAULT_STAGES,
        choices=["all", "download", "preprocess", "format", "embed", "eval", "compare"],
        help="Workflow stages to run in order. Use 'all' for preprocess, format, embed, eval, compare.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")

    parser.add_argument("--download-dir", type=str, default=None, help="Override data download directory.")
    parser.add_argument("--physionet-user", type=str, default=None, help="PhysioNet username for restricted datasets.")
    parser.add_argument("--skip-restricted", action="store_true", help="Skip restricted datasets during download.")

    parser.add_argument("--embed-chunks", type=int, default=6, help="Number of embedding chunks to run.")
    parser.add_argument("--embed-batch-size", type=int, default=None, help="Embedding batch size override.")
    parser.add_argument("--embed-num-workers", type=int, default=None, help="Embedding DataLoader worker override.")

    parser.add_argument("--eval-run-type", choices=["full", "specified"], default="full", help="Eval run type.")
    parser.add_argument("--eval-data", type=str, default=None, help="Dataset/task for specified eval.")
    parser.add_argument("--eval-model", type=str, default=None, help="Model for specified eval.")
    parser.add_argument("--eval-train-pct", type=float, default=None, help="Train percentage for specified eval.")

    parser.add_argument("--compare-data", type=str, default=None, help="Dataset/task for one comparison run.")
    parser.add_argument("--compare-best-model", type=str, default="D_BETA", help="Reference model for comparison.")
    parser.add_argument("--compare-train-pct", type=float, default=1.0, help="Train percentage for comparison.")
    return parser.parse_args()


def normalize_stages(stages):
    if "all" in stages:
        return DEFAULT_STAGES
    return stages


def main():
    args = parse_args()
    stages = normalize_stages(args.stages)

    for stage in stages:
        if stage == "download":
            cmd = ["bash", "scripts/download_public_data.sh"]
            if args.download_dir:
                cmd.extend(["--dir", args.download_dir])
            if args.physionet_user:
                cmd.extend(["--physionet-user", args.physionet_user])
            if args.skip_restricted:
                cmd.append("--skip-restricted")
            run_command(cmd, dry_run=args.dry_run)

        elif stage == "preprocess":
            cmd = add_config([sys.executable, "-m", "benchmark.preprocess.process_all_public"], args.config)
            run_command(cmd, dry_run=args.dry_run)

        elif stage == "format":
            cmd = add_config([sys.executable, "-m", "benchmark.preprocess.public.format_dataset"], args.config)
            run_command(cmd, dry_run=args.dry_run)

        elif stage == "embed":
            for chunk in range(args.embed_chunks):
                cmd = add_config([sys.executable, "-m", "benchmark.preprocess.public.embed_data"], args.config)
                cmd.extend(["--chunk", str(chunk)])
                if args.embed_batch_size is not None:
                    cmd.extend(["--batch_size", str(args.embed_batch_size)])
                if args.embed_num_workers is not None:
                    cmd.extend(["--num_workers", str(args.embed_num_workers)])
                run_command(cmd, dry_run=args.dry_run)

        elif stage == "eval":
            cmd = add_config([sys.executable, "-m", "benchmark.eval"], args.config)
            cmd.extend(["--run_type", args.eval_run_type])
            if args.eval_run_type == "specified":
                missing = [
                    name
                    for name, value in {
                        "--eval-data": args.eval_data,
                        "--eval-model": args.eval_model,
                        "--eval-train-pct": args.eval_train_pct,
                    }.items()
                    if value is None
                ]
                if missing:
                    raise SystemExit(f"specified eval requires: {', '.join(missing)}")
                cmd.extend([
                    "--data", args.eval_data,
                    "--model", args.eval_model,
                    "--train_pct", str(args.eval_train_pct),
                ])
            run_command(cmd, dry_run=args.dry_run)

        elif stage == "compare":
            cmd = add_config([sys.executable, "-m", "benchmark.compare"], args.config)
            if args.compare_data:
                cmd.extend([
                    "--data", args.compare_data,
                    "--best_model", args.compare_best_model,
                    "--train_pct", str(args.compare_train_pct),
                ])
            run_command(cmd, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
