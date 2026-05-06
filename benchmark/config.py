import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Optional


DEFAULT_CONFIG_PATH = "configs/local.example.json"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_paths() -> dict:
    base = _repo_root() / "benchmark_data"
    return {
        "raw_data_dir": str(base / "raw"),
        "processed_dir": str(base / "processed"),
        "embeddings_dir": str(base / "embeddings"),
        "results_dir": str(base / "results"),
        "comparison_dir": str(base / "comparisons"),
        "model_weights_dir": str(_repo_root() / "model_weights"),
        "physionet_dir": str(_repo_root() / "physionet.org" / "files"),
        "dataset_roots": {},
    }


def load_config(path: Optional[str] = None) -> SimpleNamespace:
    cfg = _default_paths()

    config_path = path or os.environ.get("ECG_BENCHMARK_CONFIG")
    if config_path:
        with open(config_path, "r") as f:
            user_cfg = json.load(f)
        cfg.update({k: v for k, v in user_cfg.items() if v is not None})

    env_overrides = {
        "ECG_RAW_DATA_DIR": "raw_data_dir",
        "ECG_PROCESSED_DIR": "processed_dir",
        "ECG_EMBEDDINGS_DIR": "embeddings_dir",
        "ECG_RESULTS_DIR": "results_dir",
        "ECG_COMPARISON_DIR": "comparison_dir",
        "ECG_MODEL_WEIGHTS_DIR": "model_weights_dir",
        "PHYSIONET_DATA_DIR": "physionet_dir",
    }
    for env_name, key in env_overrides.items():
        if os.environ.get(env_name):
            cfg[key] = os.environ[env_name]

    cfg.setdefault("dataset_roots", {})
    return SimpleNamespace(**cfg)


def add_config_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=str,
        default=os.environ.get("ECG_BENCHMARK_CONFIG"),
        help="Path to a JSON config with local data, embedding, result, and weight directories.",
    )


def apply_config_to_args(args) -> None:
    cfg = load_config(getattr(args, "config", None))
    for key, value in vars(cfg).items():
        if not hasattr(args, key) or getattr(args, key) is None:
            setattr(args, key, value)


def dataset_root(config, dataset: str, *relative_parts: str) -> str:
    roots = getattr(config, "dataset_roots", {}) or {}
    if dataset in roots and roots[dataset]:
        return roots[dataset]
    return os.path.join(config.physionet_dir, *relative_parts)
