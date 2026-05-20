import json
import os
from collections.abc import Iterable, Sequence

import numpy as np

from src.models.weights import get_weights_path


def model_done_path(split_dir: str, model_name: str) -> str:
    return os.path.join(split_dir, f"{model_name}_done.json")


def model_weight_metadata(model_name: str, weights_dir: str):
    if "Random" in model_name:
        return None

    weights = get_weights_path(model_name, weights_dir)
    if isinstance(weights, tuple):
        return list(weights)

    return weights


def expected_model_done_meta(
    *,
    dataset_name: str,
    split: str,
    model_name: str,
    seed: int,
    weights_dir: str,
    emb_dtype,
) -> dict:
    return {
        "dataset": dataset_name,
        "split": split,
        "model": model_name,
        "seed": seed,
        "model_weights": model_weight_metadata(model_name, weights_dir),
        "emb_dtype": str(np.dtype(emb_dtype)),
    }


def expected_meta_by_model(
    *,
    dataset_name: str,
    split: str,
    model_names: Iterable[str],
    seed: int,
    weights_dir: str,
    emb_dtype,
) -> dict[str, dict]:
    return {
        model_name: expected_model_done_meta(
            dataset_name=dataset_name,
            split=split,
            model_name=model_name,
            seed=seed,
            weights_dir=weights_dir,
            emb_dtype=emb_dtype,
        )
        for model_name in model_names
    }


def model_outputs_exist(
    split_dir: str,
    model_name: str,
    expected_meta: dict | None = None,
) -> bool:
    output_files_exist = (
        os.path.exists(os.path.join(split_dir, f"{model_name}.npy"))
        and os.path.exists(os.path.join(split_dir, f"{model_name}_y.npy"))
        and os.path.exists(model_done_path(split_dir, model_name))
    )

    if not output_files_exist:
        return False

    if expected_meta is None:
        return True

    try:
        with open(model_done_path(split_dir, model_name), "r") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False

    if payload.get("done") is not True:
        return False

    for key, value in expected_meta.items():
        if payload.get(key) != value:
            return False

    return True


def pending_models_for_split(
    out_root: str,
    dataset_name: str,
    split: str,
    model_names: Iterable[str],
    expected_meta_by_model: dict[str, dict] | None = None,
) -> list[str]:
    split_dir = os.path.join(out_root, dataset_name, split)
    return [
        mname
        for mname in model_names
        if not model_outputs_exist(
            split_dir,
            mname,
            None if expected_meta_by_model is None else expected_meta_by_model[mname],
        )
    ]


def models_needed_for_embedding(
    *,
    out_root: str,
    dataset_names: Iterable[str],
    splits: Iterable[str],
    model_names: Sequence[str],
    seed: int,
    weights_dir: str,
    emb_dtype,
) -> list[str]:
    needed = set()

    for dataset_name in dataset_names:
        for split in splits:
            expected_meta = expected_meta_by_model(
                dataset_name=dataset_name,
                split=split,
                model_names=model_names,
                seed=seed,
                weights_dir=weights_dir,
                emb_dtype=emb_dtype,
            )
            needed.update(
                pending_models_for_split(
                    out_root,
                    dataset_name,
                    split,
                    model_names,
                    expected_meta,
                )
            )

    return [model_name for model_name in model_names if model_name in needed]


def save_model_done(split_dir: str, model_name: str, meta: dict) -> None:
    done_payload = dict(meta)
    done_payload["model"] = model_name
    done_payload["done"] = True

    with open(model_done_path(split_dir, model_name), "w") as f:
        json.dump(done_payload, f, indent=2)
