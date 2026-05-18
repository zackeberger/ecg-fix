# src/metrics/stats_tests.py

import os
import pickle
import hashlib
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.metrics.utils import (
    MODEL_DISPLAY_NAMES,
    MODEL_ORDER,
    metric_stats_path,
    output_path,
    safe_name,
)

DEFAULT_N_BOOT = 1000
DEFAULT_N_PERM = 1000
DEFAULT_ALPHA = 0.05


P_TEST_RUNS = [
    ("PTBXL_super", 0.01),
    ("PTBXL_super", 0.1),
    ("PTBXL_super", 1.0),

    ("PTBXL_sub", 0.01),
    ("PTBXL_sub", 0.1),
    ("PTBXL_sub", 1.0),

    ("PTBXL_form", 0.01),
    ("PTBXL_form", 0.1),
    ("PTBXL_form", 1.0),

    ("PTBXL_rhythm", 0.01),
    ("PTBXL_rhythm", 0.1),
    ("PTBXL_rhythm", 1.0),

    ("CPSC", 0.01),
    ("CPSC", 0.1),
    ("CPSC", 1.0),

    ("CSN", 0.01),
    ("CSN", 0.1),
    ("CSN", 1.0),

    ("ECHO_NEXT", 0.01),
    ("ECHO_NEXT", 0.1),
    ("ECHO_NEXT", 1.0),
   
]

TARGET_LABELS = [
    "NORM",
    "IMI",
    "AMI",
    "STTC",
    "LVH",
    "CLBBB",
    "STE_",
    "TAB_",
    "PRC(S)",
    "shd_moderate_or_greater_flag",
    "lvef_lte_45_flag",
    "tricuspid_regurgitation_moderate_or_greater_flag",
]


def _load_metric_stats(path: str) -> dict[str, Any] | None:
    if not os.path.exists(path):
        print(f"Skipping missing metric file: {path}")
        return None

    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Skipping unreadable metric file: {path} ({e})")
        return None


def _load_available_metric_stats(
    results_dir: str,
    dataset: str,
    train_pct: float,
) -> dict[str, dict[str, Any]]:
    stats = {}

    for model in MODEL_ORDER:
        path = metric_stats_path(results_dir, dataset, train_pct, model)
        obj = _load_metric_stats(path)

        if obj is not None:
            stats[model] = obj

    return stats


def _metric_key_for_best_model(label: str | None) -> str:
    if label is None:
        return "macro_auc_boot_mean"

    return f"{label}_auc_boot_mean"


def _choose_best_model_from_saved_auc(
    metric_stats: dict[str, dict[str, Any]],
    label: str | None,
) -> str | None:
    """
    Choose best model using saved bootstrap AUROC means.

    Macro:
        macro_auc_boot_mean

    Label:
        {label}_auc_boot_mean
    """
    metric_key = _metric_key_for_best_model(label)

    best_model = None
    best_score = -np.inf

    for model, stats in metric_stats.items():
        value = stats.get(metric_key)

        if value is None:
            continue

        try:
            value = float(value)
        except Exception:
            continue

        if not np.isfinite(value):
            continue

        if value > best_score:
            best_score = value
            best_model = model

    if best_model is None:
        print(f"No saved metric found for best-model selection: {metric_key}")
        return None

    print(
        f"Best model by saved {metric_key}: "
        f"{best_model} ({best_score:.4f})"
    )

    return best_model

def _seed_from_job(*parts) -> int:
    s = "__".join(str(p) for p in parts)
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def _load_output(path: str) -> dict[str, Any] | None:
    if not os.path.exists(path):
        print(f"Skipping missing file: {path}")
        return None

    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Skipping unreadable file: {path} ({e})")
        return None


def _load_available_results(
    results_dir: str,
    dataset: str,
    train_pct: float,
) -> dict[str, dict[str, Any]]:
    results = {}

    for model in MODEL_ORDER:
        path = output_path(results_dir, dataset, train_pct, model)
        obj = _load_output(path)

        if obj is not None:
            results[model] = obj

    return results


def _get_vocab(results: dict[str, dict[str, Any]]) -> dict | None:
    for obj in results.values():
        vocab = obj.get("vocab")

        if vocab is not None:
            return vocab

    return None


def _as_label_index(vocab: dict, label: str) -> int | None:
    if label not in vocab:
        return None

    return int(vocab[label])


def _metric_macro(y_true: np.ndarray, y_pred: np.ndarray, metric: str) -> float:
    scores = []

    for i in range(y_true.shape[1]):
        try:
            if metric == "auc":
                score = roc_auc_score(y_true[:, i], y_pred[:, i])
            elif metric == "auprc":
                score = average_precision_score(y_true[:, i], y_pred[:, i])
            else:
                raise ValueError(f"Unknown metric: {metric}")

            scores.append(score)
        except Exception:
            scores.append(np.nan)

    return float(np.nanmean(scores))


def _metric_label(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: str,
    label_idx: int,
) -> float:
    if metric == "auc":
        return float(roc_auc_score(y_true[:, label_idx], y_pred[:, label_idx]))

    if metric == "auprc":
        return float(average_precision_score(y_true[:, label_idx], y_pred[:, label_idx]))

    raise ValueError(f"Unknown metric: {metric}")


def _compute_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: str,
    label_idx: int | None,
) -> float:
    if label_idx is None:
        return _metric_macro(y_true, y_pred, metric)

    return _metric_label(y_true, y_pred, metric, label_idx)


def _valid_label_indices(y_true: np.ndarray) -> list[int]:
    valid = []

    for i in range(y_true.shape[1]):
        pos_idx = np.flatnonzero(y_true[:, i] == 1)
        neg_idx = np.flatnonzero(y_true[:, i] == 0)

        if len(pos_idx) > 0 and len(neg_idx) > 0:
            valid.append(i)

    return valid


def _metric_label_on_indices(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: str,
    label_idx: int,
    indices: np.ndarray,
) -> float:
    if metric == "auc":
        return float(roc_auc_score(y_true[indices, label_idx], y_pred[indices, label_idx]))

    if metric == "auprc":
        return float(average_precision_score(y_true[indices, label_idx], y_pred[indices, label_idx]))

    raise ValueError(f"Unknown metric: {metric}")


def _stratified_bootstrap_label_indices(
    y_true: np.ndarray,
    label_idx: int,
    rng: np.random.Generator,
) -> np.ndarray | None:
    pos_idx = np.flatnonzero(y_true[:, label_idx] == 1)
    neg_idx = np.flatnonzero(y_true[:, label_idx] == 0)

    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return None

    bs_pos = rng.choice(pos_idx, size=len(pos_idx), replace=True)
    bs_neg = rng.choice(neg_idx, size=len(neg_idx), replace=True)
    return np.concatenate([bs_pos, bs_neg])


def _stratified_bootstrap_diff_once(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    metric: str,
    label_idx: int | None,
    rng: np.random.Generator,
) -> float:
    if label_idx is not None:
        idx = _stratified_bootstrap_label_indices(y_true, label_idx, rng)

        if idx is None:
            return np.nan

        a = _metric_label_on_indices(y_true, y_pred_a, metric, label_idx, idx)
        b = _metric_label_on_indices(y_true, y_pred_b, metric, label_idx, idx)
        return a - b

    deltas = []

    for i in _valid_label_indices(y_true):
        idx = _stratified_bootstrap_label_indices(y_true, i, rng)

        if idx is None:
            continue

        try:
            a = _metric_label_on_indices(y_true, y_pred_a, metric, i, idx)
            b = _metric_label_on_indices(y_true, y_pred_b, metric, i, idx)
            delta = a - b

            if np.isfinite(delta):
                deltas.append(delta)
        except Exception:
            continue

    if not deltas:
        return np.nan

    return float(np.mean(deltas))


def _bootstrap_diff(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    metric: str,
    label_idx: int | None,
    n_boot: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    deltas = []

    for _ in range(n_boot):
        try:
            delta = _stratified_bootstrap_diff_once(
                y_true=y_true,
                y_pred_a=y_pred_a,
                y_pred_b=y_pred_b,
                metric=metric,
                label_idx=label_idx,
                rng=rng,
            )

            if np.isfinite(delta):
                deltas.append(delta)
        except Exception:
            continue

    if not deltas:
        return np.nan, np.nan, np.nan

    deltas = np.asarray(deltas, dtype=float)

    return (
        float(np.mean(deltas)),
        float(np.percentile(deltas, 2.5)),
        float(np.percentile(deltas, 97.5)),
    )


def _permutation_test(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    metric: str,
    label_idx: int | None,
    n_perm: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    try:
        obs_a = _compute_metric(y_true, y_pred_a, metric, label_idx)
        obs_b = _compute_metric(y_true, y_pred_b, metric, label_idx)
        obs_delta = obs_a - obs_b
    except Exception:
        return np.nan, np.nan

    if not np.isfinite(obs_delta):
        return np.nan, np.nan

    n = len(y_true)
    perm_deltas = []

    for _ in range(n_perm):
        swap = rng.random(n) < 0.5

        pa = y_pred_a.copy()
        pb = y_pred_b.copy()

        pa[swap], pb[swap] = pb[swap].copy(), pa[swap].copy()

        try:
            a = _compute_metric(y_true, pa, metric, label_idx)
            b = _compute_metric(y_true, pb, metric, label_idx)
            delta = a - b

            if np.isfinite(delta):
                perm_deltas.append(delta)
        except Exception:
            continue

    if not perm_deltas:
        return obs_delta, np.nan

    perm_deltas = np.asarray(perm_deltas, dtype=float)
    p_value = (1.0 + np.sum(np.abs(perm_deltas) >= abs(obs_delta))) / (
        len(perm_deltas) + 1.0
    )

    return float(obs_delta), float(p_value)


def _format_diff(mean: float, lo: float, hi: float, digits: int = 3) -> str:
    if not np.isfinite(mean) or not np.isfinite(lo) or not np.isfinite(hi):
        return "---"

    return f"{mean:+.{digits}f} [{lo:+.{digits}f}, {hi:+.{digits}f}]"


def _negate_formatted_diff(value: str) -> str:
    if value == "---":
        return "---"

    match = re.fullmatch(
        r"([+-]\d+\.(\d+)) \[([+-]\d+\.\d+), ([+-]\d+\.\d+)\]",
        value,
    )
    if match is None:
        return "---"

    mean_s, decimals, lo_s, hi_s = match.groups()
    digits = len(decimals)
    mean = float(mean_s)
    lo = float(lo_s)
    hi = float(hi_s)
    return f"{-mean:+.{digits}f} [{-hi:+.{digits}f}, {-lo:+.{digits}f}]"


def _format_p(p: float, digits: int = 4) -> str:
    if not np.isfinite(p):
        return "---"

    return f"{p:.{digits}f}"


def _format_bool(x) -> str:
    if x is None:
        return "---"

    return str(bool(x))


def _compare_one_pair(job: dict) -> dict:
    dataset = job["dataset"]
    train_pct = job["train_pct"]
    target = job["target"]
    metric = job["metric"]
    model_a = job["model_a"]
    model_b = job["model_b"]
    label_idx = job["label_idx"]
    n_boot = job["n_boot"]
    n_perm = job["n_perm"]
    alpha = job["alpha"]
    results_dir = job["results_dir"]
    seed = job["seed"]

    path_a = output_path(results_dir, dataset, train_pct, model_a)
    path_b = output_path(results_dir, dataset, train_pct, model_b)

    obj_a = _load_output(path_a)
    obj_b = _load_output(path_b)

    base = {
        "dataset": dataset,
        "train_pct": train_pct,
        "target": target,
        "metric": metric,
        "model_a": model_a,
        "model_b": model_b,
        "bootstrap_diff": "---",
        "p_value": "---",
        "within_noise": "---",
        "sig": "---",
    }

    if obj_a is None or obj_b is None:
        return base

    try:
        y_true = np.asarray(obj_a["y_true"])
        y_true_b = np.asarray(obj_b["y_true"])

        if y_true.shape != y_true_b.shape or not np.array_equal(y_true, y_true_b):
            print(f"Skipping mismatched y_true: {dataset} {train_pct} {model_a} {model_b}")
            return base

        y_pred_a = np.asarray(obj_a["y_pred"])
        y_pred_b = np.asarray(obj_b["y_pred"])

        if y_pred_a.shape != y_pred_b.shape:
            print(f"Skipping mismatched y_pred: {dataset} {train_pct} {model_a} {model_b}")
            return base


        boot_seed = _seed_from_job(
            "bootstrap",
            seed,
            dataset,
            train_pct,
            target,
            metric,
            model_a,
            model_b,
        )

        perm_seed = _seed_from_job(
            "permutation",
            seed,
            dataset,
            train_pct,
            target,
            metric,
            model_a,
            model_b,
        )

        boot_rng = np.random.default_rng(boot_seed)
        perm_rng = np.random.default_rng(perm_seed)

        mean, lo, hi = _bootstrap_diff(
            y_true=y_true,
            y_pred_a=y_pred_a,
            y_pred_b=y_pred_b,
            metric=metric,
            label_idx=label_idx,
            n_boot=n_boot,
            rng=boot_rng,
        )

        _, p = _permutation_test(
            y_true=y_true,
            y_pred_a=y_pred_a,
            y_pred_b=y_pred_b,
            metric=metric,
            label_idx=label_idx,
            n_perm=n_perm,
            rng=perm_rng,
        )

        within_noise = None
        if np.isfinite(lo) and np.isfinite(hi):
            within_noise = bool(lo <= 0 <= hi)

        sig = None
        if np.isfinite(p):
            sig = bool(p <= alpha)

        base.update(
            {
                "bootstrap_diff": _format_diff(mean, lo, hi),
                "p_value": _format_p(p),
                "within_noise": _format_bool(within_noise),
                "sig": _format_bool(sig),
            }
        )

        return base

    except Exception as e:
        print(f"Skipping failed comparison {dataset} {train_pct} {target} {metric} {model_a} vs {model_b}: {e}")
        return base


def _empty_matrix() -> pd.DataFrame:
    display_names = [MODEL_DISPLAY_NAMES.get(m, m) for m in MODEL_ORDER]
    return pd.DataFrame("---", index=display_names, columns=display_names)


def _matrix_from_rows(rows: list[dict], value_key: str) -> pd.DataFrame:
    df = _empty_matrix()

    for m in MODEL_ORDER:
        name = MODEL_DISPLAY_NAMES.get(m, m)
        df.loc[name, name] = "---"

    for row in rows:
        a = MODEL_DISPLAY_NAMES.get(row["model_a"], row["model_a"])
        b = MODEL_DISPLAY_NAMES.get(row["model_b"], row["model_b"])
        value = row[value_key]

        if value_key == "bootstrap_diff":
            # Directional: row model minus column model.
            df.loc[a, b] = value
            df.loc[b, a] = _negate_formatted_diff(value)
        else:
            # p-values and boolean decisions are symmetric.
            df.loc[a, b] = value
            df.loc[b, a] = value

    return df


def _save_matrices(
    rows: list[dict],
    comparison_dir: str,
    dataset: str,
    train_pct: float,
    target: str,
    metric: str,
    alpha: float,
) -> None:
    out_dir = os.path.join(
        comparison_dir,
        "stats_tests",
        safe_name(dataset),
        f"trainpct_{safe_name(train_pct)}",
        safe_name(target),
        metric,
    )
    os.makedirs(out_dir, exist_ok=True)

    outputs = {
        "p_values": _matrix_from_rows(rows, "p_value"),
        "bootstrap_diff": _matrix_from_rows(rows, "bootstrap_diff"),
        "within_noise": _matrix_from_rows(rows, "within_noise"),
        f"sig_alpha_{safe_name(alpha)}": _matrix_from_rows(rows, "sig"),
    }

    for name, df in outputs.items():
        path = os.path.join(out_dir, f"{name}.csv")
        df.to_csv(path)
        print(f"Saved p-test table: {path}")


def _targets_for_run(
    results_dir: str,
    dataset: str,
    train_pct: float,
) -> list[dict]:
    """
    Build macro + label targets.

    Best model is selected from saved metric files:
        macro_auc_boot_mean
        {label}_auc_boot_mean

    No AUROC is recomputed for best-model selection.
    """
    results = _load_available_results(results_dir, dataset, train_pct)
    metric_stats = _load_available_metric_stats(results_dir, dataset, train_pct)

    if not results:
        print(f"No output files found for {dataset}, train_pct={train_pct}")
        return []

    if not metric_stats:
        print(f"No metric files found for {dataset}, train_pct={train_pct}")
        return []

    vocab = _get_vocab(results)

    targets = []

    macro_best_model = _choose_best_model_from_saved_auc(
        metric_stats=metric_stats,
        label=None,
    )

    if macro_best_model is not None:
        targets.append(
            {
                "target": "macro",
                "label_idx": None,
                "best_model": macro_best_model,
            }
        )

    if vocab is None:
        print(
            f"No vocab found for {dataset}, train_pct={train_pct}; "
            "only macro tests can run."
        )
        return targets

    for label in TARGET_LABELS:
        label_idx = _as_label_index(vocab, label)

        if label_idx is None:
            continue

        label_best_model = _choose_best_model_from_saved_auc(
            metric_stats=metric_stats,
            label=label,
        )

        if label_best_model is None:
            continue

        targets.append(
            {
                "target": label,
                "label_idx": label_idx,
                "best_model": label_best_model,
            }
        )

    return targets
    
def _jobs_for_target(
    results_dir: str,
    dataset: str,
    train_pct: float,
    target: str,
    label_idx: int | None,
    best_model: str,
    metric: str,
    n_boot: int,
    n_perm: int,
    alpha: float,
    seed: int,
) -> list[dict]:
    if best_model not in MODEL_ORDER:
        print(f"Skipping unknown best model {best_model} for {dataset} {target}")
        return []

    jobs = []

    for other_model in MODEL_ORDER:
        if other_model == best_model:
            continue

        jobs.append(
            {
                "results_dir": results_dir,
                "dataset": dataset,
                "train_pct": train_pct,
                "target": target,
                "label_idx": label_idx,
                "metric": metric,
                "model_a": best_model,
                "model_b": other_model,
                "n_boot": n_boot,
                "n_perm": n_perm,
                "alpha": alpha,
                "seed": seed,
            }
        )

    return jobs


def export_stats_tests_for_run(
    config: dict,
    dataset: str,
    train_pct: float,
    alpha: float = DEFAULT_ALPHA,
    n_boot: int | None = None,
    n_perm: int | None = None,
) -> None:
    results_dir = config["results_dir"]
    comparison_dir = config["tables_dir"]

    n_boot = int(
        n_boot
        if n_boot is not None
        else config.get("stats_tests", {}).get("n_boot", DEFAULT_N_BOOT)
    )
    n_perm = int(
        n_perm
        if n_perm is not None
        else config.get("stats_tests", {}).get("n_perm", DEFAULT_N_PERM)
    )
    max_workers = int(config.get("multi_process_eval", 1))
    seed = int(config.get("seed", 42))

    targets = _targets_for_run(
        results_dir=results_dir,
        dataset=dataset,
        train_pct=train_pct,
    )

    for target_info in targets:
        target = target_info["target"]
        label_idx = target_info["label_idx"]
        target_best_model = target_info["best_model"]

        for metric in ["auc", "auprc"]:
            jobs = _jobs_for_target(
                results_dir=results_dir,
                dataset=dataset,
                train_pct=train_pct,
                target=target,
                label_idx=label_idx,
                best_model=target_best_model,
                metric=metric,
                n_boot=n_boot,
                n_perm=n_perm,
                alpha=alpha,
                seed=seed,
            )

            if not jobs:
                continue

            print(
                f"Running p-tests: dataset={dataset}, train_pct={train_pct}, "
                f"target={target}, metric={metric}, "
                f"best_by_auroc={target_best_model}, jobs={len(jobs)}"
            )

            rows = []

            if max_workers > 1:
                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    futures = [
                        executor.submit(_compare_one_pair, job)
                        for job in jobs
                    ]

                    for fut in as_completed(futures):
                        rows.append(fut.result())
            else:
                for job in jobs:
                    rows.append(_compare_one_pair(job))

            _save_matrices(
                rows=rows,
                comparison_dir=comparison_dir,
                dataset=dataset,
                train_pct=train_pct,
                target=target,
                metric=metric,
                alpha=alpha,
            )


def export_stats_tests(
    config: dict,
) -> None:
    """
    Export pairwise p-test CSV matrices.

    Best model is selected dynamically from saved files using AUROC.

    For each run, target, and metric, writes four CSVs:
        1. p_values
        2. bootstrap_diff
        3. within_noise
        4. sig_alpha_<alpha>

    Metrics:
        auc
        auprc

    Targets:
        macro
        selected labels in TARGET_LABELS that exist in the dataset vocab

    Missing result files:
        skipped and written as ---
    """


    alpha=float(config.get("stats_tests", {}).get("alpha", 0.05))
    n_boot=int(config.get("stats_tests", {}).get("n_boot", 1000))
    n_perm=int(config.get("stats_tests", {}).get("n_perm", 1000))

    for dataset, train_pct in P_TEST_RUNS:
        export_stats_tests_for_run(
            config=config,
            dataset=dataset,
            train_pct=float(train_pct),
            alpha=alpha,
            n_boot=n_boot,
            n_perm=n_perm,
        )

    print("✅ p-test export complete.")
