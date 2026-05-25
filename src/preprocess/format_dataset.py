import ast
import json
import os
from collections import Counter
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from src.registry import get_preprocess_datasets

SPLIT_ORDER = ("train", "val", "test")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_csv_required(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing required CSV: {path}")
    return pd.read_csv(path)


def processed_done_path(out_dir: str, task_name: str) -> str:
    return os.path.join(out_dir, f"{task_name}.done.json")


def processed_outputs_exist(out_dir: str, task_name: str) -> bool:
    return (
        os.path.exists(os.path.join(out_dir, f"{task_name}_metadata_final.csv"))
        and os.path.exists(os.path.join(out_dir, f"{task_name}_label_prevalence.csv"))
        and os.path.exists(os.path.join(out_dir, f"{task_name}_summary_stats.json"))
        and os.path.exists(processed_done_path(out_dir, task_name))
    )


def skip_if_processed(out_dir: str, task_name: str) -> bool:
    if processed_outputs_exist(out_dir, task_name):
        print(f"⏭️  Skipping {task_name}; processed outputs already exist")
        return True
    return False


def mark_processed_done(out_dir: str, task_name: str, meta: dict) -> None:
    path = processed_done_path(out_dir, task_name)
    tmp_path = path + ".tmp"

    payload = {
        "done": True,
        "task": task_name,
        **meta,
    }

    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2)

    os.replace(tmp_path, path)

def save_json(obj: dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def split_names_in_order(df: pd.DataFrame) -> List[str]:
    if "split" not in df.columns:
        return []

    existing = [str(x) for x in df["split"].dropna().unique()]
    ordered = [s for s in SPLIT_ORDER if s in existing]
    ordered += sorted(s for s in existing if s not in SPLIT_ORDER)
    return ordered


def safe_sum(df: pd.DataFrame, col: str) -> Optional[int]:
    if col not in df.columns:
        return None
    return int(df[col].fillna(0).sum())


def safe_min(df: pd.DataFrame, col: str) -> Optional[int]:
    if col not in df.columns or df[col].dropna().empty:
        return None
    return int(df[col].dropna().min())


def safe_max(df: pd.DataFrame, col: str) -> Optional[int]:
    if col not in df.columns or df[col].dropna().empty:
        return None
    return int(df[col].dropna().max())


def basic_summary_stats(df: pd.DataFrame) -> dict:
    return {
        "N": int(len(df)),
        "zero_count": safe_sum(df, "all_zero"),
        "nan_count": safe_sum(df, "had_nan"),
    }


def length_summary_stats(df: pd.DataFrame) -> dict:
    summary = basic_summary_stats(df)
    summary.update({
        "short_count": safe_sum(df, "was_padded"),
        "long_count": safe_sum(df, "was_cropped"),
        "max_length": safe_max(df, "orig_len"),
        "min_length": safe_min(df, "orig_len"),
    })
    return summary


def overlapping_flag_counts(df: pd.DataFrame, col_to_name: Dict[str, str]) -> dict:
    counts = {}
    for col, out_name in col_to_name.items():
        if col in df.columns:
            counts[out_name] = int((df[col] == 1).sum())
    return counts


def apply_filters(
    df: pd.DataFrame,
    filters: Sequence[Tuple[str, Callable[[pd.DataFrame], pd.Series]]],
) -> Tuple[pd.DataFrame, dict]:
    """
    Applies filters sequentially and returns disjoint removal counts.

    Each tuple is: (output_count_name, keep_mask_function).
    """
    current = df.copy()
    removed_counts = {}

    for count_name, keep_mask_fn in filters:
        before = len(current)
        keep_mask = keep_mask_fn(current)
        current = current[keep_mask].copy()
        removed_counts[count_name] = int(before - len(current))

    removed_counts["removed_total"] = int(len(df) - len(current))
    return current.reset_index(drop=True), removed_counts


def add_stratified_random_split(
    df: pd.DataFrame,
    label_cols: Sequence[str],
    seed: int,
    train_frac: float = 0.7,
    val_frac: float = 0.1,
    test_frac: Optional[float] = None,
) -> pd.DataFrame:
    """
    Reproducible split with minimum label coverage.

    Procedure:
      1. For every label, seed at least one positive example into train/val/test.
      2. Randomly distribute all remaining rows while preserving target split sizes.
    """
    df = df.copy()
    n = len(df)

    if test_frac is None:
        test_frac = 1.0 - train_frac - val_frac

    split_names = np.array(["train", "val", "test"], dtype=object)
    fracs = np.array([train_frac, val_frac, test_frac], dtype=float)

    if not np.isclose(fracs.sum(), 1.0):
        raise ValueError(f"Split fractions must sum to 1. Got {fracs.sum()}")

    if n < 3:
        raise RuntimeError("Need at least 3 rows for train/val/test split.")

    rng = np.random.default_rng(seed)

    Y = (
        df[list(label_cols)]
        .fillna(0)
        .astype(float)
        .to_numpy()
    )
    Y = (Y > 0).astype(np.int8)

    label_pos_counts = Y.sum(axis=0)

    zero_labels = [
        label_cols[j]
        for j, c in enumerate(label_pos_counts)
        if c == 0
    ]

    rare_labels = [
        label_cols[j]
        for j, c in enumerate(label_pos_counts)
        if 0 < c < 3
    ]

    if zero_labels:
        raise RuntimeError(
            "These labels have 0 positives, so coverage is impossible: "
            f"{zero_labels}"
        )

    if rare_labels:
        raise RuntimeError(
            "These labels have fewer than 3 positives, so at least one positive "
            "in train/val/test is impossible: "
            f"{rare_labels}"
        )

    # Target split sizes.
    raw_sizes = fracs * n
    target_sizes = np.floor(raw_sizes).astype(int)
    remainder = n - int(target_sizes.sum())

    if remainder > 0:
        order = np.argsort(-(raw_sizes - target_sizes))
        for s in order[:remainder]:
            target_sizes[s] += 1

    split_id = np.full(n, -1, dtype=int)
    current_sizes = np.zeros(3, dtype=int)

    # ------------------------------------------------------------------
    # Step 1: seed one positive example per label into each split
    # ------------------------------------------------------------------

    # Rare labels first, because they are hardest to place.
    label_order = np.argsort(label_pos_counts)

    for label_idx in label_order:
        label_name = label_cols[label_idx]

        for s in range(3):
            # If this split already has this label because of a previous
            # multilabel row, no need to add another one.
            already_has_label = np.any(
                (split_id == s) & (Y[:, label_idx] == 1)
            )

            if already_has_label:
                continue

            candidates = np.where(
                (split_id == -1) &
                (Y[:, label_idx] == 1)
            )[0]

            if len(candidates) == 0:
                raise RuntimeError(
                    f"Could not seed label {label_name} into split "
                    f"{split_names[s]}. Try a different seed or inspect "
                    "highly overlapping rare labels."
                )

            chosen = int(rng.choice(candidates))

            split_id[chosen] = s
            current_sizes[s] += 1

    # ------------------------------------------------------------------
    # Step 2: randomly distribute the remaining rows
    # ------------------------------------------------------------------

    unassigned = np.where(split_id == -1)[0]
    rng.shuffle(unassigned)

    remaining_capacity = target_sizes - current_sizes

    if remaining_capacity.sum() != len(unassigned):
        raise RuntimeError(
            "Internal size mismatch: remaining split capacity does not match "
            "number of unassigned rows."
        )

    remaining_split_slots = np.concatenate([
        np.full(remaining_capacity[s], s, dtype=int)
        for s in range(3)
    ])

    rng.shuffle(remaining_split_slots)

    split_id[unassigned] = remaining_split_slots

    df["split"] = split_names[split_id]

    # ------------------------------------------------------------------
    # Final checks
    # ------------------------------------------------------------------

    final_counts = np.vstack([
        Y[split_id == s].sum(axis=0)
        for s in range(3)
    ])

    missing = []

    for s in range(3):
        for j, col in enumerate(label_cols):
            if final_counts[s, j] < 1:
                missing.append((split_names[s], col))

    if missing:
        raise RuntimeError(
            "Final split is missing positive examples for some labels: "
            f"{missing}"
        )

    print("\nSplit sizes:")
    for s in range(3):
        print(
            f"  {split_names[s]}: "
            f"{(split_id == s).sum()} / target {target_sizes[s]}"
        )

    print("\nLabel counts per split:")
    for j, col in enumerate(label_cols):
        print(
            f"  {col:12s} "
            f"train={final_counts[0, j]:5d} "
            f"val={final_counts[1, j]:5d} "
            f"test={final_counts[2, j]:5d}"
        )

    return df

def require_label_cols(df: pd.DataFrame, prefix: Optional[str] = None, suffix: Optional[str] = None) -> List[str]:
    if prefix is not None:
        label_cols = [c for c in df.columns if c.startswith(prefix)]
    elif suffix is not None:
        label_cols = [c for c in df.columns if c.endswith(suffix)]
    else:
        raise ValueError("Either prefix or suffix must be provided")

    if not label_cols:
        raise RuntimeError("No label columns found")
    return label_cols


def prevalence_df(
    df: pd.DataFrame,
    label_cols: Sequence[str],
    split_name: str,
    clean_label: Callable[[str], str],
) -> pd.DataFrame:
    denom = max(len(df), 1)
    counts = df[list(label_cols)].sum(axis=0).astype(int)

    return pd.DataFrame({
        "label": [clean_label(c) for c in label_cols],
        f"{split_name}_count": counts.values,
        f"{split_name}_prevalence": counts.values / denom,
    })


def make_prevalence_table(
    df: pd.DataFrame,
    label_cols: Sequence[str],
    clean_label: Callable[[str], str],
) -> pd.DataFrame:
    table = prevalence_df(df, label_cols, "full", clean_label)

    for split_name in split_names_in_order(df):
        split_df = df[df["split"] == split_name]
        split_prev = prevalence_df(split_df, label_cols, split_name, clean_label)
        table = table.merge(split_prev, on="label", how="left")

    return table


def write_processed_outputs(
    *,
    task_name: str,
    df_raw: pd.DataFrame,
    df_final: pd.DataFrame,
    out_dir: str,
    label_cols: Sequence[str],
    removed_counts: dict,
    original_flag_counts: dict,
    summary_fn: Callable[[pd.DataFrame], dict],
    clean_label: Callable[[str], str],
    extra_summary_fields: Optional[dict] = None,
) -> None:
    ensure_dir(out_dir)

    orig_n = len(df_raw)
    final_n = len(df_final)

    print(f"Original N: {orig_n}")
    print(f"Final N:    {final_n}")
    print("Removed:", removed_counts)

    if "split" in df_final.columns:
        print("\nSplit distribution:")
        print(df_final["split"].value_counts())

    final_meta_csv = os.path.join(out_dir, f"{task_name}_metadata_final.csv")
    df_final.to_csv(final_meta_csv, index=False)
    print(f"Saved metadata → {final_meta_csv}")

    prevalence_csv = os.path.join(out_dir, f"{task_name}_label_prevalence.csv")
    make_prevalence_table(df_final, label_cols, clean_label).to_csv(prevalence_csv, index=False)
    print(f"Saved prevalence → {prevalence_csv}")

    summary = {
        "task": task_name,
        "dataset_counts": {
            "original": int(orig_n),
            "final": int(final_n),
            **removed_counts,
        },
        "original_flag_counts": original_flag_counts,
        "original_summary_stats": summary_fn(df_raw),
        "final_summary_stats": summary_fn(df_final),
        "splits": {
            split_name: summary_fn(df_final[df_final["split"] == split_name])
            for split_name in split_names_in_order(df_final)
        },
    }

    if extra_summary_fields:
        summary.update(extra_summary_fields)

    summary_json = os.path.join(out_dir, f"{task_name}_summary_stats.json")
    save_json(summary, summary_json)
    print(f"Saved summary stats → {summary_json}")
    mark_processed_done(
        out_dir,
        task_name,
        {
            "original_n": int(orig_n),
            "final_n": int(final_n),
            "removed_counts": removed_counts,
        },
    )


def parse_missing_codes(value) -> List[str]:
    if pd.isna(value):
        return []
    if isinstance(value, list):
        return value

    text = str(value).strip()
    if text == "" or text.lower() == "nan":
        return []

    try:
        parsed = ast.literal_eval(text)
        return parsed if isinstance(parsed, list) else []
    except Exception:
        return [text]


def process_cpsc(raw_data_dir: str, processed_dir: str, seed: int) -> None:
    print("\n===== CPSC =====")

    task_name = "cpsc2018"
    if skip_if_processed(processed_dir, task_name):
        return

    df_raw = read_csv_required(os.path.join(raw_data_dir, "cpsc2018_metadata.csv"))
    label_cols = require_label_cols(df_raw, prefix="label_")

    original_flag_counts = overlapping_flag_counts(df_raw, {
        "was_padded": "short_flag_total",
        "had_nan": "nan_flag_total",
        "all_zero": "all_zero_flag_total",
    })

    df_final, removed_counts = apply_filters(df_raw, [
        ("removed_short", lambda d: d["was_padded"] == 0),
        ("removed_nan", lambda d: d["had_nan"] == 0),
        ("removed_all_zero", lambda d: d["all_zero"] == 0),
    ])
    df_final = add_stratified_random_split(
        df_final,
        label_cols=label_cols,
        seed=seed,
    )

    write_processed_outputs(
        task_name="cpsc2018",
        df_raw=df_raw,
        df_final=df_final,
        out_dir=processed_dir,
        label_cols=label_cols,
        removed_counts=removed_counts,
        original_flag_counts=original_flag_counts,
        summary_fn=length_summary_stats,
        clean_label=lambda c: c.replace("label_", ""),
    )


def process_echonext(raw_data_dir: str, processed_dir: str) -> None:
    print("\n===== ECHO_NEXT =====")

    task_name = "echonext"
    if skip_if_processed(processed_dir, task_name):
        return

    df_raw = read_csv_required(os.path.join(raw_data_dir, "echonext_metadata.csv"))
    label_cols = require_label_cols(df_raw, suffix="_flag")

    if "split" not in df_raw.columns:
        raise RuntimeError("ECHO_NEXT metadata must already contain a 'split' column")

    original_flag_counts = overlapping_flag_counts(df_raw, {
        "had_nan": "had_nan_flag_total",
        "all_zero": "all_zero_flag_total",
    })

    filters = []
    if "had_nan" in df_raw.columns:
        filters.append(("removed_nan", lambda d: d["had_nan"] == 0))

    df_final, removed_counts = apply_filters(df_raw, filters)

    write_processed_outputs(
        task_name="echonext",
        df_raw=df_raw,
        df_final=df_final,
        out_dir=processed_dir,
        label_cols=label_cols,
        removed_counts=removed_counts,
        original_flag_counts=original_flag_counts,
        summary_fn=basic_summary_stats,
        clean_label=lambda c: c.replace("_flag", ""),
    )


def process_csn(raw_data_dir: str, processed_dir: str, seed: int) -> None:
    print("\n===== CSN =====")

    task_name = "csn"
    if skip_if_processed(processed_dir, task_name):
        return

    df_raw = read_csv_required(os.path.join(raw_data_dir, "csn_metadata.csv"))
    label_cols = require_label_cols(df_raw, prefix="label_")

    if "missing_codes" in df_raw.columns:
        df_raw["_missing_codes_list"] = df_raw["missing_codes"].apply(parse_missing_codes)
    else:
        df_raw["_missing_codes_list"] = [[] for _ in range(len(df_raw))]

    df_raw["_has_unknown_codes"] = df_raw["_missing_codes_list"].apply(lambda x: len(x) > 0)

    unknown_code_counter = Counter()
    for codes in df_raw["_missing_codes_list"]:
        unknown_code_counter.update(codes)

    unknown_codes_found = dict(sorted(
        unknown_code_counter.items(),
        key=lambda item: (-item[1], item[0]),
    ))
    rows_with_unknown = int(df_raw["_has_unknown_codes"].sum())

    print(f"Rows with unknown codes: {rows_with_unknown}")
    print(f"Unique unknown codes found: {len(unknown_codes_found)}")

    original_flag_counts = overlapping_flag_counts(df_raw, {
        "was_padded": "was_padded_flag_total",
        "had_nan": "had_nan_flag_total",
        "all_zero": "all_zero_flag_total",
    })
    original_flag_counts.update({
        "rows_with_unknown_codes": rows_with_unknown,
        "unique_unknown_codes": int(len(unknown_codes_found)),
    })

    filters = [
        ("removed_unknown_codes", lambda d: ~d["_has_unknown_codes"]),
    ]

    if "was_padded" in df_raw.columns:
        filters.append(("removed_short", lambda d: d["was_padded"] == 0))
    if "had_nan" in df_raw.columns:
        filters.append(("removed_nan", lambda d: d["had_nan"] == 0))
    if "all_zero" in df_raw.columns:
        filters.append(("removed_all_zero", lambda d: d["all_zero"] == 0))

    df_final, removed_counts = apply_filters(df_raw, filters)
    df_final = df_final.drop(columns=["_missing_codes_list", "_has_unknown_codes"], errors="ignore")
    df_final = add_stratified_random_split(
        df_final,
        label_cols=label_cols,
        seed=seed,
    )

    write_processed_outputs(
        task_name="csn",
        df_raw=df_raw,
        df_final=df_final,
        out_dir=processed_dir,
        label_cols=label_cols,
        removed_counts=removed_counts,
        original_flag_counts=original_flag_counts,
        summary_fn=basic_summary_stats,
        clean_label=lambda c: c.replace("label_", ""),
        extra_summary_fields={"unknown_codes_found": unknown_codes_found},
    )


def process_ptbxl_task(raw_data_dir: str, processed_dir: str, prefix: str, label_type: str) -> None:
    task_name = f"{prefix}_{label_type}"
    print(f"\n----- {task_name} -----")

    if skip_if_processed(processed_dir, task_name):
        return

    df_raw = read_csv_required(os.path.join(raw_data_dir, f"{task_name}_metadata.csv"))
    label_cols = require_label_cols(df_raw, prefix="label_")

    if "split" not in df_raw.columns:
        raise RuntimeError(f"{task_name} metadata must contain a 'split' column")

    original_flag_counts = overlapping_flag_counts(df_raw, {
        "was_padded": "was_padded_flag_total",
        "had_nan": "had_nan_flag_total",
        "all_zero": "all_zero_flag_total",
    })

    filters = []
    if "was_padded" in df_raw.columns:
        filters.append(("removed_short", lambda d: d["was_padded"] == 0))
    if "had_nan" in df_raw.columns:
        filters.append(("removed_nan", lambda d: d["had_nan"] == 0))
    if "all_zero" in df_raw.columns:
        filters.append(("removed_all_zero", lambda d: d["all_zero"] == 0))

    df_final, removed_counts = apply_filters(df_raw, filters)

    write_processed_outputs(
        task_name=task_name,
        df_raw=df_raw,
        df_final=df_final,
        out_dir=processed_dir,
        label_cols=label_cols,
        removed_counts=removed_counts,
        original_flag_counts=original_flag_counts,
        summary_fn=basic_summary_stats,
        clean_label=lambda c: c.replace("label_", ""),
    )


def process_ptbxl_family(raw_data_dir: str, processed_dir: str, prefix: str, label_types: Iterable[str]) -> None:
    print(f"\n===== {prefix.upper()} =====")
    for label_type in label_types:
        process_ptbxl_task(raw_data_dir, processed_dir, prefix, label_type)



def format_main(args, config: dict) -> None:
    raw_data_dir = config["raw_data_dir"]
    processed_dir = config["processed_dir"]
    ensure_dir(processed_dir)

    # Seed used in experiments.
    base_seed = int(config.get("seed", 42))
    cpsc_seed = int(config.get("cpsc_seed", base_seed))
    csn_seed = int(config.get("csn_seed", base_seed))

    format_datasets = get_preprocess_datasets(args.datasets)

    print(f"Formatting required datasets: {', '.join(format_datasets)}")

    if "CPSC" in format_datasets:
        process_cpsc(raw_data_dir, processed_dir, seed=cpsc_seed)

    if "ECHO_NEXT" in format_datasets:
        process_echonext(raw_data_dir, processed_dir)

    if "CSN" in format_datasets:
        process_csn(raw_data_dir, processed_dir, seed=csn_seed)

    if "PTBXL" in format_datasets:
        process_ptbxl_family(
            raw_data_dir,
            processed_dir,
            prefix="ptbxl",
            label_types=["form", "rhythm", "diagnostic_class", "diagnostic_subclass"],
        )

    if "PTBXL_C" in format_datasets:
        process_ptbxl_family(
            raw_data_dir,
            processed_dir,
            prefix="ptbxl_c",
            label_types=["form", "rhythm", "diagnostic_subclass"],
        )

    print("\n✅ All processed metadata exports complete.")


