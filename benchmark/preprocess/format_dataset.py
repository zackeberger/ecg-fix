import ast
import json
import os
from collections import Counter
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


SPLIT_ORDER = ("train", "val", "test")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_csv_required(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing required CSV: {path}")
    return pd.read_csv(path)


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


def add_random_split(
    df: pd.DataFrame,
    seed: int,
    train_frac: float = 0.7,
    val_frac: float = 0.1,
) -> pd.DataFrame:
    """Adds deterministic 70/10/20 train/val/test split by default."""
    df = df.copy()
    n = len(df)

    rng = np.random.default_rng(seed=seed)
    perm = rng.permutation(n)

    train_end = int(train_frac * n)
    val_end = int((train_frac + val_frac) * n)

    split = np.empty(n, dtype=object)
    split[perm[:train_end]] = "train"
    split[perm[train_end:val_end]] = "val"
    split[perm[val_end:]] = "test"

    df["split"] = split
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
    df_final = add_random_split(df_final, seed=seed)

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
    if "ecg_index" in df_raw.columns:
        filters.append(("cannot_load", lambda d: d["ecg_index"] != -100000))

    df_final, removed_counts = apply_filters(df_raw, filters)
    df_final = df_final.drop(columns=["_missing_codes_list", "_has_unknown_codes"], errors="ignore")
    df_final = add_random_split(df_final, seed=seed)

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



def main(config: dict) -> None:
    raw_data_dir = config["raw_data_dir"]
    processed_dir = config["processed_dir"]
    ensure_dir(processed_dir)

    base_seed = int(config.get("seed", 42))
    cpsc_seed = int(config.get("cpsc_seed", base_seed))
    csn_seed = int(config.get("csn_seed", 774635088))

    print(f"Using CPSC seed: {cpsc_seed}")
    print(f"Using CSN seed:  {csn_seed}")

    process_cpsc(raw_data_dir, processed_dir, seed=cpsc_seed)
    process_echonext(raw_data_dir, processed_dir)
    process_csn(raw_data_dir, processed_dir, seed=csn_seed)

    process_ptbxl_family(
        raw_data_dir,
        processed_dir,
        prefix="ptbxl",
        label_types=["form", "rhythm", "diagnostic_class", "diagnostic_subclass"],
    )

    process_ptbxl_family(
        raw_data_dir,
        processed_dir,
        prefix="ptbxl_c",
        label_types=["form", "rhythm", "diagnostic_subclass"],
    )

    print("\n✅ All processed metadata exports complete.")


if __name__ == "__main__":
    with open("configs/config.json", "r") as f:
        config = json.load(f)
    main(config)
