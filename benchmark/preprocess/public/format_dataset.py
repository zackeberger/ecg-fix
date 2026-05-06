import argparse
import ast
import json
import multiprocessing as mp
import os
from collections import Counter

import numpy as np
import pandas as pd

from benchmark.config import add_config_arg, load_config

_parser = argparse.ArgumentParser(add_help=False)
add_config_arg(_parser)
_args, _unknown = _parser.parse_known_args()
CONFIG = load_config(_args.config)

# =======================
# CPSC
# =======================
print("===== CPSC =====")

cpsc_csv = os.path.join(CONFIG.raw_data_dir, "cpsc2018_metadata.csv")
out_dir = CONFIG.processed_dir
os.makedirs(out_dir, exist_ok=True)

# -----------------------
# load original data
# -----------------------
df_raw = pd.read_csv(cpsc_csv)
orig_N = len(df_raw)

# original flag totals (can overlap!)
orig_flag_counts = {
    "short_flag_total": int((df_raw["was_padded"] == 1).sum()),
    "nan_flag_total": int((df_raw["had_nan"] == 1).sum()),
    "all_zero_flag_total": int((df_raw["all_zero"] == 1).sum()),
}

# -----------------------
# filter step-by-step (disjoint removed counts)
# -----------------------
df_1 = df_raw[df_raw["was_padded"] == 0]
N_after_short = len(df_1)

df_2 = df_1[df_1["had_nan"] == 0]
N_after_nan = len(df_2)

df_3 = df_2[df_2["all_zero"] == 0]
N_after_all_zero = len(df_3)

# finalize filtered df
df = df_3.reset_index(drop=True)
final_N = len(df)

# disjoint removal counts based on filtering order
removed_counts = {
    "removed_short": int(orig_N - N_after_short),
    "removed_nan": int(N_after_short - N_after_nan),
    "removed_all_zero": int(N_after_nan - N_after_all_zero),
    "removed_total": int(orig_N - final_N),
}

print(f"Original N: {orig_N}")
print(f"Final N: {final_N}")
print("Removed:", removed_counts)

# -----------------------
# setup
# -----------------------
N = len(df)
label_cols = [c for c in df.columns if c.startswith("label_")]

# -----------------------
# splits (70 / 10 / 20)
# -----------------------
rng = np.random.default_rng(seed=42)
perm = rng.permutation(N)

train_end = int(0.7 * N)
val_end = int(0.8 * N)

split = np.empty(N, dtype=object)
split[perm[:train_end]] = "train"
split[perm[train_end:val_end]] = "val"
split[perm[val_end:]] = "test"

df["split"] = split

# -----------------------
# save finalized metadata
# -----------------------
final_meta_csv = f"{out_dir}/cpsc2018_metadata_final.csv"
df.to_csv(final_meta_csv, index=False)
print(f"Saved metadata → {final_meta_csv}")

# -----------------------
# prevalence helper
# -----------------------
def prevalence_df(df_subset, name):
    clean_labels = [c.replace("label_", "") for c in label_cols]
    counts = df_subset[label_cols].sum(axis=0).astype(int)
    prev = counts / len(df_subset)

    return pd.DataFrame({
        "label": clean_labels,
        f"{name}_count": counts.values,
        f"{name}_prevalence": prev.values
    })

# -----------------------
# prevalence tables
# -----------------------
prev_full  = prevalence_df(df, "full")
prev_train = prevalence_df(df[df["split"] == "train"], "train")
prev_val   = prevalence_df(df[df["split"] == "val"], "val")
prev_test  = prevalence_df(df[df["split"] == "test"], "test")

prev_all = (
    prev_full
    .merge(prev_train, on="label")
    .merge(prev_val, on="label")
    .merge(prev_test, on="label")
)

prev_csv = f"{out_dir}/cpsc2018_label_prevalence.csv"
prev_all.to_csv(prev_csv, index=False)
print(f"Saved prevalence → {prev_csv}")

# -----------------------
# summary stats helper
# -----------------------
def summary_stats(df_subset):
    return {
        "N": int(len(df_subset)),
        "zero_count": int(df_subset["all_zero"].sum()),
        "nan_count": int(df_subset["had_nan"].sum()),
        "short_count": int(df_subset["was_padded"].sum()),
        "long_count": int(df_subset["was_cropped"].sum()),
        "max_length": int(df_subset["orig_len"].max()),
        "min_length": int(df_subset["orig_len"].min()),
    }

# -----------------------
# summary JSON
# -----------------------
summary = {
    "dataset_counts": {
        "original": int(orig_N),
        "final": int(final_N),
        **removed_counts
    },
    "original_flag_counts": orig_flag_counts,   # overlaps allowed
    "original_summary_stats": summary_stats(df_raw),
    "final_summary_stats": summary_stats(df),
    "splits": {
        "train": summary_stats(df[df["split"] == "train"]),
        "val": summary_stats(df[df["split"] == "val"]),
        "test": summary_stats(df[df["split"] == "test"]),
    }
}

summary_json = f"{out_dir}/cpsc2018_summary_stats.json"
with open(summary_json, "w") as f:
    json.dump(summary, f, indent=2)

print(f"Saved summary stats → {summary_json}")

# -----------------------
# sanity check
# -----------------------
print(df["split"].value_counts(normalize=True))



print()
print("=====ECHO_NEXT======")
print()

echo_next_csv = os.path.join(CONFIG.raw_data_dir, "echonext_metadata.csv")
out_dir = CONFIG.processed_dir
os.makedirs(out_dir, exist_ok=True)

df_raw = pd.read_csv(echo_next_csv)
orig_N = len(df_raw)

label_cols = [c for c in df_raw.columns if c.endswith("_flag")]
if len(label_cols) == 0:
    raise RuntimeError("No *_flag label columns found in ECHO_NEXT metadata CSV")

def summary_stats(df_subset):
    def safe_sum(col):
        return int(df_subset[col].sum()) if col in df_subset.columns else None

    def safe_max(col):
        return int(df_subset[col].max()) if col in df_subset.columns else None

    def safe_min(col):
        return int(df_subset[col].min()) if col in df_subset.columns else None

    return {
        "N": int(len(df_subset)),
        "zero_count": safe_sum("all_zero"),
        "nan_count": safe_sum("had_nan"),
    }


orig_flag_counts = {}
for col in [ "had_nan", "all_zero"]:
    if col in df_raw.columns:
        orig_flag_counts[f"{col}_flag_total"] = int((df_raw[col] == 1).sum())


df_2 = df_raw[df_raw["had_nan"] == 0] if "had_nan" in df_raw.columns else df_raw
N_after_nan = len(df_2)

df = df_2.reset_index(drop=True)
final_N = len(df)

removed_counts = {
    "removed_nan": int(orig_N - N_after_nan),
    "removed_total": int(orig_N - final_N),
}

print(f"Original N: {orig_N}")
print(f"Final N:    {final_N}")
print("Removed:", removed_counts)


if "split" not in df.columns:
    raise RuntimeError("ECHO_NEXT metadata does NOT contain a 'split' column (expected it to already exist).")

print("\nSplit distribution (final):")
print(df["split"].value_counts())

final_meta_csv = f"{out_dir}/echonext_metadata_final.csv"
df.to_csv(final_meta_csv, index=False)
print(f"\nSaved metadata → {final_meta_csv}")

# -----------------------
# prevalence helper (remove '_flag' suffix)
# -----------------------
def prevalence_df(df_subset, name):
    clean_labels = [c.replace("_flag", "") for c in label_cols]
    counts = df_subset[label_cols].sum(axis=0).astype(int)
    prev = counts / max(len(df_subset), 1)

    return pd.DataFrame({
        "label": clean_labels,
        f"{name}_count": counts.values,
        f"{name}_prevalence": prev.values
    })

# -----------------------
# prevalence tables (full + each existing split)
# -----------------------
prev_all = prevalence_df(df, "full")

for split_name in sorted(df["split"].dropna().unique()):
    df_split = df[df["split"] == split_name]
    prev_split = prevalence_df(df_split, str(split_name))
    prev_all = prev_all.merge(prev_split, on="label", how="left")

prev_csv = f"{out_dir}/echonext_label_prevalence.csv"
prev_all.to_csv(prev_csv, index=False)
print(f"Saved prevalence → {prev_csv}")

# -----------------------
# summary JSON (original + final + per-split)
# -----------------------
summary = {
    "dataset_counts": {
        "original": int(orig_N),
        "final": int(final_N),
        **removed_counts
    },
    "original_flag_counts": orig_flag_counts,             # overlaps allowed
    "original_summary_stats": summary_stats(df_raw),      # BEFORE filtering
    "final_summary_stats": summary_stats(df),             # AFTER filtering
    "splits": {}
}

for split_name in sorted(df["split"].dropna().unique()):
    summary["splits"][str(split_name)] = summary_stats(df[df["split"] == split_name])

summary_json = f"{out_dir}/echonext_summary_stats.json"
with open(summary_json, "w") as f:
    json.dump(summary, f, indent=2)

print(f"Saved summary stats → {summary_json}")



print()
print("===== CSN =====")
print()

csn_csv = os.path.join(CONFIG.raw_data_dir, "csn_metadata.csv")
out_dir = CONFIG.processed_dir
os.makedirs(out_dir, exist_ok=True)

# -----------------------
# load original data
# -----------------------
df_raw = pd.read_csv(csn_csv)
orig_N = len(df_raw)

print(f"Original N: {orig_N}")

# -----------------------
# label columns
# -----------------------
label_cols = [c for c in df_raw.columns if c.startswith("label_")]
if len(label_cols) == 0:
    raise RuntimeError("No label_* columns found in CSN metadata CSV")

# -----------------------
# parse missing_codes (stored as string like "[]", "['123', '456']")
# -----------------------
def parse_missing_codes(x):
    if pd.isna(x):
        return []
    if isinstance(x, list):
        return x
    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return []
    try:
        val = ast.literal_eval(s)
        return val if isinstance(val, list) else []
    except Exception:
        # fallback: if something weird is stored, treat it as "unknown exists"
        return [s]

if "missing_codes" in df_raw.columns:
    df_raw["_missing_codes_list"] = df_raw["missing_codes"].apply(parse_missing_codes)
    df_raw["_has_unknown_codes"] = df_raw["_missing_codes_list"].apply(lambda x: len(x) > 0)
else:
    df_raw["_missing_codes_list"] = [[] for _ in range(orig_N)]
    df_raw["_has_unknown_codes"] = False

# unknown codes found in ORIGINAL data
unknown_code_counter = Counter()
for codes in df_raw["_missing_codes_list"]:
    unknown_code_counter.update(codes)

unknown_codes_found = dict(sorted(unknown_code_counter.items(), key=lambda kv: (-kv[1], kv[0])))
num_rows_with_unknown = int(df_raw["_has_unknown_codes"].sum())

print(f"Rows with unknown codes: {num_rows_with_unknown}")
if len(unknown_codes_found) > 0:
    print(f"Unique unknown codes found: {len(unknown_codes_found)}")
else:
    print("No unknown codes found ✅")

# -----------------------
# helper: robust summary stats
# -----------------------
def summary_stats(df_subset):
    def safe_sum(col):
        return int(df_subset[col].sum()) if col in df_subset.columns else None

    def safe_max(col):
        return int(df_subset[col].max()) if col in df_subset.columns else None

    def safe_min(col):
        return int(df_subset[col].min()) if col in df_subset.columns else None

    return {
        "N": int(len(df_subset)),
        "zero_count": safe_sum("all_zero"),
        "nan_count": safe_sum("had_nan"),
    }

# original flag totals (overlaps allowed)
orig_flag_counts = {}
for col in ["was_padded", "had_nan", "all_zero"]:
    if col in df_raw.columns:
        orig_flag_counts[f"{col}_flag_total"] = int((df_raw[col] == 1).sum())

orig_flag_counts["rows_with_unknown_codes"] = num_rows_with_unknown
orig_flag_counts["unique_unknown_codes"] = int(len(unknown_codes_found))

# -----------------------
# FILTER step-by-step (disjoint removed counts)
# Order: unknown -> short -> nan -> all_zero
# -----------------------
df_0 = df_raw.copy()
N0 = len(df_0)

# 1) drop unknown codes
df_1 = df_0[~df_0["_has_unknown_codes"]]
N1 = len(df_1)

# 2) drop padded (short)
df_2 = df_1[df_1["was_padded"] == 0] if "was_padded" in df_1.columns else df_1
N2 = len(df_2)

# 3) drop NaNs
df_3 = df_2[df_2["had_nan"] == 0] if "had_nan" in df_2.columns else df_2
N3 = len(df_3)

# 4) drop all_zero
df_4 = df_3[df_3["all_zero"] == 0] if "all_zero" in df_3.columns else df_3
N4 = len(df_4)

# 4) drop all_invalid
df_5 = df_4[df_4["ecg_index"] != -100000] if "ecg_index" in df_4.columns else df_4
N5 = len(df_5)

df = df_5.reset_index(drop=True)
final_N = len(df)

removed_counts = {
    "removed_unknown_codes": int(N0 - N1),
    "removed_short": int(N1 - N2),
    "removed_nan": int(N2 - N3),
    "removed_all_zero": int(N3 - N4),
    "cannot_load": int(N4 - N5),
    "removed_total": int(orig_N - final_N),
}

print(f"Final N: {final_N}")
print("Removed counts (disjoint):", removed_counts)

# -----------------------
# find a reusable random seed for CSN so at least 1 example in each 
# -----------------------
# -----------------------
# split validity check
# -----------------------
def is_valid_split(split, df, label_cols):
    df_local = df.copy()
    df_local["split"] = split

    for split_name in ["train", "val", "test"]:
        subset = df_local[df_local["split"] == split_name]
        if len(subset) == 0:
            return False
        if not (subset[label_cols].sum(axis=0) > 0).all():
            return False
    return True


# -----------------------
# check a single seed
# -----------------------
def check_seed(args):
    seed, df, label_cols = args
    N = len(df)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(N)

    train_end = int(0.7 * N)
    val_end = int(0.8 * N)

    split = np.empty(N, dtype=object)
    split[perm[:train_end]] = "train"
    split[perm[train_end:val_end]] = "val"
    split[perm[val_end:]] = "test"

    if is_valid_split(split, df, label_cols):
        return seed
    return None


# -----------------------
# parallel seed search
# -----------------------
def find_seed_parallel(df, label_cols, max_tries=100_000, n_workers=None):
    rng = np.random.default_rng()
    seeds = rng.integers(0, 1_000_000_000, size=max_tries)

    with mp.Pool(processes=n_workers) as pool:
        for result in pool.imap_unordered(
            check_seed,
            [(int(seed), df, label_cols) for seed in seeds],
            chunksize=100
        ):
            if result is not None:
                pool.terminate()
                pool.join()
                print(f"✅ Found valid seed: {result}")
                return result

    raise RuntimeError("❌ No valid seed found — increase max_tries")



#best_seed = find_seed_parallel(df, label_cols)
#print("FINAL SEED:", best_seed)


# -----------------------
# add splits (70 / 10 / 20)
# -----------------------
N = len(df)
rng = np.random.default_rng(seed=774635088)
perm = rng.permutation(N)

train_end = int(0.7 * N)
val_end = int(0.8 * N)

split = np.empty(N, dtype=object)
split[perm[:train_end]] = "train"
split[perm[train_end:val_end]] = "val"
split[perm[val_end:]] = "test"

df["split"] = split

print("\nSplit distribution (final):")
print(df["split"].value_counts())

# -----------------------
# save finalized metadata
# -----------------------
final_meta_csv = f"{out_dir}/csn_metadata_final.csv"
df.to_csv(final_meta_csv, index=False)
print(f"\nSaved metadata → {final_meta_csv}")

# -----------------------
# prevalence helper (remove 'label_' prefix)
# -----------------------
def prevalence_df(df_subset, name):
    clean_labels = [c.replace("label_", "") for c in label_cols]
    counts = df_subset[label_cols].sum(axis=0).astype(int)
    prev = counts / max(len(df_subset), 1)

    return pd.DataFrame({
        "label": clean_labels,
        f"{name}_count": counts.values,
        f"{name}_prevalence": prev.values
    })

# -----------------------
# prevalence tables
# -----------------------
prev_full  = prevalence_df(df, "full")
prev_train = prevalence_df(df[df["split"] == "train"], "train")
prev_val   = prevalence_df(df[df["split"] == "val"], "val")
prev_test  = prevalence_df(df[df["split"] == "test"], "test")

prev_all = (
    prev_full
    .merge(prev_train, on="label")
    .merge(prev_val, on="label")
    .merge(prev_test, on="label")
)

prev_csv = f"{out_dir}/csn_label_prevalence.csv"
prev_all.to_csv(prev_csv, index=False)
print(f"Saved prevalence → {prev_csv}")

# -----------------------
# summary JSON
# -----------------------
summary = {
    "dataset_counts": {
        "original": int(orig_N),
        "final": int(final_N),
        **removed_counts
    },
    "original_flag_counts": orig_flag_counts,             # overlaps allowed
    "unknown_codes_found": unknown_codes_found,           # code -> frequency across rows
    "original_summary_stats": summary_stats(df_raw),      # BEFORE filtering
    "final_summary_stats": summary_stats(df),             # AFTER filtering
    "splits": {
        "train": summary_stats(df[df["split"] == "train"]),
        "val": summary_stats(df[df["split"] == "val"]),
        "test": summary_stats(df[df["split"] == "test"]),
    },
}

summary_json = f"{out_dir}/csn_summary_stats.json"
with open(summary_json, "w") as f:
    json.dump(summary, f, indent=2)

print(f"Saved summary stats → {summary_json}")

print()
print("===== PTBXL (4 subtasks) =====")
print()

raw_dir = CONFIG.raw_data_dir
out_dir = CONFIG.processed_dir
os.makedirs(out_dir, exist_ok=True)

label_types = ["form", "rhythm", "diagnostic_class", "diagnostic_subclass"]


# -----------------------
# helpers
# -----------------------
def summary_stats(df_subset: pd.DataFrame):
    """Robust summary stats (handles missing columns)."""

    def safe_sum(col):
        return int(df_subset[col].sum()) if col in df_subset.columns else None

    def safe_max(col):
        return int(df_subset[col].max()) if col in df_subset.columns else None

    def safe_min(col):
        return int(df_subset[col].min()) if col in df_subset.columns else None

    return {
        "N": int(len(df_subset)),
        "zero_count": safe_sum("all_zero"),
        "nan_count": safe_sum("had_nan"),
    }


def prevalence_df(df_subset: pd.DataFrame, label_cols, name: str):
    """Counts + prevalence for label_* columns. Removes label_ prefix in output."""
    clean_labels = [c.replace("label_", "") for c in label_cols]

    counts = df_subset[label_cols].sum(axis=0).astype(int)
    prev = counts / max(len(df_subset), 1)

    return pd.DataFrame({
        "label": clean_labels,
        f"{name}_count": counts.values,
        f"{name}_prevalence": prev.values
    })


def process_one_label_type(label_type: str):
    print()
    print(f"----- PTBXL: {label_type} -----")

    in_csv = os.path.join(raw_dir, f"ptbxl_{label_type}_metadata.csv")
    if not os.path.exists(in_csv):
        raise FileNotFoundError(f"Missing: {in_csv}")

    df_raw = pd.read_csv(in_csv)
    orig_N = len(df_raw)

    # label columns
    label_cols = [c for c in df_raw.columns if c.startswith("label_")]
    if len(label_cols) == 0:
        raise RuntimeError(f"No label_* columns found in {in_csv}")

    # ensure split exists (PTB-XL already has it)
    if "split" not in df_raw.columns:
        raise RuntimeError(f"'split' column missing in {in_csv}")

    # original flag totals (can overlap)
    orig_flag_counts = {}
    for col in ["was_padded", "had_nan", "all_zero"]:
        if col in df_raw.columns:
            orig_flag_counts[f"{col}_flag_total"] = int((df_raw[col] == 1).sum())

    # -----------------------
    # FILTER step-by-step (disjoint removal counts)
    # Order: short -> nan -> all_zero
    # -----------------------
    df_0 = df_raw
    N0 = len(df_0)

    df_1 = df_0[df_0["was_padded"] == 0] if "was_padded" in df_0.columns else df_0
    N1 = len(df_1)

    df_2 = df_1[df_1["had_nan"] == 0] if "had_nan" in df_1.columns else df_1
    N2 = len(df_2)

    df_3 = df_2[df_2["all_zero"] == 0] if "all_zero" in df_2.columns else df_2
    N3 = len(df_3)

    df = df_3.reset_index(drop=True)
    final_N = len(df)

    removed_counts = {
        "removed_short": int(N0 - N1),
        "removed_nan": int(N1 - N2),
        "removed_all_zero": int(N2 - N3),
        "removed_total": int(orig_N - final_N),
    }

    print(f"Original N: {orig_N}")
    print(f"Final N:    {final_N}")
    print("Removed (disjoint):", removed_counts)
    print("Final split distribution:")
    print(df["split"].value_counts())

    # -----------------------
    # save finalized metadata
    # -----------------------
    final_meta_csv = os.path.join(out_dir, f"ptbxl_{label_type}_metadata_final.csv")
    df.to_csv(final_meta_csv, index=False)
    print(f"Saved metadata → {final_meta_csv}")

    # -----------------------
    # prevalence (full + per split)
    # -----------------------
    prev_full = prevalence_df(df, label_cols, "full")

    # merge splits dynamically in case split naming differs
    prev_all = prev_full
    for split_name in sorted(df["split"].dropna().unique()):
        df_split = df[df["split"] == split_name]
        prev_split = prevalence_df(df_split, label_cols, str(split_name))
        prev_all = prev_all.merge(prev_split, on="label", how="left")

    prev_csv = os.path.join(out_dir, f"ptbxl_{label_type}_label_prevalence.csv")
    prev_all.to_csv(prev_csv, index=False)
    print(f"Saved prevalence → {prev_csv}")

    # -----------------------
    # summary JSON (original + final + per split)
    # -----------------------
    summary = {
        "task": f"ptbxl_{label_type}",
        "dataset_counts": {
            "original": int(orig_N),
            "final": int(final_N),
            **removed_counts
        },
        "original_flag_counts": orig_flag_counts,          # overlaps allowed
        "original_summary_stats": summary_stats(df_raw),   # BEFORE filtering
        "final_summary_stats": summary_stats(df),          # AFTER filtering
        "splits": {}
    }

    for split_name in sorted(df["split"].dropna().unique()):
        summary["splits"][str(split_name)] = summary_stats(df[df["split"] == split_name])

    summary_json = os.path.join(out_dir, f"ptbxl_{label_type}_summary_stats.json")
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved summary stats → {summary_json}")


# -----------------------
# run all 4 subtasks
# -----------------------
for lt in label_types:
    process_one_label_type(lt)

print("\n✅ PTB-XL processed exports done for all 4 subtasks.")



print()
print("===== PTBXL C (4 subtasks) =====")
print()

raw_dir = CONFIG.raw_data_dir
out_dir = CONFIG.processed_dir
os.makedirs(out_dir, exist_ok=True)

label_types = ["form", "rhythm", "diagnostic_subclass"]


# -----------------------
# helpers
# -----------------------
def summary_stats(df_subset: pd.DataFrame):
    """Robust summary stats (handles missing columns)."""

    def safe_sum(col):
        return int(df_subset[col].sum()) if col in df_subset.columns else None

    def safe_max(col):
        return int(df_subset[col].max()) if col in df_subset.columns else None

    def safe_min(col):
        return int(df_subset[col].min()) if col in df_subset.columns else None

    return {
        "N": int(len(df_subset)),
        "zero_count": safe_sum("all_zero"),
        "nan_count": safe_sum("had_nan"),
    }


def prevalence_df(df_subset: pd.DataFrame, label_cols, name: str):
    """Counts + prevalence for label_* columns. Removes label_ prefix in output."""
    clean_labels = [c.replace("label_", "") for c in label_cols]

    counts = df_subset[label_cols].sum(axis=0).astype(int)
    prev = counts / max(len(df_subset), 1)

    return pd.DataFrame({
        "label": clean_labels,
        f"{name}_count": counts.values,
        f"{name}_prevalence": prev.values
    })


def process_one_label_type(label_type: str):
    print()
    print(f"----- PTBXL: {label_type} -----")

    in_csv = os.path.join(raw_dir, f"ptbxl_c_{label_type}_metadata.csv")
    if not os.path.exists(in_csv):
        raise FileNotFoundError(f"Missing: {in_csv}")

    df_raw = pd.read_csv(in_csv)
    orig_N = len(df_raw)

    # label columns
    label_cols = [c for c in df_raw.columns if c.startswith("label_")]
    if len(label_cols) == 0:
        raise RuntimeError(f"No label_* columns found in {in_csv}")

    # ensure split exists (PTB-XL already has it)
    if "split" not in df_raw.columns:
        raise RuntimeError(f"'split' column missing in {in_csv}")

    # original flag totals (can overlap)
    orig_flag_counts = {}
    for col in ["was_padded", "had_nan", "all_zero"]:
        if col in df_raw.columns:
            orig_flag_counts[f"{col}_flag_total"] = int((df_raw[col] == 1).sum())

    # -----------------------
    # FILTER step-by-step (disjoint removal counts)
    # Order: short -> nan -> all_zero
    # -----------------------
    df_0 = df_raw
    N0 = len(df_0)

    df_1 = df_0[df_0["was_padded"] == 0] if "was_padded" in df_0.columns else df_0
    N1 = len(df_1)

    df_2 = df_1[df_1["had_nan"] == 0] if "had_nan" in df_1.columns else df_1
    N2 = len(df_2)

    df_3 = df_2[df_2["all_zero"] == 0] if "all_zero" in df_2.columns else df_2
    N3 = len(df_3)

    df = df_3.reset_index(drop=True)
    final_N = len(df)

    removed_counts = {
        "removed_short": int(N0 - N1),
        "removed_nan": int(N1 - N2),
        "removed_all_zero": int(N2 - N3),
        "removed_total": int(orig_N - final_N),
    }

    print(f"Original N: {orig_N}")
    print(f"Final N:    {final_N}")
    print("Removed (disjoint):", removed_counts)
    print("Final split distribution:")
    print(df["split"].value_counts())

    # -----------------------
    # save finalized metadata
    # -----------------------
    final_meta_csv = os.path.join(out_dir, f"ptbxl_c_{label_type}_metadata_final.csv")
    df.to_csv(final_meta_csv, index=False)
    print(f"Saved metadata → {final_meta_csv}")

    # -----------------------
    # prevalence (full + per split)
    # -----------------------
    prev_full = prevalence_df(df, label_cols, "full")

    # merge splits dynamically in case split naming differs
    prev_all = prev_full
    for split_name in sorted(df["split"].dropna().unique()):
        df_split = df[df["split"] == split_name]
        prev_split = prevalence_df(df_split, label_cols, str(split_name))
        prev_all = prev_all.merge(prev_split, on="label", how="left")

    prev_csv = os.path.join(out_dir, f"ptbxl_c_{label_type}_label_prevalence.csv")
    prev_all.to_csv(prev_csv, index=False)
    print(f"Saved prevalence → {prev_csv}")

    # -----------------------
    # summary JSON (original + final + per split)
    # -----------------------
    summary = {
        "task": f"ptbxl_{label_type}",
        "dataset_counts": {
            "original": int(orig_N),
            "final": int(final_N),
            **removed_counts
        },
        "original_flag_counts": orig_flag_counts,          # overlaps allowed
        "original_summary_stats": summary_stats(df_raw),   # BEFORE filtering
        "final_summary_stats": summary_stats(df),          # AFTER filtering
        "splits": {}
    }

    for split_name in sorted(df["split"].dropna().unique()):
        summary["splits"][str(split_name)] = summary_stats(df[df["split"] == split_name])

    summary_json = os.path.join(out_dir, f"ptbxl_c_{label_type}_summary_stats.json")
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved summary stats → {summary_json}")


# -----------------------
# run all 4 subtasks
# -----------------------
for lt in label_types:
    process_one_label_type(lt)

print("\n✅ PTB-XL processed exports done for all 3 subtasks.")
