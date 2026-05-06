import os
import ast
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import numpy as np
import pandas as pd
import wfdb
from tqdm import tqdm
from numpy.lib.format import open_memmap
from benchmark.config import dataset_root, load_config


TARGET_LEN = 5000  # 10s at 500Hz (PTB-XL HR is 5000 samples)


# ----------------------------
# helpers
# ----------------------------
def resolve_ptbxl_root(base_dir: str) -> str:
    """
    Accepts either direct PTB-XL folder or PhysioNet mirror root.
    Tries:
      1) base_dir contains ptbxl_database.csv + scp_statements.csv
      2) base_dir/files/ptb-xl/<version>/
      3) recursive search
    """
    direct_csv = os.path.join(base_dir, "ptbxl_database.csv")
    direct_scp = os.path.join(base_dir, "scp_statements.csv")
    if os.path.exists(direct_csv) and os.path.exists(direct_scp):
        return base_dir

    ptbxl_dir = os.path.join(base_dir, "files", "ptb-xl")
    if os.path.isdir(ptbxl_dir):
        for version in sorted(os.listdir(ptbxl_dir), reverse=True):
            cand = os.path.join(ptbxl_dir, version)
            csv_path = os.path.join(cand, "ptbxl_database.csv")
            scp_path = os.path.join(cand, "scp_statements.csv")
            if os.path.exists(csv_path) and os.path.exists(scp_path):
                return cand

    for root, _dirs, files in os.walk(base_dir):
        if "ptbxl_database.csv" in files and "scp_statements.csv" in files:
            return root

    raise FileNotFoundError(f"Could not locate PTB-XL under base_dir={base_dir}")


def fold_to_split(strat_fold: int) -> str:
    """
    PTB-XL split convention:
      - val: fold 9
      - test: fold 10
      - train: all else
    """
    if int(strat_fold) == 9:
        return "val"
    if int(strat_fold) == 10:
        return "test"
    return "train"


def crop_or_pad(ecg: np.ndarray, target_len: int = TARGET_LEN):
    """
    ecg: (12, T)
    return (ecg_fixed, orig_len, was_cropped, was_padded, pad_amount)
    """
    T = ecg.shape[1]
    orig_len = int(T)
    was_cropped = int(T > target_len)
    was_padded = int(T < target_len)
    pad_amount = max(0, target_len - T)

    if T > target_len:
        start = (T - target_len) // 2
        ecg = ecg[:, start:start + target_len]
    elif T < target_len:
        out = np.zeros((12, target_len), dtype=ecg.dtype)
        out[:, :T] = ecg
        ecg = out

    return ecg, orig_len, was_cropped, was_padded, pad_amount


def build_vocab_and_mapping(scp_df: pd.DataFrame, label_type: str):
    """
    label_type in {'form', 'rhythm', 'diagnostic_class', 'diagnostic_subclass'}
    returns: vocab(list[str]), scp_to_target(dict scp_code -> target or None)
    """
    if scp_df.columns[0] != "scp_code":
        scp_df = scp_df.rename(columns={scp_df.columns[0]: "scp_code"})

    scp_df = scp_df.copy()

    # normalize boolean-like columns
    for col in ["form", "rhythm", "diagnostic"]:
        if col in scp_df.columns:
            scp_df[col] = scp_df[col].fillna(0).astype(float)

    if label_type in {"form", "rhythm"}:
        if label_type not in scp_df.columns:
            raise ValueError(f"Column {label_type} missing in scp_statements.csv")

        category_scp = set(scp_df.loc[scp_df[label_type] == 1.0, "scp_code"].astype(str).tolist())
        vocab = sorted(list(category_scp))
        scp_to_target = {str(code): (str(code) if str(code) in category_scp else None)
                         for code in scp_df["scp_code"].astype(str).tolist()}
        return vocab, scp_to_target

    if label_type in {"diagnostic_class", "diagnostic_subclass"}:
        if label_type not in scp_df.columns:
            raise ValueError(f"Column {label_type} missing in scp_statements.csv")

        scp_to_target = {}
        classes = set()

        for _, r in scp_df.iterrows():
            code = str(r["scp_code"])
            target = str(r[label_type]).strip()
            if target.lower() == "nan" or target == "":
                scp_to_target[code] = None
            else:
                scp_to_target[code] = target
                classes.add(target)

        vocab = sorted(list(classes))
        return vocab, scp_to_target

    raise ValueError(f"Unknown label_type={label_type}")


# ----------------------------
# multiprocessing ECG export
# ----------------------------
_G = {}


def _init_worker(items, dataset_root):
    global _G
    _G["items"] = items
    _G["dataset_root"] = dataset_root


def _process_one_ecg(i: int):
    """
    Returns:
      (i, ecg_fixed_np, meta_dict)
    or None if unreadable.
    """
    item = _G["items"][i]
    dataset_root = _G["dataset_root"]

    rel_path = item["filename_hr"]
    base = os.path.join(dataset_root, rel_path)

    try:
        sig, _ = wfdb.rdsamp(base)               # (T, 12)
        ecg = np.ascontiguousarray(sig.T, dtype=np.float32)  # (12, T)
    except Exception:
        return None

    if ecg.ndim != 2 or ecg.shape[0] != 12:
        return None

    ecg_fixed, orig_len, was_cropped, was_padded, pad_amount = crop_or_pad(ecg, TARGET_LEN)
    if ecg_fixed.shape != (12, TARGET_LEN):
        return None

    had_nan = int(not np.isfinite(ecg_fixed).all())

    meta = {
        "ecg_id": int(item["ecg_id"]),
        "patient_id": int(item["patient_id"]) if "patient_id" in item else -1,
        "filename_hr": rel_path,
        "strat_fold": int(item["strat_fold"]),
        "split": fold_to_split(int(item["strat_fold"])),
        "orig_len": orig_len,
        "final_len": TARGET_LEN,
        "was_cropped": was_cropped,
        "was_padded": was_padded,
        "pad_amount": pad_amount,
        "had_nan": had_nan,
    }

    # --- stats ---
    # per-lead (axis=1 → across time)
    lead_means = np.mean(ecg_fixed, axis=1)   # (12,)
    lead_stds  = np.std(ecg_fixed, axis=1)    # (12,)

    # global (flatten everything)
    ecg_mean = float(np.mean(ecg_fixed))
    ecg_std  = float(np.std(ecg_fixed))

    meta["ecg_mean"] = ecg_mean
    meta["ecg_std"] = ecg_std
    meta["lead_means"] =  lead_means.tolist()   # store as list for JSON/CSV
    meta["lead_stds"] =  lead_stds.tolist()
    return (i, ecg_fixed, meta)


def export_ptbxl_ecg_500hz(dataset_root: str, out_dir: str, n_workers: int = 8):
    """
    Exports ECG once:
      - ptbxl_ecg_500hz.npy  (N, 12, 5000) float16
      - ptbxl_valid_idx.npy  (K,)
      - ptbxl_base_metadata.csv
    """
    ptbxl_csv = os.path.join(dataset_root, "ptbxl_database.csv")
    df = pd.read_csv(ptbxl_csv)

    # must have these cols
    for col in ["ecg_id", "strat_fold", "filename_hr"]:
        if col not in df.columns:
            raise ValueError(f"Missing column {col} in ptbxl_database.csv")

    # patient_id exists in PTB-XL, but just in case
    if "patient_id" not in df.columns:
        df["patient_id"] = -1

    items = df[["ecg_id", "patient_id", "strat_fold", "filename_hr"]].to_dict("records")
    N = len(items)

    out_ecg = os.path.join(out_dir, "ptbxl_ecg_500hz.npy")
    out_valid = os.path.join(out_dir, "ptbxl_valid_idx.npy")
    out_base_meta = os.path.join(out_dir, "ptbxl_base_metadata.csv")

    print(f"Exporting PTB-XL ECG 500Hz | N={N} | workers={n_workers}")
    ecg_mm = open_memmap(out_ecg, mode="w+", dtype=np.float16, shape=(N, 12, TARGET_LEN))
    valid_mask = np.zeros(N, dtype=np.uint8)
    meta_rows = []

    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_worker,
        initargs=(items, dataset_root),
    ) as ex:
        futures = [ex.submit(_process_one_ecg, i) for i in range(N)]

        for fut in tqdm(as_completed(futures), total=N, desc="ECG export"):
            r = fut.result()
            if r is None:
                continue

            i, ecg_np, meta = r
            ecg_mm[i] = ecg_np.astype(np.float16, copy=False)
            valid_mask[i] = 1
            meta["ecg_index"] = i
            meta_rows.append(meta)

    del ecg_mm

    valid_idx = np.flatnonzero(valid_mask)
    np.save(out_valid, valid_idx)

    base_meta_df = pd.DataFrame(meta_rows).sort_values("ecg_index").reset_index(drop=True)
    base_meta_df.to_csv(out_base_meta, index=False)

    print("\nECG export complete ✅")
    print(f"Saved ECG -> {out_ecg}")
    print(f"Saved valid_idx -> {out_valid}")
    print(f"Saved base metadata -> {out_base_meta}")
    print(f"Valid rows: {len(valid_idx)} / {N}")


def export_ptbxl_labels_and_metadata(dataset_root: str, out_dir: str, label_type: str):
    """
    Creates per-label-type:
      - ptbxl_{label_type}_labels.npy   (N, C) uint8
      - ptbxl_{label_type}_metadata.csv (includes split + label_ columns)
      - ptbxl_{label_type}_class_to_index.json
    Uses base metadata + valid_idx (hard mask).
    """
    assert label_type in {"form", "rhythm", "diagnostic_class", "diagnostic_subclass"}

    # load base/meta
    base_meta = pd.read_csv(os.path.join(out_dir, "ptbxl_base_metadata.csv"))
    valid_idx = np.load(os.path.join(out_dir, "ptbxl_valid_idx.npy")).astype(np.int64)

    # load raw ptbxl df + scp
    df = pd.read_csv(os.path.join(dataset_root, "ptbxl_database.csv"))
    scp = pd.read_csv(os.path.join(dataset_root, "scp_statements.csv"))

    vocab, scp_to_target = build_vocab_and_mapping(scp, label_type)
    class_to_index = {c: i for i, c in enumerate(vocab)}
    C = len(vocab)
    N = len(df)

    out_lbl = os.path.join(out_dir, f"ptbxl_{label_type}_labels.npy")
    out_meta = os.path.join(out_dir, f"ptbxl_{label_type}_metadata.csv")
    out_map = os.path.join(out_dir, f"ptbxl_{label_type}_class_to_index.json")

    with open(out_map, "w") as f:
        json.dump({k: int(v) for k, v in class_to_index.items()}, f, indent=2, sort_keys=True)

    # prealloc labels
    lbl_mm = open_memmap(out_lbl, mode="w+", dtype=np.uint8, shape=(N, C))

    # build meta rows by iterating ONLY valid indices (fast + safe)
    # we fill labels for invalid rows with 0 by default
    meta_rows = []

    print(f"\nExporting labels for {label_type} | classes={C}")
    for i in tqdm(valid_idx, desc=f"Labels {label_type}"):
        row = df.iloc[int(i)]

        # parse scp_codes dict
        try:
            scp_codes = ast.literal_eval(row["scp_codes"])
            active_scp = list(scp_codes.keys())
        except Exception:
            active_scp = []

        active_targets = set()
        for sc in active_scp:
            t = scp_to_target.get(str(sc))
            if t is None:
                continue
            active_targets.add(str(t))

        y = np.zeros(C, dtype=np.uint8)
        for t in active_targets:
            j = class_to_index.get(t, None)
            if j is not None:
                y[j] = 1

        lbl_mm[int(i)] = y

        num_labels = int(y.sum())
        all_zero = int(num_labels == 0)

        meta_rows.append({
            "ecg_index": int(i),
            "num_labels": num_labels,
            "all_zero": all_zero,
            **{f"label_{vocab[j]}": int(y[j]) for j in range(C)}
        })

    del lbl_mm

    # merge base meta (split info) + per task label meta
    task_meta = pd.DataFrame(meta_rows)

    # hard-safe: keep only base rows that are valid
    base_meta = base_meta[base_meta["ecg_index"].isin(valid_idx)]

    merged = base_meta.merge(task_meta, on="ecg_index", how="inner")
    merged = merged.sort_values("ecg_index").reset_index(drop=True)

    merged.to_csv(out_meta, index=False)

    print(f"Saved labels -> {out_lbl}")
    print(f"Saved metadata -> {out_meta}")
    print(f"Saved class map -> {out_map}")


def main(config=None):
    config = config or load_config()
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    # You can pass either PTB-XL folder OR physionet mirror root.
    base_dir = dataset_root(config, "PTBXL", "ptb-xl", "1.0.3")
    out_dir = config.raw_data_dir
    os.makedirs(out_dir, exist_ok=True)

    dataset_root = resolve_ptbxl_root(base_dir)

    # 1) export ECG once (500Hz)
    export_ptbxl_ecg_500hz(dataset_root, out_dir, n_workers=min(os.cpu_count() or 1, 40))

    # 2) export label npy + metadata csv for each label type
    for lt in ["form", "rhythm", "diagnostic_class", "diagnostic_subclass"]:
        export_ptbxl_labels_and_metadata(dataset_root, out_dir, lt)

    print("\n✅ PTB-XL export done for all 4 label types.")

if __name__ == "__main__":
    main()
    
