import os
import re
import glob
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import numpy as np
import pandas as pd
import wfdb
from tqdm import tqdm
from numpy.lib.format import open_memmap


DX_LINE_RE = re.compile(r"^Dx\s*:\s*(.*)$", re.IGNORECASE)
TARGET_LEN = 5000


# ----------------------------
# multiprocessing worker state
# ----------------------------
_G = {}


def _init_worker(paths, vocab, class_to_index, codes_classes):
    global _G
    _G["paths"] = paths
    _G["vocab"] = vocab
    _G["class_to_index"] = class_to_index
    _G["codes_classes"] = codes_classes


def _find_dx_codes(header) -> list:
    """Return list of SNOMED codes from the Dx line in header comments."""
    for comment in header.comments:
        m = DX_LINE_RE.match(comment.strip())
        if m:
            raw = m.group(1).strip()
            if not raw:
                return []
            return [c.strip() for c in raw.split(",") if c.strip()]
    return []


def _crop_or_pad_numpy(ecg: np.ndarray, target_len: int = TARGET_LEN) -> tuple[np.ndarray, int, int, int]:
    """
    ecg: (12, T)
    Returns: (ecg_fixed, was_cropped, was_padded, pad_amount)
    """
    T = ecg.shape[1]
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

    return ecg, was_cropped, was_padded, pad_amount


def _process_one(i: int):
    """
    Returns:
      (i, ecg_np, label_np, meta_dict)
    or None if sample should be skipped.
    """
    base = _G["paths"][i]
    vocab = _G["vocab"]
    class_to_index = _G["class_to_index"]
    codes_classes = _G["codes_classes"]

    # ---- read header ----
    try:
        header = wfdb.rdheader(base)
    except Exception:
        return None

    codes = _find_dx_codes(header)

    # ---- build acronym labels ----
    acronyms = sorted(
        {codes_classes[c] if c in codes_classes else str(c) for c in codes}
    )
    # label vector (uint8 multi-hot)
    y = np.zeros(len(vocab), dtype=np.uint8)
    missing_codes = []
    for lab in acronyms:
        j = class_to_index.get(lab, None)
        if j is not None:
            y[j] = 1
        else:
            missing_codes.append(lab)


    # ---- read signal ----
    try:
        signal, _ = wfdb.rdsamp(base)  # (T, leads)
        ecg = np.ascontiguousarray(signal.T, dtype=np.float32)  # (leads, T)
        record = wfdb.rdrecord(base)
    #    print(record.units)
        assert record.units == ['mV'] * 12
    except Exception:
        return None

    # ---- enforce 12 lead ----
    if ecg.ndim != 2 or ecg.shape[0] != 12:
        return None

    orig_len = int(ecg.shape[1])
    if orig_len <= 0:
        return None

    # flags before modification
    had_nan = int((not np.isfinite(ecg).all()) or (not np.isfinite(y).all()))

    # ---- crop or pad to 5000 ----
    ecg_fixed, was_cropped, was_padded, pad_amount = _crop_or_pad_numpy(ecg, TARGET_LEN)

    if ecg_fixed.shape != (12, TARGET_LEN):
        return None

    num_labels = int(y.sum())
    all_zero = int(num_labels == 0)

    record_id = os.path.basename(base)

    meta = {
        "record_id": record_id,
        "orig_len": orig_len,
        "final_len": TARGET_LEN,
        "was_cropped": was_cropped,
        "was_padded": was_padded,
        "pad_amount": pad_amount,
        "had_nan": had_nan,
        "all_zero": all_zero,
        "num_labels": num_labels,
        "missing_codes": missing_codes
    }

    # expand label columns
    for j, name in enumerate(vocab):
        meta[f"label_{name}"] = int(y[j])

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

    return (i, ecg_fixed, y, meta)


def _resolve_records_root(base_dir: str) -> str:
    c = os.path.join(base_dir, "WFDBRecords")
    if os.path.isdir(c) and glob.glob(os.path.join(c, "**", "*.hea"), recursive=True):
        return c
    raise FileNotFoundError("Could not locate WFDBRecords with .hea files under base_dir")


def _find_hea_bases(records_root: str) -> list[str]:
    bases = []
    for dirpath, _, filenames in os.walk(records_root):
        for fn in filenames:
            if fn.lower().endswith(".hea"):
                bases.append(os.path.join(dirpath, os.path.splitext(fn)[0]))
    bases.sort()
    return bases


ZERO_COUNT_LABELS = {
    "2AVB2",
    "AVNRT",
    "IDC",
    "LBBB",
    "LBBBB",
    "LVQRSCL",
    "LVQRSLL",
    "MI",
    "MIBW",
    "MIFW",
    "MILW",
    "SAAWR",
    "WAVN"
}

LESS_THAN_2_COUNT_LABELS = {
   "3AVB", 
   "ABI"
}

def main(config):

    base_dir =  config["dataset_roots"]["CSN"]
    out_dir = config["raw_data_dir"]
    os.makedirs(out_dir, exist_ok=True)

    done_file = os.path.join(out_dir, "csn.done")

    if os.path.exists(done_file):
        print(f"✅ Skipping CSN export, found {done_file}")
        return

    out_ecg_npy = os.path.join(out_dir, "csn_ecg.npy")
    out_label_npy = os.path.join(out_dir, "csn_labels.npy")
    out_meta_csv = os.path.join(out_dir, "csn_metadata.csv")
    out_class_map = os.path.join(out_dir, "csn_class_to_index.json")

    # ---- load code map ----
    codes_path = os.path.join(base_dir, "ConditionNames_SNOMED-CT.csv")
    snomed_codes = pd.read_csv(codes_path)

    # class vocab + mapping code->acronym
    vocab = sorted(set(snomed_codes["Acronym Name"].astype(str).tolist()))
    vocab = [v for v in vocab if v not in ZERO_COUNT_LABELS and v not in LESS_THAN_2_COUNT_LABELS]

    class_to_index = {c: i for i, c in enumerate(vocab)}
    codes_classes = {str(code): str(name) for code, name in zip(snomed_codes["Snomed_CT"], snomed_codes["Acronym Name"])}

    with open(out_class_map, "w") as f:
        json.dump({k: int(v) for k, v in class_to_index.items()}, f, indent=2, sort_keys=True)
    print(f"Saved class_to_index -> {out_class_map}")

    # ---- find all WFDB record bases ----
    records_root = _resolve_records_root(base_dir)
    paths = _find_hea_bases(records_root)

    if len(paths) == 0:
        raise FileNotFoundError("No .hea files found under WFDBRecords")

    N_total = len(paths)
    C = len(vocab)

    print(f"Found {N_total} records.")
    print(f"Num classes: {C}")

    # ---- preallocate memmaps ----
    ecg_mm = open_memmap(out_ecg_npy, mode="w+", dtype=np.float16, shape=(N_total, 12, TARGET_LEN))
    label_mm = open_memmap(out_label_npy, mode="w+", dtype=np.uint8, shape=(N_total, C))

    meta_rows = []

    n_workers =  config["preprocess_workers"]
    print(f"Using {n_workers} workers...")

    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_worker,
        initargs=(paths, vocab, class_to_index, codes_classes),
    ) as ex:

        futures = [ex.submit(_process_one, i) for i in range(N_total)]

        for fut in tqdm(as_completed(futures), total=N_total, desc="CSN Processing + writing"):
            r = fut.result()
            if r is None:
                continue

            i, ecg_np, y_np, meta = r
            ecg_mm[i] = ecg_np.astype(np.float16, copy=False)
            label_mm[i] = y_np.astype(np.uint8, copy=False)
            meta["ecg_index"] = i

            meta_rows.append(meta)

    # flush
    del ecg_mm
    del label_mm

    meta_df = pd.DataFrame(meta_rows)
    meta_df = meta_df.sort_values("ecg_index").reset_index(drop=True)

    meta_df.to_csv(out_meta_csv, index=False)

    print("\nExport complete ✅")
    print(f"Saved ECGs      -> {out_ecg_npy}")
    print(f"Saved labels    -> {out_label_npy}")
    print(f"Saved metadata  -> {out_meta_csv}")

    with open(done_file, "w") as f:
        f.write("done\n")

    print("\nExport complete")
