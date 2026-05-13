import os
import re
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import numpy as np
import pandas as pd
import wfdb
import torch
import torch.nn.functional as F
from numpy.lib.format import open_memmap
from tqdm import tqdm
from benchmark.preprocess.ecg_utils import crop_ecg

DX_RE = re.compile(r"^Dx\s*:\s*(.*)$", re.IGNORECASE)

VALID_SNOMED_CODES = {
    "426783006": "NSR",    # Normal sinus rhythm
    "164889003": "AF",     # Atrial fibrillation
    "270492004": "IAVB",   # 1st degree AV block
    "164909002": "LBBB",   # Left bundle branch block
    "59118001":  "RBBB",   # Right bundle branch block
    "284470004": "PAC",    # Premature atrial contraction
    "164884008": "PVC",    # Premature ventricular contraction
    "429622005": "STD",    # ST depression
    "164931005": "STE",    # ST elevation
}

# ----------------------------
# multiprocessing worker state
# ----------------------------
_G = {}

def _init_worker(paths, vocab, class_to_index):
    global _G
    _G["paths"] = paths
    _G["vocab"] = vocab
    _G["class_to_index"] = class_to_index

def _extract_codes(header):
    for comment in header.comments:
        m = DX_RE.match(comment)
        if m:
            return [x.strip() for x in m.group(1).split(",") if x.strip()]
    return []


TARGET_LEN = 5000

def _process_one(i: int):
    """
    Returns:
      (i, ecg_np, label_np, meta_dict_without_ecg_index)
    or None if sample is malformed / unreadable.
    """
    base = _G["paths"][i]
    vocab = _G["vocab"]
    class_to_index = _G["class_to_index"]

    # ---- read header ----
    try:
        header = wfdb.rdheader(base)
    except Exception:
        return None

    codes = _extract_codes(header)

    # ---- label vector ----
    y = torch.zeros(len(vocab), dtype=torch.float32)
    for c in codes:
        if c in VALID_SNOMED_CODES:
            lbl = VALID_SNOMED_CODES[c]
            j = class_to_index[lbl]
            y[j] = 1.0
        else:
            print("Unknown:", c)

    # ---- read signal ----
    
    try:
        signal, _ = wfdb.rdsamp(base)
        ecg = np.ascontiguousarray(signal.T, dtype=np.float32)  # (12, T)
        ecg = torch.tensor(ecg, dtype=torch.float32)
        record = wfdb.rdrecord(base)
        assert record.units == ['mV'] * 12
    except Exception:
        return None

    # ---- flags ----
    had_nan = int((not torch.isfinite(ecg).all()) or (not torch.isfinite(y).all()))

    # ---- shape checks ----
    if ecg.ndim != 2:
        return None
    if ecg.shape[0] != 12:
        return None

    orig_len = int(ecg.shape[1])
    if orig_len <= 0:
        return None

    was_cropped = int(orig_len > TARGET_LEN)
    was_padded  = int(orig_len < TARGET_LEN)
    pad_amount  = max(0, TARGET_LEN - orig_len)

    # ---- crop OR pad ----
    if orig_len > TARGET_LEN:
        ecg = crop_ecg(ecg, target_len=TARGET_LEN)
    elif orig_len < TARGET_LEN:
        #center padd
        pad_amount = TARGET_LEN - orig_len
        pad_left = pad_amount // 2
        pad_right = pad_amount - pad_left

        # pad last dimension: (left, right)
        ecg = F.pad(ecg, (pad_left, pad_right), mode="constant", value=0.0)

    # ---- final enforce ----
    if tuple(ecg.shape) != (12, TARGET_LEN):
        return None

    # ---- label flags ----
    num_labels = int(y.sum().item())
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
    }

    for j, name in enumerate(vocab):
        meta[f"label_{name}"] = int(y[j].item())
    
    # --- stats ---
    # per-lead (axis=1 → across time)
    lead_means = np.mean(ecg.numpy(), axis=1)   # (12,)
    lead_stds  = np.std(ecg.numpy(), axis=1)    # (12,)

    # global (flatten everything)
    ecg_mean = float(np.mean(ecg.numpy()))
    ecg_std  = float(np.std(ecg.numpy()))

    meta["ecg_mean"] = ecg_mean
    meta["ecg_std"] = ecg_std
    meta["lead_means"] =  lead_means.tolist()   # store as list for JSON/CSV
    meta["lead_stds"] =  lead_stds.tolist()

    return (i, ecg.numpy(), y.numpy(), meta)




def main(config):
    base_dir =  config["dataset_roots"]["PTBXL"]
    out_dir = config["raw_data_dir"]
    os.makedirs(out_dir, exist_ok=True)

    dataset_root = resolve_ptbxl_root(base_dir)

    # 1) export ECG once (500Hz)
    ecg_done = os.path.join(out_dir, "ptbxl_ecg_500hz.done")
    if os.path.exists(ecg_done):
        print(f"✅ Skipping ECG export, found {ecg_done}")
    else:
        export_ptbxl_ecg_500hz(
            dataset_root,
            out_dir,
            n_workers=config["preprocess_workers"]
        )
        with open(ecg_done, "w") as f:
            f.write("done\n")
        print(f"Saved done marker -> {ecg_done}")


    # 2) export label npy + metadata csv for each label type
    for lt in ["form", "rhythm", "diagnostic_class", "diagnostic_subclass"]:
        label_done = os.path.join(out_dir, f"ptbxl_{lt}.done")

        if os.path.exists(label_done):
            print(f"✅ Skipping {lt}, found {label_done}")
            continue

        export_ptbxl_labels_and_metadata(dataset_root, out_dir, lt)

        with open(label_done, "w") as f:
            f.write("done\n")

        print(f"Saved done marker -> {label_done}")


    print("\n✅ PTB-XL export done for all 4 label types.")


def main(config):

    base_dir =  config["dataset_roots"]["CPSC"]
    out_dir = config["raw_data_dir"]
    os.makedirs(out_dir, exist_ok=True)

    done_file = os.path.join(out_dir, "cpsc2018.done")

    if os.path.exists(done_file):
        print(f"✅ Skipping CPSC 2018 export, found {done_file}")
        return

    out_ecg_npy   = os.path.join(out_dir, "cpsc2018_ecg.npy")
    out_label_npy = os.path.join(out_dir, "cpsc2018_labels.npy")
    out_meta_csv  = os.path.join(out_dir, "cpsc2018_metadata.csv")
    out_class_index_json = os.path.join(out_dir, "cpsc2018_class_to_index.json")

    # ---- build paths (fast scan) ----
    paths = []
    for dirpath, _, filenames in os.walk(base_dir):
        for fn in filenames:
            if fn.lower().endswith(".hea"):
                paths.append(os.path.join(dirpath, os.path.splitext(fn)[0]))
    paths.sort()

    if len(paths) == 0:
        raise FileNotFoundError("No .hea files found under base_dir")

    # ---- vocab/class mapping (same as your dataset) ----
    vocab = sorted(set(VALID_SNOMED_CODES.values()))
    class_to_index = {c: i for i, c in enumerate(vocab)}

    # Save class->index mapping as JSON
    with open(out_class_index_json, "w") as f:
        json.dump(
            {k: int(v) for k, v in class_to_index.items()},
            f,
            indent=2,
            sort_keys=True
        )
    print(f"Saved class_to_index to: {out_class_index_json}")

    # ---- multiprocessing ----
    n_workers = config["preprocess_workers"]
    print(f"Found {len(paths)} records. Using {n_workers} workers...")

    N_total = len(paths)
    C = len(vocab)

    # make disk-backed .npy files
    # float16 = 2x smaller + faster writes
    ecg_mm = open_memmap(out_ecg_npy, mode="w+", dtype=np.float32, shape=(N_total, 12, TARGET_LEN))

    # labels stored as uint8 instead of float32 (tiny)
    label_mm = open_memmap(out_label_npy, mode="w+", dtype=np.uint8, shape=(N_total, C))

    meta_rows = []

    with ProcessPoolExecutor(max_workers=n_workers,initializer=_init_worker,
            initargs=(paths, vocab, class_to_index),
        ) as ex:

        futures = [ex.submit(_process_one, i) for i in range(N_total)]

        for fut in tqdm(as_completed(futures), total=N_total, desc="Processing + writing"):
            r = fut.result()
            if r is None:
                continue

            i, ecg_np, label_np, meta = r

            # write directly into the preallocated .npy file
            ecg_mm[i] = ecg_np.astype(np.float16, copy=False)
            label_mm[i] = label_np.astype(np.uint8, copy=False)

            meta["ecg_index"] = i   # <-- IMPORTANT: direct index into memmap
            meta_rows.append(meta)

    del ecg_mm
    del label_mm

    # write metadata
    meta_df = pd.DataFrame(meta_rows).sort_values("ecg_index")
    meta_df.to_csv(out_meta_csv, index=False)

    print("\nExport complete")

    with open(done_file, "w") as f:
        f.write("done\n")

    print("\nExport complete")
