#!/usr/bin/env python3
import os
import random
import pickle
import numpy as np
import wfdb
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# =========================
# CONFIG
# =========================
ROOT = Path("/storage/shared/mimic-iv/raw/ecgs/1.0")
OUT  = Path("/storage/shared/mimic-iv/preprocessed_mimic4ecg_clocs")

FS = 500
SEG_10S = 5000

DESIRED_LEADS = [
    "I","II","III","aVR","aVL","aVF",
    "V1","V2","V3","V4","V5","V6"
]

VAL_FRACTION = 0.1
RANDOM_SEED = 42
FRAME_DTYPE = np.float16

N_WORKERS = max(1, cpu_count() - 2)
TASK = "contrastive_msml"
LEADS_NAME = "leads_all"

# =========================
# HELPERS
# =========================
def get_subject_id(record_path: Path) -> int:
    # .../pXXXXXX/sXXXXXX/XXXXXX
    return int(record_path.parts[-3][1:])

def scan_record(record: Path):
    """
    Header-only scan: check lead availability + compute #10s segments.
    Returns minimal metadata for the writing stage.
    """
    try:
        header = wfdb.rdheader(str(record))
        lead_names = list(header.sig_name)

        if not all(l in lead_names for l in DESIRED_LEADS):
            return None

        idx = [lead_names.index(l) for l in DESIRED_LEADS]
        n_samples = int(header.sig_len)
        n_10s = n_samples // SEG_10S
        if n_10s <= 0:
            return None

        pid = get_subject_id(record)
        return (str(record), idx, n_10s, pid)

    except Exception:
        return None

# Globals set in initializer (per worker)
G_TRAIN_FRAMES = None
G_TRAIN_LABELS = None
G_TRAIN_PIDS   = None
G_VAL_FRAMES   = None
G_VAL_LABELS   = None
G_VAL_PIDS     = None

def init_writer(train_frames_path, train_labels_path, train_pids_path,
                val_frames_path, val_labels_path, val_pids_path):
    global G_TRAIN_FRAMES, G_TRAIN_LABELS, G_TRAIN_PIDS
    global G_VAL_FRAMES, G_VAL_LABELS, G_VAL_PIDS

    # r+ so workers can write
    G_TRAIN_FRAMES = np.load(train_frames_path, mmap_mode="r+")
    G_TRAIN_LABELS = np.load(train_labels_path, mmap_mode="r+")
    G_TRAIN_PIDS   = np.load(train_pids_path,   mmap_mode="r+")

    G_VAL_FRAMES = np.load(val_frames_path, mmap_mode="r+")
    G_VAL_LABELS = np.load(val_labels_path, mmap_mode="r+")
    G_VAL_PIDS   = np.load(val_pids_path,   mmap_mode="r+")

def write_record(job):
    """
    job = (record_str, lead_idx, n_10s, pid, split, start_offset)
    Each worker writes into its own disjoint slice => safe.
    """
    record_str, lead_idx, n_10s, pid, split, start_offset = job
    try:
        sig, _ = wfdb.rdsamp(record_str, channels=lead_idx)  # (N,12) time-major

        # pick correct memmaps
        if split == "train":
            frames_mm = G_TRAIN_FRAMES
            labels_mm = G_TRAIN_LABELS
            pids_mm   = G_TRAIN_PIDS
        else:
            frames_mm = G_VAL_FRAMES
            labels_mm = G_VAL_LABELS
            pids_mm   = G_VAL_PIDS

        # write segments
        base = start_offset
        for i in range(n_10s):
            s0 = i * SEG_10S
            s1 = (i + 1) * SEG_10S
            frame = sig[s0:s1, :]  # (5000,12)

            if frame.shape != (SEG_10S, 12):
                continue
            if not np.isfinite(frame).all():
                continue

            frames_mm[base + i] = frame.astype(FRAME_DTYPE, copy=False)
            labels_mm[base + i] = 0
            pids_mm[base + i]   = pid

        return n_10s

    except Exception:
        return 0

def main():
    # =========================
    # FIND RECORDS
    # =========================
    records = [hea.with_suffix("") for hea in ROOT.rglob("*.hea")]
    print(f"Total records discovered: {len(records)}")

    # =========================
    # SPLIT SUBJECTS
    # =========================
    subjects = sorted({get_subject_id(r) for r in records})
    random.seed(RANDOM_SEED)
    random.shuffle(subjects)
    n_val = int(len(subjects) * VAL_FRACTION)
    val_subjects = set(subjects[:n_val])

    print(f"Train subjects: {len(subjects) - n_val}, Val subjects: {n_val}")
    print(f"Workers: {N_WORKERS}")

    # =========================
    # PASS 1: header-only scan
    # =========================
    meta = []
    with Pool(N_WORKERS) as pool:
        for out in tqdm(pool.imap_unordered(scan_record, records, chunksize=64), total=len(records)):
            if out is not None:
                meta.append(out)

    if not meta:
        raise RuntimeError("No valid records found (missing leads / too short).")

    # assign split per record
    meta2 = []
    for record_str, idx, n_10s, pid in meta:
        split = "val" if pid in val_subjects else "train"
        meta2.append((record_str, idx, n_10s, pid, split))

    # compute offsets into the big arrays
    train_offsets = {}
    val_offsets   = {}

    train_total = 0
    val_total = 0
    jobs = []
    for record_str, idx, n_10s, pid, split in meta2:
        if split == "train":
            start = train_total
            train_total += n_10s
            train_offsets[record_str] = start
        else:
            start = val_total
            val_total += n_10s
            val_offsets[record_str] = start

        jobs.append((record_str, idx, n_10s, pid, split, start))

    print(f"Total 10s segments => train: {train_total:,}, val: {val_total:,}")

    # =========================
    # OUTPUT DIR + MEMMAP FILES
    # =========================
    save_dir = OUT / "patient_data" / TASK / LEADS_NAME
    save_dir.mkdir(parents=True, exist_ok=True)

    train_frames_path = save_dir / "train_frames.npy"
    train_labels_path = save_dir / "train_labels.npy"
    train_pids_path   = save_dir / "train_pids.npy"

    val_frames_path = save_dir / "val_frames.npy"
    val_labels_path = save_dir / "val_labels.npy"
    val_pids_path   = save_dir / "val_pids.npy"

    # Create memmap-backed .npy arrays (fast random access later)
    train_frames = np.lib.format.open_memmap(
        train_frames_path, mode="w+", dtype=FRAME_DTYPE, shape=(train_total, SEG_10S, 12)
    )
    train_labels = np.lib.format.open_memmap(
        train_labels_path, mode="w+", dtype=np.int64, shape=(train_total,)
    )
    train_pids = np.lib.format.open_memmap(
        train_pids_path, mode="w+", dtype=np.int64, shape=(train_total,)
    )

    val_frames = np.lib.format.open_memmap(
        val_frames_path, mode="w+", dtype=FRAME_DTYPE, shape=(val_total, SEG_10S, 12)
    )
    val_labels = np.lib.format.open_memmap(
        val_labels_path, mode="w+", dtype=np.int64, shape=(val_total,)
    )
    val_pids = np.lib.format.open_memmap(
        val_pids_path, mode="w+", dtype=np.int64, shape=(val_total,)
    )

    # initialize labels
    train_labels[:] = 0
    val_labels[:]   = 0

    # flush headers
    del train_frames, train_labels, train_pids, val_frames, val_labels, val_pids

    # =========================
    # PASS 2: parallel read + direct write
    # =========================
    with Pool(
        N_WORKERS,
        initializer=init_writer,
        initargs=(
            str(train_frames_path), str(train_labels_path), str(train_pids_path),
            str(val_frames_path),   str(val_labels_path),   str(val_pids_path),
        ),
    ) as pool:
        written = 0
        for n in tqdm(pool.imap_unordered(write_record, jobs, chunksize=8), total=len(jobs)):
            written += int(n)

    print(f"✅ Done writing (expected ~{train_total+val_total:,} segments, processed jobs wrote ~{written:,}).")

    # =========================
    # INDEX (compat style)
    # =========================
    index = {
        "ecg": {
            1.0: {
                "train": {"labelled": {}},
                "val": {}
            }
        }
    }

    index["ecg"][1.0]["train"]["labelled"] = {
        "format": "npy_memmap_v1",
        "frames": "train_frames.npy",
        "labels": "train_labels.npy",
        "pids":   "train_pids.npy",
        "length": int(train_total),
        "frame_dtype": str(np.dtype(FRAME_DTYPE)),
        "shape": [SEG_10S, 12],
        "task": TASK,
        "leads": DESIRED_LEADS,
    }

    index["ecg"][1.0]["val"] = {
        "format": "npy_memmap_v1",
        "frames": "val_frames.npy",
        "labels": "val_labels.npy",
        "pids":   "val_pids.npy",
        "length": int(val_total),
        "frame_dtype": str(np.dtype(FRAME_DTYPE)),
        "shape": [SEG_10S, 12],
        "task": TASK,
        "leads": DESIRED_LEADS,
    }

    # write same index into 3 pkls for old code paths
    for name in ["frames_phases_mimiciv.pkl", "labels_phases_mimiciv.pkl", "pid_phases_mimiciv.pkl"]:
        with open(save_dir / name, "wb") as f:
            pickle.dump(index, f, protocol=4)

    print(f"✅ Saved memmap dataset to: {save_dir}")
    print("✅ Index PKLs written for compatibility")

if __name__ == "__main__":
    main()
