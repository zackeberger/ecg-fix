import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from tqdm import tqdm
from numpy.lib.format import open_memmap

from benchmark.preprocess.ecg_utils import resample_ecg_linear


ECHONEXT_MEAN = np.array(
    [
        5.3913547, 4.08127771, -1.18338815, -4.72457889,
        0.59169407, -0.59169407, -4.25676191, -1.44444217,
        -0.34211766, 3.35517074, 5.18700818, 5.53535442,
    ],
    dtype=np.float32,
)

ECHONEXT_STD = np.array(
    [
        31.1319959, 29.16014537, 30.41666735, 26.0615433,
        15.20833367, 15.20833367, 38.35645889, 53.62205647,
        58.92661479, 49.45070532, 43.92990916, 38.29998916,
    ],
    dtype=np.float32,
)

UNNORMALIZE_ECHONEXT = True
TARGET_LEN = 5000
SPLITS = ["train", "val", "test"]

ECHONEXT_BINARY_LABELS = [
    "lvef_lte_45_flag",
    "lvwt_gte_13_flag",
    "aortic_stenosis_moderate_or_greater_flag",
    "aortic_regurgitation_moderate_or_greater_flag",
    "mitral_regurgitation_moderate_or_greater_flag",
    "tricuspid_regurgitation_moderate_or_greater_flag",
    "pulmonary_regurgitation_moderate_or_greater_flag",
    "rv_systolic_dysfunction_moderate_or_greater_flag",
    "pericardial_effusion_moderate_large_flag",
    "pasp_gte_45_flag",
    "tr_max_gte_32_flag",
    "shd_moderate_or_greater_flag",
]


_G = {}


def echonext_unnormalize(ecg_12xT: np.ndarray) -> np.ndarray:
    return ecg_12xT * ECHONEXT_STD[:, None] + ECHONEXT_MEAN[:, None]


def _init_worker(wave_paths: dict):
    """
    Each worker opens its own mmap handles once.
    This avoids passing huge waveform arrays through multiprocessing queues.
    """
    global _G
    _G["waves"] = {
        split: np.load(path, mmap_mode="r")
        for split, path in wave_paths.items()
    }


def _process_one(task):
    """
    Returns:
      (global_i, ecg_np, label_np, meta_dict)
    or None if malformed.

    task:
      (split, local_i, global_i, row_dict, label_np)
    """
    split, local_i, global_i, row_dict, label_np = task

    try:
        waves = _G["waves"][split]

        # EchoNext waveforms: (N, 1, 2500, 12)
        w = waves[local_i]
        ecg = w.squeeze(0).T.astype(np.float32, copy=False)  # (12, 2500)

        if UNNORMALIZE_ECHONEXT:
            ecg = echonext_unnormalize(ecg)

        ecg = resample_ecg_linear(ecg, target_len=TARGET_LEN)  # (12, 5000)

        if ecg.shape != (12, TARGET_LEN):
            return None

        label = np.asarray(label_np, dtype=np.uint8)

        had_nan = int((not np.isfinite(ecg).all()) or (not np.isfinite(label).all()))
        num_labels = int(label.sum())
        all_zero = int(num_labels == 0)

        lead_means = np.mean(ecg, axis=1)
        lead_stds = np.std(ecg, axis=1)

        row = dict(row_dict)
        row["ecg_index"] = int(global_i)
        row["orig_len"] = 2500
        row["final_len"] = TARGET_LEN
        row["had_nan"] = had_nan
        row["all_zero"] = all_zero
        row["num_labels"] = num_labels
        row["split"] = split
        row["ecg_mean"] = float(np.mean(ecg))
        row["ecg_std"] = float(np.std(ecg))
        row["lead_means"] = lead_means.tolist()
        row["lead_stds"] = lead_stds.tolist()

        return global_i, ecg, label, row

    except Exception:
        return None


def _resolve_echonext_root(config: dict) -> str:
    if "dataset_roots" not in config or "ECHO_NEXT" not in config["dataset_roots"]:
        raise KeyError("config must contain config['dataset_roots']['ECHO_NEXT']")
    return config["dataset_roots"]["ECHO_NEXT"]


def _build_tasks(df: pd.DataFrame, wave_paths: dict):
    """Build deterministic global indexing: train -> val -> test."""
    tasks = []
    split_sizes = {}
    offset = 0

    for split in SPLITS:
        df_split = df[df["split"] == split].reset_index(drop=True)
        if len(df_split) == 0:
            raise RuntimeError(f"No rows in metadata for split={split}")

        if not os.path.exists(wave_paths[split]):
            raise FileNotFoundError(f"Missing waveform file: {wave_paths[split]}")

        waves = np.load(wave_paths[split], mmap_mode="r")
        if len(waves) != len(df_split):
            raise RuntimeError(
                f"Mismatch split={split}: metadata rows={len(df_split)} vs waveforms={len(waves)}"
            )

        split_sizes[split] = len(df_split)

        label_values = df_split[ECHONEXT_BINARY_LABELS].values.astype(np.uint8)
        records = df_split.to_dict("records")

        for local_i, row in enumerate(records):
            global_i = offset + local_i
            tasks.append((split, local_i, global_i, row, label_values[local_i]))

        offset += len(df_split)

    return tasks, split_sizes, offset


def main(config: dict):
    out_dir = config["raw_data_dir"]
    base_dir = _resolve_echonext_root(config)
    n_workers = int(config.get("preprocess_workers", os.cpu_count() or 1))
    os.makedirs(out_dir, exist_ok=True)

    done_file = os.path.join(out_dir, "echonext.done")
    if os.path.exists(done_file):
        print(f"✅ Skipping EchoNext export, found {done_file}")
        return

    in_meta_csv = os.path.join(base_dir, "echonext_metadata_100k.csv")

    wave_paths = {
        "train": os.path.join(base_dir, "EchoNext_train_waveforms.npy"),
        "val": os.path.join(base_dir, "EchoNext_val_waveforms.npy"),
        "test": os.path.join(base_dir, "EchoNext_test_waveforms.npy"),
    }

    out_ecg_npy = os.path.join(out_dir, "echonext_ecg.npy")
    out_label_npy = os.path.join(out_dir, "echonext_labels.npy")
    out_meta_csv = os.path.join(out_dir, "echonext_metadata.csv")
    out_valid_idx = os.path.join(out_dir, "echonext_valid_idx.npy")
    out_class_map = os.path.join(out_dir, "echonext_class_to_index.json")

    df = pd.read_csv(in_meta_csv)
    df["split"] = df["split"].replace({"valid": "val"})
    df = df[df["split"].isin(SPLITS)].reset_index(drop=True)

    for col in ECHONEXT_BINARY_LABELS:
        if col not in df.columns:
            raise ValueError(f"Missing label column in metadata: {col}")

    tasks, split_sizes, n_total = _build_tasks(df, wave_paths)
    n_classes = len(ECHONEXT_BINARY_LABELS)

    print("Split sizes:", split_sizes)
    print("Total N:", n_total)
    print(f"Using {n_workers} workers...")

    class_to_index = {c: i for i, c in enumerate(ECHONEXT_BINARY_LABELS)}
    with open(out_class_map, "w") as f:
        json.dump(class_to_index, f, indent=2, sort_keys=True)
    print(f"Saved class map -> {out_class_map}")

    ecg_mm = open_memmap(out_ecg_npy, mode="w+", dtype=np.float16, shape=(n_total, 12, TARGET_LEN))
    lbl_mm = open_memmap(out_label_npy, mode="w+", dtype=np.uint8, shape=(n_total, n_classes))

    meta_rows = []
    valid_idx = []

    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_worker,
        initargs=(wave_paths,),
    ) as ex:
        futures = [ex.submit(_process_one, task) for task in tasks]

        for fut in tqdm(as_completed(futures), total=len(futures), desc="EchoNext processing + writing"):
            result = fut.result()
            if result is None:
                continue

            global_i, ecg_np, label_np, meta = result
            ecg_mm[global_i] = ecg_np.astype(np.float16, copy=False)
            lbl_mm[global_i] = label_np.astype(np.uint8, copy=False)
            meta_rows.append(meta)
            valid_idx.append(global_i)

    del ecg_mm
    del lbl_mm

    valid_idx = np.asarray(sorted(valid_idx), dtype=np.int64)
    np.save(out_valid_idx, valid_idx)

    meta_df = pd.DataFrame(meta_rows).sort_values("ecg_index").reset_index(drop=True)
    meta_df.to_csv(out_meta_csv, index=False)

    with open(done_file, "w") as f:
        f.write("done\n")

    print("\nExport complete ✅")
    print(f"Saved ECGs      -> {out_ecg_npy}")
    print(f"Saved labels    -> {out_label_npy}")
    print(f"Saved metadata  -> {out_meta_csv}")
    print(f"Saved valid_idx -> {out_valid_idx}")
    print(f"Saved done file -> {done_file}")