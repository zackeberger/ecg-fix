import os
import json
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from numpy.lib.format import open_memmap

from benchmark.config import dataset_root, load_config
from benchmark.preprocess.ecg_utils import resample_ecg_linear

# ============================================================
# EchoNext un-normalization (invert per-lead z-score)
# NOTE: This restores "pre-zscore scale" (still truncated/clipped).
# Source mean/std from IntroECG preprocessing notebook. :contentReference[oaicite:1]{index=1}
# ============================================================
ECHONEXT_MEAN = np.array(
    [5.3913547, 4.08127771, -1.18338815, -4.72457889, 0.59169407, -0.59169407,
     -4.25676191, -1.44444217, -0.34211766, 3.35517074, 5.18700818, 5.53535442],
    dtype=np.float32
)

ECHONEXT_STD = np.array(
    [31.1319959, 29.16014537, 30.41666735, 26.0615433, 15.20833367, 15.20833367,
     38.35645889, 53.62205647, 58.92661479, 49.45070532, 43.92990916, 38.29998916],
    dtype=np.float32
)

UNNORMALIZE_ECHONEXT = True  # <--- set False to keep z-scored waveforms

def echonext_unnormalize(ecg_12xT: np.ndarray) -> np.ndarray:
    """
    ecg_12xT: (12, T) float array in normalized (z-score) scale
    returns:  (12, T) in pre-zscore scale
    """
    return ecg_12xT * ECHONEXT_STD[:, None] + ECHONEXT_MEAN[:, None]


# ============================================================
# EchoNext binary label vocabulary
# ============================================================
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


TARGET_LEN = 5000

def main(config=None):
    config = config or load_config()
    out_dir = config.raw_data_dir
    base_dir = dataset_root(config, "ECHO_NEXT", "echonext", "1.1.0")

    in_meta_csv = os.path.join(base_dir, "echonext_metadata_100k.csv")

    # these are your per-split waveform files
    wave_paths = {
        "train": os.path.join(base_dir, "EchoNext_train_waveforms.npy"),
        "val":   os.path.join(base_dir, "EchoNext_val_waveforms.npy"),
        "test":  os.path.join(base_dir, "EchoNext_test_waveforms.npy"),
    }

    os.makedirs(out_dir, exist_ok=True)

    out_ecg_npy   = os.path.join(out_dir, "echonext_ecg.npy")
    out_label_npy = os.path.join(out_dir, "echonext_labels.npy")
    out_meta_csv  = os.path.join(out_dir, "echonext_metadata.csv")
    out_valid_idx = os.path.join(out_dir, "echonext_valid_idx.npy")
    out_class_map = os.path.join(out_dir, "echonext_class_to_index.json")

    # ---------------------------------------------------------
    # Load metadata CSV
    # ---------------------------------------------------------
    df = pd.read_csv(in_meta_csv)

    # normalize split naming (some pipelines use "valid")
    df["split"] = df["split"].replace({"valid": "val"})

    # only keep expected splits
    df = df[df["split"].isin(["train", "val", "test"])].reset_index(drop=True)

    # ensure all label cols exist
    for c in ECHONEXT_BINARY_LABELS:
        if c not in df.columns:
            raise ValueError(f"Missing label column in metadata: {c}")

    # ---------------------------------------------------------
    # Check sizes from waveform files
    # ---------------------------------------------------------
    split_sizes = {}
    for split in ["train", "val", "test"]:
        df_split = df[df["split"] == split]
        if len(df_split) == 0:
            raise RuntimeError(f"No rows in metadata for split={split}")

        if not os.path.exists(wave_paths[split]):
            raise FileNotFoundError(f"Missing waveform file: {wave_paths[split]}")

        waves = np.load(wave_paths[split], mmap_mode="r")  # (N, 1, 2500, 12)
        if len(waves) != len(df_split):
            raise RuntimeError(
                f"Mismatch split={split}: metadata rows={len(df_split)} vs waveforms={len(waves)}"
            )

        split_sizes[split] = len(waves)

    N_total = split_sizes["train"] + split_sizes["val"] + split_sizes["test"]
    C = len(ECHONEXT_BINARY_LABELS)

    print("Split sizes:", split_sizes)
    print("Total N:", N_total)

    # ---------------------------------------------------------
    # Save class mapping
    # ---------------------------------------------------------
    class_to_index = {c: i for i, c in enumerate(ECHONEXT_BINARY_LABELS)}
    with open(out_class_map, "w") as f:
        json.dump(class_to_index, f, indent=2, sort_keys=True)
    print(f"Saved class map -> {out_class_map}")

    # ---------------------------------------------------------
    # Preallocate output memmaps
    # ---------------------------------------------------------
    # ECG: float16 is smaller + faster I/O, float32 if you want max precision
    ecg_mm = open_memmap(out_ecg_npy, mode="w+", dtype=np.float16, shape=(N_total, 12, TARGET_LEN))
    lbl_mm = open_memmap(out_label_npy, mode="w+", dtype=np.uint8,  shape=(N_total, C))


    # we'll build metadata rows here
    meta_rows = []

    # ---------------------------------------------------------
    # Write in split order (train -> val -> test)
    # ---------------------------------------------------------
    offset = 0
    for split in ["train", "val", "test"]:
        df_split = df[df["split"] == split].reset_index(drop=True)
        waves = np.load(wave_paths[split], mmap_mode="r")  # (N, 1, 2500, 12)

        print(f"\nProcessing split={split} | N={len(df_split)} | offset={offset}")

        for i in tqdm(range(len(df_split)), desc=f"Export {split}"):
            global_i = offset + i

            # ---------------------------------------------
            # ECG: (1, 2500, 12) -> (12, 2500) -> resample -> (12, 5000)
            # ---------------------------------------------
            w = waves[i]  # (1, 2500, 12)
            ecg = w.squeeze(0).T.astype(np.float32, copy=False)  # (12, 2500)

            if UNNORMALIZE_ECHONEXT:
                ecg = echonext_unnormalize(ecg) 

            # resample to 5000
            ecg = resample_ecg_linear(ecg, target_len=TARGET_LEN)  # (12, 5000)
            

            if ecg.shape != (12, TARGET_LEN):
                continue  # skip malformed

            # ---------------------------------------------
            # Labels: multi-hot binary vector
            # ---------------------------------------------
            label = df_split.loc[i, ECHONEXT_BINARY_LABELS].values.astype(np.uint8)

            # flags
            had_nan = int((not np.isfinite(ecg).all()) or (not np.isfinite(label).all()))
            num_labels = int(label.sum())
            all_zero = int(num_labels == 0)

            # ---------------------------------------------
            # Write memmap
            # ---------------------------------------------
            ecg_mm[global_i] = ecg.astype(np.float16, copy=False)
            lbl_mm[global_i] = label


            # ---------------------------------------------
            # Metadata row: keep original cols + add ours
            # ---------------------------------------------
            row = df_split.loc[i].to_dict()
            row["ecg_index"] = global_i
            row["orig_len"] = 2500
            row["final_len"] = TARGET_LEN
            row["had_nan"] = had_nan
            row["all_zero"] = all_zero
            row["num_labels"] = num_labels
            # ensure split is present and correct
            row["split"] = split

            # --- stats ---
            # per-lead (axis=1 → across time)
            lead_means = np.mean(ecg, axis=1)   # (12,)
            lead_stds  = np.std(ecg, axis=1)    # (12,)

            # global (flatten everything)
            ecg_mean = float(np.mean(ecg))
            ecg_std  = float(np.std(ecg))

            row["ecg_mean"] = ecg_mean
            row["ecg_std"] = ecg_std
            row["lead_means"] =  lead_means.tolist()   # store as list for JSON/CSV
            row["lead_stds"] =  lead_stds.tolist()

            meta_rows.append(row)

        offset += len(df_split)

    # flush memmaps
    del ecg_mm
    del lbl_mm


    meta_df = pd.DataFrame(meta_rows)
    meta_df = meta_df.sort_values("ecg_index").reset_index(drop=True)
    meta_df.to_csv(out_meta_csv, index=False)

    print("\nExport complete ✅")
    print(f"Saved ECGs      -> {out_ecg_npy}")
    print(f"Saved labels    -> {out_label_npy}")
    print(f"Saved metadata  -> {out_meta_csv}")
    print(f"Saved valid_idx -> {out_valid_idx}")


if __name__ == "__main__":
    main()
