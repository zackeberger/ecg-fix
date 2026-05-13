import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Dict
import pandas as pd
from benchmark.preprocess.utils import *
from benchmark.config import load_config


class CPSCNPYDataset(Dataset):
    """
    Dataset for preprocessed CPSC ECGs saved as memmapped .npy files,
    using BOTH valid_idx.npy (hard mask) + metadata.csv (optional filtering).

    Files expected in root_dir:
      - cpsc2018_ecg.npy               (N_total, 12, 5000) float16/float32
      - cpsc2018_labels.npy            (N_total, C) uint8
      - cpsc2018_valid_idx.npy         (K,) indices that were successfully written
      - cpsc2018_metadata.csv          metadata for valid saved samples (ecg_index column)
      - cpsc2018_class_to_index.json   class mapping
    """

    def __init__(
        self,
        root_dir: str = None,
        meta_path : str = None,
        split: str = "train",
        transform=None,
        seed: int = 42,
    ):
        if split == "valid":
            split = "val"
        config = load_config()
        root_dir = root_dir or config.raw_data_dir
        meta_path = meta_path or os.path.join(config.processed_dir, "cpsc2018_metadata_final.csv")
        assert split in {"train", "val", "test"}

        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.seed = seed

        # ---- load class mapping ----
        class_map_path = os.path.join(root_dir, "cpsc2018_class_to_index.json")
        with open(class_map_path, "r") as f:
            self.class_to_index: Dict[str, int] = json.load(f)

        self.vocab = list(self.class_to_index.keys())

        # ---- memmap ECG + labels ----
        self.ecg = np.load(os.path.join(root_dir, "cpsc2018_ecg.npy"), mmap_mode="r")
        self.labels = np.load(os.path.join(root_dir, "cpsc2018_labels.npy"), mmap_mode="r")

        # ---- load metadata CSV ----
        meta_df = pd.read_csv(meta_path)

        self.ecg_metrics  = compute_global_metrics(meta_df)

        # ---- OPTIONAL user filter on metadata ----
        def metadata_filter(df):
            df = df[df["split"] == split]
            return df

        if metadata_filter is not None:
            meta_df = metadata_filter(meta_df)

        if len(meta_df) == 0:
            raise RuntimeError("No samples left after applying valid_idx + metadata_filter")

        self.indices = meta_df["ecg_index"].to_numpy(dtype=np.int64)


    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = int(self.indices[idx])

        ecg = self.ecg[real_idx]       # (12, 5000)
        label = self.labels[real_idx]  # (C,)

        ecg = torch.from_numpy(ecg).float()
        label = torch.from_numpy(label).float()

        if self.transform:
            ecg = self.transform(ecg)

        return ecg, label

    def get_num_classes_and_vocab(self) -> Tuple[int, List[str], Dict[str, int]]:
        return len(self.vocab), self.vocab, self.class_to_index
