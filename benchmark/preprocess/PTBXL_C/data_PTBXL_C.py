import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Tuple, List, Dict, Callable, Optional
from benchmark.preprocess.utils import *


class PTBXL_C_NPYDataset(Dataset):
    """
    PTB-XL dataset loader for shared ECG memmap + per-task label memmap.

    Shared files:
      - ptbxl_ecg_500hz.npy
      - ptbxl_valid_idx.npy

    Per label_type:
      - ptbxl_{label_type}_labels.npy
      - ptbxl_{label_type}_metadata.csv  (must contain split + ecg_index)
      - ptbxl_{label_type}_class_to_index.json
    """

    def __init__(
        self,
        label_type: str,
        root_dir: str = None,
        meta_dir : str = None,
        split: str = "train",
        transform=None,
    ):
        if split == "valid":
            split = "val"
        config = load_config()
        root_dir = root_dir or config["raw_data_dir"]
        meta_dir = meta_dir or config["processed_dir"]
        assert split in {"train", "val", "test"}
        assert label_type in {"form", "rhythm", "diagnostic_class", "diagnostic_subclass"}

        self.root_dir = root_dir
        self.label_type = label_type
        self.split = split
        self.transform = transform

        # class map
        map_path = os.path.join(root_dir, f"ptbxl_c_{label_type}_class_to_index.json")
        with open(map_path, "r") as f:
            self.class_to_index: Dict[str, int] = json.load(f)
        self.vocab = list(self.class_to_index.keys())

        # memmaps
        self.ecg = np.load(os.path.join(root_dir, "ptbxl_ecg_500hz.npy"), mmap_mode="r")
        self.labels = np.load(os.path.join(root_dir, f"ptbxl_{label_type}_labels.npy"), mmap_mode="r")

        # metadata for this task
        meta = pd.read_csv(os.path.join(meta_dir, f"ptbxl_c_{label_type}_metadata_final.csv"))
        meta["split"] = meta["split"].replace({"valid": "val"})

        self.ecg_metrics = compute_global_metrics(meta)

        # hard filter + split selection
        meta = meta[meta["split"] == split]


        if len(meta) == 0:
            raise RuntimeError(f"No samples left for label_type={label_type}, split={split}")

        self.indices = meta["ecg_index"].to_numpy(dtype=np.int64)
        self.filtered_metadata = meta.reset_index(drop=True)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = int(self.indices[idx])

        x = self.ecg[real_idx]       # (12, 5000)
        y = self.labels[real_idx]    # (C,)

        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()

        if self.transform:
            x = self.transform(x)

        return x, y

    def get_num_classes_and_vocab(self) -> Tuple[int, List[str], Dict[str, int]]:
        return len(self.vocab), self.vocab, self.class_to_index
