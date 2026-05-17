import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Tuple, List, Dict, Callable, Optional
from src.preprocess.utils import compute_global_metrics
from src.utils import load_config


class CSNNPYDataset(Dataset):
    """
    Unified CSN dataset saved as:
      - csn_ecg.npy
      - csn_labels.npy
      - csn_metadata.csv (contains split column train/val/test)
      - csn_class_to_index.json
    """

    def __init__(
        self,
        root_dir: str = None,
        meta_path : str = None,
        split: str = "train",
        transform=None,
    ):
        if split == "valid":
            split = "val"
        config = load_config()
        root_dir = root_dir or config["raw_data_dir"]
        meta_path = meta_path or os.path.join(config["processed_dir"], "csn_metadata_final.csv")
        assert split in {"train", "val", "test"}

        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        # class map
        class_map_path = os.path.join(root_dir, "csn_class_to_index.json")
        with open(class_map_path, "r") as f:
            self.class_to_index: Dict[str, int] = json.load(f)
        self.vocab = list(self.class_to_index.keys())

        # memmaps
        self.ecg = np.load(os.path.join(root_dir, "csn_ecg.npy"), mmap_mode="r")
        self.labels = np.load(os.path.join(root_dir, "csn_labels.npy"), mmap_mode="r")

        # metadata
        meta_df = pd.read_csv(meta_path)
        meta_df["split"] = meta_df["split"].replace({"valid": "val"})

        self.ecg_metrics  = compute_global_metrics(meta_df)

        # choose split
        meta_df = meta_df[meta_df["split"] == split]


        if len(meta_df) == 0:
            raise RuntimeError(f"No samples left after filtering for split={split}")

        self.indices = meta_df["ecg_index"].to_numpy(dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = int(self.indices[idx])

        ecg = self.ecg[real_idx]       # (12, 5000)
        y = self.labels[real_idx]      # (C,)

        ecg = torch.from_numpy(ecg).float()
        y = torch.from_numpy(y).float()

        if self.transform:
            ecg = self.transform(ecg)

        return ecg, y

    def get_num_classes_and_vocab(self) -> Tuple[int, List[str], Dict[str, int]]:
        return len(self.vocab), self.vocab, self.class_to_index
