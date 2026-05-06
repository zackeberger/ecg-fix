import json
import numpy as np
import torch
from torch.utils.data import Dataset
import os
from benchmark.config import load_config

public_mapping = {
        # PTB-XL subtasks
        "PTBXL_rhythm": "ptbxl_rhythm_class_to_index.json",
        "PTBXL_form": "ptbxl_form_class_to_index.json",
        "PTBXL_super": "ptbxl_diagnostic_class_class_to_index.json",
        "PTBXL_sub": "ptbxl_diagnostic_subclass_class_to_index.json",

        "PTBXL_C_rhythm": "ptbxl_c_rhythm_class_to_index.json",
        "PTBXL_C_form": "ptbxl_c_form_class_to_index.json",
        "PTBXL_C_sub": "ptbxl_c_diagnostic_subclass_class_to_index.json",

        # other datasets
        "ECHO_NEXT": "echonext_class_to_index.json",
        "CPSC": "cpsc2018_class_to_index.json",
        "CSN": "csn_class_to_index.json",
    }

private_mapping = {
        "PCWP_BINARY": "apollo_pcwp_binary_class_to_index.json",
        "PAP_BINARY": "apollo_pap_binary_class_to_index.json",
        "HF_ALL_BINARY": "pulsehf_future_1y_any_below40_all_class_to_index.json",
        "HF_NO_HFREF_BINARY": "pulsehf_future_1y_any_below40_not_hfref_class_to_index.json",
    }

def resolve_class_to_index_and_root(dataset_name: str, config=None) -> str:
    config = config or load_config()
    if dataset_name  in private_mapping: 
        base = config.raw_data_dir
        name = private_mapping[dataset_name]
        root = config.embeddings_dir
    elif dataset_name  in public_mapping: 
        base = config.raw_data_dir
        name = public_mapping[dataset_name]
        root = config.embeddings_dir
    else:
        raise ValueError(
            f"Unknown dataset_name='{dataset_name}'. "
          #  f"Expected one of: {sorted(mapping.keys())}"
        )

    return os.path.join(base, name), root


class PrecomputedEmbeddingDataset(Dataset):
    def __init__(self, dataset_name, split, model_name, config=None):
        vocab_path, root = resolve_class_to_index_and_root(dataset_name, config=config)
        self.split_dir = os.path.join(root, dataset_name, split)

        meta_path = os.path.join(self.split_dir, "meta.json")
        with open(meta_path, "r") as f:
            self.meta = json.load(f)

        self.N = self.meta["N"]

        # embeddings
        emb_shape = tuple(self.meta["embeddings"][model_name]["shape"])
        emb_dtype = np.dtype(self.meta["emb_dtype"])
        emb_path = os.path.join(self.split_dir, f"{model_name}.npy")
        self.X = np.memmap(emb_path, mode="r", dtype=emb_dtype, shape=emb_shape)

        # labels
        y_shape = (self.N, *self.meta["y_shape"])
        y_path = os.path.join(self.split_dir, f"{model_name}_y.npy")
        self.y = np.memmap(y_path, mode="r", dtype=emb_dtype, shape=y_shape)

        with open(vocab_path, "r") as f:
            self.vocab = json.load(f)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx]).float()
        y = torch.from_numpy(self.y[idx])
        return x, y

def load_embeddings(args, split):
    return PrecomputedEmbeddingDataset(args.data, split, args.model, config=args)

def collect_embeddings(loader):
    X, y = [], []
    for batch in loader:
        emb, labels = batch   # adjust to your dataset output
        X.append(emb.cpu().numpy())
        y.append(labels.cpu().numpy())
    return np.vstack(X), np.vstack(y)
