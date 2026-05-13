import os
import copy
import json
import math
import random
from typing import Iterable, Sequence

import numpy as np
import torch
from numpy.lib.format import open_memmap
from torch.utils.data import DataLoader
from tqdm import tqdm

from benchmark.models.models import *
from benchmark.preprocess.CPSC.data_CPSC import CPSCNPYDataset
from benchmark.preprocess.CSN.data_CSN import CSNNPYDataset
from benchmark.preprocess.ECHO_NEXT.data_ECHO_NEXT import EchoNextNPYDataset
from benchmark.preprocess.PTBXL.data_PTBXL import PTBXLNPYDataset
from benchmark.preprocess.PTBXL_C.data_PTBXL_C import PTBXL_C_NPYDataset


DEFAULT_DATASETS = [
    # "ECHO_NEXT",
    "PTBXL_form",
  "PTBXL_super",
  "PTBXL_sub",
  "PTBXL_rhythm",
#    "PTBXL_C_form",
#    "PTBXL_C_rhythm",
#    "PTBXL_C_sub",
  "CPSC",
   "CSN",
]

CORE_MODELS = ["D_BETA", "MERL", "CLOCS","HeartLang", "KED"]
SPLITS = ["train", "val", "test"]

BASE_EMBEDDING_SHAPES = {
    "MERL": (512,),
    "D_BETA": (768,),
    "CLOCS": (256, 2),
    "KED": (768,),
    "HeartLang": (768,),
}


def load_config(path: str = "configs/config.json") -> dict:
    with open(path, "r") as f:
        return json.load(f)


def get_embedding_config(config: dict) -> dict:
    embedding_cfg = dict(config.get("embedding", {}))

    defaults = {
        "seed": config.get("seed", 42),
        "num_chunks": 6,          # split all models into groups and run all groups sequentially
        "batch_size": 256,
        "num_workers": 8,
        "datasets": DEFAULT_DATASETS,
        "splits": SPLITS,
        "emb_dtype": "float32",
    }

    for key, value in defaults.items():
        embedding_cfg.setdefault(key, value)


    return embedding_cfg


def build_random_model_names() -> tuple[list[str], list[str]]:
    random_resnet = []
    random_vit = []

    for hz in ["500Hz"]: #, "250Hz", "100Hz"
        for z_score in ["Z_score_sample", "Z_score_none"]: #, "Z_score_dataset",
            for band in ["No_Bandpass"]: #"Add_Bandpass", 
                for backbone in [ "Resnet18"]: #"Vit",
                    name = f"Random_{hz}_{z_score}_{band}_{backbone}"
                    if backbone == "Resnet18":
                        random_resnet.append(name)
                    else:
                        random_vit.append(name)

    return random_resnet, random_vit


def build_embedding_shapes() -> dict:
    shapes = dict(BASE_EMBEDDING_SHAPES)
    random_resnet, random_vit = build_random_model_names()

    for name in random_resnet + random_vit:
        shapes[name] = (512,)

    return shapes


def build_all_models() -> list[str]:
    random_resnet, random_vit = build_random_model_names()
    return CORE_MODELS + random_resnet + random_vit


def chunk_list(items: Sequence[str], chunk_size: int) -> list[list[str]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")

    return [
        list(items[i:i + chunk_size])
        for i in range(0, len(items), chunk_size)
    ]


def build_model_chunks(embedding_cfg: dict) -> list[list[str]]:
    """Split all models into groups of chunk_size and run every group sequentially."""
    all_models = build_all_models()
    return chunk_list(all_models, int(embedding_cfg["chunk_size"]))

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_models(config: dict, embedding_cfg: dict, model_names: Sequence[str], device: str) -> dict:
    models = {}
    seed = int(embedding_cfg["seed"])

    for model_name in model_names:
        print(f"Loading model: {model_name}")

        set_seed(seed)
        model = create_embedding_model(model_name, config["model_weights_dir"])
        model = model.to(device)
        model.eval()

        for param in model.parameters():
            param.requires_grad = False

        models[model_name] = model

    return models


def extract_ecg_dbeta_features(model, ecgs):
    ecg_model = model.ecg_encoder
    pooler = model.unimodal_ecg_pooler
    proj = model.multi_modal_ecg_proj
    class_embedding = model.class_embedding

    uni_modal_ecg_feats, ecg_padding_mask = ecg_model.get_embeddings(ecgs, padding_mask=None)
    cls_emb = class_embedding.repeat((len(uni_modal_ecg_feats), 1, 1))
    uni_modal_ecg_feats = torch.cat([cls_emb, uni_modal_ecg_feats], dim=1)
    uni_modal_ecg_feats = ecg_model.get_output(uni_modal_ecg_feats, ecg_padding_mask)
    out = proj(uni_modal_ecg_feats)
    return pooler(out)


def get_embeddings(model_name: str, model, x: torch.Tensor):
    with torch.no_grad():
        if model_name == "D_BETA":
            return extract_ecg_dbeta_features(model, x)

        if model_name == "CLOCS":
            x1, x2 = torch.chunk(x, 2, dim=2)      # each: (B, 12, 2500)
            x = torch.stack([x1, x2], dim=-1)      # (B, 12, 2500, 2)
            x = x.permute(0, 2, 1, 3)              # (B, 2500, 12, 2)
            return model.get_embedding(x)

        if model_name == "KED":
            return model(x).mean(dim=2)

        return model(x)


def make_dataset(dataset_name: str, split: str):
    if dataset_name == "CSN":
        return CSNNPYDataset(split=split)
    if dataset_name == "ECHO_NEXT":
        return EchoNextNPYDataset(split=split)
    if dataset_name == "CPSC":
        return CPSCNPYDataset(split=split)
    if dataset_name == "PTBXL_form":
        return PTBXLNPYDataset("form", split=split)
    if dataset_name == "PTBXL_super":
        return PTBXLNPYDataset("diagnostic_class", split=split)
    if dataset_name == "PTBXL_sub":
        return PTBXLNPYDataset("diagnostic_subclass", split=split)
    if dataset_name == "PTBXL_rhythm":
        return PTBXLNPYDataset("rhythm", split=split)
    if dataset_name == "PTBXL_C_form":
        return PTBXL_C_NPYDataset("form", split=split)
    if dataset_name == "PTBXL_C_sub":
        return PTBXL_C_NPYDataset("diagnostic_subclass", split=split)
    if dataset_name == "PTBXL_C_rhythm":
        return PTBXL_C_NPYDataset("rhythm", split=split)

    raise ValueError(f"Unknown dataset: {dataset_name}")


def model_done_path(split_dir: str, model_name: str) -> str:
    return os.path.join(split_dir, f"{model_name}.done.json")


def model_outputs_exist(split_dir: str, model_name: str) -> bool:
    return (
        os.path.exists(os.path.join(split_dir, f"{model_name}.npy"))
        and os.path.exists(os.path.join(split_dir, f"{model_name}_y.npy"))
        and os.path.exists(model_done_path(split_dir, model_name))
    )


def pending_models_for_split(out_root: str, dataset_name: str, split: str, models: dict) -> list[str]:
    split_dir = os.path.join(out_root, dataset_name, split)
    return [mname for mname in models if not model_outputs_exist(split_dir, mname)]


def save_split_meta(split_dir: str, meta: dict) -> None:
    with open(os.path.join(split_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


def save_model_done(split_dir: str, model_name: str, meta: dict) -> None:
    done_payload = dict(meta)
    done_payload["model"] = model_name
    done_payload["done"] = True

    with open(model_done_path(split_dir, model_name), "w") as f:
        json.dump(done_payload, f, indent=2)


def save_embeddings_for_split(
    *,
    ds,
    dataset_name: str,
    split: str,
    models: dict,
    out_root: str,
    batch_size: int,
    num_workers: int,
    device: str,
    embedding_shapes: dict,
    seed: int,
    emb_dtype=np.float32,
) -> None:
    split_dir = os.path.join(out_root, dataset_name, split)
    os.makedirs(split_dir, exist_ok=True)

    pending_names = pending_models_for_split(out_root, dataset_name, split, models)
    if not pending_names:
        print(f"⏭️  Skipping {dataset_name}/{split}; all selected models are done")
        return

    pending_models = {name: models[name] for name in pending_names}

    for model_name, model in pending_models.items():
        if "Z_score_dataset" in model_name:
            global_mean = ds.ecg_metrics["global_mean"]
            global_std = ds.ecg_metrics["global_std"]
            model.set_stats(global_mean, global_std)


    generator = torch.Generator()
    generator.manual_seed(seed)

    # from torch.utils.data import Subset
    # max_examples = 1000

    # n = min(max_examples, len(ds))
    # ds = Subset(ds, range(n))

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        drop_last=False,
        generator=generator,
    )
    
    _, y0 = ds[0]
    if torch.is_tensor(y0):
        y_shape = tuple(y0.cpu().numpy().shape)
    else:
        y_shape = tuple(np.asarray(y0).shape)
    n = len(ds)

    emb_mms = {}
    y_mms = {}

    for model_name in pending_names:
        emb_shape = embedding_shapes[model_name]
        emb_mms[model_name] = open_memmap(
            os.path.join(split_dir, f"{model_name}.npy"),
            mode="w+",
            dtype=emb_dtype,
            shape=(n, *emb_shape),
        )
        y_mms[model_name] = open_memmap(
            os.path.join(split_dir, f"{model_name}_y.npy"),
            mode="w+",
            dtype=np.uint8,
            shape=(n, *y_shape),
        )

    ptr = 0
    for bx, by in tqdm(loader, desc=f"{dataset_name}/{split}", unit="batch"):
        b = bx.shape[0]
        bx = bx.to(device, non_blocking=True)
        by_np = by.cpu().numpy().astype(np.uint8, copy=False)

        with torch.no_grad():
            for model_name, model in pending_models.items():
                emb = get_embeddings(model_name, model, bx)
                emb = emb.detach().cpu().numpy().astype(emb_dtype, copy=False)
                emb_mms[model_name][ptr:ptr + b] = emb
                y_mms[model_name][ptr:ptr + b] = by_np

        ptr += b

    for mm in emb_mms.values():
        mm.flush()
    for mm in y_mms.values():
        mm.flush()

    meta = {
        "dataset": dataset_name,
        "split": split,
        "N": n,
        "seed": seed,
        "shuffle": False,
        "emb_dtype": str(np.dtype(emb_dtype)),
        "y_shape": list(y_shape),
        "models": pending_names,
        "embeddings": {
            model_name: {"shape": [n, *list(embedding_shapes[model_name])]}
            for model_name in pending_names
        },
    }

    save_split_meta(split_dir, meta)

    for model_name in pending_names:
        save_model_done(split_dir, model_name, meta)

    print(f"✅ Saved embeddings to: {split_dir}")


def embed_dataset(
    *,
    dataset_name: str,
    splits: Iterable[str],
    models: dict,
    config: dict,
    embedding_cfg: dict,
    device: str,
    embedding_shapes: dict,
) -> None:
    out_root = config["embeddings_dir"]
    batch_size = int(embedding_cfg["batch_size"])
    num_workers = int(embedding_cfg["num_workers"])
    seed = int(embedding_cfg["seed"])
    emb_dtype = np.dtype(embedding_cfg["emb_dtype"])

    for split in splits:
        split_dir = os.path.join(out_root, dataset_name, split)
        pending = pending_models_for_split(out_root, dataset_name, split, models)

        if not pending:
            print(f"⏭️  Skipping {dataset_name}/{split}; already done")
            continue

        print(f"🚀 Embedding {dataset_name}/{split}; pending models: {pending}")
        ds = make_dataset(dataset_name, split)

        save_embeddings_for_split(
            ds=ds,
            dataset_name=dataset_name,
            split=split,
            models=models,
            out_root=out_root,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
            embedding_shapes=embedding_shapes,
            seed=seed,
            emb_dtype=emb_dtype,
        )


def embed_main(config) -> None:
    embedding_cfg = get_embedding_config(config)

    os.makedirs(config["embeddings_dir"], exist_ok=True)

    seed = int(embedding_cfg["seed"])
    set_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Seed: {seed}")

    embedding_shapes = build_embedding_shapes()
    model_chunks = build_model_chunks(embedding_cfg)
    print(f"Num model chunks: {len(model_chunks)}")

    for chunk_id, selected_model_names in enumerate(model_chunks):
        print("\n" + "=" * 80)
        print(f"Running model chunk {chunk_id + 1}/{len(model_chunks)}")
        print(f"Models: {selected_model_names}")
        print("=" * 80)

        set_seed(seed)
        models = create_models(config, embedding_cfg, selected_model_names, device)

        for dataset_name in embedding_cfg["datasets"]:
            embed_dataset(
                dataset_name=dataset_name,
                splits=embedding_cfg["splits"],
                models=models,
                config=config,
                embedding_cfg=embedding_cfg,
                device=device,
                embedding_shapes=embedding_shapes,
            )

        del models
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("✅ Embedding export complete for all chunks.")


