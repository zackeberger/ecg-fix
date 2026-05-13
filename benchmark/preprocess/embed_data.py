import argparse
import copy
import json
import math
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from benchmark.config import add_config_arg, apply_config_to_args
from benchmark.models.models import *
from benchmark.preprocess.public.CPSC.data_CPSC import CPSCNPYDataset
from benchmark.preprocess.public.CSN.data_CSN import CSNNPYDataset
from benchmark.preprocess.public.ECHO_NEXT.data_ECHO_NEXT import EchoNextNPYDataset
from benchmark.preprocess.public.PTBXL.data_PTBXL import PTBXLNPYDataset
from benchmark.preprocess.public.PTBXL_C.data_PTBXL_C import PTBXL_C_NPYDataset

def get_args():
    parser = argparse.ArgumentParser()
    add_config_arg(parser)
    parser.add_argument(
        "--model_weights_dir",
        type=str,
        default=None,
        help="Base directory pointing to physionet.org mirror or PTB-XL dataset root",
    )
    parser.add_argument("--seed", type=int, default=42, help="Global random seed")
    parser.add_argument("--chunk", type=int, default=0, help="Global random seed")
    parser.add_argument("--model", type=str, default="MERL", 
        choices=[ "MERL","CLOCS", "D_BETA", "KED", "HeartLang"], help="Type of model to use")
    parser.add_argument("--batch_size", type=int, default=256, help="Embedding batch size")
    parser.add_argument("--num_workers", type=int, default=8, help="DataLoader workers")
    args = parser.parse_args()
    apply_config_to_args(args)

    return args


args = get_args()

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
batch_size = args.batch_size
num_workers = args.num_workers
out_dir = args.embeddings_dir
os.makedirs(out_dir, exist_ok=True)

ALL_DATASETS = reversed([
    "ECHO_NEXT",
    "CPSC",
    "CSN",
    "PTBXL_form",
    "PTBXL_super",
    "PTBXL_sub",
    "PTBXL_C_form",
    "PTBXL_C_rhythm",
    "PTBXL_C_sub",
    "PTBXL_rhythm",
])

models = {}
embedding_shape = {"MERL": (512,), "D_BETA": (768, ), "CLOCS": (256,2), "KED":(768,) 
, "HeartLang": (768, )}


Random_resnet =[]
Random_vit =[]

for hz in ["500Hz", "250Hz", "100Hz"]:
    for z_score in ["Z_score_sample", "Z_score_dataset","Z_score_none"]:
        for band in ["Add_Bandpass", "No_Bandpass"]:
            for model in ["Vit", "Resnet18"]:
                name = f"Random_{hz}_{z_score}_{band}_{model}"
                if model == "Resnet18":
                    embedding_shape[name] = (512,)
                    Random_resnet.append(name)
                elif  model == "Vit":
                    embedding_shape[name] = (512,) #fix
                    Random_vit.append(name)


core = ["D_BETA", "MERL", "CLOCS"]
new = ["HeartLang", "KED"]
all_models = core + Random_resnet + Random_vit +new

def chunk_list(lst, num_chunks):
    chunk_size = math.ceil(len(lst) / num_chunks)
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

chunks = chunk_list(all_models, 6)

def get_embedings(model_name, model, x):
    #X shape (B, 12, 5000)
    with torch.no_grad():
        if model_name == "D_BETA":
            return extract_ecg_DBeta_features(model, x)  # [B, embedding_dim]
        elif model_name == "CLOCS":
            #need (B, 2500, 12, 2)
            x1, x2 = torch.chunk(x, 2, dim=2)   # each (B, 12, 2500)
            # stack as views
            x = torch.stack([x1, x2], dim=-1)   # (B, 12, 2500, 2)
            # reorder to (B, 2500, 12, 2)
            x = x.permute(0, 2, 1, 3)
            return model.get_embedding(x)
        elif model_name == "KED":
            output= model(x).mean(dim=2)
            return output
        else:
            return model(x)

def extract_ecg_DBeta_features(model, ecgs):
    #B, 12, 5000
    num_ecgs = len(ecgs)
    ecg_model = model.ecg_encoder
    pooler = model.unimodal_ecg_pooler
    proj = model.multi_modal_ecg_proj
    class_embedding = model.class_embedding
    

    uni_modal_ecg_feats, ecg_padding_mask = (
        ecg_model.get_embeddings(ecgs, padding_mask=None)
    )
    
    cls_emb = class_embedding.repeat((len(uni_modal_ecg_feats), 1, 1))
    uni_modal_ecg_feats = torch.cat([cls_emb, uni_modal_ecg_feats], dim=1)
    uni_modal_ecg_feats = ecg_model.get_output(uni_modal_ecg_feats, ecg_padding_mask)
    out = proj(uni_modal_ecg_feats)
    ecg_features = pooler(out)
    
    return ecg_features


def embeddings_exist(out_root, dataset_name, split, models):
    split_dir = os.path.join(out_root, dataset_name, split)
    meta_path = os.path.join(split_dir, "meta.json")

    if not os.path.exists(meta_path):
        return False

    for mname in models.keys():
        if not os.path.exists(os.path.join(split_dir, f"{mname}.npy")):
            return False

        if not os.path.exists(os.path.join(split_dir, f"{mname}_y.npy")):
            return False

        if not os.path.exists(os.path.join(split_dir, f"{mname}_done_meta.json")):
            return False
            

    return True


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
def save_embeddings_for_split(
    ds,
    dataset_name: str,
    split: str,
    models: dict,
    out_root: str,
    batch_size: int,
    num_workers: int,
    device: str,
    emb_dtype=np.float32,
):
    """
    Saves:
      out_root/dataset_name/split/
        meta.json
        y.npy
        MERL.npy
        Random.npy
        D_BETA.npy
        CLOCS.npy
    """

    split_dir = os.path.join(out_root, dataset_name, split)
    os.makedirs(split_dir, exist_ok=True)

    g = torch.Generator()
    g.manual_seed(42)

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(split=="train"),
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker, 
        drop_last=False,
        generator=g, 
    )
    global_mean = ds.ecg_metrics["global_mean"]
    global_std= ds.ecg_metrics["global_std"]

    # ---- Peek 1 batch to infer shapes ----
    _, y0 = next(iter(loader))
    y0_np = y0.cpu().numpy()
    y_shape = tuple(y0_np.shape[1:])


    N = len(ds)

    # ---- Allocate memmaps ----
    emb_mms = {}
    y_mms = {}
    for mname, shp in embedding_shape.items():
        if mname in models:
            path = os.path.join(split_dir, f"{mname}.npy")
            emb_mms[mname] = np.memmap(
                path, mode="w+", dtype=emb_dtype, shape=(N, *shp)
            )
            y_path = os.path.join(split_dir, f"{mname}_y.npy")
            y_mms[mname] = np.memmap(y_path, mode="w+", dtype=emb_dtype, shape=(N, *y_shape))

    for mname, m in models.items():  
        if "Z_score_dataset" in mname:
            print(mname)
            models[mname].set_stats(global_mean,global_std)

    # ---- Write batches ----
    ptr = 0
    for bx, by in tqdm(loader, desc=f"{dataset_name}/{split}", unit="batch"):
        b = bx.shape[0]
        bx = bx.to(device, non_blocking=True)

        with torch.no_grad():
            for mname, m in models.items():
                emb = get_embedings(mname, m, bx)   # [B, ...]
                emb = emb.detach().cpu().numpy().astype(emb_dtype, copy=False)
                emb_mms[mname][ptr:ptr + b] = emb
                y_mms[mname][ptr:ptr + b] = by.cpu().numpy().astype(np.float32, copy=False)

        ptr += b

    # flush to disk
    for mm in emb_mms.values():
        mm.flush()
    for mm in y_mms.values():
        mm.flush()

    # ---- Save metadata ----
    meta = {
        "dataset": dataset_name,
        "split": split,
        "N": N,
        "emb_dtype": str(np.dtype(emb_dtype)),
        "y_shape": list(y_shape),
        "embeddings": {
            mname: {"shape": [N, *list(shp)]} for mname, shp in embedding_shape.items()
        },
    }
    with open(os.path.join(split_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    done = {"done"}
    for mname, m in models.items():  
        with open(os.path.join(split_dir, f"{mname}_done_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

    print(f"✅ Saved embeddings to: {split_dir}")


def make_dataset(data_name, split):
    if data_name == "CSN":
        return CSNNPYDataset(split=split)
    elif data_name == "ECHO_NEXT":
        return EchoNextNPYDataset(split=split)
    elif data_name == "CPSC":
        return CPSCNPYDataset(split=split)
    elif data_name == "PTBXL_form":
        return PTBXLNPYDataset("form", split=split)
    elif data_name == "PTBXL_super":
        return PTBXLNPYDataset("diagnostic_class", split=split)
    elif data_name == "PTBXL_sub":
        return PTBXLNPYDataset("diagnostic_subclass", split=split)
    elif data_name == "PTBXL_rhythm":
        return PTBXLNPYDataset("rhythm", split=split)
    elif data_name == "PTBXL_C_form":
        return PTBXL_C_NPYDataset("form", split=split)
    elif data_name == "PTBXL_C_sub":
        return PTBXL_C_NPYDataset("diagnostic_subclass", split=split)
    elif data_name == "PTBXL_C_rhythm":
        return PTBXL_C_NPYDataset("rhythm", split=split)
    else:
        raise ValueError(f"Unknown dataset: {data_name}")

def main():
    for m in chunks[args.chunk]:
        args_copy = copy.deepcopy(args)
        args_copy.model= m
        print(args_copy)

        if  "Random" in m:
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
        embedd_model = create_embedding_model(args_copy)
        embedd_model = embedd_model.to(device)
        embedd_model.eval()
        for param in embedd_model.parameters():
            param.requires_grad = False
        models[m] = embedd_model

def embedd_data(dataset_name):
    for split in ["train", "val", "test"]:

        if embeddings_exist(out_dir, dataset_name, split, models):
            print(f"⏭️  Skipping {dataset_name}/{split} (already done)")
            continue

        print(f"🚀 Embedding {dataset_name}/{split}")

        ds = make_dataset(dataset_name, split)
        save_embeddings_for_split(
            ds=ds,
            dataset_name=dataset_name,
            split=split,
            models=models,
            out_root=out_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
            emb_dtype=np.float32,   # saves lots of space
        )
            

if __name__ =="__main__":
    main()
    for dset in ALL_DATASETS:
        embedd_data(dset)
        
    


