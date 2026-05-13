import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy

import numpy as np
import torch
import torch.multiprocessing as mp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Subset
from benchmark.preprocess.load_embeddings import collect_embeddings, load_embeddings
from benchmark.utils import log_dataset_stats, log_metrics_stats, save_metrics, save_stats
import json

mp.set_sharing_strategy("file_system")


def split_clocs_views(X: np.ndarray):
    """
    Supports CLOCS embeddings shaped:
      - (N, 2, D)
      - (N, D, 2)
    Returns:
      X1, X2 each of shape (N, D)
    """
    if X.ndim != 3:
        raise ValueError(f"Expected CLOCS X to be 3D, got shape={X.shape}")

    if X.shape[1] == 2:
        # (N, 2, D)
        return X[:, 0, :], X[:, 1, :]
    elif X.shape[2] == 2:
        # (N, D, 2)
        return X[:, :, 0], X[:, :, 1]
    else:
        raise ValueError(f"Can't find 2-view dimension in CLOCS X shape={X.shape}")

def prepare_train_data(X: np.ndarray, y: np.ndarray, model_name: str):
    """
    For CLOCS: stack views -> (2N, D), duplicate labels -> (2N, C)
    For others: unchanged
    """
    if model_name != "CLOCS":
        return X, y

    X1, X2 = split_clocs_views(X)
    X_out = np.concatenate([X1, X2], axis=0)
    y_out = np.concatenate([y, y], axis=0)
    return X_out, y_out

def predict_proba(clf, X: np.ndarray, model_name: str, binary: bool):
    """
    For CLOCS: average view probs.
    For binary tasks: return positive-class probability only.
    """
    if model_name == "CLOCS":
        X1, X2 = split_clocs_views(X)
        p1 = clf.predict_proba(X1)
        p2 = clf.predict_proba(X2)
        probs = 0.5 * (p1 + p2)
    else:
        probs = clf.predict_proba(X)

    if binary:
        # probs shape: (N, 2) → keep P(class=1)
        return probs[:, 1]
    return probs

def build_classifier(
    *,
    scale: bool,
    C: float,
    class_weight,
    seed: int,
):
    steps = []

    if scale:
        steps.append(("scaler", StandardScaler()))

    steps.append((
        "clf",
        OneVsRestClassifier(
            LogisticRegression(
                C=C,
                max_iter=10000,
                random_state=seed,
                class_weight=class_weight,
                solver='lbfgs',
            ),
            n_jobs=10,
        )
    ))

    return Pipeline(steps)



def safe_name(x) -> str:
    return str(x).replace("/", "_").replace(" ", "_").replace(".", "p")


def eval_done_dir(args) -> str:
    return os.path.join(args.results_dir, "done")

def eval_done_path(args) -> str:
    fname = (
        f"{safe_name(args.data)}__"
        f"{safe_name(args.model)}__"
        f"trainpct_{safe_name(args.train_pct)}__"
        f"seed_{safe_name(args.seed)}.done.json"
    )
    return os.path.join(eval_done_dir(args), fname)


def eval_is_done(args) -> bool:
    return os.path.exists(eval_done_path(args))

def mark_eval_done(args, summary: dict) -> None:
    done_dir = eval_done_dir(args)
    os.makedirs(done_dir, exist_ok=True)

    done_path = eval_done_path(args)
    tmp_path = done_path + ".tmp"

    payload = {
        "done": True,
        "data": args.data,
        "model": args.model,
        "train_pct": args.train_pct,
        "seed": args.seed,
        **summary,
    }

    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2)

    os.replace(tmp_path, done_path)

def multilabel_bce_loss(y_true, y_prob, eps=1e-15):
    y_true = (np.asarray(y_true) > 0).astype(np.float64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    y_prob = np.clip(y_prob, eps, 1.0 - eps)

    return float(
        -np.mean(
            y_true * np.log(y_prob)
            + (1.0 - y_true) * np.log(1.0 - y_prob)
        )
    )

def eval_model(args):
    """Train the model with early stopping based on validation loss"""
    #----train/val ----
    if eval_is_done(args):
        print(
            f"⏭️  Skipping done eval: "
            f"{args.data} / {args.model} / train_pct={args.train_pct} / seed={args.seed}"
        )
        return

    dataset_train = load_embeddings(args, "train")
    num_train = int(len(dataset_train)* args.train_pct)

    g = torch.Generator()
    g.manual_seed(args.seed)
    indices_train = torch.randperm(len(dataset_train), generator=g)[:num_train]
    subset_train = Subset(dataset_train, indices_train)

    subset_train_loader = torch.utils.data.DataLoader(
            subset_train,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=False,
            prefetch_factor=args.prefetch_factor
        )

    dataset_valid = load_embeddings(args, "val")

    valid_loader = torch.utils.data.DataLoader(
            dataset_valid,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=False,
            prefetch_factor=args.prefetch_factor
        )

    X_train, y_train = collect_embeddings(subset_train_loader)
    X_val, y_val = collect_embeddings(valid_loader)
    X_train_fit, y_train_fit = prepare_train_data(X_train, y_train, args.model)


    grid = {
    "scale": [True, False],
    "C": [0.01, 0.1, 1.0, 10.0],
    "class_weight": [None, "balanced"],
    }

    best_loss = float("inf")
    best_clf = None
    best_cfg = None

    for scale in grid["scale"]:
        for C in grid["C"]:
            for cw in grid["class_weight"]:
                clf = build_classifier(
                    scale=scale,
                    C=C,
                    class_weight=cw,
                    seed=args.seed,
                )

                clf.fit(X_train_fit, y_train_fit)

                val_probs = predict_proba(clf, X_val, args.model, binary=False)
                val_loss = multilabel_bce_loss(y_val, val_probs)


                if val_loss < best_loss:
                    best_loss = val_loss
                    best_clf = deepcopy(clf)
                    best_cfg = {
                        "scale": scale,
                        "C": C,
                        "class_weight": cw,
                    }
    eval_summary = {
        "best_val_loss": best_loss,
        "best_scale": best_cfg["scale"],
        "best_C": best_cfg["C"],
        "best_class_weight": str(best_cfg["class_weight"]),
    }

    dataset_test = load_embeddings(args, "test")

    test_loader = torch.utils.data.DataLoader(
            dataset_test,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=False,
            prefetch_factor=args.prefetch_factor
        )

    X_test, y_true = collect_embeddings(test_loader)


    y_pred = predict_proba(best_clf, X_test, args.model, binary=False)

    test_loss = multilabel_bce_loss(y_true, y_pred)
    eval_summary["test_loss"] = test_loss

    dataset_stats = log_dataset_stats(args, subset_train, dataset_valid, dataset_test)
    metric_stats = log_metrics_stats(args, y_true, y_pred, dataset_test.vocab)
    metric_stats.update(eval_summary)
    save_metrics(args, y_true, y_pred,  dataset_test.vocab)
    save_stats(args, dataset_stats, metric_stats)

    mark_eval_done(args, metric_stats)
    print(
        f"Done: "
        f"{args.data} / {args.model} / train_pct={args.train_pct} / seed={args.seed}"
    )


RUNS = [  "PTBXL"] #, "ECHO_NEXT","CSN","CPSC"


def run_one(cfg):
    args = get_args_raw_ecg()
    args.data = cfg["data"]
    args.model = cfg["model"]
    args.train_pct = cfg["train_pct"]
    args.seed = cfg["seed"]
    args.results_dir = cfg["results_dir"]
    args.label_type =""

    eval_model(args)



models = ["D_BETA", "MERL", "CLOCS", "KED", "HeartLang"]

for hz in ["500Hz"]: #, "250Hz", "100Hz"
    for z_score in ["Z_score_sample", "Z_score_none"]: #"Z_score_dataset",
        for band in ["No_Bandpass"]: #"Add_Bandpass", 
            for model in [ "Resnet18"]: #"Vit",
                name = f"Random_{hz}_{z_score}_{band}_{model}"
                models.append(name)


def main_eval(config):
    args = get_args_raw_ecg()
    configs = []
    if args.run_type == "specified":
        eval_model(args)
    elif args.run_type == "full":
        for data in RUNS:
            if "PTBXL" in data:
                label_types = ['form'] #'rhythm','super', 'sub'
            elif  data == "PTBXL_C":
                label_types = [ "form"]#"rhythm", , "sub"
            else:
                label_types = ['']

            for label_type in label_types:   
                for model in models:
                    for train_pct in [.01, .1,  1]: 
                        args = get_args_raw_ecg()
                        if label_type != "":
                            dataset_name = f"{data}_{label_type}"
                        else:
                            dataset_name = data
                        
                        cfg = {
                                "data": dataset_name,
                                "model": model,
                                "train_pct": train_pct,
                                "seed": args.seed,
                                "results_dir": config["results_dir"],
                            }

                        if config["multi_process_eval"] > 1:
                            configs.append(cfg)
                        else:
                            run_one(cfg)
        if config["multi_process_eval"] > 1:
            max_workers = config["multi_process_eval"]
            print(f"Launching {len(configs)} runs using {max_workers} workers")

            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(run_one, cfg) for cfg in configs]

                for fut in as_completed(futures):
                    try:
                        fut.result()
                    except Exception as e:
                        print("❌ Run failed:", e)


def get_args_raw_ecg():
    parser = argparse.ArgumentParser(description="Train/evaluate models on ECG dataset")

    # Run naming / logging
    parser.add_argument("--name", type=str, default=None, help="Optional run name for logging and model saving")
    parser.add_argument("--notes", type=str, default="", help="Additional notes")

    parser.add_argument(
        "--run_type",
        type=str,
        default="full",
        choices=["full", "specified"],
        help="Run full test on model or just specified",
    )

    # Data settings
    parser.add_argument(
        "--data",
        type=str,
        default="PTBXL",
        help="Data",
    )

    parser.add_argument("--model", type=str, default="D_BETA", help="Embedding model name")
    parser.add_argument("--train_pct", type=float, default=1.0, help="Fraction of train split used for probe fitting")
    parser.add_argument("--label_type", type=str, default="", help="Optional label type metadata")

    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loader workers")

    parser.add_argument("--prefetch_factor", type=int, default=4, help="Number of data loader workers")

    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for pytorch/merl probe")


    # Reproducibility
    parser.add_argument("--seed", type=int, default=42, help="Global random seed")

    args = parser.parse_args()

    return args




if __name__ == "__main__":
    main()
