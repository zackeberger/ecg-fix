import gc
import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import numpy as np
import torch
import torch.multiprocessing as mp
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Subset
from types import SimpleNamespace

from src.preprocess.load_embeddings import collect_embeddings, load_embeddings
from src.metrics.logging import log_dataset_stats, log_metrics, save_outputs, save_stats
from src.metrics.utils import dataset_stats_path, metric_stats_path, output_path, safe_name
import json
from src.registry import (
    normalize_datasets,
     normalize_models
)
from src.utils import eval_config_hash
import warnings
from tqdm import tqdm
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings(
    "ignore",
    message=r"Label .* is present in all training examples\.",
    category=UserWarning,
    module=r"sklearn\.multiclass",
)

warnings.filterwarnings(
    "ignore",
    category=ConvergenceWarning,
    module=r"sklearn\.linear_model\._logistic",
)


warnings.filterwarnings(
    "ignore",
    category=UndefinedMetricWarning,
    module=r"sklearn\.metrics\._ranking",
)

warnings.filterwarnings(
    "ignore",
    message=r"No positive class found in y_true.*",
    category=UserWarning,
    module=r"sklearn\.metrics\._ranking",
)



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
    sklearn_n_jobs: int,
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
            n_jobs=sklearn_n_jobs,
        )
    ))

    return Pipeline(steps)



def eval_done_dir(args) -> str:
    return os.path.join(args.results_dir, "done")

def eval_done_path(args) -> str:
    fname = (
        f"{safe_name(args.data)}__"
        f"{safe_name(args.model)}__"
        f"trainpct_{safe_name(args.train_pct)}__"
        f"seed_{safe_name(args.seed)}_done.json"
    )
    return os.path.join(eval_done_dir(args), fname)


def eval_is_done(args) -> bool:
    done_path = eval_done_path(args)
    if not os.path.exists(done_path):
        return False

    try:
        with open(done_path, "r") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False

    if payload.get("done") is not True:
        return False

    expected = {
        "data": args.data,
        "model": args.model,
        "train_pct": args.train_pct,
        "seed": args.seed,
        "config_hash": args.config_hash,
        "n_boot": args.n_boot,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            return False

    expected_outputs = [
        output_path(args.results_dir, args.data, args.train_pct, args.model),
        dataset_stats_path(args.results_dir, args.data, args.train_pct, args.model),
        metric_stats_path(args.results_dir, args.data, args.train_pct, args.model),
    ]
    return all(os.path.exists(path) for path in expected_outputs)

def mark_eval_done(args, summary: dict) -> None:
    done_dir = eval_done_dir(args)
    os.makedirs(done_dir, exist_ok=True)

    done_path = eval_done_path(args)
    tmp_path = f"{done_path}.tmp"

    payload = {
        "done": True,
        "data": args.data,
        "model": args.model,
        "train_pct": args.train_pct,
        "seed": args.seed,
        "config_hash": args.config_hash,
        "n_boot": args.n_boot,
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
    print(f"Running Dataset:{args.data} Model:{args.model} Train PCT:{args.train_pct} ")
    # Train/validation
    if eval_is_done(args):
        print(
            f"⏭️  Skipping eval already done: "
            f"{args.data} / {args.model} / train_pct={args.train_pct} / seed={args.seed}"
        )
        return

    dataset_train = load_embeddings(args, "train")
    num_train = int(len(dataset_train)* args.train_pct)

    g = torch.Generator()
    g.manual_seed(args.seed)
    indices_train = torch.randperm(len(dataset_train), generator=g)[:num_train]
    subset_train = Subset(dataset_train, indices_train)

    loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": False,
        "persistent_workers": False,
    }

    subset_train_loader = torch.utils.data.DataLoader(
            subset_train,
            **loader_kwargs,
        )

    dataset_valid = load_embeddings(args, "val")

    valid_loader = torch.utils.data.DataLoader(
            dataset_valid,
            **loader_kwargs,
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
                    sklearn_n_jobs=args.sklearn_n_jobs,
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
            **loader_kwargs,
        )

    X_test, y_true = collect_embeddings(test_loader)


    y_pred = predict_proba(best_clf, X_test, args.model, binary=False)

    save_outputs(args, y_true, y_pred,  dataset_test.vocab)
    test_loss = multilabel_bce_loss(y_true, y_pred)
    

    eval_summary["test_loss"] = test_loss
    dataset_stats = log_dataset_stats(args, subset_train, dataset_valid, dataset_test)
    metric_stats = log_metrics(
        y_true,
        y_pred,
        dataset_test.vocab,
        n_bootstrap=args.n_boot,
        seed=args.seed,
    )
    metric_stats.update(eval_summary)
    save_stats(args, dataset_stats, metric_stats)

    mark_eval_done(args, metric_stats)

    del (
        X_train,
        X_val,
        X_train_fit,
        y_train,
        y_val,
        y_train_fit,
        X_test,
        y_true,
        y_pred,
        subset_train_loader,
        valid_loader,
        test_loader,
        dataset_train,
        subset_train,
        dataset_valid,
        dataset_test,
        best_clf,
    )
    gc.collect()

    print(
        f"Done: "
        f"{args.data} / {args.model} / train_pct={args.train_pct} / seed={args.seed}"
    )



def make_eval_args(cfg: dict):
    """
    Build the args object expected by eval_model().
    """
    return SimpleNamespace(
        run_type="specified",
        data=cfg["data"],
        model=cfg["model"],
        train_pct=cfg["train_pct"],

        results_dir=cfg["results_dir"],
        embeddings_dir=cfg["embeddings_dir"],

        num_workers=cfg["num_workers"],
        batch_size=cfg["batch_size"],
        n_boot= cfg["n_boot"],
        sklearn_n_jobs=cfg["sklearn_n_jobs"],

        seed=cfg["seed"],
        config_hash=cfg["config_hash"],
    )


def run_one(cfg: dict):
    eval_args = make_eval_args(cfg)
    eval_model(eval_args)


def log_eval_failure(results_dir: str, cfg: dict, exc: Exception) -> None:
    log_dir = os.path.join(results_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    payload = {
        "event": "eval_failed",
        "config": cfg,
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback": "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        ),
    }

    log_path = os.path.join(log_dir, "eval_failures.jsonl")
    with open(log_path, "a") as f:
        f.write(json.dumps(payload) + "\n")

    print(f"Logged eval failure to {log_path}")


def main_eval(args, config):
    """
    Run evaluation using args.datasets and args.models from main.py.

    args.datasets should already be normalized in main.py, e.g.
        PTBXL -> PTBXL_form, PTBXL_super, PTBXL_sub, PTBXL_rhythm

    args.models can be:
        all, D_BETA, MERL, Random, Random_Resnet, exact random model name, etc.
    """
    selected_datasets = normalize_datasets(args.datasets)
    selected_models = normalize_models(args.models)

    train_pcts = [0.01, 0.1, 1.0]

    eval_cfg = config.get("eval", {})
    eval_batch_size = int(eval_cfg.get("batch_size", 256))
    eval_num_workers = int(eval_cfg.get("num_workers", 0))
    sklearn_n_jobs = int(eval_cfg.get("sklearn_n_jobs", 1))
    n_boot=int(config.get("stats_tests", {}).get("n_boot", 1000))

    seed = int(config.get("seed", 42))
    config_hash = eval_config_hash(config)
    max_workers = int(config.get("multi_process_eval", 1))

    print(f"Selected eval datasets: {selected_datasets}")
    print(f"Selected eval models: {selected_models}")
    print(f"Selected train percentages: {train_pcts}")

    configs = []

    for dataset_name in selected_datasets:
        for model_name in selected_models:
            for train_pct in train_pcts:
                cfg = {
                    "n_boot": n_boot,
                    "data": dataset_name,
                    "model": model_name,
                    "train_pct": float(train_pct),
                    "seed": seed,
                    "config_hash": config_hash,
                    "results_dir": config["results_dir"],
                    "embeddings_dir": config["embeddings_dir"],
                    "batch_size": eval_batch_size,
                    "num_workers": eval_num_workers,
                    "sklearn_n_jobs": sklearn_n_jobs,
                }

                if max_workers > 1:
                    configs.append(cfg)
                else:
                    configs.append(cfg)

    if max_workers > 1:
        print(f"Launching {len(configs)} eval runs using {max_workers} workers")

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(run_one, cfg): cfg
                for cfg in configs
            }

            for fut in tqdm(
                as_completed(futures.keys()),
                total=len(futures),
                desc="Evaluating",
                unit="run",
            ):
                try:
                    fut.result()
                except Exception as e:
                    print("❌ Run failed:", e)
                    log_eval_failure(config["results_dir"], futures[fut], e)

    else:
        for cfg in tqdm(configs, desc="Evaluating", unit="run"):
            run_one(cfg)
