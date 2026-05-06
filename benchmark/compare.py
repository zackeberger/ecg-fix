import os
import pickle
import itertools
import numpy as np
import pandas as pd
import argparse

from multiprocessing import Pool, cpu_count

from sklearn.metrics import roc_auc_score, average_precision_score
from benchmark.config import add_config_arg, apply_config_to_args


# =========================
# CONFIG
# =========================
BASE_PATH = None
SAVEBASE_PATH = None

N_BOOT = 1000
N_PERM = 1000

#LABELS = ["NORM","IMI","AMI","STTC","LVH","CLBBB", "shd_moderate_or_greater_flag", "lvef_lte_45_flag", "tricuspid_regurgitation_moderate_or_greater_flag"]


LABELS = ["NORM","IMI","AMI","STTC","LVH","CLBBB", "STE_", "TAB_", "PRC(S)", "shd_moderate_or_greater_flag", "lvef_lte_45_flag", "tricuspid_regurgitation_moderate_or_greater_flag"]
LABELS_pair = ["D_BETA","D_BETA","MERL","MERL","KED","D_BETA", "MERL", "MERL", "D_BETA", "MERL", "MERL", "MERL"]

MODELS = set(
   ["Random_500Hz_Z_score_none_No_Bandpass_Resnet18",
    "CLOCS",
    "KED",
    "HeartLang",
    "MERL",
    "D_BETA"
])

# =========================
# LOAD RESULTS
# =========================
def load_all_models(data, train_pct, results_dir):
    path = os.path.join(results_dir, data, str(train_pct))
    results = {}

    for file in os.listdir(path):
        if file.endswith(".pkl"):
            model = file.replace(".pkl", "")
            if model in MODELS:
                with open(os.path.join(path, file), "rb") as f:
                    results[model] = pickle.load(f)

    print(f"Loaded {len(results)} models from {path}")
    return results


# =========================
# METRICS
# =========================
def compute_metrics(y_true, y_pred):
    n_labels = y_true.shape[1]

    aurocs = []
    auprcs = []

    for i in range(n_labels):
        try:
            aurocs.append(roc_auc_score(y_true[:, i], y_pred[:, i]))
            auprcs.append(average_precision_score(y_true[:, i], y_pred[:, i]))
        except:
            aurocs.append(np.nan)
            auprcs.append(np.nan)

    return {
        "auroc_per_label": np.array(aurocs),
        "auprc_per_label": np.array(auprcs),
        "auroc_macro": np.nanmean(aurocs),
        "auprc_macro": np.nanmean(auprcs),
    }


# =========================
# BOOTSTRAP
# =========================
def bootstrap_diff(y_true, y_pred_A, y_pred_B, metric_fn, n_boot=N_BOOT):
    N = len(y_true)

    # 🔥 vectorized bootstrap indices
    idx = np.random.randint(0, N, size=(n_boot, N))

    deltas = np.empty(n_boot)

    for b in range(n_boot):
        yt = y_true[idx[b]]
        pA = y_pred_A[idx[b]]
        pB = y_pred_B[idx[b]]

        deltas[b] = metric_fn(yt, pA) - metric_fn(yt, pB)

    mean = deltas.mean()
    lo, hi = np.percentile(deltas, [2.5, 97.5])

    return mean, lo, hi


# =========================
# PERMUTATION TEST
# =========================
def permutation_test(y_true, y_pred_A, y_pred_B, metric_fn, n_perm=N_PERM):
    mA = metric_fn(y_true, y_pred_A)
    mB = metric_fn(y_true, y_pred_B)
    delta_obs = mA - mB

    N = len(y_true)
    deltas = np.zeros(n_perm)

    for r in range(n_perm):
        swap = np.random.rand(N) < 0.5

        pA = y_pred_A.copy()
        pB = y_pred_B.copy()

        pA[swap], pB[swap] = pB[swap], pA[swap]

        mA_r = metric_fn(y_true, pA)
        mB_r = metric_fn(y_true, pB)

        deltas[r] = mA_r - mB_r

    p = (1 + np.sum(np.abs(deltas) >= abs(delta_obs))) / (n_perm + 1)

    return delta_obs, p


# =========================
# METRIC WRAPPERS
# =========================
def auroc_macro(y_true, y_pred):
    return compute_metrics(y_true, y_pred)["auroc_macro"]


def auprc_macro(y_true, y_pred):
    return compute_metrics(y_true, y_pred)["auprc_macro"]


def auroc_label(i):
    def fn(y_true, y_pred):
        return roc_auc_score(y_true[:, i], y_pred[:, i])
    return fn


def auprc_label(i):
    def fn(y_true, y_pred):
        return average_precision_score(y_true[:, i], y_pred[:, i])
    return fn

# =========================
# WORKER FUNCTION (NEW)
# =========================
def compare_pair(args):
    A, B, results, vocab = args

    print(vocab)

    rA = results[A]
    rB = results[B]

    y_true = rA["y_true"]
    y_pred_A = rA["y_pred"]
    y_pred_B = rB["y_pred"]

    print(f"Running {A} vs {B}")

    rows = []

    # -----------------
    # MACRO
    # -----------------
    for metric_name, metric_fn in [
        ("AUROC", auroc_macro),
        ("AUPRC", auprc_macro),
    ]:
        mean, lo, hi = bootstrap_diff(
            y_true, y_pred_A, y_pred_B, metric_fn
        )

        delta_obs, p = permutation_test(
            y_true, y_pred_A, y_pred_B, metric_fn
        )

        within_noise = (lo <= 0 <= hi)

        sig = (p <=.05)

        rows.append({
            "model_A": A,
            "model_B": B,
            "metric": metric_name,
            "label": "macro",
            "ci_delta_mean": mean,
            "ci_delta_low": lo,
            "ci_delta_high": hi,
            "ci_delta_within_noise": within_noise,
            "p_value": p,
            "p_value_sig": sig,
            
        })

    # -----------------
    # PER LABEL
    # -----------------


    for label_name, i in vocab.items():
        if label_name not in LABELS:
            continue 
        for metric_name, metric_fn in [
            ("AUROC", auroc_label(i)),
            ("AUPRC", auprc_label(i)),
        ]:
            try:
                mean, lo, hi = bootstrap_diff(
                    y_true, y_pred_A, y_pred_B, metric_fn
                )

                delta_obs, p = permutation_test(
                    y_true, y_pred_A, y_pred_B, metric_fn
                )

                within_noise = (lo <= 0 <= hi)

                sig = (p <=.05)

                rows.append({
                    "model_A": A,
                    "model_B": B,
                    "metric": metric_name,
                    "label": label_name,
                    "ci_delta_mean": mean,
                    "ci_delta_low": lo,
                    "ci_delta_high": hi,
                    "ci_delta_within_noise": within_noise,
                    "p_value": p,
                    "p_value_sig": sig,
                    
                })

            except Exception:
                pass

    return rows


def compare_all(data, best_model, train_pct, results_dir, comparison_dir, save_csv=True):
    results = load_all_models(data, train_pct, results_dir)
    models = sorted(list(results.keys()))

    # ---- get vocab ----
    vocab = None
    for m in models:
        if "vocab" in results[m]:
            vocab = results[m]["vocab"]
            break

    if vocab is None:
        raise ValueError("No vocab found in saved results!")

    for m in models:
        if not np.array_equal(vocab, results[m]["vocab"]):
            raise ValueError(f"Vocab mismatch for model {m}")

    print(f"Using {len(vocab)} labels")

    best_models = set([best_model])
    for l, m in zip(LABELS, LABELS_pair):
        if l in vocab.keys():
            best_models.add(m)
    # -----------------
    # MULTIPROCESSING
    # -----------------
    pairs = list(itertools.combinations(models, 2))
    args = [(A, B, results, vocab) for A, B in pairs if A in best_models or B in best_models]

    all_rows = []  # ✅ accumulate results


    log_every = 30
    counter = 0

    with Pool(cpu_count()//2) as pool:
        for result in pool.imap_unordered(compare_pair, args):
            # result = list of rows from one pair

            all_rows.extend(result)  # ✅ keep for CSV

            counter += 1
            if counter % log_every == 0:
                print(f"Compared {counter}/{len(args)} model pairs")

    # -----------------
    # FINAL DATAFRAME
    # -----------------
    df = pd.DataFrame(all_rows)

    # -----------------
    # SAVE CSV
    # -----------------
    if save_csv:
        os.makedirs(comparison_dir, exist_ok=True)
        save_path = os.path.join(comparison_dir, f"comparison_{data}_{train_pct}.csv")
        df.to_csv(save_path, index=False)
        print(f"\nSaved results to {save_path}")

    return df

run = [("PTBXL_super", "D_BETA", .01), 
        ("PTBXL_super", "D_BETA", .1), 
        ("PTBXL_super", "MERL", 1),

        ("ECHO_NEXT", "D_BETA", .01), 
        ("ECHO_NEXT", "MERL", .1), 
         ("ECHO_NEXT", "D_BETA", 1), 

        ("PTBXL_sub", "D_BETA", .01), 
         ("PTBXL_sub", "D_BETA", .1), 
        ("PTBXL_sub", "D_BETA", 1),
         ("PTBXL_form", "D_BETA", .01), 
        ("PTBXL_form", "D_BETA", .1), 
        ("PTBXL_form", "MERL", 1), 
        ("PTBXL_rhythm", "D_BETA", .01), 
        ("PTBXL_rhythm", "D_BETA", .1), 
        ("PTBXL_rhythm", "D_BETA", 1), 
        #  ("CPSC", "D_BETA", .01), 
        # ("CPSC", "D_BETA", .1), 
        # ("CPSC", "D_BETA", 1), 
        # ("CSN", "D_BETA", .01), 
        # ("CSN", "D_BETA", .1), 
        # ("CSN", "D_BETA", 1), 
]
# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare saved ECG benchmark model results.")
    add_config_arg(parser)
    parser.add_argument("--data", type=str, default=None, help="Dataset/task to compare. If omitted, runs the built-in comparison list.")
    parser.add_argument("--best_model", type=str, default="D_BETA", help="Reference model used to select comparison pairs.")
    parser.add_argument("--train_pct", type=float, default=1.0, help="Training percentage results folder to compare.")
    parser.add_argument("--n_boot", type=int, default=N_BOOT, help="Bootstrap iterations.")
    parser.add_argument("--n_perm", type=int, default=N_PERM, help="Permutation iterations.")
    args = parser.parse_args()
    apply_config_to_args(args)
    N_BOOT = args.n_boot
    N_PERM = args.n_perm

    if args.data:
        compare_all(args.data, args.best_model, args.train_pct, args.results_dir, args.comparison_dir)
    else:
        for data, best_model, train_pct in run:
            compare_all(data, best_model, train_pct, args.results_dir, args.comparison_dir)
