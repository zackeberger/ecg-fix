import os
import pickle
import random

import numpy as np
import scipy.stats
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

try:
    import neurokit2 as nk
except ImportError:
    nk = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False




def get_embedings(model_name, model, x):
    with torch.no_grad():
        if model_name == "D_BETA":
            return extract_ecg_DBeta_features(model, x)  # [B, embedding_dim]
        else:
            return model(x)


def extract_hr_stats_neurokit2(X, sampling_rate):
    """
    Extract mean HR and HRV from batched 12-lead ECG using NeuroKit2.

    Parameters
    ----------
    X : np.ndarray
        Array of shape (B, 12, L) containing ECG signals.
    sampling_rate : int or float
        Sampling rate in Hz.

    Returns
    -------
    stats : np.ndarray
        Array of shape (B, 2) with mean HR and HRV per ECG.
        Columns: [mean_hr, hrv]
    """
    if nk is None:
        raise ImportError("neurokit2 is required for heart-rate feature extraction. Install requirements.txt first.")

    B, C, L = X.shape
    results = []

    for b in range(B):
        # Take one representative lead (e.g., lead II is index 1)
        if C == 12:
            ecg_signal = X[b, 1, :]  
        elif C == 1:
            ecg_signal = X[b, 0, :]  
        

        try:
            # Process ECG
            ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=sampling_rate)
            _, rpeaks = nk.ecg_peaks(ecg_cleaned, sampling_rate=sampling_rate)

             # --- Try full HRV first ---
            try:
                hrv_indices = nk.hrv(rpeaks, sampling_rate=sampling_rate, show=False)
            except Exception:
                # --- Fallback: time-domain HRV only ---
                hrv_indices = nk.hrv_time(rpeaks, sampling_rate=sampling_rate, show=False)


            hrv = (
                hrv_indices["HRV_SDNN"].values[0]
                if "HRV_SDNN" in hrv_indices
                else np.nan
            )

            mean_hr = (
                hrv_indices["HRV_MeanNN"].values[0]
                if "HRV_MeanNN" in hrv_indices
                else np.nan
            )

            results.append([mean_hr, hrv])

        #    print([60000/mean_hr, hrv])
        except Exception as e:
      #      print(e)
            # In case NeuroKit fails on a noisy signal
            results.append([np.nan, np.nan])

    return np.array(results)


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
    
def save_metrics(args, y_true, y_pred, vocab):
    save_dict = {
        "y_true": y_true,
        "y_pred": y_pred,
        "vocab": vocab,
    }
    path = os.path.join(args.results_dir, args.data, str(args.train_pct))
    os.makedirs(path, exist_ok=True)

    save_path = os.path.join(path, f"{args.model}.pkl")

    with open(save_path, "wb") as f:
        pickle.dump(save_dict, f)

    print(f"Saved metrics to {save_path}")

def log_metrics_stats(args, y_true, y_pred, vocab):
    if "regression" in args.label_type:
        return log_metrics_regression(y_true, y_pred, vocab)
    else:
        return log_metrics_prob(y_true, y_pred, vocab)

def log_metrics_prob(y_true, y_pred, vocab, n_bootstrap=1000, seed=42):
    """
    Computes per-label metrics + macro AUC and a stratified bootstrap
    (per label) for macro AUC and per-label AUC CIs.

    Stratified bootstrap (per label):
      - For each label i, sample with replacement separately from the
        positive and negative sets to preserve prevalence for that label.
      - Compute that label's AUC on the resampled set.
      - Macro AUC for that bootstrap replicate = mean over labels that are valid.

    Args:
        y_true: (N, C) binary matrix.
        y_pred: (N, C) predicted probabilities/scores.
        vocab:  dict {label_name: column_index}
        n_bootstrap: number of bootstrap replicates
        seed: RNG seed
    """
    num_classes = y_pred.shape[1]
    assert num_classes == len(vocab.keys()), "Vocab size different than keys check"

    logging = {}

    # --- Macro-AUC (point estimate from full data) ---
    aucs = []
    auprcs = []
    for label, i in vocab.items():
        # counts
        pos_count = int(np.sum(y_true[:, i] == 1))
        neg_count = int(np.sum(y_true[:, i] == 0))
        logging[label + "_true_count_pos"] = pos_count
        logging[label + "_true_count_neg"] = neg_count

        # AUC
        try:
            auc = roc_auc_score(y_true[:, i], y_pred[:, i])
            aucs.append(auc)
            logging[label+"_auc"]  = auc
        except ValueError as v:
            print(v)
            logging[label+"_auc"]  = np.nan

        # Accuracy at 0.5 threshold
        try:
            y_pred_binary = (y_pred[:, i] >= 0.5).astype(int)
            logging[label + "_acc"] = accuracy_score(y_true[:, i], y_pred_binary)
        except ValueError:
            logging[label + "_acc"] = np.nan

        #auprc 
        try:
            auprc = average_precision_score(y_true[:, i], y_pred[:, i])
            auprcs.append(auprc)
            logging[label + "_auprc"] = auprc
        except ValueError:
            logging[label + "_auprc"] = np.nan

    macro_auc = np.mean(aucs) if len(aucs) else np.nan
    macro_auprc = np.mean(auprcs) if len(auprcs) else np.nan
    print(f"Macro-AUC: {macro_auc:.4f}" if np.isfinite(macro_auc) else "Macro-AUC: nan")
    print(f"Macro-AUPRC: {macro_auprc:.4f}" if np.isfinite(macro_auprc) else "Macro-AUPRC: nan")

    # --- Precompute pos/neg indices per label for stratified bootstrap ---
    label_pos_idx = {}
    label_neg_idx = {}
    valid_labels = []  # labels with both classes present
    for label, i in vocab.items():
        pos_idx = np.flatnonzero(y_true[:, i] == 1)
        neg_idx = np.flatnonzero(y_true[:, i] == 0)
        label_pos_idx[label] = pos_idx
        label_neg_idx[label] = neg_idx
        if len(pos_idx) > 0 and len(neg_idx) > 0:
            valid_labels.append(label)

    # --- Stratified bootstrapping (per label) ---
    bootstrapped_auc_macro_scores = []
    bootstrapped_auprc_macro_scores = []
    per_label_auc_scores = {label: [] for label in vocab.keys()}
    per_label_auprc_scores = {label: [] for label in vocab.keys()}

    if len(valid_labels) == 0:
        # No label has both classes; nothing to bootstrap
        bootstrapped_scores = np.array([])
    else:
        rng = np.random.default_rng(123)
        for _ in range(n_bootstrap):
            aucs_bs = []
            auprcs_bs = []

            # For each label independently, draw a stratified bootstrap sample
            for label in valid_labels:
                i = vocab[label]
                pos_idx = label_pos_idx[label]
                neg_idx = label_neg_idx[label]

                n_pos = len(pos_idx)
                n_neg = len(neg_idx)

                # sample WITH replacement preserving class counts
                bs_pos = rng.choice(pos_idx, size=n_pos, replace=True)
                bs_neg = rng.choice(neg_idx, size=n_neg, replace=True)
                bs_indices = np.concatenate([bs_pos, bs_neg])

                try:
                    auc = roc_auc_score(y_true[bs_indices, i], y_pred[bs_indices, i])
                    auprc = average_precision_score(y_true[bs_indices, i], y_pred[bs_indices, i])
                    if not np.isnan(auc) and not np.isnan(auprc):
                        per_label_auc_scores[label].append(auc)
                        per_label_auprc_scores[label].append(auprc)
                        aucs_bs.append(auc)
                        auprcs_bs.append(auprc)
                except:
                    continue            

            bootstrapped_auc_macro_scores.append(np.mean(aucs_bs))
            bootstrapped_auprc_macro_scores.append(np.mean(auprcs_bs))

        bootstrapped_auc_scores = np.array(bootstrapped_auc_macro_scores)
        bootstrapped_auprc_scores = np.array(bootstrapped_auprc_macro_scores)

    # --- Aggregate bootstrap stats ---
    if bootstrapped_auc_scores.size > 0:
        mean_auc = float(np.mean(bootstrapped_auc_scores))
        std_auc = float(np.std(bootstrapped_auc_scores))
        ci_lower = float(np.percentile(bootstrapped_auc_scores, 2.5))
        ci_upper = float(np.percentile(bootstrapped_auc_scores, 97.5))
        print(f"Stratified-Boot Macro-AUC: mean={mean_auc:.4f}, std={std_auc:.4f}, 95% CI=({ci_lower:.4f}, {ci_upper:.4f})")
    else:
        mean_auc = std_auc = ci_lower = ci_upper = np.nan
        print("Stratified-Boot Macro-AUC: not available (no valid labels with both classes).")

    if bootstrapped_auprc_scores.size > 0:
        mean_auprc = float(np.mean(bootstrapped_auprc_scores))
        std_auprc = float(np.std(bootstrapped_auprc_scores))
        ci_lower_auprc = float(np.percentile(bootstrapped_auprc_scores, 2.5))
        ci_upper_auprc = float(np.percentile(bootstrapped_auprc_scores, 97.5))
        print(f"Stratified-Boot Macro-AUPRC: mean={mean_auprc:.4f}, std={std_auprc:.4f}, 95% CI=({ci_lower_auprc:.4f}, {ci_upper_auprc:.4f})")
    else:
        mean_auprc = std_auprc = ci_lower_auprc = ci_upper_auprc = np.nan
        print("Stratified-Boot Macro-AUPRC: not available (no valid labels with both classes).")

    for label in vocab.keys():
        auc_scores = per_label_auc_scores[label]
        auprc_scores = per_label_auprc_scores[label]
        if len(auc_scores) > 0:
            logging[label + "_auc_boot_mean"] = float(np.mean(auc_scores))
            logging[label + "_auc_boot_std"]  = float(np.std(auc_scores))
            logging[label + "_auc_ci_low"]    = float(np.percentile(auc_scores, 2.5))
            logging[label + "_auc_ci_high"]   = float(np.percentile(auc_scores, 97.5))
            logging[label + "_auprc_boot_mean"] = float(np.mean(auprc_scores))
            logging[label + "_auprc_boot_std"]  = float(np.std(auprc_scores))
            logging[label + "_auprc_ci_low"]    = float(np.percentile(auprc_scores, 2.5))
            logging[label + "_auprc_ci_high"]   = float(np.percentile(auprc_scores, 97.5))
        else:
            logging[label + "_auc_boot_mean"] = np.nan
            logging[label + "_auc_boot_std"]  = np.nan
            logging[label + "_auc_ci_low"]    = np.nan
            logging[label + "_auc_ci_high"]   = np.nan
            logging[label + "_auprc_boot_mean"] = np.nan
            logging[label + "_auprc_boot_std"]  = np.nan
            logging[label + "_auprc_ci_low"]    = np.nan
            logging[label + "_auprc_ci_high"]   = np.nan

    logging.update({
        "macro_auc": float(macro_auc) if np.isfinite(macro_auc) else np.nan,
        "macro_auc_boot_mean": mean_auc,
        "macro_auc_boot_std": std_auc,
        "macro_auc_boot_ci_low": ci_lower,
        "macro_auc_boot_ci_high": ci_upper,
        "num_classes": num_classes,
        "num_bootstrap": n_bootstrap,
        "num_valid_labels_for_boot": len(valid_labels),
        "macro_auprc": float(macro_auprc) if np.isfinite(macro_auprc) else np.nan,
        "macro_auprc_boot_mean": mean_auprc,
        "macro_auprc_boot_std": std_auprc,
        "macro_auprc_boot_ci_low": ci_lower_auprc,
        "macro_auprc_boot_ci_high": ci_upper_auprc,
    })

    return logging
        

def log_metrics_regression(
    y_true,
    y_pred,
    vocab,
    n_bootstrap=1000,
    seed=42,
):
    """
    Regression analogue of log_metrics_prob.

    Args:
        y_true: (N, C) ground-truth regression targets
        y_pred: (N, C) predicted values
        vocab:  dict {label_name: column_index}
        n_bootstrap: number of bootstrap replicates
        seed: RNG seed
    """

    num_targets = y_pred.shape[1]
    assert num_targets == len(vocab), "Vocab size mismatch"

    logging = {}

    # --------------------------------------------------
    # Point estimates (full data)
    # --------------------------------------------------
    per_target_rmse = []
    per_target_mae = []
    per_target_r2 = []
    per_target_pearson = []
    per_target_spearman = []

    for label, i in vocab.items():
        yt = y_true[:, i]
        yp = y_pred[:, i]

        mse = mean_squared_error(yt, yp)
        rmse = float(np.sqrt(mse))
        mae = float(mean_absolute_error(yt, yp))
        r2 = float(r2_score(yt, yp))

        # Correlations
        pearson_r = float(np.corrcoef(yt, yp)[0, 1]) if np.std(yt) > 0 else np.nan
        spearman_r = float(
            scipy.stats.spearmanr(yt, yp, nan_policy="omit").correlation
        )

        logging[label + "_rmse"] = rmse
        logging[label + "_mae"] = mae
        logging[label + "_r2"] = r2
        logging[label + "_pearson_r"] = pearson_r
        logging[label + "_spearman_r"] = spearman_r

        per_target_rmse.append(rmse)
        per_target_mae.append(mae)
        per_target_r2.append(r2)
        per_target_pearson.append(pearson_r)
        per_target_spearman.append(spearman_r)

    # Macro (mean over targets)
    macro_rmse = float(np.nanmean(per_target_rmse))
    macro_mae = float(np.nanmean(per_target_mae))
    macro_r2 = float(np.nanmean(per_target_r2))
    macro_pearson = float(np.nanmean(per_target_pearson))
    macro_spearman = float(np.nanmean(per_target_spearman))

    print(f"Macro RMSE: {macro_rmse:.4f}")
    print(f"Macro MAE: {macro_mae:.4f}")
    print(f"Macro R2: {macro_r2:.4f}")
    print(f"Macro Pearson r: {macro_pearson:.4f}")
    print(f"Macro Spearman r: {macro_spearman:.4f}")

    # --------------------------------------------------
    # Bootstrap (paired over samples)
    # --------------------------------------------------
    N = y_true.shape[0]

    rng_global = np.random.default_rng(seed)
    bootstrap_seeds = rng_global.integers(0, 1_000_000, size=n_bootstrap)

    boot_rmse = []
    boot_mae = []
    boot_r2 = []
    boot_pearson = []
    boot_spearman = []

    per_target_boot = {
        label: {
            "rmse": [],
            "mae": [],
            "r2": [],
            "pearson": [],
            "spearman": [],
        }
        for label in vocab
    }

    for boot in bootstrap_seeds:
        rng = np.random.default_rng(boot)
        bs_idx = rng.choice(N, size=N, replace=True)

        rmse_bs = []
        mae_bs = []
        r2_bs = []
        pearson_bs = []
        spearman_bs = []

        for label, i in vocab.items():
            yt = y_true[bs_idx, i]
            yp = y_pred[bs_idx, i]

            mse = mean_squared_error(yt, yp)
            rmse = float(np.sqrt(mse))
            mae = float(mean_absolute_error(yt, yp))
            r2 = float(r2_score(yt, yp))

            pearson_r = float(np.corrcoef(yt, yp)[0, 1]) if np.std(yt) > 0 else np.nan
            spearman_r = float(
                scipy.stats.spearmanr(yt, yp, nan_policy="omit").correlation
            )

            per_target_boot[label]["rmse"].append(rmse)
            per_target_boot[label]["mae"].append(mae)
            per_target_boot[label]["r2"].append(r2)
            per_target_boot[label]["pearson"].append(pearson_r)
            per_target_boot[label]["spearman"].append(spearman_r)

            rmse_bs.append(rmse)
            mae_bs.append(mae)
            r2_bs.append(r2)
            pearson_bs.append(pearson_r)
            spearman_bs.append(spearman_r)

        boot_rmse.append(np.nanmean(rmse_bs))
        boot_mae.append(np.nanmean(mae_bs))
        boot_r2.append(np.nanmean(r2_bs))
        boot_pearson.append(np.nanmean(pearson_bs))
        boot_spearman.append(np.nanmean(spearman_bs))

    # --------------------------------------------------
    # Aggregate bootstrap stats
    # --------------------------------------------------
    def ci(x):
        return float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))

    logging.update({
        "macro_rmse": macro_rmse,
        "macro_mae": macro_mae,
        "macro_r2": macro_r2,
        "macro_pearson_r": macro_pearson,
        "macro_spearman_r": macro_spearman,

        "macro_rmse_boot_mean": float(np.mean(boot_rmse)),
        "macro_rmse_boot_std": float(np.std(boot_rmse)),
        "macro_rmse_ci_low": ci(boot_rmse)[0],
        "macro_rmse_ci_high": ci(boot_rmse)[1],

        "macro_mae_boot_mean": float(np.mean(boot_mae)),
        "macro_mae_boot_std": float(np.std(boot_mae)),
        "macro_mae_ci_low": ci(boot_mae)[0],
        "macro_mae_ci_high": ci(boot_mae)[1],

        "macro_r2_boot_mean": float(np.mean(boot_r2)),
        "macro_r2_boot_std": float(np.std(boot_r2)),
        "macro_r2_ci_low": ci(boot_r2)[0],
        "macro_r2_ci_high": ci(boot_r2)[1],

        "macro_pearson_boot_mean": float(np.mean(boot_pearson)),
        "macro_pearson_ci_low": ci(boot_pearson)[0],
        "macro_pearson_ci_high": ci(boot_pearson)[1],

        "macro_spearman_boot_mean": float(np.mean(boot_spearman)),
        "macro_spearman_ci_low": ci(boot_spearman)[0],
        "macro_spearman_ci_high": ci(boot_spearman)[1],

        "num_targets": num_targets,
        "num_bootstrap": n_bootstrap,
    })

    # Per-target bootstrap summaries
    for label in vocab:
        for metric in per_target_boot[label]:
            scores = per_target_boot[label][metric]
            logging[f"{label}_{metric}_boot_mean"] = float(np.nanmean(scores))
            logging[f"{label}_{metric}_boot_std"] = float(np.nanstd(scores))
            logging[f"{label}_{metric}_ci_low"] = float(np.nanpercentile(scores, 2.5))
            logging[f"{label}_{metric}_ci_high"] = float(np.nanpercentile(scores, 97.5))

    return logging

def _collect_label_stats(dataset, vocab):
    """
    Iterates once over dataset and collects label counts.
    Returns:
        N: total samples
        pos_counts: dict[label -> count]
        neg_counts: dict[label -> count]
    """
    num_classes = len(vocab)
    pos_counts = {label: 0 for label in vocab}
    neg_counts = {label: 0 for label in vocab}

    N = len(dataset)

    for i in range(N):
        _, y = dataset[i]

        if isinstance(y, torch.Tensor):
            y = y.cpu().numpy()

        for label, j in vocab.items():
            if y[j] == 1:
                pos_counts[label] += 1
            else:
                neg_counts[label] += 1

    return N, pos_counts, neg_counts


def log_dataset_stats(args, subset_train, dataset_valid, dataset_test):
    """
    Logs dataset sizes and per-label prevalence for train / valid / test.

    Args:
        args: argparse namespace (for metadata only)
        subset_train: training dataset or Subset
        dataset_valid: validation dataset
        dataset_test: test dataset
    """

    logging = {}

    # All datasets share vocab
    vocab = dataset_valid.vocab
    num_classes = len(vocab)

    datasets = {
        "train": subset_train,
        "valid": dataset_valid,
        "test": dataset_test,
    }

    for split, dataset in datasets.items():
        N, pos_counts, neg_counts = _collect_label_stats(dataset, vocab)

        logging[f"{split}_num_samples"] = N

        for label in vocab.keys():
            pos = pos_counts[label]
            neg = neg_counts[label]
            prev = pos / N if N > 0 else np.nan

            logging[f"{split}_{label}_pos_count"] = pos
            logging[f"{split}_{label}_neg_count"] = neg
            logging[f"{split}_{label}_prevalence"] = prev

    logging.update({
        "num_classes": num_classes,
        "dataset_name": args.data,
        "label_type": args.label_type,
    })

    # Console summary (optional but useful)
    print("Dataset statistics:")
    for split in datasets:
        print(f"  {split}: {logging[f'{split}_num_samples']} samples")

    return logging
