from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    roc_auc_score,
)
import os
import pickle
import torch
import numpy as np

from src.metrics.utils import dataset_stats_path, metric_stats_path, model_result_dir, output_path

def save_outputs(args, y_true, y_pred, vocab):
    save_dict = {
        "y_true": y_true,
        "y_pred": y_pred,
        "vocab": vocab,
    }
    path = model_result_dir(args.results_dir, args.data, args.train_pct, args.model)
    os.makedirs(path, exist_ok=True)

    save_path = output_path(args.results_dir, args.data, args.train_pct, args.model)

    with open(save_path, "wb") as f:
        pickle.dump(save_dict, f)

    print(f"Saved outputs to {save_path}")

def save_stats(args, dataset_stats, metric_stats):
    path = model_result_dir(args.results_dir, args.data, args.train_pct, args.model)
    os.makedirs(path, exist_ok=True)

    save_path_dataset = dataset_stats_path(args.results_dir, args.data, args.train_pct, args.model)
    with open(save_path_dataset, "wb") as f:
        pickle.dump(dataset_stats, f)

    print(f"Saved dataset_stats to {save_path_dataset}")

    save_path_metric = metric_stats_path(args.results_dir, args.data, args.train_pct, args.model)
    with open(save_path_metric, "wb") as f:
        pickle.dump(metric_stats, f)

    print(f"Saved metric_stats to {save_path_metric}")


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
    })

  #  print("Dataset statistics:")
 #   for split in datasets:
  #      print(f"  {split}: {logging[f'{split}_num_samples']} samples")

    return logging


def log_metrics(y_true, y_pred, vocab, n_bootstrap=1000, seed=42):
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
            #print(v)
            logging[label+"_auc"]  = np.nan

        # Accuracy at 0.5 threshold
        try:
            y_pred_binary = (y_pred[:, i] >= 0.5).astype(int)
            logging[label + "_acc"] = accuracy_score(y_true[:, i], y_pred_binary)
        except ValueError:
            logging[label + "_acc"] = np.nan

        # AUPRC
        try:
            auprc = average_precision_score(y_true[:, i], y_pred[:, i])
            auprcs.append(auprc)
            logging[label + "_auprc"] = auprc
        except ValueError:
            logging[label + "_auprc"] = np.nan

    macro_auc = np.mean(aucs) if len(aucs) else np.nan
    macro_auprc = np.mean(auprcs) if len(auprcs) else np.nan
   #(f"Macro-AUC: {macro_auc:.4f}" if np.isfinite(macro_auc) else "Macro-AUC: nan")
   # print(f"Macro-AUPRC: {macro_auprc:.4f}" if np.isfinite(macro_auprc) else "Macro-AUPRC: nan")

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
        rng = np.random.default_rng(seed)
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
                except Exception:
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
    #    print(f"Stratified-Boot Macro-AUC: mean={mean_auc:.4f}, std={std_auc:.4f}, 95% CI=({ci_lower:.4f}, {ci_upper:.4f})")
    else:
        mean_auc = std_auc = ci_lower = ci_upper = np.nan
    #    print("Stratified-Boot Macro-AUC: not available (no valid labels with both classes).")

    if bootstrapped_auprc_scores.size > 0:
        mean_auprc = float(np.mean(bootstrapped_auprc_scores))
        std_auprc = float(np.std(bootstrapped_auprc_scores))
        ci_lower_auprc = float(np.percentile(bootstrapped_auprc_scores, 2.5))
        ci_upper_auprc = float(np.percentile(bootstrapped_auprc_scores, 97.5))
   #     print(f"Stratified-Boot Macro-AUPRC: mean={mean_auprc:.4f}, std={std_auprc:.4f}, 95% CI=({ci_lower_auprc:.4f}, {ci_upper_auprc:.4f})")
    else:
        mean_auprc = std_auprc = ci_lower_auprc = ci_upper_auprc = np.nan
     #   print("Stratified-Boot Macro-AUPRC: not available (no valid labels with both classes).")

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



def _collect_label_stats(dataset, vocab):
    """
    Iterates once over dataset and collects label counts.
    Returns:
        N: total samples
        pos_counts: dict[label -> count]
        neg_counts: dict[label -> count]
    """
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
