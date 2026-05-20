import csv
import os
import pickle
from typing import Any

import numpy as np

from src.metrics.stats_tests import export_stats_tests
from src.metrics.paths import (
    MODEL_DISPLAY_NAMES,
    MODEL_ORDER,
    RANDOM_MODEL,
    metric_stats_path,
)
from src.registry import build_random_model_names


# =============================================================================
# Constants
# =============================================================================

DATASET_ORDER = [
    "PTBXL_super",
    "PTBXL_sub",
    "PTBXL_form",
    "PTBXL_rhythm",
    "CPSC",
    "CSN",
    "ECHO_NEXT",
]

DATASET_DISPLAY_NAMES = {
    "PTBXL_super": "PTB-XL Super",
    "PTBXL_sub": "PTB-XL Sub",
    "PTBXL_form": "PTB-XL Form",
    "PTBXL_rhythm": "PTB-XL Rhythm",
    "CPSC": "CPSC2018",
    "CSN": "CSN",
    "ECHO_NEXT": "EchoNext",
}

TRAIN_PCTS = [0.01, 0.1, 1.0]


# =============================================================================
# Label aliases and table specs
# =============================================================================

ECHONEXT_BINARY_LABELS = [
    "lvef_lte_45_flag",
    "lvwt_gte_13_flag",
    "aortic_stenosis_moderate_or_greater_flag",
    "aortic_regurgitation_moderate_or_greater_flag",
    "mitral_regurgitation_moderate_or_greater_flag",
    "tricuspid_regurgitation_moderate_or_greater_flag",
    "pulmonary_regurgitation_moderate_or_greater_flag",
    "rv_systolic_dysfunction_moderate_or_greater_flag",
    "pericardial_effusion_moderate_large_flag",
    "pasp_gte_45_flag",
    "tr_max_gte_32_flag",
    "shd_moderate_or_greater_flag",
]

ECHONEXT_LABEL_ALIASES = {
    "lvef_lte_45_flag": "LVEF ≤ 45",
    "lvwt_gte_13_flag": "LVWT ≥ 13",
    "aortic_stenosis_moderate_or_greater_flag": "AS",
    "aortic_regurgitation_moderate_or_greater_flag": "AR",
    "mitral_regurgitation_moderate_or_greater_flag": "MR",
    "tricuspid_regurgitation_moderate_or_greater_flag": "TR",
    "pulmonary_regurgitation_moderate_or_greater_flag": "PR",
    "rv_systolic_dysfunction_moderate_or_greater_flag": "RV Dysfunction",
    "pericardial_effusion_moderate_large_flag": "Pericardial Effusion",
    "pasp_gte_45_flag": "PASP ≥ 45",
    "tr_max_gte_32_flag": "TR Max ≥ 3.2",
    "shd_moderate_or_greater_flag": "SHD",
}

TASK_TABLE_SPECS = {
    "PTBXL_super": {
        "labels": None,
        "aliases": {},
    },
    "PTBXL_sub": {
        "labels": None,
        "aliases": {},
    },
    "PTBXL_form": {
        "labels": None,
        "aliases": {},
    },
    "PTBXL_rhythm": {
        "labels": None,
        "aliases": {},
    },
    "CPSC": {
        "labels": None,
        "aliases": {},
    },
    "CSN": {
        "labels": None,
        "aliases": {},
    },
    "ECHO_NEXT": {
        "labels": ECHONEXT_BINARY_LABELS,
        "aliases": ECHONEXT_LABEL_ALIASES,
    },
}

SMALL_TASK_TABLE_SPECS = {
    "PTBXL_sub_main_subset": {
        "dataset": "PTBXL_sub",
        "train_pct": 1.0,
        "labels": ["NORM", "IMI", "AMI", "STTC", "LVH", "CLBBB"],
        "aliases": {},
    },
    "PTBXL_form_low_support": {
        "dataset": "PTBXL_form",
        "train_pct": 1.0,
        "labels": ["PRC(S)", "STE_", "TAB_"],
        "aliases": {},
    },
    "ECHO_NEXT_main_subset": {
        "dataset": "ECHO_NEXT",
        "train_pct": 1.0,
        "labels": [
            "shd_moderate_or_greater_flag",
            "lvef_lte_45_flag",
            "tricuspid_regurgitation_moderate_or_greater_flag",
        ],
        "aliases": {
            "shd_moderate_or_greater_flag": "SHD",
            "lvef_lte_45_flag": "LVEF ≤ 45",
            "tricuspid_regurgitation_moderate_or_greater_flag": "TR",
        },
    },
}

CLEAN_COMPARISON_TABLE_SPECS = {
    "ptbxl_macroauc_change_sub": {
        "orig_dataset": "PTBXL_sub",
        "clean_dataset": "PTBXL_C_sub",
        "train_pct": 1.0,
    },
    "ptbxl_macroauc_change_rhythm": {
        "orig_dataset": "PTBXL_rhythm",
        "clean_dataset": "PTBXL_C_rhythm",
        "train_pct": 1.0,
    },
    "ptbxl_macroauc_change_form": {
        "orig_dataset": "PTBXL_form",
        "clean_dataset": "PTBXL_C_form",
        "train_pct": 1.0,
    },
}


# =============================================================================
# Path and IO helpers
# =============================================================================

def _export_dir(config: dict) -> str:
    path = config["tables_dir"]
    os.makedirs(path, exist_ok=True)
    return path


def _export_subdir(config: dict, *parts: str) -> str:
    path = os.path.join(_export_dir(config), *parts)
    os.makedirs(path, exist_ok=True)
    return path


def _metric_path(config: dict, dataset: str, train_pct: float, model: str) -> str:
    return metric_stats_path(
        config["results_dir"],
        dataset,
        train_pct,
        model,
    )


def _safe_read_pickle(path: str) -> dict[str, Any] | None:
    if not os.path.exists(path):
        print(f"Skipping missing file: {path}")
        return None

    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Skipping unreadable file: {path} ({e})")
        return None


def _write_csv(path: str, headers: list[str], rows: list[dict]) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved CSV: {path}")
    return path


# =============================================================================
# Formatting helpers
# =============================================================================

def _is_missing(x) -> bool:
    if x is None:
        return True

    try:
        return bool(np.isnan(x))
    except TypeError:
        return False


def _fmt_value(x, digits: int = 3) -> str:
    if _is_missing(x):
        return "---"

    return f"{float(x):.{digits}f}"


def _fmt_metric_with_ci(
    stats: dict | None,
    metric_key: str,
    low_key: str,
    high_key: str,
    digits: int = 3,
) -> str:
    if stats is None:
        return "---"

    value = stats.get(metric_key)
    low = stats.get(low_key)
    high = stats.get(high_key)

    if _is_missing(value):
        return "---"

    value_s = _fmt_value(value, digits)

    if _is_missing(low) or _is_missing(high):
        return value_s

    return f"{value_s} [{_fmt_value(low, digits)}, {_fmt_value(high, digits)}]"


def _fmt_auc_auprc_cell(
    stats: dict | None,
    label: str,
    digits: int = 3,
) -> str:
    auc = _fmt_metric_with_ci(
        stats,
        metric_key=f"{label}_auc_boot_mean",
        low_key=f"{label}_auc_ci_low",
        high_key=f"{label}_auc_ci_high",
        digits=digits,
    )

    auprc = _fmt_metric_with_ci(
        stats,
        metric_key=f"{label}_auprc_boot_mean",
        low_key=f"{label}_auprc_ci_low",
        high_key=f"{label}_auprc_ci_high",
        digits=digits,
    )

    if auc == "---" and auprc == "---":
        return "---"

    return f"{auc} / {auprc}"

def _fmt_macro_auc_auprc_cell(stats: dict | None, digits: int = 3) -> str:
    auc = _fmt_metric_with_ci(
        stats,
        metric_key="macro_auc_boot_mean",
        low_key="macro_auc_boot_ci_low",
        high_key="macro_auc_boot_ci_high",
        digits=digits,
    )

    auprc = _fmt_metric_with_ci(
        stats,
        metric_key="macro_auprc_boot_mean",
        low_key="macro_auprc_boot_ci_low",
        high_key="macro_auprc_boot_ci_high",
        digits=digits,
    )

    if auc == "---" and auprc == "---":
        return "---"

    return f"{auc} / {auprc}"

def _fmt_delta(orig_stats: dict | None, clean_stats: dict | None, digits: int = 3) -> str:
    orig = _get_metric_value(orig_stats, "macro_auc_boot_mean")
    clean = _get_metric_value(clean_stats, "macro_auc_boot_mean")

    if orig is None or clean is None:
        return "---"

    return f"{clean - orig:+.{digits}f}"


def _fmt_mean_std(values: list[float], digits: int = 2) -> str:
    values = [float(v) for v in values if v is not None and np.isfinite(v)]

    if len(values) == 0:
        return "---"

    mean = float(np.mean(values))
    std = float(np.std(values))

    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def _fmt_single_value(value: float | None, digits: int = 2) -> str:
    if value is None or not np.isfinite(value):
        return "---"

    return f"{float(value):.{digits}f}"


def _train_pct_label(train_pct: float) -> str:
    if float(train_pct) == 1.0:
        return "100%"

    return f"{int(round(float(train_pct) * 100))}%"


def _train_pct_for_filename(train_pct: float) -> str:
    return str(float(train_pct)).replace(".", "p")


def _label_display_name(label: str, aliases: dict[str, str] | None = None) -> str:
    if aliases is None:
        aliases = {}

    return aliases.get(label, label)


# =============================================================================
# Metric extraction and label inference
# =============================================================================

def _get_metric_value(stats: dict | None, key: str) -> float | None:
    if stats is None:
        return None

    value = stats.get(key)

    if _is_missing(value):
        return None

    return float(value)


def _macro_auc_from_metric_file(
    config: dict,
    dataset: str,
    train_pct: float,
    model: str,
) -> float | None:
    stats = _safe_read_pickle(_metric_path(config, dataset, train_pct, model))
    return _get_metric_value(stats, "macro_auc_boot_mean")


def _infer_labels_from_stats(stats: dict | None) -> list[str]:
    if stats is None:
        return []

    labels = []

    for key in stats.keys():
        if key.endswith("_auc_boot_mean"):
            label = key[: -len("_auc_boot_mean")]

            if not label.startswith("macro"):
                labels.append(label)

    return sorted(set(labels))


def _infer_labels_for_dataset(
    config: dict,
    dataset: str,
    train_pct: float,
) -> list[str]:
    labels = []

    for model in MODEL_ORDER:
        stats = _safe_read_pickle(_metric_path(config, dataset, train_pct, model))
        labels.extend(_infer_labels_from_stats(stats))

    return sorted(set(labels))


# =============================================================================
# Macro table export
# =============================================================================

def export_macro_table(config: dict) -> str:
    """
    Export combined macro table.

    CSV shape:
        Dataset, Train %, Random, CLOCS, KED, HeartLang, MERL, D-BETA

    Each cell:
        AUC [low, high] / AUPRC [low, high]
    """
    headers = ["Dataset", "Train %"] + [
        MODEL_DISPLAY_NAMES.get(model, model)
        for model in MODEL_ORDER
    ]

    rows = []

    for dataset in DATASET_ORDER:
        for train_pct in TRAIN_PCTS:
            row = {
                "Dataset": DATASET_DISPLAY_NAMES.get(dataset, dataset),
                "Train %": _train_pct_label(train_pct),
            }

            for model in MODEL_ORDER:
                stats = _safe_read_pickle(_metric_path(config, dataset, train_pct, model))
                model_display = MODEL_DISPLAY_NAMES.get(model, model)
                row[model_display] = _fmt_macro_auc_auprc_cell(stats)

            rows.append(row)

    out_path = os.path.join(
        _export_subdir(config, "tables", "summary"),
        "macro_auc_auprc_table.csv",
    )
    return _write_csv(out_path, headers, rows)


# =============================================================================
# Task table export
# =============================================================================

def export_task_table(
    config: dict,
    dataset: str,
    train_pct: float = 1.0,
    aliases: dict[str, str] | None = None,
    labels: list[str] | None = None,
    output_name: str | None = None,
) -> str:
    """
    Export task-level AUC/AUPRC table.

    CSV shape:
        Method, LABEL_1, LABEL_2, ...

    Each cell:
        AUC [low, high] / AUPRC [low, high]
    """
    if labels is None:
        labels = _infer_labels_for_dataset(config, dataset, train_pct)

    if output_name is None:
        output_name = (
            f"{dataset}_task_auc_auprc_trainpct_"
            f"{_train_pct_for_filename(train_pct)}.csv"
        )

    headers = ["Method"] + [
        _label_display_name(label, aliases)
        for label in labels
    ]

    rows = []

    for model in MODEL_ORDER:
        stats = _safe_read_pickle(_metric_path(config, dataset, train_pct, model))

        row = {
            "Method": MODEL_DISPLAY_NAMES.get(model, model),
        }

        for label in labels:
            label_display = _label_display_name(label, aliases)
            row[label_display] = _fmt_auc_auprc_cell(stats, label)

        rows.append(row)

    out_path = os.path.join(
        _export_subdir(config, "tables", "tasks", dataset),
        output_name,
    )
    return _write_csv(out_path, headers, rows)


def export_all_task_tables(config: dict) -> None:
    for dataset, spec in TASK_TABLE_SPECS.items():
        export_task_table(
            config=config,
            dataset=dataset,
            train_pct=1.0,
            aliases=spec.get("aliases", {}),
            labels=spec.get("labels"),
        )


def export_subset_task_tables(config: dict) -> None:
    for name, spec in SMALL_TASK_TABLE_SPECS.items():
        export_task_table(
            config=config,
            dataset=spec["dataset"],
            train_pct=spec.get("train_pct", 1.0),
            labels=spec["labels"],
            aliases=spec.get("aliases", {}),
            output_name=f"{name}_auc_auprc.csv",
        )


# =============================================================================
# PTB-XL original vs clean comparison export
# =============================================================================

def export_clean_comparison_table(
    config: dict,
    orig_dataset: str,
    clean_dataset: str,
    output_name: str,
    train_pct: float = 1.0,
) -> str:
    """
    Export ORIG vs CLEAN macro-AUROC comparison.

    CSV shape:
        Method, ORIG, CLEAN, Delta Macro-AUROC
    """
    headers = ["Method", "ORIG", "CLEAN", "Delta Macro-AUROC"]
    rows = []

    for model in MODEL_ORDER:
        orig_stats = _safe_read_pickle(_metric_path(config, orig_dataset, train_pct, model))
        clean_stats = _safe_read_pickle(_metric_path(config, clean_dataset, train_pct, model))

        row = {
            "Method": MODEL_DISPLAY_NAMES.get(model, model),
            "ORIG": _fmt_metric_with_ci(
                orig_stats,
                metric_key="macro_auc_boot_mean",
                low_key="macro_auc_boot_ci_low",
                high_key="macro_auc_boot_ci_high",
            ),
            "CLEAN": _fmt_metric_with_ci(
                clean_stats,
                metric_key="macro_auc_boot_mean",
                low_key="macro_auc_boot_ci_low",
                high_key="macro_auc_boot_ci_high",
            ),
            "Delta Macro-AUROC": _fmt_delta(orig_stats, clean_stats),
        }

        rows.append(row)

    out_path = os.path.join(
        _export_subdir(config, "tables", "ptbxl_clean_comparison"),
        f"{output_name}.csv",
    )
    return _write_csv(out_path, headers, rows)


def export_clean_comparison_tables(config: dict) -> None:
    for output_name, spec in CLEAN_COMPARISON_TABLE_SPECS.items():
        export_clean_comparison_table(
            config=config,
            orig_dataset=spec["orig_dataset"],
            clean_dataset=spec["clean_dataset"],
            output_name=output_name,
            train_pct=spec.get("train_pct", 1.0),
        )


# =============================================================================
# Random grid export
# =============================================================================

def _random_model_groups() -> dict[str, list[str]]:
    random_resnet, random_vit = build_random_model_names()
    all_random = random_resnet + random_vit

    return {
        "All Grid": all_random,
        "ResNet Grid": random_resnet,
        "ViT Grid": random_vit,

        "500Hz": [m for m in all_random if "_500Hz_" in m],
        "250Hz": [m for m in all_random if "_250Hz_" in m],
        "100Hz": [m for m in all_random if "_100Hz_" in m],

        "Z_score_sample": [m for m in all_random if "_Z_score_sample_" in m],
        "Z_score_none": [m for m in all_random if "_Z_score_none_" in m],
        "Z_score_dataset": [m for m in all_random if "_Z_score_dataset_" in m],

        "No_Bandpass": [m for m in all_random if "_No_Bandpass_" in m],
        "Add_Bandpass": [m for m in all_random if "_Add_Bandpass_" in m],
    }


def _collect_group_macro_aucs(
    config: dict,
    dataset: str,
    train_pct: float,
    models: list[str],
) -> list[float]:
    values = []

    for model in models:
        value = _macro_auc_from_metric_file(
            config=config,
            dataset=dataset,
            train_pct=train_pct,
            model=model,
        )

        if value is not None:
            values.append(value)

    return values


def _export_random_group_table(
    config: dict,
    output_name: str,
    group_columns: list[str],
    train_pct: float = 1.0,
) -> str:
    groups = _random_model_groups()

    headers = ["Dataset", "Original"] + group_columns
    rows = []

    for dataset in DATASET_ORDER:
        original = _macro_auc_from_metric_file(
            config=config,
            dataset=dataset,
            train_pct=train_pct,
            model=RANDOM_MODEL,
        )

        row = {
            "Dataset": DATASET_DISPLAY_NAMES.get(dataset, dataset),
            "Original": _fmt_single_value(original),
        }

        for group_name in group_columns:
            values = _collect_group_macro_aucs(
                config=config,
                dataset=dataset,
                train_pct=train_pct,
                models=groups[group_name],
            )

            row[group_name] = _fmt_mean_std(values)

        rows.append(row)

    out_path = os.path.join(
        _export_subdir(config, "tables", "random_grid"),
        output_name,
    )
    return _write_csv(out_path, headers, rows)


def export_random_table(config: dict) -> None:
    train_pct = 1.0

    random_tables = {
        "random_grid_backbone.csv": [
            "All Grid",
            "ResNet Grid",
            "ViT Grid",
        ],
        "random_grid_hz.csv": [
            "500Hz",
            "250Hz",
            "100Hz",
        ],
        "random_grid_zscore.csv": [
            "Z_score_sample",
            "Z_score_none",
            "Z_score_dataset",
        ],
        "random_grid_bandpass.csv": [
            "No_Bandpass",
            "Add_Bandpass",
        ],
    }

    for output_name, group_columns in random_tables.items():
        _export_random_group_table(
            config=config,
            output_name=output_name,
            group_columns=group_columns,
            train_pct=train_pct,
        )

    print("✅ Random grid export complete.")


# =============================================================================
# Main export entrypoint
# =============================================================================

def main_export(config: dict) -> None:
    """
    Export all CSV tables.

    Outputs:
        - tables/summary/macro_auc_auprc_table.csv
        - tables/tasks/<dataset>/*_task_auc_auprc_*.csv
        - tables/ptbxl_clean_comparison/ptbxl_macroauc_change_*.csv
        - stats_tests/<dataset>/trainpct_<pct>/<target>/<metric>/*.csv
        - tables/random_grid/random_grid_*.csv

    Missing files become:
        ---
    """
    export_macro_table(config)
    export_all_task_tables(config)
    export_subset_task_tables(config)
    export_clean_comparison_tables(config)
    export_random_table(config)
    export_stats_tests(config)
    

    print("✅ CSV export complete.")
