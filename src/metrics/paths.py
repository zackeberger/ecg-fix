import os

from src.registry import MODEL_ORDER, RANDOM_MODEL

MODEL_DISPLAY_NAMES = {
    RANDOM_MODEL: "Random",
    "CLOCS": "CLOCS",
    "KED": "KED",
    "HeartLang": "HeartLang",
    "MERL": "MERL",
    "D_BETA": "D-BETA",
}


def display_model_name(model: str) -> str:
    return MODEL_DISPLAY_NAMES.get(model, model)


def safe_name(x) -> str:
    return str(x).replace("/", "_").replace(" ", "_").replace(".", "p")


def train_pct_result_dir(results_dir: str, dataset: str, train_pct: float) -> str:
    return os.path.join(results_dir, dataset, str(float(train_pct)))


def model_result_dir(
    results_dir: str,
    dataset: str,
    train_pct: float,
    model: str,
) -> str:
    return os.path.join(train_pct_result_dir(results_dir, dataset, train_pct), model)


def output_path(results_dir: str, dataset: str, train_pct: float, model: str) -> str:
    return os.path.join(model_result_dir(results_dir, dataset, train_pct, model), f"{model}.pkl")


def dataset_stats_path(
    results_dir: str,
    dataset: str,
    train_pct: float,
    model: str,
) -> str:
    return os.path.join(
        model_result_dir(results_dir, dataset, train_pct, model),
        f"{model}_dataset.pkl",
    )


def metric_stats_path(
    results_dir: str,
    dataset: str,
    train_pct: float,
    model: str,
) -> str:
    return os.path.join(
        model_result_dir(results_dir, dataset, train_pct, model),
        f"{model}_metric.pkl",
    )
