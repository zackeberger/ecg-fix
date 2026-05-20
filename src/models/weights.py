from pathlib import Path


def _resolve_weight_file(base_dir, preferred_name, legacy_names=()):
    base = Path(base_dir)
    candidates = [base / preferred_name, *(base / name for name in legacy_names)]
    for path in candidates:
        if path.exists():
            return str(path)
    return str(candidates[0])


def get_weights_path(model_name, base_dir):
    if model_name == "CLOCS":
        path = _resolve_weight_file(base_dir, "best_weights_clocs")
    elif model_name == "MERL":
        path = _resolve_weight_file(base_dir, "res18_best_encoder.pth")
    elif model_name == "KED":
        path = _resolve_weight_file(base_dir, "ked.pt")
    elif model_name == "HeartLang":
        path = _resolve_weight_file(base_dir, "heart.pth")
    elif model_name == "D_BETA":
        config_path = _resolve_weight_file(base_dir, "dbeta_config.json")
        checkpoint_path = _resolve_weight_file(base_dir, "dbeta_best.pt")
        return config_path, checkpoint_path
    else:
        raise ValueError(f"Unknown model type: {model_name}")

    return path
