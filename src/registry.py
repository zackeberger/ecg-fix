# src/registry.py

CORE_MODELS = ["D_BETA", "MERL", "CLOCS", "HeartLang", "KED"]
RANDOM_MODEL = "Random_500Hz_Z_score_none_No_Bandpass_Resnet18"

MODEL_ORDER = [
    RANDOM_MODEL,
    "CLOCS",
    "KED",
    "HeartLang",
    "MERL",
    "D_BETA",
]

COARSE_DATASET_CHOICES = [
    "PTBXL",
    "PTBXL_C",
    "CPSC",
    "CSN",
    "ECHO_NEXT",
]

EXACT_DATASET_CHOICES = [
    "PTBXL_form",
    "PTBXL_super",
    "PTBXL_sub",
    "PTBXL_rhythm",
    "PTBXL_C_form",
    "PTBXL_C_sub",
    "PTBXL_C_rhythm",
    "CPSC",
    "CSN",
    "ECHO_NEXT",
]

DATASET_GROUPS = {
    "PTBXL": [
        "PTBXL_form",
        "PTBXL_super",
        "PTBXL_sub",
        "PTBXL_rhythm",
    ],
    "PTBXL_C": [
        "PTBXL_C_form",
        "PTBXL_C_sub",
        "PTBXL_C_rhythm",
    ],
    "CPSC": ["CPSC"],
    "CSN": ["CSN"],
    "ECHO_NEXT": ["ECHO_NEXT"],
}



def build_random_model_names() -> tuple[list[str], list[str]]:
    random_resnet = []
    random_vit = []

    for hz in ["500Hz", "250Hz", "100Hz"]:
        for z_score in ["Z_score_sample", "Z_score_none", "Z_score_dataset"]:
            for band in ["No_Bandpass", "Add_Bandpass"]:
                for backbone in ["Resnet18", "Vit"]:
                    name = f"Random_{hz}_{z_score}_{band}_{backbone}"

                    if backbone == "Resnet18":
                        random_resnet.append(name)
                    else:
                        random_vit.append(name)

    return random_resnet, random_vit


def build_all_models() -> list[str]:
    random_resnet, random_vit = build_random_model_names()
    return CORE_MODELS + random_resnet + random_vit


def normalize_models(model_args) -> list[str]:
    random_resnet, random_vit = build_random_model_names()
    random_models = random_resnet + random_vit
    all_models = build_all_models()

    if model_args is None or "all" in model_args:
        return all_models

    selected = []

    for model in model_args:
        if model == "Random":
            selected.extend(random_models)
        elif model in {"Random_Resnet", "Random_Resnet18"}:
            selected.extend(random_resnet)
        elif model == "Random_Vit":
            selected.extend(random_vit)
        elif model in all_models:
            selected.append(model)
        else:
            raise ValueError(
                f"Unknown model: {model}\n\n"
                f"Valid core models: {', '.join(CORE_MODELS)}\n"
                "Valid random shortcuts: Random, Random_Resnet, Random_Resnet18, Random_Vit\n"
                "Or use an exact random model name like:\n"
                "Random_500Hz_Z_score_sample_No_Bandpass_Resnet18"
            )

    return _dedupe(selected)


def normalize_datasets(dataset_args) -> list[str]:
    if dataset_args is None or "all" in dataset_args:
        return EXACT_DATASET_CHOICES

    selected = []

    for dataset in dataset_args:
        if dataset in DATASET_GROUPS:
            selected.extend(DATASET_GROUPS[dataset])
        elif dataset in EXACT_DATASET_CHOICES:
            selected.append(dataset)
        else:
            raise ValueError(
                f"Unknown dataset: {dataset}\n\n"
                f"Valid coarse datasets: {', '.join(COARSE_DATASET_CHOICES)}\n"
                f"Valid exact datasets: {', '.join(EXACT_DATASET_CHOICES)}"
            )

    return _dedupe(selected)


def get_download_datasets(selected_datasets) -> list[str]:
    download_datasets = []

    for dataset in selected_datasets:
        if dataset.startswith("PTBXL"):
            download_datasets.append("PTBXL")
        elif dataset in {"CPSC", "CSN", "ECHO_NEXT"}:
            download_datasets.append(dataset)

    return _dedupe(download_datasets)


def _dedupe(items) -> list[str]:
    deduped = []
    seen = set()

    for item in items:
        if item not in seen:
            deduped.append(item)
            seen.add(item)

    return deduped

def get_preprocess_datasets(selected_datasets) -> list[str]:
    """
    Map exact run datasets back to the preprocessing modules they require.

    Examples:
        PTBXL_super -> PTBXL
        PTBXL_form -> PTBXL
        PTBXL_C_sub -> PTBXL_C
        CPSC -> CPSC
    """
    preprocess_datasets = []

    for dataset in selected_datasets:
        if dataset.startswith("PTBXL_C"):
            preprocess_datasets.append("PTBXL_C")

        elif dataset.startswith("PTBXL"):
            preprocess_datasets.append("PTBXL")

        elif dataset in {"CPSC", "CSN", "ECHO_NEXT"}:
            preprocess_datasets.append(dataset)

        else:
            raise ValueError(f"Unknown dataset for preprocessing: {dataset}")

    return _dedupe(preprocess_datasets)

def label_type_from_dataset(dataset_name: str) -> str:
    label_map = {
        "PTBXL_form": "form",
        "PTBXL_rhythm": "rhythm",
        "PTBXL_super": "super",
        "PTBXL_sub": "sub",
        "PTBXL_C_form": "form",
        "PTBXL_C_rhythm": "rhythm",
        "PTBXL_C_sub": "sub",
    }

    return label_map.get(dataset_name, "")
