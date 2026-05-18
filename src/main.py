import argparse
import json
import multiprocessing as mp

from src.preprocess.download_dataset import main_download
from src.preprocess.process_data import main_preprocess
from src.metrics.export_results import main_export
from src.eval import main_eval
from src.utils import load_config
from src.registry import (
    COARSE_DATASET_CHOICES,
    EXACT_DATASET_CHOICES,
    normalize_datasets,
    get_download_datasets,
)


DATASET_CHOICES = ["all"] + COARSE_DATASET_CHOICES + EXACT_DATASET_CHOICES


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run ECG benchmark pipeline.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["all"],
        choices=DATASET_CHOICES,
        help=(
            "Datasets to run.\n\n"
            "Coarse options:\n"
            "  all\n"
            "  PTBXL       Runs PTBXL_form, PTBXL_super, PTBXL_sub, PTBXL_rhythm\n"
            "  PTBXL_C     Runs PTBXL_C_form, PTBXL_C_sub, PTBXL_C_rhythm\n"
            "  CPSC\n"
            "  CSN\n"
            "  ECHO_NEXT\n\n"
            "Exact options:\n"
            "  PTBXL_form\n"
            "  PTBXL_super\n"
            "  PTBXL_sub\n"
            "  PTBXL_rhythm\n"
            "  PTBXL_C_form\n"
            "  PTBXL_C_sub\n"
            "  PTBXL_C_rhythm\n"
            "  CPSC\n"
            "  CSN\n"
            "  ECHO_NEXT\n\n"
            "Examples:\n"
            "  --datasets all\n"
            "  --datasets PTBXL\n"
            "  --datasets PTBXL_super\n"
            "  --datasets PTBXL_super CPSC\n\n"
            "Default: all"
        ),
    )

    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        help=(
            "Models to evaluate.\n\n"
            "Options:\n"
            "  all            Run all available models.\n"
            "  D_BETA         Run D-BETA model.\n"
            "  MERL           Run MERL model.\n"
            "  CLOCS          Run CLOCS model.\n"
            "  HeartLang      Run HeartLang model.\n"
            "  KED            Run KED model.\n"
            "  Random         Run all random-initialized baselines.\n"
            "  Random_Resnet  Run all random ResNet18 baselines.\n"
            "  Random_Vit     Run all random ViT baselines.\n\n"
            "Random model name format:\n"
            "  Random_{Hz}_{Z_score}_{Bandpass}_{Backbone}\n\n"
            "Random parameters:\n"
            "  Hz:        500Hz, 250Hz, 100Hz\n"
            "  Z_score:   Z_score_sample, Z_score_none, Z_score_dataset\n"
            "  Bandpass:  No_Bandpass, Add_Bandpass\n"
            "  Backbone:  Resnet18, Vit\n\n"
            "Examples:\n"
            "  --models all\n"
            "  --models D_BETA MERL HeartLang\n"
            "  --models Random\n"
            "  --models Random_Resnet\n"
            "  --models Random_500Hz_Z_score_sample_No_Bandpass_Resnet18\n\n"
            "Default: all"
        ),
    )

    parser.add_argument(
        "--physionet-user",
        default=None,
        help="PhysioNet username for restricted datasets such as ECHO_NEXT.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config()

    # Exact datasets used by preprocess / embedding / eval.
    selected_datasets = normalize_datasets(args.datasets)

    # Physical datasets needed for download.
    download_datasets = get_download_datasets(selected_datasets)

    args.datasets = selected_datasets

    print("Using config")
    print(json.dumps(config, indent=2))

    print(f"Selected run datasets: {', '.join(args.datasets)}")
    print(f"Datasets needed for download: {', '.join(download_datasets)}")
    print(f"Selected models: {', '.join(args.models)}")

    main_download(
        config,
        datasets=download_datasets,
        physionet_user=args.physionet_user,
    )

    main_preprocess(args, config)

    main_eval(args, config)

    main_export(config)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()