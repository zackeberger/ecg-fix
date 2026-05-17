import os
import shutil
import subprocess
from typing import Iterable, Optional
from pathlib import Path

DATASET_DOWNLOADS = {
    "PTBXL": {
        "url": "https://physionet.org/files/ptb-xl/1.0.3/",
        "restricted": False,
    },
    "CPSC": {
        "url": "https://physionet.org/files/challenge-2020/1.0.2/training/cpsc_2018/",
        "restricted": False,
    },
    "CSN": {
        "url": "https://physionet.org/files/ecg-arrhythmia/1.0.0/",
        "restricted": False,
    },
    "ECHO_NEXT": {
        "url": "https://physionet.org/files/echonext/1.1.0/",
        "restricted": True,
    },
}


def _check_wget() -> None:
    if shutil.which("wget") is None:
        raise RuntimeError("wget is required. Install wget, then rerun.")

def _is_downloaded(dataset_root: str) -> bool:
    """
    Return True if dataset directory already exists and is non-empty.
    """
    path = Path(dataset_root)

    if not path.exists():
        return False

    if not path.is_dir():
        return False

    return any(path.iterdir())

def _run_wget(
    url: str,
    physionet_dir: str,
    physionet_user: Optional[str] = None,
) -> None:
    _check_wget()
    os.makedirs(physionet_dir, exist_ok=True)

    cmd = [
        "wget",
        "-r",
        "-N",
        "-c",
        "-np",
        "-nH",
        "--cut-dirs=1",
        "-P",
        physionet_dir,
    ]

    if physionet_user:
        cmd.extend(["--user", physionet_user, "--ask-password"])

    cmd.append(url)

    print(f"Downloading {url}")
    subprocess.run(cmd, check=True)


def main_download(
    config: dict,
    datasets: Optional[Iterable[str]] = None,
    physionet_user: Optional[str] = None,
) -> None:
    """
    Download selected PhysioNet datasets.

    Args:
        config: Loaded config dictionary.
        datasets: Dataset names to download. If None, downloads all datasets in config.
        physionet_user: PhysioNet username for restricted datasets.
    """

    physionet_dir = config.get("physionet_dir", "./physionet.org/files")
    dataset_roots = config.get("dataset_roots", {})

    if datasets is None:
        selected_datasets = list(dataset_roots.keys())
    else:
        selected_datasets = list(datasets)


    for dataset in selected_datasets:
        if dataset == "PTBXL_C":
            dataset = "PTBXL"

        if dataset not in DATASET_DOWNLOADS:
            raise ValueError(
                f"Unknown dataset: {dataset}. "
                f"Valid options are: {', '.join(DATASET_DOWNLOADS.keys())}"
            )

        if dataset not in dataset_roots:
            raise ValueError(
                f"{dataset} is not found in config['dataset_roots']."
            )

        dataset_root = dataset_roots[dataset]

        if _is_downloaded(dataset_root):
            print(f"Skipping {dataset}; already downloaded at: {dataset_root}")
            continue

        info = DATASET_DOWNLOADS[dataset]
        url = info["url"]
        restricted = info["restricted"]

        if restricted:

            if not physionet_user:
                raise RuntimeError(
                    f"{dataset} is restricted and requires a PhysioNet login. "
                    "Accept the dataset DUA, then rerun with "
                    "--physionet-user USER "
                )

        _run_wget(
            url=url,
            physionet_dir=physionet_dir,
            physionet_user=physionet_user if restricted else None,
        )

        print(f"Done downloading {dataset} under: {physionet_dir}")

    print("Done. All requested data downloaded.")