import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from src.preprocess.CPSC import p_CPSC
from src.preprocess.CSN import p_CSN
from src.preprocess.ECHO_NEXT import p_ECHO_NEXT
from src.preprocess.PTBXL import p_PTBXL
from src.preprocess.PTBXL_C import p_PTBXL_C
from src.preprocess.format_dataset import format_main
from src.preprocess.embed_data import embed_main
import multiprocessing as mp
from src.registry import get_preprocess_datasets



def main_preprocess(args, config):
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    preprocess_datasets = get_preprocess_datasets(args.datasets)

    print(f"Preprocessing required datasets: {', '.join(preprocess_datasets)}")

    if "PTBXL" in preprocess_datasets:
        p_PTBXL.main(config)

    if "PTBXL_C" in preprocess_datasets:
        p_PTBXL_C.main(config)

    if "CPSC" in preprocess_datasets:
        p_CPSC.main(config)

    if "ECHO_NEXT" in preprocess_datasets:
        p_ECHO_NEXT.main(config)

    if "CSN" in preprocess_datasets:
        p_CSN.main(config)

    format_main(args, config)
    embed_main(args, config)

    print("All data processed")
