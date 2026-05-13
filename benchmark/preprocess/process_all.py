import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from benchmark.preprocess.CPSC import p_CPSC
from benchmark.preprocess.CSN import p_CSN
#from benchmark.preprocess.ECHO_NEXT import p_ECHO_NEXT
from benchmark.preprocess.PTBXL import p_PTBXL
from benchmark.preprocess.PTBXL_C import p_PTBXL_C
from benchmark.preprocess.format_dataset import format_main
#from benchmark.preprocess.embed_data import embed_main
import multiprocessing as mp


def main(config):
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    p_PTBXL.main(config)
    p_PTBXL_C.main(config)
    p_CPSC.main(config)
   # p_ECHO_NEXT.main(config)
    p_CSN.main(config)
    format_main(config)
   # embed_main(config)
    print("All data processed")

