from benchmark.preprocess.CPSC import p_CPSC
from benchmark.preprocess.CSN import p_CSN
from benchmark.preprocess.ECHO_NEXT import p_ECHO_NEXT
from benchmark.preprocess.PTBXL import p_PTBXL
from benchmark.preprocess.PTBXL_C import p_PTBXL_C
import multiprocessing as mp


def main(config):
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    p_PTBXL.main(config)
    p_PTBXL_C.main(config)
    p_CPSC.main(config)
    p_ECHO_NEXT.main(config)
    p_CSN.main(config)

