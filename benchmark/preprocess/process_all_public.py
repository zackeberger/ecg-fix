import argparse

from benchmark.config import add_config_arg, load_config
from benchmark.preprocess.public.CPSC import p_CPSC
from benchmark.preprocess.public.CSN import p_CSN
from benchmark.preprocess.public.ECHO_NEXT import p_ECHO_NEXT
from benchmark.preprocess.public.PTBXL import p_PTBXL
from benchmark.preprocess.public.PTBXL_C import p_PTBXL_C


def main():
    parser = argparse.ArgumentParser(description="Preprocess all public ECG benchmark datasets.")
    add_config_arg(parser)
    args = parser.parse_args()
    config = load_config(args.config)

    p_PTBXL.main(config)
    p_PTBXL_C.main(config)
    p_CPSC.main(config)
    p_ECHO_NEXT.main(config)
    p_CSN.main(config)


if __name__ == "__main__":
    main()
