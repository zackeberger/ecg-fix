import json
import multiprocessing as mp
from benchmark.preprocess.process_all import main as main_preprocess


def main():
    with open("configs/config.json", "r") as f:
        config = json.load(f)

    print(config)
    main_preprocess(config)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()