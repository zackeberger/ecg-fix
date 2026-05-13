import json
import multiprocessing as mp
from benchmark.process_data import main_preprocess
from benchmark.eval import main_eval



def main():
    with open("configs/config.json", "r") as f:
        config = json.load(f)

    print(config)
    main_preprocess(config)
    main_eval(config)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()