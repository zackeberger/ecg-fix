import random
import hashlib
import numpy as np
import torch

import json


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def load_config():
    with open("./config.json", "r") as f:
        config = json.load(f)

    return config


def stable_config_hash(config: dict) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def embedding_config_hash(config: dict) -> str:
    payload = {
        "seed": config.get("seed", 42),
    }
    return stable_config_hash(payload)


def eval_config_hash(config: dict) -> str:
    payload = {
        "seed": config.get("seed", 42),
        "stats_tests": {
            "n_boot": config.get("stats_tests", {}).get("n_boot", 1000),
        },
    }
    return stable_config_hash(payload)
