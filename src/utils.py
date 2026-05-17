import os
import pickle
import random
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