import numpy as np
import json

def compute_global_metrics(df_old):
    df = df_old[df_old["split"] == "train"]

    # -------- global ECG --------
    means = df["ecg_mean"].values
    stds  = df["ecg_std"].values

    global_mean = means.mean()
    global_var  = np.mean(stds**2 + (means - global_mean)**2)
    global_std  = np.sqrt(global_var)

    return {
        "global_mean": global_mean,
        "global_std": global_std,
    }

def load_config():
    with open("configs/config.json", "r") as f:
        config = json.load(f)
    return config