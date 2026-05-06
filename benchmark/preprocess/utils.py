import numpy as np


def compute_global_metrics(df_old):
    df = df_old[df_old["split"] == "train"]

   # print(df.columns)

    # -------- global ECG --------
    means = df["ecg_mean"].values
    stds  = df["ecg_std"].values

    global_mean = means.mean()
    global_var  = np.mean(stds**2 + (means - global_mean)**2)
    global_std  = np.sqrt(global_var)

    # -------- per-lead --------
   # lead_means = np.stack(df["lead_means"].values)  # (N, 12)
   # lead_stds  = np.stack(df["lead_stds"].values)   # (N, 12)

    # mean per lead
  #  global_lead_means = lead_means.mean(axis=0)  # (12,)

    # variance per lead
  #  global_lead_vars = np.mean(
  #      lead_stds**2 + (lead_means - global_lead_means)**2,
  #      axis=0
  #  )  # (12,)

  #  global_lead_stds = np.sqrt(global_lead_vars)

    return {
        "global_mean": global_mean,
        "global_std": global_std,
   #     "lead_means": global_lead_means,
   #     "lead_stds": global_lead_stds,
    }