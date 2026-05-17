# [ICML 2026] Position: Evaluation of ECG Representations Must Be Fixed

[Paper](https://arxiv.org/pdf/2602.17531)

This repository contains the ECG representation benchmark used to download and
preprocess public ECG datasets, generate frozen model embeddings, train linear
probes, and export metrics/statistical comparisons.

## Sources

Datasets used by the public benchmark:

- PTB-XL v1.0.3: https://physionet.org/content/ptb-xl/1.0.3/
- CPSC2018 subset from PhysioNet/CinC Challenge 2020 v1.0.2: https://physionet.org/content/challenge-2020/1.0.2/training/cpsc_2018/
- CSN / Chapman-Shaoxing-Ningbo ECG arrhythmia database v1.0.0: https://physionet.org/content/ecg-arrhythmia/1.0.0/
- EchoNext v1.1.0: https://physionet.org/content/echonext/1.1.0/

Models and upstream code references:

- D-BETA: https://github.com/manhph2211/D-BETA
- MERL: https://github.com/cheliu-computation/MERL-ICML2024
- CLOCS: https://github.com/danikiyasseh/CLOCS
- KED / ECGFM-KED: https://github.com/control-spiderman/ECGFM-KED
- HeartLang: https://github.com/PKUDigitalHealth/HeartLang

## Setup

Create and activate an environment from the repository root:

```bash
conda create -n ecg-fix-env python=3.11 -y
. /home/stultzlab/miniconda3/bin/activate
conda activate ecg-fix-env
pip install -r requirements.txt
```

The pipeline reads `config.json` from the repository root. Update paths there
before running experiments:

- `physionet_dir`: base directory for downloaded PhysioNet datasets.
- `dataset_roots`: expected location for each downloaded dataset.
- `raw_data_dir`: intermediate raw/preprocessed arrays.
- `processed_dir`: formatted metadata and split files used by loaders.
- `embeddings_dir`: generated frozen embeddings.
- `results_dir`: prediction pickles, metric pickles, and done markers.
- `tables_dir`: CSV tables and statistical comparison exports.
- `model_weights_dir`: pretrained checkpoint directory.
- `embedding.chunk_size`: how many models to embed in one loaded model group.
- `multi_process_eval`: number of parallel linear-probe evaluation workers.

## Download Data

The main pipeline downloads missing datasets automatically. For unrestricted
datasets only, run:

```bash
python -m src.main --datasets PTBXL CPSC CSN --models D_BETA
```

EchoNext is restricted. First accept the EchoNext data-use agreement on
PhysioNet, then include your PhysioNet username:

```bash
python -m src.main --datasets ECHO_NEXT --models D_BETA --physionet-user YOUR_USERNAME
```

The downloader uses `wget` and skips dataset directories that already exist and
are non-empty under the paths in `config.json`.

You can also download datasets with the bash scripts in `scripts/` if you want
to prepare data before running the Python pipeline:

```bash
# PTB-XL
bash scripts/download_ptb_xl.sh

# CPSC2018 from PhysioNet Challenge 2020
bash scripts/download_challenge_2020_cpsc2018.sh

# CSN / ECG Arrhythmia
bash scripts/download_ecg_arrhythmia.sh

# EchoNext, after accepting the PhysioNet DUA
bash scripts/download_echonext.sh --physionet-user YOUR_USERNAME
```

Each script downloads one PhysioNet dataset with `wget` into
`./physionet.org/files` by default. Pass `--dir PATH` to use a different
PhysioNet files directory:

```bash
bash scripts/download_ptb_xl.sh --dir /path/to/physionet.org/files
```

The resulting directory layout must match `dataset_roots` in `config.json`, for
example `physionet.org/files/ptb-xl/1.0.3` for PTB-XL. The EchoNext script is
restricted-access and requires `--physionet-user`; it also supports
`--skip-restricted` when you want to skip EchoNext. These scripts use the shared
`scripts/_physionet_download_common.sh` helper, so keep that file with the
download scripts.

## Download Model Weights

Install and log in to the Hugging Face CLI:

```bash
pip install -U huggingface_hub
hf auth login
```

Download the released weights into the configured `model_weights_dir`:

```bash
hf download doprakah/ecg-fix-weights --local-dir model_weights
```

Expected checkpoint filenames:

- `dbeta_config.json`
- `dbeta_best.pt`
- `res18_best_encoder.pth`
- `best_weights_clocs`
- `ked.pt`
- `heart.pth`

## Run Experiments

Run from the repository root. The full entry point is:

```bash
python -m src.main --datasets all --models all
```

This runs the complete workflow:

1. Download missing datasets.
2. Preprocess selected datasets.
3. Format train/validation/test splits.
4. Generate frozen embeddings for selected models.
5. Train and evaluate linear probes at train percentages `0.01`, `0.1`, and `1.0`.
6. Export CSV result tables and statistical tests.

Common smaller runs:

```bash
# One PTB-XL task with one model
python -m src.main --datasets PTBXL_super --models D_BETA

# All PTB-XL tasks with core pretrained models
python -m src.main --datasets PTBXL --models D_BETA MERL CLOCS HeartLang KED

# CPSC and CSN with the random baseline grid
python -m src.main --datasets CPSC CSN --models Random

# PTB-XL cleaned labels only
python -m src.main --datasets PTBXL_C --models all
```

To choose a specific GPU, set `CUDA_VISIBLE_DEVICES` before the command. For
example, use GPU 0:

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.main --datasets PTBXL_super --models D_BETA
```

Use a different GPU by changing the value, for example
`CUDA_VISIBLE_DEVICES=1`. To expose multiple GPUs to the process, pass a
comma-separated list such as `CUDA_VISIBLE_DEVICES=0,1`.

Dataset options:

- `all`
- `PTBXL`: expands to `PTBXL_form`, `PTBXL_super`, `PTBXL_sub`, `PTBXL_rhythm`
- `PTBXL_C`: expands to `PTBXL_C_form`, `PTBXL_C_sub`, `PTBXL_C_rhythm`
- `CPSC`
- `CSN`
- `ECHO_NEXT`
- exact tasks: `PTBXL_form`, `PTBXL_super`, `PTBXL_sub`, `PTBXL_rhythm`,
  `PTBXL_C_form`, `PTBXL_C_sub`, `PTBXL_C_rhythm`

Model options:

- `all`
- `D_BETA`
- `MERL`
- `CLOCS`
- `HeartLang`
- `KED`
- `Random`: all random-initialized baselines
- `Random_Resnet` or `Random_Resnet18`
- `Random_Vit`
- exact random model names such as
  `Random_500Hz_Z_score_sample_No_Bandpass_Resnet18`

## Outputs

Embedding files are written under:

```text
data/embeddings/<dataset>/<split>/<model>.npy
data/embeddings/<dataset>/<split>/<model>_y.npy
```

Evaluation files are written under:

```text
results/<dataset>/<train_pct>/<model>/<model>.pkl
results/<dataset>/<train_pct>/<model>/<model>_dataset.pkl
results/<dataset>/<train_pct>/<model>/<model>_metric.pkl
results/done/
```

CSV exports are written under `tables_dir`, which is `./metrics` in the default
`config.json`:

```text
metrics/
  tables/
    summary/
      macro_auc_auprc_table.csv
    tasks/
      <dataset>/
    ptbxl_clean_comparison/
      ptbxl_macroauc_change_*.csv
    random_grid/
      random_grid_*.csv
  stats_tests/
    <dataset>/
      trainpct_<pct>/
        <target>/
          <metric>/
            p_values.csv
            bootstrap_diff.csv
            within_noise.csv
            sig_alpha_<alpha>.csv
```

The export step creates the macro AUC/AUPRC tables, task-specific tables,
PTB-XL original-vs-clean comparisons, bootstrap/permutation statistics, and
random-grid summary tables.

Metric confidence intervals and pairwise bootstrap-difference tables use
stratified bootstrap resampling by label: positives and negatives are sampled
with replacement separately for each valid label. Macro bootstrap values average
over labels that have both positive and negative examples.

Metrics export code shares model ordering, display names, and result path
helpers through `src/metrics/utils.py`. Update that file if you add or rename
models that should appear in both result tables and statistical tests.

To rerun a specific evaluation result, delete the corresponding model result
folder, for example:

```bash
rm -r results/PTBXL_super/1.0/D_BETA
```

The done marker in `results/done/` is validated against the expected output
files, so a missing model folder will cause that evaluation to run again. To
regenerate exported CSVs, delete the corresponding folder under `metrics/`, for
example `metrics/tables/ptbxl_clean_comparison/` or
`metrics/stats_tests/PTBXL_super/trainpct_1p0/`, then rerun the pipeline.

## Notes

- The pipeline is resumable at the embedding and evaluation levels. Existing
  model outputs with done markers are skipped.
