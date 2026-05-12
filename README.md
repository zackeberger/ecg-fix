# [ICML 2026] Position: Evaluation of ECG Representations Must Be Fixed

[Paper](https://arxiv.org/pdf/2602.17531)

This repository contains the ECG representation benchmark used for preprocessing
public ECG datasets, evaluating frozen embeddings with linear probes, and
comparing model results with bootstrap/permutation statistics.
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

Install dependencies:

```bash
pip install -r requirements.txt
```

Important config fields:

- `physionet_dir`: base directory for downloaded PhysioNet datasets.
- `dataset_roots`: optional per-dataset overrides.
- `raw_data_dir`: output from dataset-specific raw preprocessing.
- `processed_dir`: filtered metadata used by dataset loaders.
- `embeddings_dir`: output from embedding generation.
- `results_dir`: evaluation outputs, including predictions and metrics.
- `comparison_dir`: CSV outputs from pairwise model comparison.
- `model_weights_dir`: pretrained checkpoint directory.

You can pass the config on each command with `--config configs/config.json`

## Download Data

bash scripts/download_all_public_data.sh --skip-restricted
bash scripts/download_echonext.sh --physionet-user YOUR_USERNAME

## Download Models
huggingface-cli download doprakah/ecg-fix-weights \
  --local-dir model_weights2

## Data Workflow

Run the full local workflow after data and checkpoints are available:

```bash
python main.py --config configs/local.json
```

To include dataset download in the same command:

```bash
python main.py --config configs/local.json --stages download all --skip-restricted
```

Run or resume selected stages:

```bash
python main.py --config configs/local.json --stages embed eval compare
```

Preprocess public datasets:

```bash
python -m benchmark.preprocess.process_all_public --config configs/local.json
python -m benchmark.preprocess.public.format_dataset --config configs/local.json
```

Generate embeddings:

```bash
python -m benchmark.preprocess.public.embed_data --config configs/local.json --chunk 0
```

`embed_data.py` splits the model list into chunks so large embedding jobs can be
run separately. Increase `--chunk` for the remaining chunks.

## Evaluation

Run one specified evaluation:

```bash
python -m benchmark.eval \
  --config configs/local.json \
  --run_type specified \
  --data PTBXL_super \
  --model D_BETA \
  --train_pct 0.1
```

Run the full benchmark grid:

```bash
python -m benchmark.eval --config configs/local.json --run_type full
```

Evaluation writes:

- prediction pickles under `results_dir/<dataset>/<train_pct>/<model>.pkl`
- local run logs under `results_dir/runs/`
- `metrics.csv` and `events.jsonl` for each run

## Comparison

Compare one dataset/task:

```bash
python -m benchmark.compare \
  --config configs/local.json \
  --data PTBXL_super \
  --best_model D_BETA \
  --train_pct 0.1
```

Comparison CSVs are written to `comparison_dir`.

## Expected Checkpoints

Place pretrained weights in `model_weights_dir` with these filenames:

- `D_BETA_config.json`
- `D_BETA.pt`
- `MERL.pt`
- `CLOCS.pt`
- `KED.pt`
- `HeartLang.pt`

## Notes

Large generated arrays, model checkpoints, and downloaded ECG datasets should not
be committed to GitHub. Keep them in local directories pointed to by the config.
