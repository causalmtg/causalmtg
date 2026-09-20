# A Causal Benchmark from Sequential Magic: The Gathering Data

This repository contains the code, metadata, benchmark data, and evaluation results for **A Causal Benchmark from Sequential Magic: The Gathering Data**.

The benchmark is constructed from sequential Magic: The Gathering draft data and is designed for evaluating causal effect estimation methods in a realistic sequential decision-making setting.

## Repository Contents

* **Code** — Scripts and notebooks for benchmark construction, data processing, and evaluation.
* **Data** — Metadata and benchmark reference effects used by the benchmark.
* **Results** — Additional results used to evaluate the benchmark and causal effect estimation methods.
* **Notebooks** — Examples illustrating benchmark usage and reproducing the reported analyses.

### Metadata

The `data/` directory contains metadata used in benchmark construction, including set and card metadata obtained from Scryfall and metadata extracted from the underlying draft data.

### Benchmark Reference Effects

The file `results/reference.effects.csv` contains the benchmark treatment–outcome-group pairs together with their reference causal effects. These reference effects constitute the benchmark targets used for evaluating causal effect estimators.

### Additional Results

The `results/` directory contains additional results used in the evaluation of the benchmark and its estimators.

## Dataset

The benchmark was constructed from draft data originally published by 17Lands. The large draft-data files hosted on Hugging Face are a **frozen copy of the source data used for benchmark construction**, provided to support reproducibility.

The benchmark data and the large underlying draft-data files are available on Hugging Face:

https://huggingface.co/datasets/causalmtg/causalmtg/

The original 17Lands draft data is available at:

https://17lands-public.s3.amazonaws.com/analysis_data/draft_data/

The frozen copy may differ from the current version of the data available from 17Lands.

## Usage

The benchmark pipeline is organized primarily by **set code**.

### Configure the data path

Before running the pipeline, edit `config/cfg.yaml` and set `data_dir` to the local directory where the MTG data and metadata should be stored.

### Download the data

The required draft data can be downloaded with:

```bash
python src/dispatch.py download
```

To download a specific set only:

```bash
python src/dispatch.py download --set-code MKM
```

The downloaded data are placed under the `data_dir` configured in `config/cfg.yaml`.

### Reference effects

The benchmark reference effects are provided in:

```text
results/reference.effects.csv
```

They do not need to be regenerated to reproduce the reported evaluation results.

To reconstruct the reference effects from the underlying data for a specific set:

```bash
python src/dispatch.py build-reference-effects MKM
```

### Extract baseline features

Before running the baseline estimators, extract and cache the features required by the estimators:

```bash
python src/dispatch.py extract-baseline-features MKM
```

Repeat for each set code being evaluated. The extracted features are cached and reused by the baseline estimation step. Running this as a separate preprocessing step is recommended, particularly when distributing the baseline estimation across multiple `--idx` runs.

### Obtain baseline estimates

Baseline causal effect estimates are generated separately for each set code. The estimators to evaluate are specified in a text file, with one estimator name per line.

For example:

```bash
python src/dispatch.py build-baseline-estimates MKM config/methods.txt
```

The estimator configurations and hyperparameter settings used for the benchmark are provided with the repository; no separate hyperparameter-tuning step is required.

The evaluation can be distributed by `card_a`. Passing `--idx IDX` evaluates the baseline estimators for the single `card_a` corresponding to that index:

```bash
python src/dispatch.py build-baseline-estimates MKM config/methods.txt --idx 0
```

Run the command for each required index and combine the resulting outputs to obtain the complete evaluation for the set.

Available estimators can be listed with:

```bash
python src/dispatch.py list-available-estimators
```

### Run diagnostics

Two diagnostic procedures are provided for each set code:

```bash
python src/dispatch.py independence-diagnostic MKM
python src/dispatch.py instrument-diagnostic MKM
```

### Aggregate results across sets

After processing the individual set codes, results can be summarized across multiple sets:

```bash
python src/dispatch.py create-summary <topic> <set-codes> <methods-file>
```

For example:

```bash
python src/dispatch.py create-summary evaluation MKM,DSK,DFT,BLB config/methods.txt
```

Set codes are supplied as a comma-separated list.

### Reproducibility workflow

```text
Configure data path
       │
       ▼
Download data
       │
       ▼
Extract and cache baseline features
       │
       ▼
Run baseline estimators
       │
       ├── one run per card_a (--idx)
       ▼
Combine per-card_a results
       │
       ├── per set code
       ▼
Aggregate results across sets
```

The reference effects required for evaluation are already provided in `results/reference.effects.csv`.

## License

This project is distributed under the following terms:

* **Code:** The software and scripts in this repository are licensed under the [MIT License](LICENSE).
* **Drafts Datasets:** The frozen benchmark datasets derived from 17Lands are licensed under the [Creative Commons Attribution 4.0 International (CC BY 4.0)](LICENSE-DATA).
* **Card Metadata (Local `data/` folder):** The set and card metadata contained within this repository is the intellectual property of Wizards of the Coast. It is distributed strictly under the Wizards of the Coast Fan Content Policy and is explicitly excluded from the CC BY 4.0 license.

## Disclaimers & Attribution

* **Wizards of the Coast:** This project is unofficial Fan Content permitted under the Fan Content Policy. It is not approved or endorsed by Wizards of the Coast. Portions of the materials used are property of Wizards of the Coast LLC. © Wizards of the Coast LLC.
* **17Lands:** The base draft data used to construct this benchmark was retrieved from [17Lands](https://www.17lands.com). 17Lands is not affiliated with and does not endorse this project or its findings.
* **Scryfall:** Scryfall was used as a source for set and card metadata included in the benchmark. Scryfall is not affiliated with and does not endorse this project or its findings.