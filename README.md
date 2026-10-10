# Tabular benchmark extension

A reproducible experiment comparing tree models and a neural network on three processed datasets from **Grinsztajn, Oyallon and Varoquaux (2022)**. The experiment changes training size, feature selection and model settings to investigate when a simple baseline can improve.

Project owner: **Sanjida Alam**, University of Canberra.

This repository contains the experiment code and its evidence. The written critique and presentation are delivered separately and are excluded from Git.

## Relationship to the paper

Paper: *Why do tree-based models still outperform deep learning on typical tabular data?* NeurIPS 2022, Datasets and Benchmarks.

- [Published paper](https://proceedings.neurips.cc/paper_files/paper/2022/file/0378c7692da36807bdec87ab043cdadc-Paper-Datasets_and_Benchmarks.pdf)
- [Authors' repository](https://github.com/LeoGrin/tabular-benchmark)
- [Authors' processed datasets](https://huggingface.co/datasets/inria-soda/tabular-benchmark)

This is an **independent, small extension of the experimental protocol**, not a complete reproduction of the authors' pipeline or a claim to beat their published results. It uses the authors' processed numerical classification data. Their original Python 3.8, PyTorch and Weights & Biases pipeline is replaced here by a local CPU runner with CSV logging. No original source code has been copied. The authors retain credit for the paper, benchmark and data preparation.

## Research questions

1. Does using more training data improve each model on these datasets?
2. Does retaining the top half of features help an MLP relative to trees?
3. Can validation-selected model settings improve on a full-feature Random Forest baseline?

The configuration was fixed before the full experiment. Every comparison includes unsuccessful changes as well as improvements.

## Experiment design

| Component | Setting |
|---|---|
| Datasets | credit, electricity, MagicTelescope |
| Models | Random Forest, XGBoost, HistGradientBoosting, sklearn MLP |
| Requested training sizes | 1,000, 3,000, up to 10,000 rows |
| Feature conditions | All columns, top half ranked by a training-only Random Forest |
| Random seeds | 17, 42, 91 |
| Split | Approximately 70% train, 9% validation, 21% test, stratified |
| Search budget | Four predefined configurations per model and condition |
| Primary metric | Accuracy |
| Additional metrics | Balanced accuracy, F1, ROC AUC |
| Hardware policy | CPU only, two numerical threads |
| Full run | 216 conditions, 864 model fits, 432 test records |

MagicTelescope has fewer than 10,000 available training rows. Its largest balanced subset contains 9,362 rows. Electricity has seven original features, so the half-feature condition keeps four. Credit and MagicTelescope keep five of ten.

Training subsets are balanced and nested within each seed. Validation and test sets stay fixed when training size changes. Extra training-pool rows beyond the size cap are unused. Feature importance is recalculated on the current training subset, without validation or test labels.

The MLP uses a training-fitted Gaussian quantile transform and an internal 10% early-stopping split. That internal split differs from the held-out tuning validation set. It runs for at most 200 epochs with patience 15. This is not the paper's PyTorch MLP, ResNet, FT-Transformer or SAINT implementation. HistGradientBoosting disables internal early stopping for all configurations. The remaining three tree baselines use their software defaults except common CPU/reproducibility controls and XGBoost's histogram tree method. These details matter when interpreting the word **baseline**.

## Run locally

Python 3.13.9 was used for the saved run. Create a separate environment rather than changing your other projects.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m unittest test_protocol.py
```

Start with a short run:

```powershell
python run_experiment.py --smoke --output-dir smoke-results
```

Run the complete experiment and analysis:

```powershell
python run_experiment.py --output-dir new-results
python analyse_results.py --results-dir new-results
```

The runner downloads public CSVs from a pinned dataset revision and checks their SHA-256 hashes. It needs an internet connection on the first run. No OpenML key, GPU, paid account or Weights & Biases account is required. Use a fresh output directory for each run so existing evidence is never overwritten or mixed with another experiment.

## Saved results

The `results/` directory contains the completed run. The outcome table is in `results/main_comparison.csv`, and all individual records remain available for inspection.

The main comparison uses the largest training subset. It compares the full-feature baseline Random Forest with the best **validation** configuration across the four tuned model families and two feature conditions, separately for each dataset and seed. Test accuracy does not determine that choice. This modified policy searches more total candidates than a single untuned baseline, so improvement measures the benefit of the whole selection procedure, not a fair-budget causal effect of one model. Equal four-trial budgets apply to the per-model comparisons.

`results/feature_effect.csv` compares tuned full-feature and half-feature models within the same family and seed. `results/summary.csv` retains every dataset, training size, feature condition and baseline/tuned setting. Figures show mean accuracy and, where plotted, one standard deviation across the three seeds. Those seeds use overlapping data and do not give independent confidence intervals.

## File guide

| File | Purpose |
|---|---|
| `experiment.json` | Fixed design, data revision and checksums |
| `run_experiment.py` | Download, split, feature selection, training and evaluation |
| `analyse_results.py` | Validation-selected comparisons, summary tables and plots |
| `test_protocol.py` | Checks data isolation, balanced nested samples and repeatable selection |
| `requirements.txt` | Versions used in the recorded experiment |
| `results/trials.csv` | Every candidate's validation score, settings, runtime and warning status |
| `results/results.csv` | Test scores for each condition's baseline and validation-selected model |
| `results/validation_selected_comparisons.csv` | Main comparison for each seed, with selected settings |
| `results/environment.json` | Software versions, CPU policy and run timestamps |
| `results/datasets.json` | Dataset sources, hashes, target mappings and dimensions |
| `results/splits/` | Row indices for the recorded splits |
| `results/figures/` | Plots generated from the recorded numbers |

Prediction arrays are generated locally in `results/predictions/` and excluded from Git. Downloaded datasets are excluded too. Results and settings can be reproduced from the retained split indices and hashes.

## Limitations

- Three datasets and one small MLP cannot represent the whole 45-dataset benchmark or all deep learning architectures.
- Four predefined configurations are a much smaller search than the paper's approximately 400 random-search iterations. The search spaces are practical choices rather than samples from the authors' exact spaces.
- The processed targets are balanced and some original classes, rows or features have been removed. Scores do not represent the original data distribution or deployment performance.
- Shuffled splits of electricity observations assess the benchmark's classification task. They do not assess forecasting on future dates and may benefit from nearby observations crossing partitions.
- Random Forest importance can favour certain variables and reflect correlated features. Dropping half the features can remove useful information.
- Every predeclared ablation is reported. These test scores are diagnostic evidence, not a new tuning signal for further adaptive searches.
- Timing describes this CPU run. It cannot establish the paper's GPU-versus-CPU speed claims.
- Small differences across three overlapping splits should not be treated as statistically significant.

## Citation

Grinsztajn, L., Oyallon, E., & Varoquaux, G. (2022). Why do tree-based models still outperform deep learning on typical tabular data? *Advances in Neural Information Processing Systems, 35*, 507-520.

Dataset revision: `8d0ff9103525b7e3579b180230fddb3186258301`.

## Repository description

Suggested GitHub description: **Reproducible extension of the NeurIPS 2022 tabular benchmark, studying training size, feature selection and model tuning on three datasets.**
