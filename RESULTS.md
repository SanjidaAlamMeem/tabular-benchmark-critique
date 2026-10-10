# Recorded experiment results

The fixed experiment ran on 10 October 2026 using Python 3.13.9 on Windows 11, with two CPU threads. It completed 216 conditions and 864 candidate fits in 293.87 seconds. No MLP convergence warnings occurred. The full records are retained rather than only the successful conditions.

## Main validation-selected comparison

All values are mean test accuracy across seeds 17, 42 and 91. The baseline is the full-feature default Random Forest at the largest training size. The modified policy selects a model family, feature condition and trial using validation accuracy at that same size.

| Dataset | Training rows | Baseline accuracy | Modified accuracy | Change in percentage points |
|---|---:|---:|---:|---:|
| credit | 10,000 | 77.17% | 77.37% | +0.20 |
| electricity | 10,000 | 84.70% | 85.47% | +0.77 |
| MagicTelescope | 9,362 | 85.74% | 85.77% | +0.02 |

Every electricity split improved, by +0.77, +0.97 and +0.58 points. Credit changed by +0.97, +0.57 and -0.94 points. MagicTelescope changed by +0.04, -0.07 and +0.11 points. The credit and MagicTelescope averages are small relative to their variability. No significance claim is made.

The selected models differ by seed. For example, electricity seed 17 selected Random Forest with four features and trial 2; seed 42 selected Random Forest with seven features and trial 2; seed 91 selected HistGradientBoosting with seven features and trial 2. Validation selection is not an oracle for the best test model.

The modified policy searches across more candidates than the single untuned baseline. It measures the practical benefit of that entire selection procedure. It does not isolate an equal-budget model effect or establish an improvement over the authors' published benchmark.

## Full-feature models at the largest training size

The following models each select among four configurations using validation accuracy. These are the appropriate equal-trial-budget model comparisons.

| Dataset | Random Forest | XGBoost | HistGradientBoosting | MLP |
|---|---:|---:|---:|---:|
| credit | 77.44% | 77.86% | 77.68% | 77.33% |
| electricity | 85.37% | 85.47% | 85.53% | 80.06% |
| MagicTelescope | 85.68% | 86.01% | 85.97% | 85.86% |

Trees have a clear advantage over this MLP on electricity, while the models are much closer on the other two datasets. This does not evaluate the paper's Transformer models.

## Training size and feature selection

With full features, MLP accuracy at 1,000 rows and at the largest size changes as follows:

- electricity: 75.49% to 80.06%.
- MagicTelescope: 82.23% to 85.86%.
- credit: 76.81% to 77.33%, with 77.42% at 3,000 rows.

Larger training sets often help, but the credit trend is not monotonic. Subset variation, optimisation and the restricted tuning budget can affect that trend.

Retaining the top half of features changes tuned MLP accuracy at the largest size by -4.22 points on credit, -0.94 on electricity and -1.13 on MagicTelescope. It changes tuned Random Forest accuracy by -5.16, +0.07 and -1.00 points, respectively. The selected half is not guaranteed to consist only of useful or nonredundant features, and the discarded half is not guaranteed to be uninformative.

## Interpretation limits

Only three processed binary classification datasets, three overlapping splits and four small search spaces are used. The small MLP is not an exact copy of the authors' neural architectures. Electricity observations use random rather than future-date partitions. All results describe the processed benchmark task rather than deployment on the original population. Standard deviations are descriptive and are not independent confidence intervals.
