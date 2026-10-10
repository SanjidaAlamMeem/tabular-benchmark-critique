"""Summarise predeclared comparisons without choosing settings on test scores."""

# Read and validate the experiment records before calculating comparisons.
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Use stable file locations and consistent model colours across all result figures.
ROOT = Path(__file__).resolve().parent
COLOURS = {'RandomForest': '#192c48', 'XGBoost': '#00a0b2',
           'HistGradientBoosting': '#8a6f44', 'MLP': '#ad5477'}


# Make report figures from retained test records, with uncertainty stated explicitly.
def plot_report_evidence(results, summary, paired_features, directory):
    datasets = [('credit', 'Credit'), ('electricity', 'Electricity'), ('MagicTelescope', 'MAGIC')]
    models = ['RandomForest', 'XGBoost', 'HistGradientBoosting', 'MLP']
    names = {'RandomForest': 'Random Forest', 'XGBoost': 'XGBoost',
             'HistGradientBoosting': 'Histogram boosting', 'MLP': 'MLP'}
    markers = {'RandomForest': 'o', 'XGBoost': 's', 'HistGradientBoosting': '^', 'MLP': 'D'}
    curves = summary[(summary.setting == 'tuned') & (summary.feature_fraction == 1.0)].copy()
    curves.to_csv(directory / 'learning_curve_plot_data.csv', index=False)
    paired_features[['dataset', 'seed', 'model', 'accuracy_full', 'accuracy_half', 'difference_pp']].to_csv(
        directory / 'paired_feature_plot_data.csv', index=False)

    # Use actual training rows and mean plus/minus one sample standard deviation.
    # The bands describe overlapping splits; they are not confidence intervals.
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.45), sharey=True)
    low = np.floor(((curves.accuracy_mean - curves.accuracy_sd) * 100).min() / 5) * 5
    high = np.ceil(((curves.accuracy_mean + curves.accuracy_sd) * 100).max() / 5) * 5
    for ax, (dataset, label) in zip(axes, datasets):
        for model in models:
            group = curves[(curves.dataset == dataset) & (curves.model == model)].sort_values('train_size')
            x = group.train_size.to_numpy()
            y = group.accuracy_mean.to_numpy() * 100
            sd = group.accuracy_sd.to_numpy() * 100
            ax.plot(x, y, marker=markers[model], markersize=5, linewidth=1.8,
                    color=COLOURS[model], label=names[model])
            ax.fill_between(x, y - sd, y + sd, color=COLOURS[model], alpha=0.11, linewidth=0)
        ax.set_title(label, fontsize=12, weight='bold')
        sizes = sorted(curves[curves.dataset == dataset].train_size.unique())
        ax.set_xticks(sizes, [f'{int(n):,}' for n in sizes], fontsize=10)
        ax.set_ylim(low, high)
        ax.grid(axis='y', color='#dddddd', linewidth=0.6)
        ax.set_xlabel('Actual training rows', fontsize=11)
    axes[0].set_ylabel('Mean test accuracy (%)', fontsize=11)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, frameon=False, fontsize=10)
    fig.tight_layout(rect=[0, 0.13, 1, 1])
    fig.savefig(directory / 'figures' / 'report_learning_curves.png', dpi=250, bbox_inches='tight')
    plt.close(fig)

    # Show all seed-level paired changes alongside each family's mean and SD.
    # Negative values mean removing half the columns reduced held-out accuracy.
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.35), sharex=True, sharey=True)
    for ax, (dataset, label) in zip(axes, datasets):
        for position, model in enumerate(models):
            values = paired_features[(paired_features.dataset == dataset) &
                                     (paired_features.model == model)].sort_values('seed').difference_pp.to_numpy()
            ax.scatter(values, position + np.linspace(-0.10, 0.10, len(values)), s=18,
                       color=COLOURS[model], alpha=0.55, zorder=3)
            ax.errorbar(values.mean(), position, xerr=values.std(ddof=1), fmt='D', markersize=6,
                        color=COLOURS[model], markeredgecolor='white', markeredgewidth=0.7,
                        capsize=3, linewidth=1.6, zorder=4)
        ax.axvline(0, color='#555555', linewidth=0.8)
        ax.set_title(label, fontsize=12, weight='bold')
        ax.set_yticks(range(4), ['RF', 'XGB', 'HGB', 'MLP'])
        ax.set_ylim(3.6, -0.6)
        ax.grid(axis='x', color='#dddddd', linewidth=0.6)
        ax.set_xlabel('Accuracy change (percentage points)', fontsize=10)
    fig.tight_layout()
    fig.savefig(directory / 'figures' / 'report_feature_effects.png', dpi=250, bbox_inches='tight')
    plt.close(fig)


# Pair baseline and modified settings on exactly the same seed and test partition.
def select_validation_winners(results):
    maximum = results['requested_train_size'].max()
    rows = []
    for (dataset, seed), group in results.groupby(['dataset', 'seed']):
        group = group[group.requested_train_size == maximum]
        baseline = group[(group.model == 'RandomForest') & (group.setting == 'baseline')
                         & (group.feature_fraction == 1.0)].iloc[0]
        # Use validation scores across models and feature conditions; resolve ties
        # by model name, then prefer all features within that model.
        alternatives = group[group.setting == 'tuned'].sort_values(
            ['validation_accuracy', 'model', 'feature_fraction'], ascending=[False, True, False], kind='stable')
        winner = alternatives.iloc[0]
        rows.append({'dataset': dataset, 'seed': seed, 'train_size': winner.train_size,
                     'baseline_accuracy': baseline.accuracy, 'modified_accuracy': winner.accuracy,
                     'difference_pp': 100 * (winner.accuracy - baseline.accuracy),
                     'selected_model': winner.model, 'selected_feature_fraction': winner.feature_fraction,
                     'selected_trial': winner.selected_trial,
                     'selected_validation_accuracy': winner.validation_accuracy})
    return pd.DataFrame(rows)


# Produce readable CSVs with descriptive spread across three repeated splits.
def analyse(directory):
    config = json.loads((directory / 'experiment.json').read_text())
    results = pd.read_csv(directory / 'results.csv')
    trials = pd.read_csv(directory / 'trials.csv')
    conditions = (len(config['datasets']) * len(config['seeds']) * len(config['training_sizes'])
                  * len(config['feature_fractions']) * len(config['models']))
    assert len(results) == conditions * 2, 'Missing test records.'
    assert len(trials) == conditions * 4, 'Missing hyperparameter trials.'
    assert not results.duplicated(['dataset','seed','requested_train_size','feature_fraction','model','setting']).any()
    assert results['accuracy'].between(0, 1).all()
    # Summarise repeated splits using means and sample SDs, without significance claims.
    summary = results.groupby(['dataset','requested_train_size','feature_fraction','model','setting']).agg(
        accuracy_mean=('accuracy','mean'), accuracy_sd=('accuracy','std'),
        validation_accuracy_mean=('validation_accuracy','mean'), f1_mean=('f1','mean'),
        roc_auc_mean=('roc_auc','mean'), search_seconds_mean=('search_seconds','mean'),
        train_size=('train_size','first'), n_features=('n_features','first')).reset_index()
    summary.to_csv(directory / 'summary.csv', index=False)
    # Report the broader validation-selected policy separately from equal-budget models.
    winners = select_validation_winners(results)
    winners.to_csv(directory / 'validation_selected_comparisons.csv', index=False)
    comparisons = winners.groupby('dataset').agg(
        baseline_accuracy=('baseline_accuracy','mean'), baseline_sd=('baseline_accuracy','std'),
        modified_accuracy=('modified_accuracy','mean'), modified_sd=('modified_accuracy','std'),
        difference_pp=('difference_pp','mean'), difference_pp_sd=('difference_pp','std')).reset_index()
    comparisons.to_csv(directory / 'main_comparison.csv', index=False)
    # Pair all-feature and half-feature scores within each dataset, seed and model.
    tuned = results[results.setting == 'tuned']
    max_rows = tuned[tuned.requested_train_size == max(config['training_sizes'])]
    full = max_rows[max_rows.feature_fraction == 1.0]
    half = max_rows[max_rows.feature_fraction == 0.5]
    effect = full.merge(half, on=['dataset','seed','model'], suffixes=('_full','_half'))
    effect['difference_pp'] = 100 * (effect.accuracy_half - effect.accuracy_full)
    feature_effect = effect.groupby(['dataset','model']).agg(
        difference_pp=('difference_pp','mean'), difference_sd=('difference_pp','std')).reset_index()
    feature_effect.to_csv(directory / 'feature_effect.csv', index=False)

    # Compare every family at the same four-trial budget with all features kept.
    # These tables prevent the broader selection policy from hiding model differences.
    fair = summary[(summary.setting == 'tuned') & (summary.feature_fraction == 1.0)
                   & (summary.requested_train_size == max(config['training_sizes']))].copy()
    fair.to_csv(directory / 'equal_budget_model_comparison.csv', index=False)
    fair.pivot(index='dataset', columns='model', values='accuracy_mean').to_csv(
        directory / 'equal_budget_accuracy_matrix.csv')

    # Pair tuning changes with each family's own default on the identical test rows.
    baseline_rows = results[results.setting == 'baseline']
    tuned_rows = results[results.setting == 'tuned']
    paired_keys = ['dataset', 'seed', 'requested_train_size', 'feature_fraction', 'model']
    paired = baseline_rows.merge(tuned_rows, on=paired_keys, suffixes=('_baseline', '_tuned'),
                                 validate='one_to_one')
    paired['difference_pp'] = 100 * (paired.accuracy_tuned - paired.accuracy_baseline)
    paired.to_csv(directory / 'paired_tuning_effects.csv', index=False)
    paired.groupby(['dataset','requested_train_size','feature_fraction','model']).agg(
        difference_pp=('difference_pp','mean'), difference_sd=('difference_pp','std'),
        improvements=('difference_pp',lambda x: int((x > 0).sum())),
        losses=('difference_pp',lambda x: int((x < 0).sum()))).reset_index().to_csv(
            directory / 'tuning_effect_summary.csv', index=False)

    # Keep endpoint learning-curve effects separate from feature-removal effects.
    smallest = tuned[(tuned.requested_train_size == min(config['training_sizes']))
                     & (tuned.feature_fraction == 1.0)]
    growth = smallest.merge(full, on=['dataset','seed','model'], suffixes=('_small','_large'),
                            validate='one_to_one')
    growth['difference_pp'] = 100 * (growth.accuracy_large - growth.accuracy_small)
    growth.groupby(['dataset','model']).agg(
        small_accuracy=('accuracy_small','mean'), large_accuracy=('accuracy_large','mean'),
        difference_pp=('difference_pp','mean'), difference_sd=('difference_pp','std')).reset_index().to_csv(
            directory / 'training_size_effect.csv', index=False)

    # Plot numeric evidence for the report and for users rerunning the repository.
    figures = directory / 'figures'
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':11, 'axes.spines.top':False,
                         'axes.spines.right':False, 'savefig.dpi':180})
    fig, axes = plt.subplots(1, len(config['datasets']), figsize=(13, 3.8), sharey=False)
    for ax, spec in zip(np.atleast_1d(axes), config['datasets']):
        data = summary[(summary.dataset == spec['name']) & (summary.setting == 'tuned')
                       & (summary.feature_fraction == 1.0)]
        for model, group in data.groupby('model'):
            group = group.sort_values('requested_train_size')
            ax.errorbar(group.requested_train_size, group.accuracy_mean * 100,
                        yerr=group.accuracy_sd * 100, label=model, marker='o', capsize=3, color=COLOURS[model])
        ax.set_title(spec['name'])
        ax.set_xlabel('Requested training rows')
        ax.set_xticks(config['training_sizes'], ['1,000','3,000','10,000*'])
        ax.set_ylabel('Test accuracy (%)')
        ax.grid(alpha=0.15)
    handles, labels = axes[0].get_legend_handles_labels() if len(config['datasets']) > 1 else axes.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, frameon=False, fontsize=10)
    fig.tight_layout(rect=[0,0.13,1,1])
    fig.savefig(figures / 'training_size.png', bbox_inches='tight')
    plt.close(fig)
    # Display every equal-budget model, not only the best average test result.
    matrix = fair.pivot(index='dataset', columns='model', values='accuracy_mean')
    fig, ax = plt.subplots(figsize=(10, 4.4))
    positions = np.arange(len(matrix))
    for i, model in enumerate(config['models']):
        ax.bar(positions + (i - 1.5) * 0.18, matrix[model] * 100, 0.18,
               label=model, color=COLOURS[model])
    ax.set_xticks(positions, matrix.index)
    ax.set_ylabel('Mean test accuracy (%)')
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, ncol=2, loc='lower right')
    fig.tight_layout()
    fig.savefig(figures / 'equal_budget_models.png', bbox_inches='tight')
    plt.close(fig)
    # Keep negative ablations visible and label changes in percentage points.
    feature_matrix = feature_effect.pivot(index='dataset', columns='model', values='difference_pp')
    fig, ax = plt.subplots(figsize=(10, 4.4))
    for i, model in enumerate(config['models']):
        ax.bar(positions + (i - 1.5) * 0.18, feature_matrix[model], 0.18,
               label=model, color=COLOURS[model])
    ax.axhline(0, color='#555555', linewidth=0.8)
    ax.set_xticks(positions, feature_matrix.index)
    ax.set_ylabel('Half features minus all features (percentage points)')
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(figures / 'feature_selection_effect.png', bbox_inches='tight')
    plt.close(fig)
    # Contrast the untuned forest with the broader policy, which has a larger search budget.
    fig, ax = plt.subplots(figsize=(9, 4))
    positions = np.arange(len(comparisons))
    ax.bar(positions - 0.18, comparisons.baseline_accuracy * 100, 0.36, label='Baseline Random Forest', color='#192c48')
    ax.bar(positions + 0.18, comparisons.modified_accuracy * 100, 0.36, label='Validation-selected modification', color='#00a0b2')
    ax.set_xticks(positions, comparisons.dataset)
    ax.set_ylabel('Test accuracy (%)')
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, loc='upper left')
    fig.tight_layout()
    fig.savefig(figures / 'baseline_vs_modified.png', bbox_inches='tight')
    plt.close(fig)
    # Export readable learning curves and paired ablations for the written critique.
    plot_report_evidence(results, summary, effect, directory)
    print(comparisons.to_string(index=False))
    print('\nFeature selection effect in percentage points:')
    print(feature_effect.to_string(index=False))
    print(f'\nVerified {conditions} conditions, {len(trials)} fits, {len(results)} test records.')


# Keep result selection deterministic when the script runs from a new machine.
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, default=ROOT / 'results')
    analyse(parser.parse_args().results_dir)
