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

ROOT = Path(__file__).resolve().parent
COLOURS = {'RandomForest': '#192c48', 'XGBoost': '#00a0b2',
           'HistGradientBoosting': '#8a6f44', 'MLP': '#ad5477'}


# Pair baseline and modified settings on exactly the same seed and test partition.
def select_validation_winners(results):
    maximum = results['requested_train_size'].max()
    rows = []
    for (dataset, seed), group in results.groupby(['dataset', 'seed']):
        group = group[group.requested_train_size == maximum]
        baseline = group[(group.model == 'RandomForest') & (group.setting == 'baseline')
                         & (group.feature_fraction == 1.0)].iloc[0]
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
    summary = results.groupby(['dataset','requested_train_size','feature_fraction','model','setting']).agg(
        accuracy_mean=('accuracy','mean'), accuracy_sd=('accuracy','std'),
        validation_accuracy_mean=('validation_accuracy','mean'), f1_mean=('f1','mean'),
        roc_auc_mean=('roc_auc','mean'), search_seconds_mean=('search_seconds','mean'),
        train_size=('train_size','first'), n_features=('n_features','first')).reset_index()
    summary.to_csv(directory / 'summary.csv', index=False)
    winners = select_validation_winners(results)
    winners.to_csv(directory / 'validation_selected_comparisons.csv', index=False)
    comparisons = winners.groupby('dataset').agg(
        baseline_accuracy=('baseline_accuracy','mean'), baseline_sd=('baseline_accuracy','std'),
        modified_accuracy=('modified_accuracy','mean'), modified_sd=('modified_accuracy','std'),
        difference_pp=('difference_pp','mean'), difference_pp_sd=('difference_pp','std')).reset_index()
    comparisons.to_csv(directory / 'main_comparison.csv', index=False)
    tuned = results[results.setting == 'tuned']
    max_rows = tuned[tuned.requested_train_size == max(config['training_sizes'])]
    full = max_rows[max_rows.feature_fraction == 1.0]
    half = max_rows[max_rows.feature_fraction == 0.5]
    effect = full.merge(half, on=['dataset','seed','model'], suffixes=('_full','_half'))
    effect['difference_pp'] = 100 * (effect.accuracy_half - effect.accuracy_full)
    feature_effect = effect.groupby(['dataset','model']).agg(
        difference_pp=('difference_pp','mean'), difference_sd=('difference_pp','std')).reset_index()
    feature_effect.to_csv(directory / 'feature_effect.csv', index=False)

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
    print(comparisons.to_string(index=False))
    print('\nFeature selection effect in percentage points:')
    print(feature_effect.to_string(index=False))
    print(f'\nVerified {conditions} conditions, {len(trials)} fits, {len(results)} test records.')


# Keep result selection deterministic when the script runs from a new machine.
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, default=ROOT / 'results')
    analyse(parser.parse_args().results_dir)
