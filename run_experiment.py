"""A small, reproducible extension of Grinsztajn et al.'s tabular benchmark.

Run this file from any directory. The configuration fixes the experiment before
training starts. Models and feature selectors never fit on the held-out test set.
This is an independent implementation, not the authors' full training pipeline.
"""

# Set thread limits before importing numerical libraries for comparable CPU runs.
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')

# Import standard utilities and the common sklearn-compatible model interface.
import argparse
import hashlib
import json
import platform
import time
import warnings
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
import scipy
import sklearn
import xgboost
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, QuantileTransformer
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier

# Resolve default file locations relative to the script, rather than the shell directory.
ROOT = Path(__file__).resolve().parent


# Download the authors' already-processed datasets and verify exact file hashes.
def load_dataset(spec, revision, directory):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{spec['name']}.csv"
    url = f"https://huggingface.co/datasets/inria-soda/tabular-benchmark/resolve/{revision}/clf_num/{spec['name']}.csv"
    if not path.exists():
        urlretrieve(url, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != spec['sha256']:
        raise ValueError(f"Dataset checksum mismatch: {path.name}")
    frame = pd.read_csv(path)
    features = frame.drop(columns=spec['target'])
    if features.isna().any().any() or not np.isfinite(features.to_numpy(dtype=float)).all():
        raise ValueError('This numerical-only experiment requires complete finite values.')
    # Encode the two class names without using target values to transform features.
    encoder = LabelEncoder().fit(frame[spec['target']])
    if len(encoder.classes_) != 2:
        raise ValueError('This experiment supports binary classification only.')
    return features, encoder.transform(frame[spec['target']]), {
        'name': spec['name'], 'rows': len(frame), 'features': len(features.columns),
        'class_mapping': {str(v): i for i, v in enumerate(encoder.classes_)},
        'source_url': url, 'sha256': digest,
    }


# Fix stratified train/validation/test partitions before any model fitting.
def make_splits(y, seed):
    indices = np.arange(len(y))
    train, remaining = train_test_split(indices, train_size=0.7, stratify=y, random_state=seed)
    validation, test = train_test_split(remaining, train_size=0.3, stratify=y[remaining], random_state=seed)
    assert not set(train) & set(validation)
    assert not set(train) & set(test)
    assert not set(validation) & set(test)
    return train, validation, test


# Build balanced nested subsets, so larger training sets contain smaller ones.
def nested_training_indices(train, y, requested_size, seed):
    rng = np.random.default_rng(seed)
    class_zero = rng.permutation(train[y[train] == 0])
    class_one = rng.permutation(train[y[train] == 1])
    size = min(requested_size, 2 * min(len(class_zero), len(class_one)))
    zero_count = size // 2
    one_count = size - zero_count
    selected = np.concatenate([class_zero[:zero_count], class_one[:one_count]])
    return rng.permutation(selected)


# Rank columns using only the current training subset and retain a fixed fraction.
def select_features(x_train, y_train, fraction, seed, threads):
    if fraction == 1.0:
        return np.arange(x_train.shape[1]), 0.0
    start = time.perf_counter()
    ranker = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=threads)
    ranker.fit(x_train, y_train)
    number = max(1, int(np.ceil(x_train.shape[1] * fraction)))
    # Stable ordering makes tied importances reproducible.
    ranked = np.argsort(-ranker.feature_importances_, kind='stable')
    return np.sort(ranked[:number]), time.perf_counter() - start


# Predeclare one baseline configuration and three alternatives for each family.
def model_candidates(name, seed, threads, train_rows):
    if name == 'RandomForest':
        parameters = [{}, {'n_estimators': 200, 'min_samples_leaf': 2},
                      {'n_estimators': 200, 'max_features': 0.8, 'min_samples_leaf': 2},
                      {'n_estimators': 200, 'max_depth': 12, 'max_features': 0.8}]
        return [(p, RandomForestClassifier(random_state=seed, n_jobs=threads, **p)) for p in parameters]
    if name == 'XGBoost':
        parameters = [{}, {'n_estimators': 200, 'max_depth': 3, 'learning_rate': 0.05},
                      {'n_estimators': 200, 'max_depth': 5, 'learning_rate': 0.1, 'subsample': 0.8},
                      {'n_estimators': 300, 'max_depth': 4, 'learning_rate': 0.05, 'reg_lambda': 5}]
        return [(p, XGBClassifier(random_state=seed, n_jobs=threads, tree_method='hist', **p)) for p in parameters]
    if name == 'HistGradientBoosting':
        parameters = [{}, {'max_iter': 200, 'max_leaf_nodes': 15, 'learning_rate': 0.05},
                      {'max_iter': 200, 'max_leaf_nodes': 31, 'l2_regularization': 1},
                      {'max_iter': 200, 'max_leaf_nodes': 15, 'l2_regularization': 5}]
        return [(p, HistGradientBoostingClassifier(random_state=seed, early_stopping=False, **p)) for p in parameters]
    if name == 'MLP':
        parameters = [{}, {'hidden_layer_sizes': (64, 32), 'alpha': 0.001},
                      {'hidden_layer_sizes': (64, 32), 'alpha': 0.01, 'learning_rate_init': 0.003},
                      {'hidden_layer_sizes': (32, 16), 'alpha': 0.001, 'learning_rate_init': 0.003}]
        candidates = []
        for p in parameters:
            # The transformer fits inside the pipeline on training rows only.
            model = Pipeline([
                ('gaussianize', QuantileTransformer(n_quantiles=min(1000, train_rows),
                     output_distribution='normal', random_state=seed)),
                ('classifier', MLPClassifier(random_state=seed, max_iter=200,
                     early_stopping=True, validation_fraction=0.1, n_iter_no_change=15, **p)),
            ])
            candidates.append((p, model))
        return candidates
    raise ValueError(f'Unknown model family: {name}')


# Measure classification performance without using test labels to choose a model.
def score_model(model, x, y):
    prediction = model.predict(x)
    probability = model.predict_proba(x)[:, 1]
    return {
        'accuracy': accuracy_score(y, prediction),
        'balanced_accuracy': balanced_accuracy_score(y, prediction),
        'f1': f1_score(y, prediction),
        'roc_auc': roc_auc_score(y, probability),
    }, prediction, probability


# Write completed records immediately, so an interrupted run retains its evidence.
def append_record(path, record):
    pd.DataFrame([record]).to_csv(path, mode='a', header=not path.exists(), index=False)


# Execute a fixed factorial design and retain full tuning and split provenance.
def run(config, data_dir, output_dir):
    # Store the planned settings and reject directories containing an earlier run.
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / 'results.csv').exists() or (output_dir / 'trials.csv').exists():
        raise FileExistsError('Choose a new output directory to avoid mixing independent runs.')
    (output_dir / 'predictions').mkdir(exist_ok=True)
    (output_dir / 'splits').mkdir(exist_ok=True)
    (output_dir / 'experiment.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    # Record library versions, CPU settings and elapsed time for reproducibility.
    environment = {'python': platform.python_version(), 'platform': platform.platform(),
                   'numpy': np.__version__, 'pandas': pd.__version__, 'scipy': scipy.__version__,
                   'scikit-learn': sklearn.__version__, 'xgboost': xgboost.__version__,
                   'threads': config['threads'], 'device': 'CPU', 'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    (output_dir / 'environment.json').write_text(json.dumps(environment, indent=2), encoding='utf-8')
    total_start = time.perf_counter()
    metadata = []
    # Keep validation and test rows fixed while changing the training subset size.
    for spec in config['datasets']:
        features, y, info = load_dataset(spec, config['dataset_revision'], data_dir)
        metadata.append(info)
        x = features.to_numpy(dtype=np.float64)
        for seed in config['seeds']:
            train_pool, validation, test = make_splits(y, seed)
            for requested_size in config['training_sizes']:
                train = nested_training_indices(train_pool, y, requested_size, seed)
                np.savez_compressed(output_dir / 'splits' / f"{spec['name']}_{seed}_{requested_size}.npz",
                                    train=train, validation=validation, test=test)
                # Fit one column ranking per training subset and feature condition.
                # Every model in that condition receives the same selected columns.
                for fraction in config['feature_fractions']:
                    columns, selection_seconds = select_features(x[train], y[train], fraction, seed, config['threads'])
                    x_train, x_validation, x_test = x[train][:, columns], x[validation][:, columns], x[test][:, columns]
                    for name in config['models']:
                        common = {'dataset': spec['name'], 'seed': seed, 'requested_train_size': requested_size,
                                  'train_size': len(train), 'validation_size': len(validation), 'test_size': len(test),
                                  'feature_fraction': fraction, 'n_features': len(columns), 'model': name}
                        best, baseline = None, None
                        search_seconds = 0.0
                        # Give each family four candidates and log warnings as well as scores.
                        for trial, (parameters, model) in enumerate(model_candidates(name, seed, config['threads'], len(train))):
                            start = time.perf_counter()
                            with warnings.catch_warnings(record=True) as captured:
                                warnings.simplefilter('always', ConvergenceWarning)
                                model.fit(x_train, y[train])
                            fit_seconds = time.perf_counter() - start
                            validation_start = time.perf_counter()
                            validation_accuracy = accuracy_score(y[validation], model.predict(x_validation))
                            validation_seconds = time.perf_counter() - validation_start
                            search_seconds += fit_seconds + validation_seconds
                            n_iter = getattr(model.named_steps['classifier'], 'n_iter_', None) if isinstance(model, Pipeline) else None
                            record = {**common, 'trial': trial, 'parameters': json.dumps(parameters),
                                      'validation_accuracy': validation_accuracy, 'fit_seconds': fit_seconds,
                                      'validation_seconds': validation_seconds, 'epochs': n_iter,
                                      'convergence_warning': any(issubclass(w.category, ConvergenceWarning) for w in captured)}
                            append_record(output_dir / 'trials.csv', record)
                            candidate = (validation_accuracy, trial, model, parameters, fit_seconds)
                            # Select by validation accuracy; ties keep the earlier trial.
                            if best is None or candidate[0] > best[0]:
                                best = candidate
                            if trial == 0:
                                baseline = candidate
                        # Test evaluation begins only after the validation search ends.
                        for label, candidate in [('tuned', best), ('baseline', baseline)]:
                            validation_accuracy, trial, model, parameters, fit_seconds = candidate
                            test_start = time.perf_counter()
                            scores, prediction, probability = score_model(model, x_test, y[test])
                            test_seconds = time.perf_counter() - test_start
                            append_record(output_dir / 'results.csv', {**common, 'setting': label, 'selected_trial': trial,
                                          'validation_accuracy': validation_accuracy, **scores,
                                          'fit_seconds': fit_seconds, 'search_seconds': search_seconds,
                                          'feature_selection_seconds': selection_seconds,
                                          'test_seconds': test_seconds, 'parameters': json.dumps(parameters),
                                          'selected_features': json.dumps(features.columns[columns].tolist())})
                            key = f"{spec['name']}_{seed}_{requested_size}_{fraction}_{name}_{label}"
                            # Retain predictions locally so the saved metrics can be checked.
                            np.savez_compressed(output_dir / 'predictions' / f'{key}.npz',
                                                row_indices=test, y_true=y[test], y_pred=prediction, probability=probability)
                        print(f"{spec['name']} seed={seed} n={len(train)} features={len(columns)} {name}: "
                              f"validation={best[0]:.4f} trial={best[1]} search={search_seconds:.1f}s", flush=True)
    # Finish the manifest only after every planned condition has completed.
    (output_dir / 'datasets.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    environment['total_seconds'] = time.perf_counter() - total_start
    environment['completed_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    (output_dir / 'environment.json').write_text(json.dumps(environment, indent=2), encoding='utf-8')
    print(f"Completed in {environment['total_seconds'] / 60:.1f} minutes.", flush=True)


# Offer a short smoke run and a complete experiment through the same entry point.
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'experiment.json')
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results')
    parser.add_argument('--smoke', action='store_true', help='One dataset, seed, size, and feature condition.')
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    if args.smoke:
        config['datasets'] = config['datasets'][:1]
        config['seeds'] = config['seeds'][:1]
        config['training_sizes'] = [1000]
        config['feature_fractions'] = [1.0]
    with threadpool_limits(limits=config['threads']):
        run(config, args.data_dir, args.output_dir)


if __name__ == '__main__':
    main()
