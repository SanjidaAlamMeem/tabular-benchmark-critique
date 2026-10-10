"""Check properties that would invalidate the experiment if they failed."""

# Test data isolation and reproducibility rather than model-specific score values.
import unittest
import numpy as np
import pandas as pd
from run_experiment import make_splits, nested_training_indices, select_features
from analyse_results import select_validation_winners


class ProtocolTests(unittest.TestCase):
    # Every row must belong to one partition, with the intended 70/9/21 ratio.
    def test_partitions_are_disjoint_and_cover_all_rows(self):
        y = np.array([0, 1] * 5000)
        train, validation, test = make_splits(y, 17)
        self.assertEqual(len(set(train) | set(validation) | set(test)), len(y))
        self.assertFalse(set(train) & set(validation))
        self.assertFalse(set(train) & set(test))
        self.assertFalse(set(validation) & set(test))
        self.assertEqual((len(train),len(validation),len(test)), (7000,900,2100))

    # Reusing the seed must preserve both the sampled rows and class balance.
    def test_training_subsets_are_nested_and_balanced(self):
        y = np.array([0, 1] * 5000)
        train, _, _ = make_splits(y, 42)
        small = nested_training_indices(train, y, 1000, 42)
        large = nested_training_indices(train, y, 3000, 42)
        self.assertTrue(set(small).issubset(set(large)))
        self.assertEqual(np.bincount(y[small]).tolist(), [500,500])
        np.testing.assert_array_equal(small, nested_training_indices(train, y, 1000, 42))

    # Check the selected column count and recovery of a known informative column.
    # The caller is responsible for passing training rows; this test uses synthetic data.
    def test_feature_selection_uses_training_input_and_keeps_column_count(self):
        rng = np.random.default_rng(5)
        x = rng.normal(size=(400, 10))
        y = (x[:, 0] > 0).astype(int)
        selected, _ = select_features(x, y, 0.5, 17, 1)
        self.assertEqual(len(selected), 5)
        self.assertIn(0, selected)
        np.testing.assert_array_equal(selected, select_features(x, y, 0.5, 17, 1)[0])

    # A deliberately better test score must never override validation selection.
    def test_model_selection_does_not_use_test_accuracy(self):
        common = {'dataset':'example', 'seed':17, 'requested_train_size':10000,
                  'train_size':10000, 'selected_trial':1}
        rows = [
            {**common,'model':'RandomForest','setting':'baseline','feature_fraction':1.0,
             'validation_accuracy':0.7,'accuracy':0.7},
            {**common,'model':'MLP','setting':'tuned','feature_fraction':1.0,
             'validation_accuracy':0.9,'accuracy':0.6},
            {**common,'model':'XGBoost','setting':'tuned','feature_fraction':1.0,
             'validation_accuracy':0.8,'accuracy':0.99},
        ]
        selected = select_validation_winners(pd.DataFrame(rows)).iloc[0]
        self.assertEqual(selected.selected_model, 'MLP')
        self.assertAlmostEqual(selected.modified_accuracy, 0.6)

    # Reordering records cannot change the documented tie-breaking rule.
    def test_selection_ties_are_deterministic(self):
        common = {'dataset':'example','seed':17,'requested_train_size':10000,
                  'train_size':10000,'selected_trial':1,'validation_accuracy':0.8,'accuracy':0.75}
        rows = [{**common,'model':'RandomForest','setting':'baseline','feature_fraction':1.0}]
        rows += [{**common,'model':model,'setting':'tuned','feature_fraction':fraction}
                 for model,fraction in [('XGBoost',1.0),('MLP',0.5),('MLP',1.0)]]
        selected = select_validation_winners(pd.DataFrame(rows).sample(frac=1,random_state=3)).iloc[0]
        self.assertEqual(selected.selected_model,'MLP')
        self.assertEqual(selected.selected_feature_fraction,1.0)


# Use standard unittest so checks do not require an additional testing framework.
if __name__ == '__main__':
    unittest.main()
