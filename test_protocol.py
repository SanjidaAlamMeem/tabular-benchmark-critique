"""Check properties that would invalidate the experiment if they failed."""

# Test data isolation and reproducibility rather than model-specific score values.
import unittest
import numpy as np
from run_experiment import make_splits, nested_training_indices, select_features


class ProtocolTests(unittest.TestCase):
    def test_partitions_are_disjoint_and_cover_all_rows(self):
        y = np.array([0, 1] * 5000)
        train, validation, test = make_splits(y, 17)
        self.assertEqual(len(set(train) | set(validation) | set(test)), len(y))
        self.assertFalse(set(train) & set(validation))
        self.assertFalse(set(train) & set(test))
        self.assertFalse(set(validation) & set(test))
        self.assertEqual((len(train),len(validation),len(test)), (7000,900,2100))

    def test_training_subsets_are_nested_and_balanced(self):
        y = np.array([0, 1] * 5000)
        train, _, _ = make_splits(y, 42)
        small = nested_training_indices(train, y, 1000, 42)
        large = nested_training_indices(train, y, 3000, 42)
        self.assertTrue(set(small).issubset(set(large)))
        self.assertEqual(np.bincount(y[small]).tolist(), [500,500])
        np.testing.assert_array_equal(small, nested_training_indices(train, y, 1000, 42))

    def test_feature_selection_uses_training_input_and_keeps_column_count(self):
        rng = np.random.default_rng(5)
        x = rng.normal(size=(400, 10))
        y = (x[:, 0] > 0).astype(int)
        selected, _ = select_features(x, y, 0.5, 17, 1)
        self.assertEqual(len(selected), 5)
        self.assertIn(0, selected)
        np.testing.assert_array_equal(selected, select_features(x, y, 0.5, 17, 1)[0])


# Use standard unittest so checks do not require an additional testing framework.
if __name__ == '__main__':
    unittest.main()
