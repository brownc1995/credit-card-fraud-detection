import unittest

import pandas as pd

from ccfd.data import get_data, train_val_test_split, scale_data, CLASS

AMOUNT = 'amount'


class TestData(unittest.TestCase):
    DELTA = 0.001

    def setUp(self):
        self.data = get_data()
        self.train_data, self.train_target, self.val_data, _, self.test_data, _ = train_val_test_split(self.data, 0.2)
        self.train_scaled, *_ = scale_data(self.train_data, self.val_data, self.test_data)

    def test_get_data(self):
        self.assertIsInstance(self.data, pd.DataFrame)
        self.assertTrue(hasattr(self.data, CLASS))
        self.assertTrue(hasattr(self.data, AMOUNT))

    def test_train_val_test_split(self):
        self.assertEqual(len(self.train_data), len(self.train_target))
        self.assertIsInstance(self.train_data, pd.DataFrame)
        self.assertIsInstance(self.train_target, pd.Series)
        self.assertEqual(round(len(self.data) * 0.8 * 0.8), len(self.train_data))

    def test_scale_data(self):
        self.assertIsInstance(self.train_scaled, pd.DataFrame)
        self.assertAlmostEqual(self.train_scaled[AMOUNT].mean(), 0, delta=self.DELTA)
        self.assertAlmostEqual(self.train_scaled[AMOUNT].std(), 1, delta=self.DELTA)
        self.assertAlmostEqual(self.train_scaled['v4'].mean(), 0, delta=self.DELTA)
        self.assertAlmostEqual(self.train_scaled['v4'].std(), 1, delta=self.DELTA)
