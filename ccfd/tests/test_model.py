import unittest

from ccfd.model import *


class TestData(unittest.TestCase):
    def setUp(self) -> None:
        self.data = pd.DataFrame({
            'class': [0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1],
        })

    def test_calc_input_bias(self):
        self.assertAlmostEqual(calc_initial_bias(self.data), -0.405465108)

    def test_set_class_weights(self):
        class_weight = {
            0: 0.8333333333333333,
            1: 1.25
        }
        self.assertEqual(set_class_weights(self.data), class_weight)

    def test_resample_steps_per_epoch(self):
        self.assertEqual(resample_steps_per_epoch(self.data), 1)
