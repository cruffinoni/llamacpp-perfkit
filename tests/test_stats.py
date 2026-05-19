from __future__ import annotations

import unittest

from llamacpp_perfkit.stats import Sample, summarize


class StatsTest(unittest.TestCase):
    def test_mean(self) -> None:
        self.assertEqual(summarize([1, 2, 3]).mean, 2)

    def test_median_odd_and_even_samples(self) -> None:
        self.assertEqual(summarize([3, 1, 2]).median, 2)
        self.assertEqual(summarize([4, 1, 3, 2]).median, 2.5)

    def test_sample_stddev(self) -> None:
        self.assertAlmostEqual(summarize([1, 2, 3]).stddev or 0, 1.0)

    def test_geometric_mean(self) -> None:
        self.assertAlmostEqual(summarize([1, 4, 16]).geometric_mean or 0, 4.0)

    def test_geometric_mean_ignores_zero_and_negative_values(self) -> None:
        self.assertAlmostEqual(summarize([1, 0, -4, 9]).geometric_mean or 0, 3.0)

    def test_p10(self) -> None:
        self.assertAlmostEqual(summarize([1, 2, 3, 4, 5]).p10 or 0, 1.4)

    def test_empty_input(self) -> None:
        summary = summarize([])
        self.assertEqual(summary.count, 0)
        self.assertIsNone(summary.mean)
        self.assertIsNone(summary.median)
        self.assertIsNone(summary.p10)
        self.assertIsNone(summary.stddev)
        self.assertIsNone(summary.geometric_mean)
        self.assertIsNone(summary.min)
        self.assertIsNone(summary.max)

    def test_sample_median_does_not_mutate_input_order(self) -> None:
        sample = Sample((3, 1, 2))
        self.assertEqual(sample.median(), 2)
        self.assertEqual(sample.values, (3, 1, 2))


if __name__ == "__main__":
    unittest.main()
