"""Guard the manuscript's mean-per-run metric convention."""

import unittest
import numpy as np
import pandas as pd
from gwfm.visualization.report_statistics import summarize_runs


class ReportStatisticsTest(unittest.TestCase):
    def frame(self):
        return pd.DataFrame(
            dict(
                method=["GWFM"] * 4,
                feature=["target"] * 4,
                seed=[11, 11, 23, 23],
                node_id=[0, 1, 0, 1],
                true_value=[0.0, 2.0, 0.0, 2.0],
                inference=[1.0, 3.0, -1.0, 1.0],
            )
        )

    def test_errors_do_not_cancel_between_seeds(self):
        nodes, metrics = summarize_runs(self.frame())
        np.testing.assert_array_equal(nodes.mean_absolute_error, [1.0, 1.0])
        self.assertEqual(metrics.mae.iloc[0], 1.0)
        self.assertEqual(metrics.r2.iloc[0], 0.0)

    def test_reject_unpaired_targets_and_duplicates(self):
        frame = self.frame()
        with self.assertRaises(ValueError):
            summarize_runs(pd.concat([frame, frame.iloc[:1]]))
        frame.loc[3, "node_id"] = 2
        with self.assertRaises(AssertionError):
            summarize_runs(frame)


if __name__ == "__main__":
    unittest.main()
