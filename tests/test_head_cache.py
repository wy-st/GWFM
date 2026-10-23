import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from gwfm import fit_mlp


class HeadCacheTests(unittest.TestCase):
    def test_changed_targets_with_same_mean_cannot_reuse_head(self):
        torch.set_num_threads(2)
        train = dict(
            x=np.arange(12, dtype=float).reshape(6, 2),
            y=np.arange(6, dtype=float).reshape(6, 1),
        )
        val = dict(x=train["x"][:3].copy(), y=train["y"][:3].copy())
        with tempfile.TemporaryDirectory() as directory, patch.object(
            fit_mlp, "GRID", [dict(hidden=4, weight_decay=0.01)]
        ), patch.object(fit_mlp, "MAX_EPOCHS", 2):
            folder = Path(directory)
            expected = fit_mlp.select_head(train, val, val["x"], folder, "target")
            np.testing.assert_array_equal(
                expected, fit_mlp.select_head(train, val, val["x"], folder, "target")
            )
            changed = dict(train, y=train["y"][::-1].copy())
            self.assertEqual(train["y"].mean(), changed["y"].mean())
            with self.assertRaisesRegex(ValueError, "Saved head inputs differ"):
                fit_mlp.select_head(changed, val, val["x"], folder, "target")


if __name__ == "__main__":
    unittest.main()
