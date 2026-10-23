import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from gwfm import training
from gwfm.ablation import run as ablation
from gwfm.utils import digest
from gwfm.visualization import data as figure_data


class ArtifactGuardsTests(unittest.TestCase):
    def test_ablation_reuse_rejects_changed_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("complementary.npz", "geometry.npz", "split.npz", "model.pt"):
                (root / name).write_bytes(b"original")
            cfg = dict(ablation="no_weighting", data="data")
            hashes = {
                name: digest(root / name)
                for name in ("complementary.npz", "geometry.npz", "split.npz")
            }
            (root / "config.json").write_text(
                json.dumps(dict(config=cfg, input_sha256=hashes))
            )
            (root / "metrics.json").write_text(
                json.dumps(dict(checkpoint_sha256=digest(root / "model.pt")))
            )
            with patch.object(training, "DATA", root), patch.object(
                ablation, "encoder_folder", return_value=root
            ):
                ablation.train("no_weighting", 11)
                (root / "split.npz").write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError, "inputs changed"):
                    ablation.train("no_weighting", 11)

    def test_data_illustration_uses_reported_masks_and_test_split(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(
            figure_data, "OUT", Path(directory)
        ):
            (Path(directory) / "source_data").mkdir()
            _, _, _, _, counts, ids, _, _ = figure_data.load()
            np.testing.assert_array_equal(counts.sum(axis=1), [600, 600, 600])
            np.testing.assert_array_equal(counts[:, 1], [77, 213, 145])
            np.testing.assert_array_equal(counts[:, 2], [34, 44, 37])
            self.assertEqual(len(ids), 120)


if __name__ == "__main__":
    unittest.main()
