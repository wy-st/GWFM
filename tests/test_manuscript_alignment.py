"""Prevent stale figures and changed table values from silently entering the final package."""

import json
import unittest

import pandas as pd
from PIL import Image
import torch

import build_figures as figures
from gwfm.paths import FIGURES, REPOSITORY
from gwfm.utils import digest
from gwfm.models.model import encoder_from_config
from gwfm.training import configuration


class ManuscriptAlignmentTests(unittest.TestCase):
    def test_reported_encoder_sizes_and_neighbor_count(self):
        feature_modality = torch.arange(3).repeat_interleave(4)
        for name, count in (("GWFM", 106297), ("PDFM", 16764)):
            encoder = encoder_from_config(feature_modality, configuration(name, 11))
            self.assertEqual(sum(p.numel() for p in encoder.parameters()), count)
            self.assertEqual(encoder.embedding_dim, 48)
            self.assertEqual(encoder.block_dim, 16)
            self.assertEqual(encoder.neighbors, 8)

    def test_reported_values_and_changed_result_rejected(self):
        summary = pd.read_csv(figures.SOURCE / "all_method_summary.csv")
        table = summary.pivot(index="key", columns="method", values="mean")
        maps = pd.read_csv(figures.SOURCE / "map_metrics.csv")
        pca = json.loads((figures.SOURCE / "pca_panels.json").read_text())
        figures.check_manuscript_values(table, summary, maps, pca)
        changed = table.copy()
        changed.loc["Heat-related burden", "GWFM"] += 0.01
        with self.assertRaises(AssertionError):
            figures.check_manuscript_values(changed, summary, maps, pca)

    def test_one_final_copy_per_figure_and_exact_schematic_artwork(self):
        actual = sorted(p.name for p in FIGURES.iterdir() if p.is_file())
        self.assertEqual(actual, sorted(figures.FINAL_FILES))
        reference = json.loads(
            (REPOSITORY / "report/figure_reference.json").read_text()
        )
        report = reference["manuscript_source"]
        self.assertEqual(digest(REPOSITORY / report["name"]), report["sha256"])
        for name, metadata in reference["static_artwork"].items():
            self.assertEqual(digest(FIGURES / name), metadata["sha256"])
        for name in actual:
            with Image.open(FIGURES / name) as image:
                image.load()
                self.assertGreater(min(image.size), 700)
                lo, hi = image.convert("L").getextrema()
                self.assertGreater(hi - lo, 200)
        self.assertEqual(len({digest(FIGURES / name) for name in actual}), 6)


if __name__ == "__main__":
    unittest.main()
