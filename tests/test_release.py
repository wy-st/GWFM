"""The cleaned package keeps only the reported generator and three trainable ablations."""

import json
import tempfile
import unittest
from pathlib import Path

import torch

from gwfm import data, training
from gwfm.ablation.model import AblationSSL, make_encoder
from gwfm.paths import DATA, REPOSITORY
from gwfm.synthetic.data import generate


class ReleaseTests(unittest.TestCase):
    def test_generator_rejects_unreported_protocol(self):
        config = json.loads((REPOSITORY / "configs/dataset.json").read_text())
        config["family"] = "stationary"
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "Only the paper"):
                generate(config["data_seed"], config, directory)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_generator_preserves_existing_files(self):
        config = json.loads((REPOSITORY / "configs/dataset.json").read_text())
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "existing.txt"
            marker.write_text("keep")
            with self.assertRaisesRegex(ValueError, "empty output directory"):
                generate(config["data_seed"], config, directory)
            self.assertEqual(marker.read_text(), "keep")

    def test_all_reported_ablations_train_on_observed_targets(self):
        graph = data.load_graph(DATA)
        graph = data.subset_graph(graph, 0)
        graph = data.standardize(graph, *data.fit_standardizer(graph))
        masks = [
            data.sample_mask(
                graph["observed"],
                graph["node_type"],
                0.3,
                torch.Generator().manual_seed(seed),
                "mixed",
            )
            for seed in (11, 23)
        ]
        for variant in ("no_weighting", "no_modality", "no_prototype"):
            with self.subTest(variant=variant):
                torch.manual_seed(11)
                config = training.configuration("GWFM", 11)
                config["ablation"] = variant
                if variant == "no_modality":
                    config.pop("modality_arch")
                if variant == "no_prototype":
                    config.update(lambda_proto=0.0, lambda_variance=0.0)
                model = AblationSSL(make_encoder(graph["feature_modality"], config))
                model.begin_epoch(1, graph, 11)
                loss, _ = model(graph, masks)
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                grads = [p.grad for p in model.parameters() if p.grad is not None]
                self.assertTrue(grads)
                self.assertTrue(all(torch.isfinite(g).all() for g in grads))
                self.assertTrue(any(g.abs().sum() > 0 for g in grads))
