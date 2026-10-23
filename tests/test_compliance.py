"""Regression checks for the assignment's data and supervision boundaries."""

import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from gwfm import data, protocol, training, generate
from gwfm.models.model import reconstruction_losses, encoder_from_config
from gwfm.models.visible_graph import VisibleGraphSSL
from gwfm.synthetic.common import split_regions
from gwfm.decoders import mlp_predict
from gwfm.paths import DATA, REPOSITORY


torch.set_num_threads(2)
torch.use_deterministic_algorithms(True)


class ComplianceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.g = data.load_graph(DATA)

    def test_hierarchy_split_edges_and_missingness(self):
        g = self.g
        self.assertEqual(g["x"].shape, (5400, 12))
        self.assertEqual(g["node_id"].unique().numel(), 5400)
        parent = g["parent"][600:]
        self.assertTrue(((parent >= 0) & (parent < 600)).all())
        self.assertTrue(torch.equal(g["split"][600:], g["split"][parent]))
        s = dict(np.load(DATA / "split.npz"))
        np.testing.assert_array_equal(s["split"], split_regions(s["group"], 1729))
        for group in np.unique(s["group"]):
            self.assertEqual(
                np.bincount(s["split"][s["group"] == group]).tolist(), [42, 6, 12]
            )
        for level in (0, 1):
            rates = 1 - g["observed"][g["node_type"] == level].float().mean(0)
            self.assertTrue(((rates >= 0.1) & (rates <= 0.4)).all())
        for code in (0, 1, 2):
            p = data.subset_graph(g, code)
            e = p["edge_index"]
            self.assertTrue((p["split"][e] == code).all())
            self.assertTrue(((e >= 0) & (e < len(p["x"]))).all())
            self.assertTrue((p["node_type"][e[0]] == p["node_type"][e[1]]).any())
            children = (p["node_type"] == 1).nonzero().flatten()
            for child in children:
                self.assertEqual(
                    int(((e[0] == child) & (e[1] == p["parent"][child])).sum()), 1
                )
        missing = dict(np.load(DATA / "missingness.npz"))
        order = np.argsort(missing["urbanicity"][:600])
        self.assertTrue(
            (
                np.diff(missing["probability_complementary"][:600][order], axis=0) <= 0
            ).all()
        )

    def test_natural_missing_targets_and_gradients_are_ignored(self):
        observed = torch.tensor(
            [[1, 0, 1], [0, 1, 1], [1, 1, 0], [1, 0, 1]], dtype=torch.bool
        )
        kind = torch.tensor([0, 0, 1, 1])
        fm = torch.arange(3).repeat_interleave(4)
        hidden = data.sample_mask(
            observed, kind, 0.3, torch.Generator().manual_seed(11), "mixed"
        )
        self.assertFalse((hidden & ~observed).any())
        inference = (
            torch.arange(48, dtype=torch.float32).reshape(4, 12).requires_grad_()
        )
        target = torch.zeros_like(inference)
        original = sum(
            reconstruction_losses(inference, target, hidden, kind, fm).values()
        )
        target[~hidden[:, fm]] = float("nan")
        changed = sum(
            reconstruction_losses(inference, target, hidden, kind, fm).values()
        )
        torch.testing.assert_close(original, changed, rtol=0, atol=0)
        changed.backward()
        self.assertTrue((inference.grad[~hidden[:, fm]] == 0).all())
        self.assertTrue(torch.isfinite(inference.grad).all())

    def test_modality_heads_and_masked_encoder_inputs(self):
        torch.manual_seed(11)
        enc = encoder_from_config(
            self.g["feature_modality"], training.configuration("GWFM", 11)
        ).eval()
        z = torch.randn(4, 48, requires_grad=True)
        for modality in range(3):
            grad = torch.autograd.grad(
                enc.decode(z)[:, modality * 4 : (modality + 1) * 4].sum(),
                z,
                retain_graph=True,
            )[0]
            keep = torch.ones(48, dtype=torch.bool)
            keep[modality * 16 : (modality + 1) * 16] = False
            self.assertTrue((grad[:, keep] == 0).all())
        p = data.subset_graph(self.g, 1)
        mean, std = data.fit_standardizer(data.subset_graph(self.g, 0))
        p = data.standardize(p, mean, std)
        mask = data.sample_mask(
            p["observed"], p["node_type"], 0.3, torch.Generator().manual_seed(11)
        )
        visible = p["observed"] & ~mask
        with torch.no_grad():
            reference = enc.encode_graph(p, visible)[0]
            poisoned = dict(p, x=p["x"].clone())
            poisoned["x"][~visible[:, p["feature_modality"]]] = float("nan")
            torch.testing.assert_close(
                enc.encode_graph(poisoned, visible)[0], reference, rtol=0, atol=0
            )
        invalid = mask | ~p["observed"]
        with self.assertRaises(ValueError):
            VisibleGraphSSL(enc)(p, [invalid, invalid])

    def test_training_loader_never_opens_outcome_or_oracle_files(self):
        original = np.load
        opened = []

        def tracked(path, *args, **kwargs):
            opened.append(Path(path).name)
            self.assertNotIn(
                Path(path).name, ("targets.npz", "truth.npz", "missingness.npz")
            )
            return original(path, *args, **kwargs)

        with patch("gwfm.data.np.load", side_effect=tracked):
            g = data.load_graph(DATA)
        self.assertNotIn("y", g)
        self.assertEqual(
            set(opened), {"complementary.npz", "geometry.npz", "split.npz"}
        )

    def test_training_only_scalers(self):
        g = {k: v.clone() for k, v in self.g.items()}
        before = data.fit_standardizer(data.subset_graph(g, 0))
        g["x"][g["split"] != 0] = 1e20
        after = data.fit_standardizer(data.subset_graph(g, 0))
        for a, b in zip(before, after):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        tr = {"x": np.arange(24.0).reshape(8, 3), "y": np.arange(8.0)[:, None]}
        va = {"x": np.ones((4, 3)) * 1000, "y": np.ones((4, 1)) * -1000}
        with tempfile.TemporaryDirectory() as directory, threadpool_limits(limits=2):
            path = Path(directory) / "head.pt"
            mlp_predict(tr, va, va["x"], 11, path, hidden=4, max_epochs=2)
            saved = torch.load(path, weights_only=True)
            np.testing.assert_array_equal(saved["x_mean"], tr["x"].mean(0))
            np.testing.assert_array_equal(saved["y_mean"], tr["y"].mean(0))

    def test_reordered_targets_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            dest = Path(directory)
            for name in ("complementary.npz", "schema.json", "split.npz"):
                shutil.copy2(DATA / name, dest / name)
            target = dict(np.load(DATA / "targets.npz"))
            target = {k: v[::-1] for k, v in target.items()}
            np.savez(dest / "targets.npz", **target)
            with patch.object(protocol, "DATA", dest), self.assertRaisesRegex(
                ValueError, "Region IDs"
            ):
                protocol.dataset()

    def test_stale_config_rejected_without_overwriting_data(self):
        for key, value in (
            ("split_seed", -1),
            ("scenarios", ["complementary", "extra_scenario"]),
            ("epochs", 300),
        ):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as directory:
                dest = Path(directory)
                metadata = json.loads((DATA / "metadata.json").read_text())
                metadata["config"][key] = value
                original = json.dumps(metadata)
                (dest / "metadata.json").write_text(original)
                with patch.object(generate, "DATA", dest), self.assertRaisesRegex(
                    ValueError, "different configuration"
                ):
                    generate.run()
                self.assertEqual(list(dest.iterdir()), [dest / "metadata.json"])
                self.assertEqual((dest / "metadata.json").read_text(), original)

    def test_checkpoint_input_mutation_rejected(self):
        folder = REPOSITORY / "checkpoints/GWFM/seed11"
        cfg = training.configuration("GWFM", 11)
        training.check_saved_run(folder, cfg)
        with patch(
            "gwfm.training.digest", return_value="changed"
        ), self.assertRaisesRegex(ValueError, "inputs changed"):
            training.check_saved_run(folder, cfg)

    def test_unsupported_fixed_settings_rejected(self):
        cfg = json.loads((REPOSITORY / "configs/gwfm.json").read_text())
        cfg["block_dim"] = 8
        with patch(
            "gwfm.training.json.loads", return_value=cfg
        ), self.assertRaisesRegex(ValueError, "fixed protocol"):
            training.configuration("GWFM", 11)


if __name__ == "__main__":
    unittest.main()
