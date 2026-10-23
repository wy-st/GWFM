import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from gwfm import current
from gwfm.paths import DATA
from gwfm.current import target_input, inputs


class CurrentProtocolTests(unittest.TestCase):
    def test_only_regional_output_block_excluded(self):
        graph = dict(
            x=torch.arange(48, dtype=torch.float32).reshape(4, 12),
            observed=torch.tensor(
                [
                    [True, True, True],
                    [True, False, True],
                    [True, True, False],
                    [True, True, True],
                ]
            ),
            node_type=torch.tensor([0, 0, 1, 1]),
            feature_modality=torch.arange(3).repeat_interleave(4),
        )
        original = graph["x"].clone()
        task, visible = target_input(graph, 0)
        self.assertTrue(torch.equal(graph["x"], original))
        self.assertFalse(visible[:2, 0].any())
        self.assertTrue(torch.equal(visible[2:], graph["observed"][2:]))
        self.assertTrue(torch.equal(visible[:, 1:], graph["observed"][:, 1:]))
        expected = original.clone()
        expected[:2, :4] = 0
        self.assertTrue(torch.equal(task["x"], expected))
        poison = original.clone()
        poison[:2, :4] = float("nan")
        replay, _ = target_input(dict(graph, x=poison), 0)
        self.assertTrue(torch.equal(task["x"], replay["x"]))

    def test_original_test_cohorts(self):
        raw, _, split, cohorts = inputs()
        self.assertEqual([len(c["test"]) for c in cohorts], [110, 80, 89])
        for m, c in enumerate(cohorts):
            self.assertTrue((split[c["test"]] == 2).all())
            self.assertTrue(raw["observed"][c["test"], m].all())

    def test_every_observed_region_is_in_its_interpolation_cohort(self):
        raw, _, split, cohorts = inputs()
        for m, c in enumerate(cohorts):
            for code, role in enumerate(("train", "val", "test")):
                expected = np.flatnonzero(
                    (raw["node_type"] == 0) & (split == code) & raw["observed"][:, m]
                )
                np.testing.assert_array_equal(np.sort(c[role]), expected)
        self.assertIn(245, cohorts[0]["train"])
        self.assertIn(534, cohorts[0]["val"])

    def test_missing_or_duplicate_interpolation_target_is_rejected(self):
        original = json.loads(
            (current.REPOSITORY / "configs/interpolation_cohorts.json").read_text()
        )
        for duplicate in (False, True):
            with self.subTest(
                duplicate=duplicate
            ), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "configs").mkdir()
                manifest = json.loads(json.dumps(original))
                ids = manifest["cohorts"][0]["train"]
                ids.remove(245)
                if duplicate:
                    ids.append(ids[0])
                (root / "configs/interpolation_cohorts.json").write_text(
                    json.dumps(manifest)
                )
                with patch.object(current, "REPOSITORY", root):
                    with self.assertRaisesRegex(
                        ValueError, "Incomplete or invalid interpolation cohort"
                    ):
                        inputs()

    def test_cohorts_do_not_load_random_view_caches(self):
        real_load = np.load
        names = []

        def tracked(path, *args, **kwargs):
            names.append(Path(path).name)
            self.assertNotIn("contexts", Path(path).parts)
            return real_load(path, *args, **kwargs)

        with patch.object(np, "load", side_effect=tracked):
            inputs()
        self.assertEqual(set(names), {"complementary.npz", "split.npz"})

    def test_health_fitting_receives_only_train_and_validation_labels(self):
        raw, schema, split, cohorts = inputs()
        y = np.load(DATA / "targets.npz")["y"]
        tr, va, te = [np.flatnonzero(split[:600] == i) for i in (0, 1, 2)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bad = y.copy()
            bad[te] = float("nan")
            np.savez(root / "targets.npz", region_id=np.arange(600), y=bad)
            folder = root / "fixed_contexts/GWFM/seed11"
            folder.mkdir(parents=True)
            z = np.random.default_rng(11).normal(size=(600, 48)).astype(np.float32)
            np.savez(folder / "natural_embeddings.npz", embedding=z)
            calls = []

            def select(train, val, query, out, label):
                j = current.TARGETS.index(label)
                np.testing.assert_array_equal(train["y"], y[tr, j : j + 1])
                np.testing.assert_array_equal(val["y"], y[va, j : j + 1])
                np.testing.assert_array_equal(query, z)
                calls.append(label)
                return np.zeros((600, 1))

            with patch.object(
                current, "inputs", return_value=(raw, schema, split, cohorts)
            ), patch.object(current, "DATA", root), patch.object(
                current, "RESULTS", root
            ), patch.object(
                current, "MODELS", ("GWFM",)
            ), patch.object(
                current, "SEEDS", (11,)
            ), patch.object(
                current, "select_head", side_effect=select
            ), patch.object(
                current, "predict_mlp", return_value=np.zeros((600, 1))
            ):
                current.fit_health_neural()
            self.assertEqual(calls, current.TARGETS)


if __name__ == "__main__":
    unittest.main()
