import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from threadpoolctl import threadpool_limits

from gwfm import data, training
from gwfm.models.model import encoder_from_config, masked_pdfm_loss
from gwfm.paths import DATA

torch.set_num_threads(2)
torch.use_deterministic_algorithms(True)


class MaskedPDFMTests(unittest.TestCase):
    def test_single_forward_and_only_hidden_observed_targets_have_gradients(self):
        observed = torch.tensor(
            [[1, 0, 1], [0, 1, 1], [1, 1, 0], [1, 0, 1]], dtype=torch.bool
        )
        fm = torch.arange(3).repeat_interleave(4)
        mask = torch.tensor(
            [[1, 0, 0], [0, 1, 1], [0, 1, 0], [1, 0, 1]], dtype=torch.bool
        )
        graph = dict(
            observed=observed,
            feature_modality=fm,
            node_type=torch.tensor([0, 0, 1, 1]),
            x=torch.zeros(4, 12),
        )
        graph["x"][~mask[:, fm]] = float("nan")
        inference = torch.ones(4, 12, requires_grad=True)

        class RecordingEncoder:
            calls = []

            def encode_graph(self, graph, visible):
                self.calls.append(visible.clone())
                return inference, inference

        encoder = RecordingEncoder()
        loss = masked_pdfm_loss(encoder, graph, mask)
        self.assertEqual(len(encoder.calls), 1)
        self.assertTrue(torch.equal(encoder.calls[0], observed & ~mask))
        self.assertEqual(
            float(loss.detach()), 1.0
        )  # Each scale's mean Huber loss is 0.5.
        loss.backward()
        self.assertTrue((inference.grad[~mask[:, fm]] == 0).all())
        self.assertTrue((inference.grad[mask[:, fm]] > 0).all())
        with self.assertRaises(ValueError):
            masked_pdfm_loss(encoder, graph, mask | ~observed)

    def test_real_encoder_cannot_see_ssl_targets_or_natural_gaps(self):
        cfg = training.configuration("PDFM", 11)
        g = data.load_graph(DATA, distance_bandwidths=cfg["distance_bandwidths"])
        mean, std = data.fit_standardizer(data.subset_graph(g, 0))
        part = data.standardize(data.subset_graph(g, 1), mean, std)
        mask = data.sample_mask(
            part["observed"],
            part["node_type"],
            cfg["mask_rate"],
            torch.Generator().manual_seed(30012),
            cfg["mask_strategy"],
        )
        visible = part["observed"] & ~mask
        enc = encoder_from_config(g["feature_modality"], cfg).eval()
        poisoned = dict(part, x=part["x"].clone())
        poisoned["x"][~visible[:, part["feature_modality"]]] = float("nan")
        with torch.no_grad():
            z, output = enc.encode_graph(part, visible)
            zz, other = enc.encode_graph(poisoned, visible)
        torch.testing.assert_close(z, zz, rtol=0, atol=0)
        torch.testing.assert_close(output, other, rtol=0, atol=0)

    def test_two_epoch_training_uses_one_view_per_epoch_and_train_partition_only(self):
        cfg = training.configuration("PDFM", 11, 2)
        gwfm = training.configuration("GWFM", 11, 2)
        for name in ("mask_strategy", "mask_rate"):
            self.assertEqual(cfg[name], gwfm[name])
        real_loss, real_sample, real_evaluate = (
            training.masked_pdfm_loss,
            training.sample_mask,
            training.reconstruction_evaluate,
        )
        losses, samples, validations = [], [], []

        def record_loss(encoder, graph, mask):
            self.assertTrue((graph["split"] == 0).all())
            losses.append(mask.clone())
            return real_loss(encoder, graph, mask)

        def record_sample(*args, **kwargs):
            result = real_sample(*args, **kwargs)
            samples.append(result.clone())
            return result

        def record_validation(encoder, graph, mask):
            self.assertTrue((graph["split"] == 1).all())
            validations.append(mask.clone())
            return real_evaluate(encoder, graph, mask)

        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            with patch.object(
                training, "encoder_dir", return_value=folder
            ), patch.object(
                training, "masked_pdfm_loss", side_effect=record_loss
            ), patch.object(
                training, "sample_mask", side_effect=record_sample
            ), patch.object(
                training, "reconstruction_evaluate", side_effect=record_validation
            ), threadpool_limits(
                limits=2
            ), contextlib.redirect_stdout(
                io.StringIO()
            ):
                training.train_pdfm(11, epochs=2)
            metrics = json.loads((folder / "metrics.json").read_text())
            self.assertEqual(metrics["training_mask_views_per_epoch"], 1)
        self.assertEqual(len(losses), 2)
        self.assertEqual(
            len(samples), 3
        )  # One fixed validation mask, then one per epoch.
        self.assertEqual(len(validations), 2)
        self.assertTrue(torch.equal(validations[0], validations[1]))
        self.assertFalse(torch.equal(losses[0], losses[1]))


if __name__ == "__main__":
    unittest.main()
