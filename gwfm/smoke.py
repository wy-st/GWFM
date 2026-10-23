"""Small generated-data → SSL → exported embedding → downstream R² self-check.

This short run validates execution only; its scores are not report results.
"""

import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.work_dir.resolve()
    if root.exists():
        parser.error("Use a new empty output directory")
    os.environ["GWFM_WORK_DIR"] = str(root)
    import numpy as np
    import torch
    from threadpoolctl import threadpool_limits
    from .paths import REPOSITORY, DATA, encoder_dir
    from .synthetic.data import generate
    from .training import train_gwfm, train_pdfm, extract_embeddings
    from .data import load_graph, subset_graph, standardize
    from .current import target_input
    from .ablation.model import make_encoder
    from .decoders import mlp_predict
    from .baselines.knn import coordinate_knn
    from .utils import save_json, score

    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    config = json.loads((REPOSITORY / "configs/dataset.json").read_text())
    config.update(n_regions=200, n_subregions=1600, knot_grid=11)
    root.mkdir(parents=True)
    save_json(root / "smoke_config.json", config)
    with threadpool_limits(limits=2):
        generate(config["data_seed"], config, str(DATA))
        g = load_graph(DATA)
        nr = int((g["node_type"] == 0).sum())
        split = g["split"][:nr].numpy()
        tr, va, te = [np.flatnonzero(split == i) for i in (0, 1, 2)]
        assert [len(tr), len(va), len(te)] == [140, 20, 40]
        for code in (0, 1, 2):
            part = subset_graph(g, code)
            assert (part["split"][part["edge_index"]] == code).all()
        with np.load(DATA / "targets.npz") as target:
            np.testing.assert_array_equal(
                target["region_id"], g["node_id"][:nr].numpy()
            )
            y = target["y"]
        records = []
        for name, train in [("GWFM", train_gwfm), ("PDFM", train_pdfm)]:
            train(11, 2)
            extract_embeddings(name, 11)
            with np.load(encoder_dir(name, 11) / "region_embeddings.npz") as exported:
                np.testing.assert_array_equal(
                    exported["node_id"], g["node_id"][:nr].numpy()
                )
                z = exported["embedding"]
            assert z.shape == (nr, 48) and np.isfinite(z).all()
            pred, selection = mlp_predict(
                dict(x=z[tr], y=y[tr]),
                dict(x=z[va], y=y[va]),
                z[te],
                11,
                root / f"{name}_head.pt",
                hidden=8,
                max_epochs=3,
            )
            np.savez_compressed(
                root / f"{name}_inferences.npz",
                region_id=te,
                inference=pred,
                target=y[te],
            )
            for j in range(2):
                metric = score(y[te, j], pred[:, j], y[tr, j].std())
                assert np.isfinite(metric["r2"])
                records.append(dict(method=name, target=j, **metric))
            saved = torch.load(encoder_dir(name, 11) / "model.pt", weights_only=True)
            g = load_graph(
                DATA, distance_bandwidths=saved["config"]["distance_bandwidths"]
            )
            enc = (
                make_encoder(g["feature_modality"], saved["config"])
                .eval()
                .requires_grad_(False)
            )
            enc.load_state_dict(saved["encoder_state"])
            for m in range(3):
                contexts = []
                for code in (0, 1, 2):
                    part = standardize(
                        subset_graph(g, code), saved["mean"], saved["std"]
                    )
                    if m == 0:
                        with torch.no_grad():
                            natural = enc.encode_graph(part, part["observed"])[0]
                        regions = part["node_type"] == 0
                        np.testing.assert_array_equal(
                            natural[regions].numpy(), z[part["node_id"][regions]]
                        )
                    task, visible = target_input(part, m)
                    ids = part["node_id"]
                    keep = (part["node_type"] == 0) & part["observed"][:, m]
                    assert not visible[part["node_type"] == 0, m].any()
                    with torch.no_grad():
                        embedding = enc.encode_graph(task, visible)[0][keep].numpy()
                    rows = ids[keep].numpy()
                    contexts.append(
                        dict(x=embedding, y=g["x"][rows, m * 4 : (m + 1) * 4].numpy())
                    )
                train_context, val_context, test_context = contexts
                value, _ = mlp_predict(
                    train_context,
                    val_context,
                    test_context["x"],
                    11,
                    root / f"{name}_interpolation_m{m}.pt",
                    hidden=8,
                    max_epochs=3,
                )
                for j in range(4):
                    metric = score(
                        test_context["y"][:, j],
                        value[:, j],
                        train_context["y"][:, j].std(),
                    )
                    assert np.isfinite(metric["r2"])
                    records.append(
                        dict(
                            method=name,
                            task="interpolation",
                            target=m * 4 + j,
                            **metric,
                        )
                    )
        xy = g["pos"][:nr].numpy()
        pred = coordinate_knn(
            xy[tr], y[tr], np.ones(len(tr), bool), xy[te], 8, y[tr].mean(0)
        )
        for j in range(2):
            records.append(
                dict(
                    method="KNN",
                    target=j,
                    **score(y[te, j], pred[:, j], y[tr, j].std()),
                )
            )
        save_json(
            root / "smoke_results.json",
            dict(
                status="passed",
                report_results=False,
                encoder_epochs=2,
                decoder_epochs=3,
                region_counts=[140, 20, 40],
                results=records,
            ),
        )
        print(json.dumps(records, indent=2), flush=True)


if __name__ == "__main__":
    main()
