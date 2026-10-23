"""Fit the requested 16D probes on the current frozen task embeddings."""

import json

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from gwfm.current import inputs
from gwfm.paths import DATA, SEEDS, TARGETS
from gwfm.fit_mlp import select_head
from gwfm.decoders import physical_inference, predict_mlp
from gwfm.utils import digest, save_json, score

from gwfm.paths import ROOT

CURRENT = ROOT / "results/current"
OUT = ROOT / "results/figure_probes"


def main():
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    OUT.mkdir(parents=True, exist_ok=True)
    raw, schema, split, cohorts = inputs()
    tr, va, te = [np.flatnonzero(split[:600] == k) for k in (0, 1, 2)]
    with np.load(DATA / "targets.npz") as f:
        np.testing.assert_array_equal(f["region_id"], raw["node_id"][:600])
        train_y, val_y = f["y"][tr].copy(), f["y"][va].copy()
    source_files = [
        p
        for base in ("data", "checkpoints", "results/current")
        for p in (ROOT / base).rglob("*")
        if p.is_file()
    ]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in source_files}
    save_json(OUT / "inputs_before.json", hashes)
    heads = 0
    with threadpool_limits(limits=2):
        for model in ("PDFM", "GWFM"):
            for seed in SEEDS:
                source = CURRENT / "fixed_contexts" / model / f"seed{seed}"
                with np.load(source / "natural_embeddings.npz") as natural:
                    np.testing.assert_array_equal(
                        natural["region_id"], raw["node_id"][:600]
                    )
                    embedding = natural["embedding"]
                for block in (1, 2, 3):
                    columns = slice((block - 1) * 16, block * 16)
                    out = OUT / "runs" / f"{model}_Modal-{block}_seed{seed}"
                    out.mkdir(parents=True, exist_ok=True)
                    payload = {}
                    for m, cohort in enumerate(cohorts):
                        with np.load(source / f"modality{m}.npz") as c:
                            assert "test_y" not in c
                            for role, code in (("train", 0), ("val", 1), ("test", 2)):
                                np.testing.assert_array_equal(
                                    c[role + "_node_id"], cohort[role]
                                )
                                assert (split[cohort[role]] == code).all()
                                assert not c[role + "_visible"][:, m].any()
                            train = dict(x=c["train_x"][:, columns], y=c["train_y"])
                            val = dict(x=c["val_x"][:, columns], y=c["val_y"])
                            value = select_head(
                                train, val, c["test_x"][:, columns], out, f"modality{m}"
                            )
                            np.testing.assert_array_equal(
                                value,
                                predict_mlp(
                                    out / f"modality{m}.pt", c["test_x"][:, columns]
                                ),
                            )
                        value = physical_inference(
                            value, schema["features"][m * 4 : (m + 1) * 4]
                        )
                        for k in range(4):
                            payload[f"test_{m*4+k}_node_id"] = cohort["test"]
                            payload[f"test_{m*4+k}_inference"] = value[:, k]
                        heads += 1
                    x = embedding[:, columns]
                    values = []
                    for j, target in enumerate(TARGETS):
                        value = select_head(
                            dict(x=x[tr], y=train_y[:, j : j + 1]),
                            dict(x=x[va], y=val_y[:, j : j + 1]),
                            x,
                            out,
                            target,
                        )
                        np.testing.assert_array_equal(
                            value, predict_mlp(out / f"{target}.pt", x)
                        )
                        values.append(value[:, 0])
                        heads += 1
                    payload["health_inference"] = np.column_stack(values)
                    payload["health_region_id"] = np.arange(600)
                    np.savez_compressed(out / "inferences.npz", **payload)
                    print("Frozen probe:", model, seed, f"Modal-{block}", flush=True)
    assert heads == 90
    frozen = {
        str(p.relative_to(OUT)): digest(p)
        for p in (OUT / "runs").rglob("*")
        if p.is_file()
    }
    save_json(OUT / "frozen_before_test.json", frozen)
    targets = np.load(DATA / "targets.npz")["y"]
    scales = np.load(DATA / "standardizer.npz")["scale"][0]
    attributes, health, details = [], [], []
    for model in ("PDFM", "GWFM"):
        for seed in SEEDS:
            for block in (1, 2, 3):
                method = f"{model} (Modal-{block})"
                with np.load(
                    OUT
                    / "runs"
                    / f"{model}_Modal-{block}_seed{seed}"
                    / "inferences.npz"
                ) as f:
                    for j, feature in enumerate(schema["features"]):
                        ids, value = f[f"test_{j}_node_id"], f[f"test_{j}_inference"]
                        np.testing.assert_array_equal(ids, cohorts[j // 4]["test"])
                        assert (split[ids] == 2).all() and raw["observed"][
                            ids, j // 4
                        ].all()
                        y = raw["x"][ids, j]
                        attributes.append(
                            dict(
                                method=method,
                                seed=seed,
                                feature=feature,
                                modality=("climate", "pollution", "sociodemographic")[
                                    j // 4
                                ],
                                **score(y, value, scales[j]),
                            )
                        )
                        details.extend(
                            dict(
                                method=method,
                                seed=seed,
                                task="interpolation",
                                feature=feature,
                                region_id=int(i),
                                true_value=float(a),
                                inference=float(b),
                            )
                            for i, a, b in zip(ids, y, value)
                        )
                    np.testing.assert_array_equal(f["health_region_id"], np.arange(600))
                    for j, target in enumerate(TARGETS):
                        y, value = targets[te, j], f["health_inference"][te, j]
                        health.append(
                            dict(
                                method=method,
                                seed=seed,
                                target=target,
                                **score(y, value, targets[tr, j].std()),
                            )
                        )
                        details.extend(
                            dict(
                                method=method,
                                seed=seed,
                                task="inference",
                                feature=target,
                                region_id=int(i),
                                true_value=float(a),
                                inference=float(b),
                            )
                            for i, a, b in zip(te, y, value)
                        )
    pd.DataFrame(attributes).to_csv(OUT / "feature_metrics_by_seed.csv", index=False)
    pd.DataFrame(health).to_csv(OUT / "health_metrics_by_seed.csv", index=False)
    pd.DataFrame(details).to_csv(OUT / "test_inferences.csv", index=False)
    selections = list((OUT / "runs").rglob("*_selection.json"))
    assert len(selections) == 90
    for path in selections:
        record = json.loads(path.read_text())
        assert len(record["candidates"]) == 6 and record["input_dim"] == 16
    for name, value in hashes.items():
        assert digest(ROOT / name) == value, name
    for name, value in frozen.items():
        assert digest(OUT / name) == value, name
    save_json(
        OUT / "verification.json",
        dict(
            status="passed",
            selected_heads=90,
            candidate_fits=540,
            embedding_width=16,
            encoder_retrained=False,
            additional_masks=False,
            metric_rows=len(attributes) + len(health),
            source_files_unchanged=len(hashes),
            test_scored_after_freezing=True,
            selected_heads_replayed=True,
        ),
    )
    print("All 90 block heads and 252 metric rows verified.", flush=True)


if __name__ == "__main__":
    main()
