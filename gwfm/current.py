"""Fixed attribute test cohorts, no random downstream views, explicit label exclusion."""

import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import joblib

from .paths import (
    ROOT as PROJECT,
    REPOSITORY,
    DATA,
    RESULTS as REPORT_RESULTS,
    SEEDS,
    TARGETS,
    encoder_dir as main_encoder_dir,
)
from .data import load_graph, subset_graph, standardize
from .ablation.model import make_encoder
from .fit_mlp import select_head
from .decoders import physical_inference
from .protocol import CANDIDATES
from .baselines.knn import coordinate_knn
from .baselines.idw import interpolate
from .baselines.gpr import fit_kernel, coordinate_gpr
from .contexts import MODALITIES
from .utils import digest, save_json, write_csv, score, aggregate, summarize
from .decoders import predict_mlp
from .protocol import select_embedding


ROOT = REPORT_RESULTS / "current"
RESULTS = ROOT
MODELS = ("GWFM", "PDFM", "no_weighting", "no_modality", "no_prototype")


def encoder_dir(model, seed):
    if model in ("GWFM", "PDFM"):
        return main_encoder_dir(model, seed)
    return PROJECT / "checkpoints" / "ablations" / model / f"seed{seed}"


def inputs():
    """Load and validate all observed regional cohorts against the fixed split manifest."""
    raw = dict(np.load(DATA / "complementary.npz"))
    partition = np.load(DATA / "split.npz")
    np.testing.assert_array_equal(partition["region_id"], raw["node_id"][:600])
    split = np.r_[partition["split"], partition["split"][raw["parent"][600:]]]
    schema = json.loads((DATA / "schema.json").read_text())
    manifest = json.loads(
        (REPOSITORY / "configs/interpolation_cohorts.json").read_text()
    )
    if manifest["protocol"] != "all_observed_target_excluded_interpolation":
        raise ValueError(
            "Interpolation requires complete naturally observed regional cohorts"
        )
    for name, expected in manifest["array_sha256"].items():
        filename, key = name.split(":")
        values = raw[key] if filename == "complementary.npz" else partition[key]
        if hashlib.sha256(values.tobytes()).hexdigest() != expected:
            raise ValueError(
                "Fixed interpolation cohorts require the prescribed dataset and split"
            )
    cohorts = [
        {role: np.asarray(ids, dtype=np.int64) for role, ids in cohort.items()}
        for cohort in manifest["cohorts"]
    ]
    if len(cohorts) != raw["observed"].shape[1]:
        raise ValueError("Interpolation must include every modality")
    for m, cohort in enumerate(cohorts):
        for role, code in [("train", 0), ("val", 1), ("test", 2)]:
            ids = cohort[role]
            expected = np.flatnonzero(
                (raw["node_type"] == 0) & (split == code) & raw["observed"][:, m]
            )
            if not np.array_equal(np.sort(ids), expected):
                raise ValueError(
                    f"Incomplete or invalid interpolation cohort: modality={m}, split={role}; "
                    f"missing={np.setdiff1d(expected, ids).tolist()}, "
                    f"extra={np.setdiff1d(ids, expected).tolist()}"
                )
        assert not np.intersect1d(cohort["train"], cohort["test"]).size
    return raw, schema, split, cohorts


def target_input(graph, modality):
    """Remove the regional output-label block before any encoder operation."""
    regions = graph["node_type"] == 0
    cols = graph["feature_modality"] == modality
    visible = graph["observed"].clone()
    visible[regions, modality] = False
    x = graph["x"].clone()
    x[regions[:, None] & cols[None, :]] = 0.0
    assert torch.equal(visible[~regions], graph["observed"][~regions])
    assert torch.equal(
        visible[:, np.arange(3) != modality],
        graph["observed"][:, np.arange(3) != modality],
    )
    return dict(graph, x=x), visible


@torch.no_grad()
def build_contexts():
    """Encode each split after excluding the regional target modality; save no test labels."""
    (ROOT / "logs").mkdir(parents=True, exist_ok=True)
    raw, schema, split, cohorts = inputs()
    audit = []
    for model in MODELS:
        for seed in SEEDS:
            saved = torch.load(encoder_dir(model, seed) / "model.pt", weights_only=True)
            graph = load_graph(
                DATA, distance_bandwidths=saved["config"]["distance_bandwidths"]
            )
            enc = (
                make_encoder(graph["feature_modality"], saved["config"])
                .eval()
                .requires_grad_(False)
            )
            enc.load_state_dict(saved["encoder_state"])
            folder = RESULTS / "fixed_contexts" / model / f"seed{seed}"
            folder.mkdir(parents=True, exist_ok=True)
            for m, cohort in enumerate(cohorts):
                payload = {}
                for code, role in enumerate(("train", "val", "test")):
                    part = standardize(
                        subset_graph(graph, code), saved["mean"], saved["std"]
                    )
                    task_graph, visible = target_input(part, m)
                    embedding = enc.encode_graph(task_graph, visible)[0].numpy()
                    ids = cohort[role]
                    lookup = {int(n): i for i, n in enumerate(part["node_id"])}
                    rows = np.array([lookup[int(n)] for n in ids])
                    assert not visible[rows, m].any()
                    payload[role + "_node_id"] = ids
                    payload[role + "_x"] = embedding[rows]
                    payload[role + "_visible"] = visible[rows].numpy()
                    if role != "test":
                        payload[role + "_y"] = raw["x"][ids, m * 4 : (m + 1) * 4]
                    # Changing every regional target value must not change any task embedding.
                    poison = part["x"].clone()
                    selected = (part["node_type"] == 0)[:, None] & (
                        part["feature_modality"] == m
                    )[None, :]
                    poison[selected] = float("nan")
                    changed_graph, changed_visible = target_input(
                        dict(part, x=poison), m
                    )
                    replay = enc.encode_graph(changed_graph, changed_visible)[0].numpy()
                    np.testing.assert_array_equal(embedding, replay)
                    audit.append(
                        dict(
                            model=model,
                            seed=seed,
                            modality=m,
                            partition=role,
                            query_count=len(ids),
                            regional_target_input_cells=0,
                            target_poison_embedding_max_abs=0.0,
                        )
                    )
                np.savez_compressed(folder / f"modality{m}.npz", **payload)
            print("Fixed target-excluded contexts:", model, seed, flush=True)
    save_json(ROOT / "logs/target_exclusion_checks.json", audit)


def fit_neural():
    """Select one four-output MLP per modality on validation data, then freeze test predictions."""
    _, schema, _, cohorts = inputs()
    for model in MODELS:
        for seed in SEEDS:
            for block, columns in {"full": slice(0, 48)}.items():
                method = model if block == "full" else f"{model} ({block})"
                folder = RESULTS / "runs" / f"{method}_seed{seed}"
                folder.mkdir(parents=True, exist_ok=True)
                payload = {}
                for m in range(3):
                    ctx = np.load(
                        RESULTS
                        / "fixed_contexts"
                        / model
                        / f"seed{seed}"
                        / f"modality{m}.npz"
                    )
                    inference = select_head(
                        dict(x=ctx["train_x"][:, columns], y=ctx["train_y"]),
                        dict(x=ctx["val_x"][:, columns], y=ctx["val_y"]),
                        ctx["test_x"][:, columns],
                        folder,
                        f"level0_modality{m}",
                    )
                    inference = physical_inference(
                        inference, schema["features"][m * 4 : (m + 1) * 4]
                    )
                    for k in range(4):
                        j = m * 4 + k
                        payload[f"test_{j}_node_id"] = cohorts[m]["test"]
                        payload[f"test_{j}_inference"] = inference[:, k]
                np.savez_compressed(folder / "inferences.npz", **payload)
                print("Attribute inferences frozen:", method, seed, flush=True)


def simple_predict(method, cfg, raw, donors, queries, j):
    available = np.ones(len(donors), dtype=bool)
    fallback = raw["x"][donors, j].mean()
    if method == "KNN":
        return coordinate_knn(
            raw["pos"][donors],
            raw["x"][donors, j : j + 1],
            available,
            raw["pos"][queries],
            cfg["neighbors"],
            np.array([fallback]),
        )[:, 0]
    return interpolate(
        raw["pos"][donors],
        raw["x"][donors, j],
        available,
        raw["pos"][queries],
        "idw",
        cfg["neighbors"],
        power=cfg["power"],
        fallback=fallback,
    )


def fit_spatial():
    """Tune coordinate-only baselines with training donors and validation targets."""
    raw, schema, split, cohorts = inputs()
    for method in ("KNN", "IDW", "GPR"):
        folder = RESULTS / "runs" / f"{method}_seed0"
        folder.mkdir(parents=True, exist_ok=True)
        payload, selected, kernels, candidates = {}, {}, {}, []
        for j, name in enumerate(schema["features"]):
            tr, va, te = [cohorts[j // 4][role] for role in ("train", "val", "test")]
            assert (
                (split[tr] == 0).all()
                and (split[va] == 1).all()
                and (split[te] == 2).all()
            )
            assert not np.intersect1d(tr, np.r_[va, te]).size
            if method == "GPR":
                trials = []
                for family in ("RBF", "Matern32"):
                    kernel, info = fit_kernel(
                        raw["pos"][tr], raw["x"][tr, j], np.ones(len(tr), bool), family
                    )
                    pred, _ = coordinate_gpr(
                        raw["pos"][tr],
                        raw["x"][tr, j],
                        np.ones(len(tr), bool),
                        raw["pos"][va],
                        kernel,
                        info["mean"],
                        info["scale"],
                    )
                    pred = physical_inference(pred[:, None], [name])[:, 0]
                    loss = float(np.mean((pred - raw["x"][va, j]) ** 2))
                    trials.append((loss, kernel, info))
                    candidates.append(dict(feature=name, validation_mse=loss, **info))
                loss, kernel, info = min(trials, key=lambda x: x[0])
                kernels[name] = (kernel, info)
                selected[name] = dict(validation_mse=loss, **info)
                pred, _ = coordinate_gpr(
                    raw["pos"][tr],
                    raw["x"][tr, j],
                    np.ones(len(tr), bool),
                    raw["pos"][te],
                    kernel,
                    info["mean"],
                    info["scale"],
                )
                pred = physical_inference(pred[:, None], [name])[:, 0]
            else:
                grid = CANDIDATES[method]
                losses = [
                    float(
                        np.mean(
                            (
                                simple_predict(method, c, raw, tr, va, j)
                                - raw["x"][va, j]
                            )
                            ** 2
                        )
                    )
                    for c in grid
                ]
                winner = int(np.argmin(losses))
                selected[name] = dict(
                    config=grid[winner], validation_mse=losses[winner]
                )
                candidates.extend(
                    dict(feature=name, config=json.dumps(c), validation_mse=loss)
                    for c, loss in zip(grid, losses)
                )
                pred = simple_predict(method, grid[winner], raw, tr, te, j)
            payload[f"test_{j}_inference"] = pred
            payload[f"test_{j}_node_id"] = te
            payload[f"donor_{j}_node_id"] = tr
        np.savez_compressed(folder / "inferences.npz", **payload)
        save_json(folder / "selection.json", selected)
        write_csv(folder / "validation_candidates.csv", candidates)
        if kernels:
            joblib.dump(kernels, folder / "selected_kernels.joblib")
        print("Spatial attribute inferences frozen:", method, flush=True)


def method_runs():
    for method in ("KNN", "IDW", "GPR"):
        yield method, (0,)
    for model in MODELS:
        yield model, SEEDS


def freeze():
    files = [
        p
        for directory in ("fixed_contexts", "runs", "inference")
        for p in (RESULTS / directory).rglob("*")
        if p.is_file()
    ]
    save_json(
        ROOT / "logs/inference_freeze.json",
        {str(p.relative_to(ROOT)): digest(p) for p in files},
    )


def evaluate():
    """Score frozen interpolation predictions on exactly the registered observed test regions."""
    raw, schema, split, cohorts = inputs()
    scales = np.load(DATA / "standardizer.npz")["scale"][0]
    rows, details = [], []
    for method, seeds in method_runs():
        for seed in seeds:
            saved = np.load(
                RESULTS / "runs" / f"{method}_seed{seed}" / "inferences.npz"
            )
            for j, name in enumerate(schema["features"]):
                ids = saved[f"test_{j}_node_id"]
                inference = saved[f"test_{j}_inference"]
                np.testing.assert_array_equal(ids, cohorts[j // 4]["test"])
                assert (split[ids] == 2).all() and raw["observed"][ids, j // 4].all()
                target = raw["x"][ids, j]
                rows.append(
                    dict(
                        method=method,
                        seed=seed,
                        epochs=0,
                        scenario="fixed_test",
                        pattern="target_excluded",
                        missing_count="any",
                        level="region",
                        modality=MODALITIES[j // 4],
                        feature=name,
                        unit=schema["units"][j],
                        n_nodes=len(ids),
                        n_families=len(ids),
                        **score(target, inference, scales[j]),
                    )
                )
                details.extend(
                    dict(
                        method=method,
                        seed=seed,
                        feature=name,
                        region_id=int(n),
                        true_value=float(y),
                        inference=float(p),
                    )
                    for n, y, p in zip(ids, target, inference)
                )
    macro = aggregate(rows)
    for name, records in [
        ("feature_metrics_by_seed", rows),
        ("macro_metrics_by_seed", macro),
        ("feature_metrics_summary", summarize(rows, True)),
        ("macro_metrics_summary", summarize(macro)),
    ]:
        write_csv(RESULTS / f"{name}.csv", records)
    pd.DataFrame(details).to_csv(RESULTS / "test_inferences.csv", index=False)
    print(
        pd.DataFrame(summarize(macro))
        .query("modality == 'all'")[["method", "r2_mean", "r2_sd"]]
        .to_string(index=False),
        flush=True,
    )


@torch.no_grad()
def natural_contexts():
    """Replay natural-input embeddings independently for each train/validation/test graph."""
    checks = []
    for model in MODELS:
        for seed in SEEDS:
            folder = encoder_dir(model, seed)
            saved = torch.load(folder / "model.pt", weights_only=True)
            graph = load_graph(
                DATA, distance_bandwidths=saved["config"]["distance_bandwidths"]
            )
            enc = (
                make_encoder(graph["feature_modality"], saved["config"])
                .eval()
                .requires_grad_(False)
            )
            enc.load_state_dict(saved["encoder_state"], strict=True)
            original = np.load(folder / "embeddings.npz")
            np.testing.assert_array_equal(original["node_id"], graph["node_id"].numpy())
            z = np.empty_like(original["embedding"])
            for code in range(3):
                part = standardize(
                    subset_graph(graph, code), saved["mean"], saved["std"]
                )
                rows = (graph["split"] == code).numpy()
                z[rows] = enc.encode_graph(part, part["observed"])[0].numpy()
                np.testing.assert_array_equal(z[rows], original["embedding"][rows])
                checks.append(
                    dict(
                        model=model,
                        seed=seed,
                        partition=code,
                        artificial_exclusions=0,
                        max_abs_difference=0.0,
                    )
                )
            out = (
                RESULTS
                / "fixed_contexts"
                / model
                / f"seed{seed}"
                / "natural_embeddings.npz"
            )
            np.savez_compressed(
                out, embedding=z[:600], region_id=graph["node_id"][:600].numpy()
            )
    save_json(ROOT / "logs/natural_embedding_checks.json", checks)


def fit_health_neural():
    """Fit separate health-target heads on frozen 48D regional embeddings."""
    raw, _, split, _ = inputs()
    tr, va = [np.flatnonzero(split[:600] == i) for i in (0, 1)]
    with np.load(DATA / "targets.npz") as targets:
        np.testing.assert_array_equal(targets["region_id"], raw["node_id"][:600])
        train_y, val_y = targets["y"][tr].copy(), targets["y"][va].copy()
    for model in MODELS:
        for seed in SEEDS:
            x = np.load(
                RESULTS
                / "fixed_contexts"
                / model
                / f"seed{seed}"
                / "natural_embeddings.npz"
            )["embedding"]
            if model in ("GWFM", "PDFM"):
                x = select_embedding(x)
            out = RESULTS / "inference/runs" / f"{model}_seed{seed}"
            out.mkdir(parents=True, exist_ok=True)
            values = []
            for j, name in enumerate(TARGETS):
                value = select_head(
                    dict(x=x[tr], y=train_y[:, j : j + 1]),
                    dict(x=x[va], y=val_y[:, j : j + 1]),
                    x,
                    out,
                    name,
                )
                np.testing.assert_array_equal(value, predict_mlp(out / f"{name}.pt", x))
                values.append(value[:, 0])
            np.savez_compressed(
                out / "inferences.npz",
                inference=np.column_stack(values),
                region_id=np.arange(600),
            )
            print("Health inference heads:", model, seed, flush=True)


def fit_health_spatial():
    from unittest.mock import patch
    from . import fit_spatial as spatial

    raw, _, split, _ = inputs()
    tr, va = [np.flatnonzero(split[:600] == i) for i in (0, 1)]
    with np.load(DATA / "targets.npz") as targets:
        np.testing.assert_array_equal(targets["region_id"], raw["node_id"][:600])
        train_y, val_y = targets["y"][tr].copy(), targets["y"][va].copy()
    with patch.object(spatial, "ROOT", RESULTS):
        spatial.fit_outcomes(raw, split)
    out = RESULTS / "inference/runs/GPR_seed0"
    out.mkdir(parents=True, exist_ok=True)
    values, selections, kernels = [], [], {}
    for j, name in enumerate(TARGETS):
        trials = []
        for family in ("RBF", "Matern32"):
            kernel, info = fit_kernel(
                raw["pos"][tr], train_y[:, j], np.ones(len(tr), bool), family
            )
            p, _ = coordinate_gpr(
                raw["pos"][tr],
                train_y[:, j],
                np.ones(len(tr), bool),
                raw["pos"][va],
                kernel,
                info["mean"],
                info["scale"],
            )
            trials.append((float(np.mean((p - val_y[:, j]) ** 2)), kernel, info))
        loss, kernel, info = min(trials, key=lambda x: x[0])
        p, _ = coordinate_gpr(
            raw["pos"][tr],
            train_y[:, j],
            np.ones(len(tr), bool),
            raw["pos"][:600],
            kernel,
            info["mean"],
            info["scale"],
        )
        selections.append(
            dict(
                target=name,
                validation_mse=loss,
                selected=info,
                candidates=[dict(validation_mse=t[0], **t[2]) for t in trials],
            )
        )
        kernels[name] = (kernel, info)
        values.append(p)
    np.savez_compressed(
        out / "inferences.npz",
        inference=np.column_stack(values),
        region_id=np.arange(600),
        donor_region_id=tr,
    )
    save_json(out / "selection.json", selections)
    joblib.dump(kernels, out / "kernels.joblib")


def evaluate_health():
    raw, _, split, _ = inputs()
    tr, te = [np.flatnonzero(split[:600] == i) for i in (0, 2)]
    with np.load(DATA / "targets.npz") as targets:
        np.testing.assert_array_equal(targets["region_id"], raw["node_id"][:600])
        y = targets["y"]
    rows, records = [], []
    for method, seeds in method_runs():
        for seed in seeds:
            values = np.load(
                RESULTS / "inference/runs" / f"{method}_seed{seed}" / "inferences.npz"
            )
            np.testing.assert_array_equal(values["region_id"], np.arange(600))
            if method in ("KNN", "IDW", "GPR"):
                np.testing.assert_array_equal(values["donor_region_id"], tr)
            for j, name in enumerate(TARGETS):
                p = values["inference"][:, j]
                rows.append(
                    dict(
                        method=method,
                        seed=seed,
                        target=name,
                        **score(y[te, j], p[te], y[tr, j].std()),
                    )
                )
                records.extend(
                    dict(
                        method=method,
                        seed=seed,
                        target=name,
                        region_id=int(i),
                        true_value=float(y[i, j]),
                        inference=float(p[i]),
                    )
                    for i in te
                )
    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "inference/test_metrics.csv", index=False)
    pd.DataFrame(records).to_csv(RESULTS / "inference/test_inferences.csv", index=False)
    summary = (
        frame.groupby(["method", "target"], sort=False)
        .agg(r2_mean=("r2", "mean"), r2_sd=("r2", "std"), seeds=("seed", "size"))
        .reset_index()
    )
    summary.to_csv(RESULTS / "inference/test_summary.csv", index=False)
    print(summary.to_string(index=False), flush=True)


def verify_current():
    """Replay all selected full-embedding heads and recompute the 252 stored R-squared values."""
    raw, schema, split, cohorts = inputs()
    targets = np.load(DATA / "targets.npz")
    np.testing.assert_array_equal(targets["region_id"], raw["node_id"][:600])
    train_ids = np.flatnonzero(split[:600] == 0)
    heads = 0
    for model in MODELS:
        for seed in SEEDS:
            for m in range(3):
                ctx = np.load(
                    RESULTS
                    / "fixed_contexts"
                    / model
                    / f"seed{seed}"
                    / f"modality{m}.npz"
                )
                assert "test_y" not in ctx
                for role in ("train", "val", "test"):
                    np.testing.assert_array_equal(
                        ctx[role + "_node_id"], cohorts[m][role]
                    )
                    assert not ctx[role + "_visible"][:, m].any()
                    if role != "test":
                        np.testing.assert_array_equal(
                            ctx[role + "_y"],
                            raw["x"][cohorts[m][role], m * 4 : (m + 1) * 4],
                        )
                folder = RESULTS / "runs" / f"{model}_seed{seed}"
                file = folder / f"level0_modality{m}.pt"
                value = physical_inference(
                    predict_mlp(file, ctx["test_x"]),
                    schema["features"][m * 4 : (m + 1) * 4],
                )
                saved = np.load(folder / "inferences.npz")
                for j in range(4):
                    np.testing.assert_array_equal(
                        value[:, j], saved[f"test_{m*4+j}_inference"]
                    )
                head = torch.load(file, weights_only=True)
                np.testing.assert_allclose(
                    head["x_mean"], ctx["train_x"].mean(0, dtype=np.float64), atol=1e-12
                )
                np.testing.assert_allclose(
                    head["y_mean"], ctx["train_y"].mean(0), atol=1e-12
                )
                heads += 1
            natural = np.load(
                RESULTS
                / "fixed_contexts"
                / model
                / f"seed{seed}"
                / "natural_embeddings.npz"
            )
            np.testing.assert_array_equal(natural["region_id"], raw["node_id"][:600])
            x = natural["embedding"]
            folder = RESULTS / "inference/runs" / f"{model}_seed{seed}"
            saved = np.load(folder / "inferences.npz")
            np.testing.assert_array_equal(saved["region_id"], natural["region_id"])
            for j, name in enumerate(TARGETS):
                file = folder / f"{name}.pt"
                np.testing.assert_array_equal(
                    predict_mlp(file, x)[:, 0], saved["inference"][:, j]
                )
                head = torch.load(file, weights_only=True)
                np.testing.assert_allclose(
                    head["x_mean"], x[train_ids].mean(0, dtype=np.float64), atol=1e-12
                )
                np.testing.assert_allclose(
                    head["y_mean"], targets["y"][train_ids, j].mean(), atol=1e-12
                )
                heads += 1
    frames = [
        ("test_inferences.csv", "feature_metrics_by_seed.csv", "feature"),
        ("inference/test_inferences.csv", "inference/test_metrics.csv", "target"),
    ]
    cells = 0
    for record_file, metric_file, key in frames:
        records = pd.read_csv(RESULTS / record_file)
        metrics = pd.read_csv(RESULTS / metric_file)
        for (method, seed, name), group in records.groupby(["method", "seed", key]):
            ids = group.region_id.to_numpy(int)
            assert (split[ids] == 2).all() and len(np.unique(ids)) == len(ids)
            actual = (
                raw["x"][ids, schema["features"].index(name)]
                if key == "feature"
                else targets["y"][ids, TARGETS.index(name)]
            )
            np.testing.assert_allclose(group.true_value, actual, atol=1e-12)
            r2 = (
                1
                - np.square(group.true_value - group.inference).sum()
                / np.square(group.true_value - group.true_value.mean()).sum()
            )
            row = metrics[
                (metrics.method == method)
                & (metrics.seed == seed)
                & (metrics[key] == name)
            ].iloc[0]
            np.testing.assert_allclose(r2, row.r2, atol=1e-12)
            cells += 1
    assert heads == 75 and cells == 252
    return dict(
        status="passed",
        selected_heads_checked=heads,
        metric_cells_checked=cells,
        interpolation_test_counts=[110, 80, 89],
        inference_test_regions=120,
        target_exclusion_checks=len(
            json.loads((ROOT / "logs/target_exclusion_checks.json").read_text())
        ),
        natural_embedding_checks=len(
            json.loads((ROOT / "logs/natural_embedding_checks.json").read_text())
        ),
    )


def main():
    import time
    from contextlib import ExitStack
    from unittest.mock import patch
    from threadpoolctl import threadpool_limits
    from . import data, training

    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    (ROOT / "logs").mkdir(parents=True, exist_ok=True)
    immutable = [
        p
        for base in ("data", "checkpoints", "configs", "gwfm")
        for p in (PROJECT / base).rglob("*")
        if p.is_file() and p.suffix not in (".pyc",)
    ]
    hashes = {str(p.relative_to(PROJECT)): digest(p) for p in immutable}
    hashes.update(
        {
            str(p.relative_to(PROJECT)): digest(p)
            for p in (ROOT / "cohorts").glob("*.npz")
        }
    )
    save_json(ROOT / "logs/inputs_before_run.json", hashes)
    real_load = np.load
    stage = "encoding"
    accesses = []
    started = time.monotonic()

    def load(file, *args, **kwargs):
        path = Path(file)
        assert path.name not in ("truth.npz", "evaluation_masks.npz"), path
        if stage == "encoding":
            assert path.name != "targets.npz"
        accesses.append(dict(stage=stage, path=str(path.relative_to(PROJECT))))
        return real_load(file, *args, **kwargs)

    with threadpool_limits(limits=2), ExitStack() as stack:
        stack.enter_context(patch.object(np, "load", side_effect=load))
        for module in (data, training):
            stack.enter_context(
                patch.object(
                    module,
                    "sample_mask",
                    side_effect=AssertionError("Random downstream views disabled"),
                )
            )
        build_contexts()
        natural_contexts()
        stage = "fit"
        fit_neural()
        fit_spatial()
        fit_health_neural()
        fit_health_spatial()
        freeze()
        stage = "score"
        evaluate()
        evaluate_health()
        report = verify_current()
    for name, h in hashes.items():
        assert digest(PROJECT / name) == h, name
    for name, h in json.loads(
        (ROOT / "logs/inference_freeze.json").read_text()
    ).items():
        assert digest(ROOT / name) == h, name
    pd.DataFrame(accesses).to_csv(ROOT / "logs/data_access.csv", index=False)
    report.update(
        seconds=time.monotonic() - started,
        random_mask_calls=0,
        pretraining_repeated=False,
        original_checkpoint_and_input_files_unchanged=len(hashes),
    )
    save_json(ROOT / "logs/verification.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
