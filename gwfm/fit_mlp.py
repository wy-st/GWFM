"""Train the final MLP heads; validation chooses capacity, test is never used here."""

import hashlib, json, shutil, time
import numpy as np
import torch
from .utils import save_json, digest
from .decoders import mlp_predict, predict_mlp

GRID = [dict(hidden=h, weight_decay=w) for h in (32, 64, 128) for w in (0.0001, 0.01)]
MAX_EPOCHS = 500
PATIENCE = 40


def select_head(train, val, query, folder, label):
    """Choose width and weight decay by validation MSE; test labels are never accepted."""
    selected = folder / f"{label}_selection.json"
    chosen = folder / f"{label}.pt"
    input_hashes = {
        f"{role}_{key}": hashlib.sha256(
            str((value.shape, value.dtype)).encode() + value.tobytes()
        ).hexdigest()
        for role, data in (("train", train), ("val", val))
        for key, value in data.items()
    }
    if (
        selected.exists()
        and json.loads(selected.read_text()).get("input_sha256") != input_hashes
    ):
        raise ValueError(
            f"Saved head inputs differ or are unverified: {selected}; use a new output directory"
        )
    if not selected.exists():
        trials = []
        start = time.monotonic()
        for config in GRID:
            path = folder / f'{label}_h{config["hidden"]}_wd{config["weight_decay"]}.pt'
            _, info = mlp_predict(
                train,
                val,
                val["x"],
                11,
                path,
                max_epochs=MAX_EPOCHS,
                patience=PATIENCE,
                min_delta=1e-6,
                **config,
            )
            trials.append(dict(**info, checkpoint=path.name))
        best = min(trials, key=lambda r: r["validation_link_mse"])
        shutil.copyfile(folder / best["checkpoint"], chosen)
        save_json(
            selected,
            dict(
                selected=best,
                candidates=trials,
                seconds=time.monotonic() - start,
                train_events=len(train["x"]),
                validation_nodes=len(val["x"]),
                input_dim=train["x"].shape[1],
                output_dim=train["y"].shape[1],
                checkpoint_sha256=digest(chosen),
                input_sha256=input_hashes,
            ),
        )
    info = json.loads(selected.read_text())
    assert digest(chosen) == info["checkpoint_sha256"]
    saved = torch.load(chosen, weights_only=True)
    np.testing.assert_allclose(
        saved["x_mean"], train["x"].mean(0, dtype=np.float64), atol=1e-12
    )
    np.testing.assert_allclose(
        saved["y_mean"], train["y"].mean(0, dtype=np.float64), atol=1e-12
    )
    vp = predict_mlp(chosen, val["x"])
    value = np.mean(((vp - val["y"]) / np.asarray(saved["y_scale"])) ** 2)
    np.testing.assert_allclose(
        value, info["selected"]["validation_link_mse"], rtol=2e-5, atol=2e-6
    )
    return predict_mlp(chosen, query)
