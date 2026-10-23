"""Train-only scaling and validation-selected two-layer MLP heads."""

from copy import deepcopy
import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler


def mlp_predict(
    train,
    val,
    query,
    seed,
    out,
    *,
    hidden=64,
    weight_decay=0.01,
    max_epochs=300,
    patience=None,
    min_delta=0.0
):
    if (
        hidden < 1
        or max_epochs < 1
        or not np.isfinite(weight_decay)
        or weight_decay < 0
        or (patience is not None and patience < 1)
        or not np.isfinite(min_delta)
        or min_delta < 0
    ):
        raise ValueError(
            "Invalid MLP capacity, regularization or early-stopping configuration"
        )
    for bucket in (train, val):
        if (
            bucket["x"].ndim != 2
            or bucket["y"].ndim != 2
            or not len(bucket["x"])
            or len(bucket["x"]) != len(bucket["y"])
            or not np.isfinite(bucket["x"]).all()
            or not np.isfinite(bucket["y"]).all()
        ):
            raise ValueError(
                "Finite, aligned nonempty regression inputs and targets required"
            )
    torch.manual_seed(seed)
    sx, sy = StandardScaler().fit(train["x"]), StandardScaler().fit(train["y"])
    tx = torch.tensor(sx.transform(train["x"]), dtype=torch.float32)
    ty = torch.tensor(sy.transform(train["y"]), dtype=torch.float32)
    vx = torch.tensor(sx.transform(val["x"]), dtype=torch.float32)
    vy = torch.tensor(sy.transform(val["y"]), dtype=torch.float32)
    model = nn.Sequential(
        nn.Linear(tx.shape[1], hidden), nn.GELU(), nn.Linear(hidden, ty.shape[1])
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.003, weight_decay=weight_decay
    )
    best, best_epoch, state = float("inf"), 0, None
    stale = 0
    for epoch in range(max_epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = (model(tx) - ty).square().mean()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            score = float((model(vx) - vy).square().mean())
        if not np.isfinite(score):
            raise FloatingPointError("Nonfinite MLP validation loss")
        if score < best - min_delta:
            best, best_epoch, state = score, epoch + 1, deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if patience is not None and stale >= patience:
            break
    model.load_state_dict(state)
    torch.save(
        dict(
            state=state,
            input_dim=tx.shape[1],
            output_dim=ty.shape[1],
            hidden_dim=hidden,
            x_mean=sx.mean_.tolist(),
            x_scale=sx.scale_.tolist(),
            y_mean=sy.mean_.tolist(),
            y_scale=sy.scale_.tolist(),
        ),
        out,
    )
    with torch.no_grad():
        pred = model(torch.tensor(sx.transform(query), dtype=torch.float32)).numpy()
    return sy.inverse_transform(pred), dict(
        best_epoch=best_epoch,
        validation_link_mse=best,
        epochs_run=epoch + 1,
        hidden=hidden,
        weight_decay=weight_decay,
        patience=patience,
        min_delta=min_delta,
    )


def predict_mlp(checkpoint, query):
    saved = torch.load(checkpoint, weights_only=True)
    if (
        query.ndim != 2
        or query.shape[1] != saved["input_dim"]
        or not np.isfinite(query).all()
    ):
        raise ValueError(
            "Finite query matrix matching the saved MLP input width required"
        )
    hidden = saved["state"]["0.weight"].shape[0]
    model = nn.Sequential(
        nn.Linear(saved["input_dim"], hidden),
        nn.GELU(),
        nn.Linear(hidden, saved["output_dim"]),
    )
    model.load_state_dict(saved["state"])
    model.eval()
    x = (query - np.asarray(saved["x_mean"])) / np.asarray(saved["x_scale"])
    with torch.no_grad():
        inference = model(torch.tensor(x, dtype=torch.float32)).numpy()
    inference *= np.asarray(saved["y_scale"])
    inference += np.asarray(saved["y_mean"])
    return inference


def physical_inference(inference, names):
    result = np.array(inference, dtype=float, copy=True)
    for j, name in enumerate(names):
        if name in LOGIT_FEATURES:
            result[:, j] = np.clip(result[:, j], 1e-12, 1 - 1e-12)
        elif name in LOG_FEATURES:
            result[:, j] = np.maximum(result[:, j], 1e-12)
    if not np.isfinite(result).all():
        raise FloatingPointError("Nonfinite raw-unit inferences")
    return result


from .contexts import LOG_FEATURES, LOGIT_FEATURES
