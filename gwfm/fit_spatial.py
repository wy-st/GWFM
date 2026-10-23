"""Coordinate-only KNN and IDW; separate donors for completion and outcomes."""

import numpy as np
from .paths import RESULTS as ROOT, DATA, TARGETS
from .protocol import CANDIDATES
from .utils import save_json
from .baselines.knn import coordinate_knn
from .baselines.idw import interpolate


def fit_outcomes(raw, split):
    tr, va = [np.flatnonzero(split[:600] == i) for i in (0, 1)]
    y = np.load(DATA / "targets.npz")["y"]
    train_y, val_y = y[tr].copy(), y[va].copy()
    del y
    for method, configs in CANDIDATES.items():
        out = ROOT / "inference/runs" / f"{method}_seed0"
        out.mkdir(parents=True, exist_ok=True)
        if (out / "inferences.npz").exists():
            continue
        inferences = []
        selected = []
        for j, name in enumerate(TARGETS):

            def predict(config, positions):
                if method == "KNN":
                    return coordinate_knn(
                        raw["pos"][tr],
                        train_y[:, j : j + 1],
                        np.ones(len(tr), bool),
                        positions,
                        config["neighbors"],
                        train_y[:, j : j + 1].mean(0),
                    )[:, 0]
                return interpolate(
                    raw["pos"][tr],
                    train_y[:, j],
                    np.ones(len(tr), bool),
                    positions,
                    "idw",
                    config["neighbors"],
                    power=config["power"],
                    fallback=train_y[:, j].mean(),
                )

            losses = [
                float(
                    np.mean((predict(c, raw["pos"][va]) - val_y[:, j]) ** 2)
                    / train_y[:, j].var()
                )
                for c in configs
            ]
            best = int(np.argmin(losses))
            selected.append(
                dict(
                    target=name,
                    selected=configs[best],
                    validation_losses=losses,
                    candidates=configs,
                )
            )
            inferences.append(predict(configs[best], raw["pos"][:600]))
        save_json(out / "selection.json", selected)
        np.savez_compressed(
            out / "inferences.npz",
            inference=np.column_stack(inferences),
            region_id=np.arange(600),
            donor_region_id=tr,
        )
