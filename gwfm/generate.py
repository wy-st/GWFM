"""Generate the prescribed data, split, training statistics, and diagnostics."""

import json
import numpy as np
from .paths import DATA, REPOSITORY
from .synthetic.data import generate
from .utils import save_json, write_csv


from .utils import moran


def evaluation_artifacts():
    raw = dict(np.load(DATA / "complementary.npz"))
    rs = np.load(DATA / "split.npz")["split"]
    split = np.r_[rs, rs[raw["parent"][len(rs) :]]]
    visible = np.where(raw["observed"].repeat(4, axis=1), raw["x"], np.nan)
    mean = np.stack(
        [
            np.nanmean(visible[(split == 0) & (raw["node_type"] == level)], axis=0)
            for level in (0, 1)
        ]
    )
    scale = np.stack(
        [
            np.nanstd(visible[(split == 0) & (raw["node_type"] == level)], axis=0)
            for level in (0, 1)
        ]
    )
    assert np.isfinite(mean).all() and (scale > 0).all()
    np.savez_compressed(DATA / "standardizer.npz", mean=mean, scale=scale)


def diagnostic_tables():
    """Oracle features below are used for descriptive figures only, never training."""
    metadata = json.loads((DATA / "metadata.json").read_text())
    names = metadata["features"]
    truth = np.load(DATA / "truth.npz")["complete"]
    geometry = np.load(DATA / "geometry.npz")
    rows = []
    for level, values in (("region", truth[:600]), ("subregion", truth[600:])):
        edges = geometry[level + "_rook"]
        edges = np.column_stack([edges, edges[::-1]])
        for j, name in enumerate(names):
            for norm in ("binary", "row"):
                rows.append(
                    dict(
                        level=level,
                        feature=name,
                        weights="rook",
                        normalization=norm,
                        moran_i=moran(values[:, j], edges, norm == "row"),
                    )
                )
    write_csv(DATA / "moran_by_attribute.csv", rows)
    missing = np.load(DATA / "missingness.npz")
    u = missing["urbanicity"]
    rows = []
    for scenario in metadata["config"]["scenarios"]:
        raw = np.load(DATA / f"{scenario}.npz")
        expected = 1 - missing["probability_" + scenario]
        for level, ids in (
            ("region", np.arange(600)),
            ("subregion", np.arange(600, 5400)),
        ):
            bins = np.array_split(ids[np.argsort(u[ids], kind="stable")], 4)
            for m, modality in enumerate(metadata["modalities"]):
                for q, part in enumerate(bins):
                    rows.append(
                        dict(
                            scenario=scenario,
                            level=level,
                            modality=modality,
                            urbanicity_quartile=q + 1,
                            n=len(part),
                            observed_n=int(raw["observed"][part, m].sum()),
                            observed_fraction=float(raw["observed"][part, m].mean()),
                            expected_observed_fraction=float(expected[part, m].mean()),
                            u_min=float(u[part].min()),
                            u_max=float(u[part].max()),
                        )
                    )
    write_csv(DATA / "observation_by_urbanicity.csv", rows)


def run():
    config = json.loads((REPOSITORY / "configs/dataset.json").read_text())
    # Never overwrite a dataset that already has trained results attached to it.
    if (DATA / "metadata.json").exists():
        saved = json.loads((DATA / "metadata.json").read_text())["config"]
        # Older metadata also recorded training options and unused missingness scenarios.
        if any(
            saved.get(key) != value
            for key, value in config.items()
            if key != "scenarios"
        ) or not set(config["scenarios"]).issubset(saved.get("scenarios", [])):
            raise ValueError(
                "Saved data use a different configuration; use a new --work-dir"
            )
        required = (
            "complementary.npz",
            "geometry.npz",
            "split.npz",
            "targets.npz",
            "schema.json",
            "standardizer.npz",
            "truth.npz",
            "missingness.npz",
            "population.npz",
            "regions.geojson",
            "subregions.geojson",
            "moran_by_attribute.csv",
            "observation_by_urbanicity.csv",
        )
        if any(not (DATA / name).is_file() for name in required):
            raise ValueError("Incomplete generated data; use a new --work-dir")
        print("Using saved data; pass a new --work-dir to regenerate.")
        return
    generate(config["data_seed"], config, str(DATA))
    metadata = json.loads((DATA / "metadata.json").read_text())
    save_json(
        DATA / "schema.json",
        dict(features=metadata["features"], units=metadata["feature_units"]),
    )
    evaluation_artifacts()
    diagnostic_tables()
