"""Fixed final experiment: 600 regions, three encoder seeds, 16D/48D probes."""

import json
import numpy as np
from .paths import DATA

CANDIDATES = {
    "KNN": [dict(neighbors=k) for k in (1, 3, 5, 8, 16, 32)],
    "IDW": [dict(neighbors=k, power=p) for k in (8, 16, 32) for p in (1.0, 2.0, 3.0)],
}


def dataset():
    raw = dict(np.load(DATA / "complementary.npz"))
    schema = json.loads((DATA / "schema.json").read_text())
    with np.load(DATA / "split.npz") as partition, np.load(
        DATA / "targets.npz"
    ) as targets:
        region_split = partition["split"]
        expected = raw["node_id"][raw["node_type"] == 0]
        if not (
            np.array_equal(partition["region_id"], expected)
            and np.array_equal(targets["region_id"], expected)
        ):
            raise ValueError(
                "Region IDs in graph, split and targets must have identical order"
            )
        if (
            targets["y"].shape != (len(expected), 2)
            or not np.isfinite(targets["y"]).all()
        ):
            raise ValueError("Expected two finite targets per region")
    split = np.r_[region_split, region_split[raw["parent"][len(region_split) :]]]
    np.testing.assert_array_equal(raw["node_id"], np.arange(len(split)))
    assert len(region_split) == 600 and len(split) == 5400
    return raw, schema, split


def select_embedding(embedding, columns=None):
    columns = np.arange(48) if columns is None else np.asarray(columns)
    if (
        embedding.ndim != 2
        or embedding.shape[1] != 48
        or not np.isfinite(embedding).all()
        or columns.ndim != 1
        or columns.dtype.kind not in "iu"
        or not len(columns)
        or (columns < 0).any()
        or (columns >= 48).any()
        or len(np.unique(columns)) != len(columns)
    ):
        raise ValueError(
            "Finite 48D embeddings and unique integer column indices required"
        )
    return embedding[:, columns]
