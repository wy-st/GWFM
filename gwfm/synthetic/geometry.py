"""Two connected, irregular polygon layers built by graph-based aggregation."""

import heapq
import json
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from shapely import (
    MultiPoint,
    STRtree,
    box,
    get_parts,
    intersection,
    set_precision,
    union_all,
    voronoi_polygons,
)
from shapely.geometry import mapping


def settlement_density(xy):
    """A synthetic settlement field controls cell density, not outcome labels."""
    return (
        0.24
        + 1.8 * np.exp(-np.square((xy - [0.29, 0.38]) / [0.16, 0.21]).sum(1) / 2)
        + 1.25 * np.exp(-np.square((xy - [0.76, 0.72]) / [0.13, 0.17]).sum(1) / 2)
    )


def sample_sites(rng, n):
    batches, total = [], 0
    while total < n:
        proposal = rng.uniform(0, 1, (max(100, 3 * (n - total)), 2))
        accepted = proposal[
            rng.uniform(0, 3.29, len(proposal)) < settlement_density(proposal)
        ]
        batches.append(accepted)
        total += len(accepted)
    return np.vstack(batches)[:n]


def rook_edges(polygons):
    tree = STRtree(polygons)
    a, b = tree.query(polygons, predicate="intersects")
    pairs = []
    for i, j in zip(a[a < b], b[a < b]):
        if polygons[i].boundary.intersection(polygons[j].boundary).length > 1e-9:
            pairs.append((int(i), int(j)))
    return np.asarray(sorted(pairs), dtype=int).T


def grow_regions(xy, edges, n_clusters, rng):
    """Multi-source shortest paths make every cluster rook-connected."""
    n = len(xy)
    if not 1 < n_clusters < n:
        raise ValueError(
            "The number of clusters must be between 2 and the number of cells"
        )
    # Farthest-site seeds suppress tiny adjacent seed cells without fixing counts.
    seeds = [int(rng.integers(n))]
    nearest = np.square(xy - xy[seeds[0]]).sum(1)
    for _ in range(1, n_clusters):
        i = int(np.argmax(nearest))
        seeds.append(i)
        nearest = np.minimum(nearest, np.square(xy - xy[i]).sum(1))
    neighbors = [[] for _ in range(n)]
    weights = np.linalg.norm(xy[edges[0]] - xy[edges[1]], axis=1)
    for (i, j), weight in zip(edges.T, weights):
        neighbors[i].append((j, weight))
        neighbors[j].append((i, weight))
    distance, label = np.full(n, np.inf), np.full(n, -1, dtype=int)
    queue = []
    for group, site in enumerate(seeds):
        distance[site], label[site] = 0.0, group
        heapq.heappush(queue, (0.0, group, site))
    while queue:
        dist, group, i = heapq.heappop(queue)
        if dist != distance[i] or label[i] != group:
            continue
        for j, weight in neighbors[i]:
            candidate = dist + weight
            if candidate < distance[j]:
                distance[j], label[j] = candidate, group
                heapq.heappush(queue, (candidate, group, j))
    if (label < 0).any():
        raise ValueError("Disconnected polygon adjacency")
    return label


def collapse(polygons, labels, n):
    return np.array(
        [union_all(polygons[labels == i], grid_size=1e-10) for i in range(n)],
        dtype=object,
    )


def representative_xy(polygons):
    # A concave polygon's centroid can lie outside; node coordinates must stay inside.
    return np.array(
        [[p.representative_point().x, p.representative_point().y] for p in polygons]
    )


def balanced_groups(xy, group_grid):
    gx, gy = group_grid
    if len(xy) % (gx * gy):
        raise ValueError(
            "Region count must be divisible by the number of stratification groups"
        )
    groups = np.empty(len(xy), dtype=int)
    for row, band in enumerate(np.array_split(np.argsort(xy[:, 1], kind="stable"), gy)):
        for col, ids in enumerate(
            np.array_split(band[np.argsort(xy[band, 0], kind="stable")], gx)
        ):
            groups[ids] = row * gx + col
    return groups


def make_geometry(seed, config):
    nr, ns = config["n_regions"], config["n_subregions"]
    na = ns * config.get("atoms_per_subregion", 3)
    if not 1 < nr < ns < na:
        raise ValueError("Require 1 < regions < subregions < construction cells")
    rng = np.random.default_rng(np.random.SeedSequence([seed, 91817]))
    sites = sample_sites(rng, na)
    domain = box(0, 0, 1, 1)
    cells = get_parts(
        voronoi_polygons(MultiPoint(sites), extend_to=domain, ordered=True)
    )
    cells = set_precision(intersection(cells, domain), 1e-10)
    atomic_edges = rook_edges(cells)
    sub_labels = grow_regions(representative_xy(cells), atomic_edges, ns, rng)
    subregions = collapse(cells, sub_labels, ns)
    sub_edges = rook_edges(subregions)
    parent = grow_regions(representative_xy(subregions), sub_edges, nr, rng)
    regions = collapse(subregions, parent, nr)
    # Ordering by parent keeps the graph convention, without any equal-count assumption.
    order = np.argsort(parent, kind="stable")
    subregions, parent = subregions[order], parent[order]
    region_xy, child_xy = representative_xy(regions), representative_xy(subregions)
    return {
        "regions": regions,
        "subregions": subregions,
        "parent": parent,
        "region_xy": region_xy,
        "child_xy": child_xy,
        "group": balanced_groups(region_xy, config["group_grid"]),
        "region_rook": rook_edges(regions),
        "subregion_rook": rook_edges(subregions),
    }


def validate_geometry(g):
    regions, children, parent = g["regions"], g["subregions"], g["parent"]
    assert (
        len(children) == len(parent)
        and parent.min() == 0
        and parent.max() == len(regions) - 1
    )
    errors = []
    for polys in [regions, children]:
        assert all(
            p.is_valid and p.geom_type == "Polygon" and p.area > 0 for p in polys
        )
        merged = union_all(polys, grid_size=1e-10)
        gap = merged.symmetric_difference(box(0, 0, 1, 1)).area
        overlap = abs(sum(p.area for p in polys) - merged.area)
        assert gap < 1e-8 and overlap < 1e-8, (gap, overlap)
        errors.append({"gap_area": gap, "overlap_area": overlap})
    outside = max(children[i].difference(regions[p]).area for i, p in enumerate(parent))
    assert outside < 1e-9
    for i, region in enumerate(regions):
        difference = region.symmetric_difference(
            union_all(children[parent == i], grid_size=1e-10)
        ).area
        assert difference < 1e-9
    for polys, xy in [(regions, g["region_xy"]), (children, g["child_xy"])]:
        from shapely import points

        assert all(p.covers(q) for p, q in zip(polys, points(xy)))
    counts = np.bincount(parent, minlength=len(regions))
    assert counts.min() >= 1 and np.unique(counts).size > 1
    for key, n in [("region_rook", len(regions)), ("subregion_rook", len(children))]:
        edge = g[key]
        adj = csr_matrix(
            (
                np.ones(2 * edge.shape[1]),
                (np.r_[edge[0], edge[1]], np.r_[edge[1], edge[0]]),
            ),
            shape=(n, n),
        )
        assert connected_components(adj)[0] == 1
    return {
        "n_regions": len(regions),
        "n_subregions": len(children),
        "children_min": int(counts.min()),
        "children_median": float(np.median(counts)),
        "children_mean": float(counts.mean()),
        "children_max": int(counts.max()),
        "children_count_distribution": {
            str(k): int((counts == k).sum()) for k in np.unique(counts)
        },
        "region_area_min": min(p.area for p in regions),
        "region_area_max": max(p.area for p in regions),
        "nonconvex_regions": sum(p.convex_hull.area - p.area > 1e-10 for p in regions),
        "nonconvex_subregions": sum(
            p.convex_hull.area - p.area > 1e-10 for p in children
        ),
        "partition_checks": errors,
        "max_child_outside_parent_area": outside,
        "region_rook_edges_undirected": g["region_rook"].shape[1],
        "subregion_rook_edges_undirected": g["subregion_rook"].shape[1],
    }


def save_geometry(g, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    report = validate_geometry(g)
    counts = np.bincount(g["parent"], minlength=len(g["regions"]))
    for layer in ["regions", "subregions"]:
        features = []
        for i, p in enumerate(g[layer]):
            props = {"id": i, "area": p.area}
            if layer == "subregions":
                props["parent_region_id"] = int(g["parent"][i])
                props["node_id"] = i + len(g["regions"])
            else:
                props["n_subregions"] = int(counts[i])
                props["group"] = int(g["group"][i])
                props["node_id"] = i
            features.append(
                {"type": "Feature", "geometry": mapping(p), "properties": props}
            )
        # Synthetic unit-square coordinates, not longitude/latitude or a real CRS.
        payload = {
            "type": "FeatureCollection",
            "coordinate_system": "synthetic planar unit square; no geographic CRS",
            "features": features,
        }
        (out / f"{layer}.geojson").write_text(
            json.dumps(payload, separators=(",", ":")) + "\n"
        )
    np.savez_compressed(
        out / "geometry.npz",
        parent=g["parent"],
        region_xy=g["region_xy"],
        child_xy=g["child_xy"],
        group=g["group"],
        child_counts=counts,
        region_rook=g["region_rook"],
        subregion_rook=g["subregion_rook"],
        region_area=np.array([p.area for p in g["regions"]]),
        child_area=np.array([p.area for p in g["subregions"]]),
    )
    (out / "geometry_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
