"""Train final encoders and explicitly export all-node embeddings."""

import json, hashlib, time
from copy import deepcopy
import numpy as np
import torch
from .paths import DATA, REPOSITORY, encoder_dir
from .utils import save_json, digest
from .data import load_graph, subset_graph, fit_standardizer, standardize, sample_mask
from .models.model import encoder_from_config, masked_pdfm_loss, reconstruction_losses
from .models.visible_graph import VisibleGraphSSL


def configuration(model, seed, epochs=300):
    cfg = json.loads((REPOSITORY / "configs" / f"{model.lower()}.json").read_text())
    cfg.update(data="data", seed=seed, epochs=epochs)
    if not np.isfinite(cfg["lr"]) or cfg["lr"] <= 0:
        raise ValueError("Learning rate must be finite and positive")
    # The retained modality fusion and auxiliary objective implement this fixed research protocol.
    fixed = {"block_dim": 16, "decoder_layers": 1, "no_child_spatial": False}
    if model == "GWFM":
        fixed.update(lambda_proto=0.05, lambda_variance=0.05, lambda_cross=0.0)
    if any(cfg.get(key) != value for key, value in fixed.items()):
        raise ValueError(f"Unsupported fixed protocol values: {fixed}")
    return cfg


def check_saved_run(folder, config):
    saved = json.loads((folder / "config.json").read_text())
    if saved["config"] != config:
        raise ValueError(
            "Saved checkpoint uses a different configuration; use a new --work-dir"
        )
    if set(saved.get("input_sha256", {})) != {
        "complementary.npz",
        "geometry.npz",
        "split.npz",
    }:
        raise ValueError(
            "Saved checkpoint lacks verified input hashes; use a new --work-dir"
        )
    if any(digest(DATA / name) != sha for name, sha in saved["input_sha256"].items()):
        raise ValueError("Saved checkpoint inputs changed; use a new --work-dir")
    if (
        digest(folder / "model.pt")
        != json.loads((folder / "metrics.json").read_text())["checkpoint_sha256"]
    ):
        raise ValueError("Saved checkpoint hash mismatch")


@torch.no_grad()
def reconstruction_evaluate(encoder, graph, mask):
    encoder.eval()
    _, pred = encoder.encode_graph(graph, graph["observed"] & ~mask)
    return {
        k: float(v)
        for k, v in reconstruction_losses(
            pred, graph["x"], mask, graph["node_type"], graph["feature_modality"]
        ).items()
    }


def train_gwfm(seed, epochs=300):
    """Train on two masked views; use validation reconstruction to select the frozen encoder."""
    cfg = configuration("GWFM", seed, epochs)
    variant = "GWFM"
    corruption = "modality_mask"
    out = encoder_dir("GWFM", seed)
    if (out / "metrics.json").exists():
        check_saved_run(out, cfg)
        return
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    start = time.monotonic()
    g = load_graph(
        DATA, graph_rule="distance", distance_bandwidths=cfg["distance_bandwidths"]
    )
    mean, std = fit_standardizer(subset_graph(g, 0))
    parts = [standardize(subset_graph(g, code), mean, std) for code in (0, 1, 2)]
    enc = encoder_from_config(g["feature_modality"], cfg)
    model = VisibleGraphSSL(enc)
    save_json(
        out / "config.json",
        dict(
            config=cfg,
            input_sha256={
                name: digest(DATA / name)
                for name in ("complementary.npz", "geometry.npz", "split.npz")
            },
        ),
    )
    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad), lr=cfg["lr"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    rng = torch.Generator().manual_seed(seed + 30001)
    val_mask = sample_mask(
        parts[1]["observed"],
        parts[1]["node_type"],
        cfg["mask_rate"],
        torch.Generator().manual_seed(seed + 10001),
    )
    fingerprint = hashlib.sha256()
    best, best_epoch, state = float("inf"), 0, None
    with (out / "history.jsonl").open("w") as log:
        for epoch in range(1, epochs + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            model.begin_epoch(epoch, parts[0], seed)
            masks = [
                sample_mask(
                    parts[0]["observed"],
                    parts[0]["node_type"],
                    cfg["mask_rate"],
                    rng,
                    cfg["mask_strategy"],
                )
                for _ in range(2)
            ]
            for mask in masks:
                fingerprint.update(mask.numpy().tobytes())
            loss, _ = model(parts[0], masks, corruption=corruption)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite SSL loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 5.0, error_if_nonfinite=True
            )
            optimizer.step()
            model.update_teacher()
            scheduler.step()
            val = reconstruction_evaluate(enc, parts[1], val_mask)
            total = sum(val.values())
            if total < best:
                best, best_epoch, state = total, epoch, deepcopy(model.state_dict())
            log.write(
                json.dumps(dict(epoch=epoch, loss=float(loss.detach()), validation=val))
                + "\n"
            )
            log.flush()
            if epoch in (1, 100, 200, 300):
                print(variant, seed, epoch, round(total, 6), flush=True)
    model.load_state_dict(state)
    model.eval()
    torch.save(
        dict(
            config=cfg,
            mean=mean,
            std=std,
            best_epoch=best_epoch,
            encoder_state=enc.state_dict(),
            training_state=model.state_dict(),
        ),
        out / "model.pt",
    )
    embedding = torch.empty(len(g["x"]), 48)
    with torch.no_grad():
        for code, part in enumerate(parts):
            embedding[g["split"] == code] = enc.encode_graph(part, part["observed"])[0]
            spatial, up, down = enc.topology(part, part["observed"])
            payload = dict(
                node_id=part["node_id"].numpy(),
                node_type=part["node_type"].numpy(),
                parent=part["parent"].numpy(),
                observed=part["observed"].numpy(),
                up=up.numpy(),
                down=down.numpy(),
            )
            for m, (edges, delta) in enumerate(spatial):
                payload.update(
                    {
                        f"m{m}_edges": edges.numpy(),
                        f"m{m}_beta_weights": enc.spatial_weights(
                            enc, part, m, edges, delta
                        ).numpy(),
                        f"m{m}_k0_weights": enc.spatial_weights(
                            enc.k0, part, m, edges, delta
                        ).numpy(),
                    }
                )
            np.savez_compressed(out / f"natural_graph_partition{code}.npz", **payload)
    np.savez_compressed(
        out / "embeddings.npz",
        embedding=embedding.numpy(),
        node_id=g["node_id"].numpy(),
        split=g["split"].numpy(),
    )
    save_json(
        out / "metrics.json",
        dict(
            best_epoch=best_epoch,
            validation_huber=best,
            seconds=time.monotonic() - start,
            parameters=sum(p.numel() for p in enc.parameters()),
            training_masks_sha256=fingerprint.hexdigest(),
            beta_bandwidth_multiplier=enc.log_bandwidth.exp()
            .clamp(0.25, 4)
            .detach()
            .tolist(),
            k0_bandwidth_multiplier=enc.k0.log_bandwidth.exp()
            .clamp(0.25, 4)
            .detach()
            .tolist(),
            checkpoint_sha256=digest(out / "model.pt"),
        ),
    )


def train_pdfm(seed, epochs=300):
    """Train the PDFM baseline on one masked view with the same observation boundaries."""
    variant = "PDFM"
    cfg = configuration("PDFM", seed, epochs)
    out = encoder_dir("PDFM", seed)
    if (out / "metrics.json").exists():
        check_saved_run(out, cfg)
        return
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    start = time.monotonic()
    g = load_graph(
        DATA, graph_rule="distance", distance_bandwidths=cfg["distance_bandwidths"]
    )
    mean, std = fit_standardizer(subset_graph(g, 0))
    parts = [standardize(subset_graph(g, c), mean, std) for c in (0, 1, 2)]
    enc = encoder_from_config(g["feature_modality"], cfg)
    save_json(
        out / "config.json",
        dict(
            config=cfg,
            input_sha256={
                n: digest(DATA / n)
                for n in ("complementary.npz", "geometry.npz", "split.npz")
            },
        ),
    )
    optimizer = torch.optim.Adam(enc.parameters(), lr=cfg["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    rng = torch.Generator().manual_seed(seed + 30001)
    fingerprint = hashlib.sha256()
    val_mask = sample_mask(
        parts[1]["observed"],
        parts[1]["node_type"],
        cfg["mask_rate"],
        torch.Generator().manual_seed(seed + 10001),
    )
    best, best_epoch, state = float("inf"), 0, None
    with (out / "history.jsonl").open("w") as f:
        for epoch in range(1, epochs + 1):
            enc.train()
            optimizer.zero_grad(set_to_none=True)
            mask = sample_mask(
                parts[0]["observed"],
                parts[0]["node_type"],
                cfg["mask_rate"],
                rng,
                cfg["mask_strategy"],
            )
            fingerprint.update(mask.numpy().tobytes())
            loss = masked_pdfm_loss(enc, parts[0], mask)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite PDFM loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                enc.parameters(), 5.0, error_if_nonfinite=True
            )
            optimizer.step()
            scheduler.step()
            validation = reconstruction_evaluate(enc, parts[1], val_mask)
            value = sum(validation.values())
            if value < best:
                best, best_epoch, state = value, epoch, deepcopy(enc.state_dict())
            f.write(
                json.dumps(
                    dict(epoch=epoch, loss=float(loss.detach()), validation=validation)
                )
                + "\n"
            )
            f.flush()
            if epoch in (1, 100, 200, 300):
                print(variant, seed, epoch, round(value, 6), flush=True)
    enc.load_state_dict(state)
    enc.eval()
    torch.save(
        dict(
            config=cfg, mean=mean, std=std, best_epoch=best_epoch, encoder_state=state
        ),
        out / "model.pt",
    )
    embedding = torch.empty(len(g["x"]), 48)
    with torch.no_grad():
        for code, part in enumerate(parts):
            embedding[g["split"] == code] = enc.encode_graph(part, part["observed"])[0]
            payload = dict(
                node_id=part["node_id"].numpy(),
                parent=part["parent"].numpy(),
                node_type=part["node_type"].numpy(),
                observed=part["observed"].numpy(),
            )
            for m, (e, w) in enumerate(enc.topology(part, part["observed"])):
                payload.update({f"m{m}_edges": e.numpy(), f"m{m}_weights": w.numpy()})
            np.savez_compressed(out / f"natural_graph_partition{code}.npz", **payload)
    np.savez_compressed(
        out / "embeddings.npz",
        embedding=embedding.numpy(),
        node_id=g["node_id"].numpy(),
        split=g["split"].numpy(),
    )
    save_json(
        out / "metrics.json",
        dict(
            best_epoch=best_epoch,
            validation_huber=best,
            parameters=sum(p.numel() for p in enc.parameters()),
            seconds=time.monotonic() - start,
            checkpoint_sha256=digest(out / "model.pt"),
            training="PDFM single-view SSL masked reconstruction; GWFM mixed sampler",
            training_masks_sha256=fingerprint.hexdigest(),
            training_mask_views_per_epoch=1,
            validation_mask_sha256=hashlib.sha256(
                val_mask.numpy().tobytes()
            ).hexdigest(),
        ),
    )


@torch.no_grad()
def extract_embeddings(model, seed):
    """Export embeddings from the saved encoder using the selected work directory."""
    from .ablation.run import encoder_folder
    from .ablation.model import make_encoder

    out = (
        encoder_dir(model, seed)
        if model in ("GWFM", "PDFM")
        else encoder_folder(model, seed)
    )
    saved = torch.load(out / "model.pt", weights_only=True)
    g = load_graph(
        DATA,
        graph_rule="distance",
        distance_bandwidths=saved["config"]["distance_bandwidths"],
    )
    enc = make_encoder(g["feature_modality"], saved["config"]).eval()
    enc.load_state_dict(saved["encoder_state"], strict=True)
    z = torch.empty(len(g["x"]), 48)
    for code in (0, 1, 2):
        part = standardize(subset_graph(g, code), saved["mean"], saved["std"])
        z[g["split"] == code] = enc.encode_graph(part, part["observed"])[0]
    if not torch.isfinite(z).all():
        raise ValueError("Nonfinite embeddings")
    np.savez_compressed(
        out / "embeddings.npz",
        embedding=z.numpy(),
        node_id=g["node_id"].numpy(),
        split=g["split"].numpy(),
    )
    np.savez_compressed(
        out / "region_embeddings.npz",
        embedding=z[g["node_type"] == 0].numpy(),
        node_id=g["node_id"][g["node_type"] == 0].numpy(),
    )
    return z.numpy()
