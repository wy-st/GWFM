"""Train the three paired ablations; downstream evaluation lives in current.py."""

from copy import deepcopy
import hashlib
import json
import time
import numpy as np
import torch
from ..paths import (
    ROOT as PROJECT,
    DATA,
    encoder_dir,
)
from ..utils import save_json, digest
from ..data import load_graph, subset_graph, fit_standardizer, standardize, sample_mask
from ..training import reconstruction_evaluate as evaluate, check_saved_run
from .model import AblationSSL, make_encoder


def encoder_folder(variant, seed):
    if variant == "full":
        return encoder_dir("GWFM", seed)
    return PROJECT / "checkpoints" / "ablations" / variant / f"seed{seed}"


def train(variant, seed):
    out = encoder_folder(variant, seed)
    cfg = json.loads((encoder_folder("full", seed) / "config.json").read_text())[
        "config"
    ]
    cfg.update(ablation=variant, data="data")
    if variant == "no_modality":
        cfg.pop("modality_arch", None)
    if variant == "no_prototype":
        cfg.update(lambda_proto=0.0, lambda_variance=0.0)
    if (out / "metrics.json").exists():
        check_saved_run(out, cfg)
        return
    out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(seed)
    start = time.monotonic()
    g = load_graph(
        DATA, graph_rule="distance", distance_bandwidths=cfg["distance_bandwidths"]
    )
    mean, std = fit_standardizer(subset_graph(g, 0))
    parts = [standardize(subset_graph(g, code), mean, std) for code in (0, 1, 2)]
    enc = make_encoder(g["feature_modality"], cfg)
    model = AblationSSL(enc)
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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)
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
        for epoch in range(1, 301):
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
            loss, terms = model(parts[0], masks)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite SSL loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 5.0, error_if_nonfinite=True
            )
            optimizer.step()
            model.update_teacher()
            scheduler.step()
            val = evaluate(enc, parts[1], val_mask)
            total = sum(val.values())
            if total < best:
                best, best_epoch, state = total, epoch, deepcopy(model.state_dict())
            log.write(
                json.dumps(
                    dict(
                        epoch=epoch,
                        loss=float(loss.detach()),
                        validation=val,
                        terms={k: float(v.detach()) for k, v in terms.items()},
                    )
                )
                + "\n"
            )
            log.flush()
            if epoch in (1, 100, 200, 300):
                print(variant, seed, epoch, round(total, 6), flush=True)
    reference = json.loads((encoder_folder("full", seed) / "metrics.json").read_text())
    assert fingerprint.hexdigest() == reference["training_masks_sha256"]
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
    restored = make_encoder(g["feature_modality"], cfg).eval()
    restored.load_state_dict(enc.state_dict())
    embeddings = torch.empty(len(g["x"]), 48)
    with torch.no_grad():
        for code, part in enumerate(parts):
            z = enc.encode_graph(part, part["observed"])[0]
            torch.testing.assert_close(
                z, restored.encode_graph(part, part["observed"])[0], rtol=0, atol=0
            )
            embeddings[g["split"] == code] = z
    np.savez_compressed(
        out / "embeddings.npz",
        embedding=embeddings.numpy(),
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
            checkpoint_sha256=digest(out / "model.pt"),
            restore_exact=True,
        ),
    )
