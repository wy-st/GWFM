"""One entry point for the final pipeline. Run `python -m gwfm --help`."""

import argparse
import os
import runpy
from pathlib import Path


def main():
    p = argparse.ArgumentParser(
        description="Final GWFM pipeline (CPU, fixed scientific protocol)."
    )
    p.add_argument(
        "stage",
        choices=[
            "generate",
            "train",
            "extract",
            "contexts",
            "decode",
            "baselines",
            "evaluate",
            "ablations",
            "figures",
            "verify",
            "current",
            "probes",
            "all",
        ],
    )
    p.add_argument(
        "--work-dir",
        type=Path,
        help="Separate output root for a clean rerun; defaults to this checkout.",
    )
    p.add_argument("--model", choices=["GWFM", "PDFM", "both"], default="both")
    p.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 37])
    p.add_argument(
        "--epochs",
        type=int,
        default=300,
        help="SSL epochs; reported experiments use 300.",
    )
    args = p.parse_args()
    if args.epochs < 1 or any(s < 0 for s in args.seeds):
        p.error("epochs must be positive; seeds must be nonnegative")
    if args.stage == "all" and (
        args.seeds != [11, 23, 37] or args.model != "both" or args.epochs != 300
    ):
        p.error(
            "all uses all final models/seeds and 300 epochs; use train for an individual run"
        )
    if args.work_dir:
        os.environ["GWFM_WORK_DIR"] = str(args.work_dir.resolve())
    os.environ.setdefault(
        "MPLCONFIGDIR", str(Path(os.environ.get("TMPDIR", "/tmp")) / "gwfm-matplotlib")
    )
    import torch
    from threadpoolctl import threadpool_limits

    torch.set_num_threads(2)
    # CPU kernels are deterministic in the recorded environment; reject unsupported nondeterministic kernels.
    torch.use_deterministic_algorithms(True)
    stages = (
        [
            "generate",
            "train",
            "ablations",
            "extract",
            "current",
            "probes",
            "figures",
            "verify",
        ]
        if args.stage == "all"
        else [args.stage]
    )
    models = ["GWFM", "PDFM"] if args.model == "both" else [args.model]
    with threadpool_limits(limits=2):
        for stage in stages:
            print("Stage:", stage, flush=True)
            if stage == "generate":
                from .generate import run

                run()
            elif stage == "train":
                from .training import train_gwfm, train_pdfm

                for model in models:
                    for seed in args.seeds:
                        (train_gwfm if model == "GWFM" else train_pdfm)(
                            seed, args.epochs
                        )
            elif stage == "extract":
                from .training import extract_embeddings

                export_models = models + (
                    ["no_weighting", "no_modality", "no_prototype"]
                    if args.model == "both"
                    else []
                )
                for model in export_models:
                    for seed in args.seeds:
                        extract_embeddings(model, seed)
            elif stage == "contexts":
                from .current import build_contexts, natural_contexts

                build_contexts()
                natural_contexts()
            elif stage == "decode":
                from .current import fit_neural, fit_health_neural

                fit_neural()
                fit_health_neural()
            elif stage == "baselines":
                from .current import fit_spatial, fit_health_spatial

                fit_spatial()
                fit_health_spatial()
            elif stage == "evaluate":
                from .current import (
                    freeze,
                    evaluate,
                    evaluate_health,
                    verify_current,
                    ROOT,
                )
                from .utils import save_json

                freeze()
                evaluate()
                evaluate_health()
                save_json(ROOT / "logs/verification.json", verify_current())
            elif stage == "ablations":
                from .ablation.run import train
                from .ablation.model import VARIANTS

                for variant in VARIANTS:
                    if variant != "full":
                        for seed in args.seeds:
                            train(variant, seed)
            elif stage == "current":
                from .current import main as downstream

                downstream()
            elif stage == "probes":
                from .paths import REPOSITORY

                runpy.run_path(
                    str(REPOSITORY / "fit_figure_probes.py"), run_name="__main__"
                )
            elif stage == "figures":
                from .paths import REPOSITORY

                runpy.run_path(
                    str(REPOSITORY / "build_figures.py"), run_name="__main__"
                )
            elif stage == "verify":
                from .current import (
                    build_contexts,
                    natural_contexts,
                    verify_current,
                    ROOT,
                )
                from .utils import save_json

                build_contexts()
                natural_contexts()
                result = verify_current()
                save_json(ROOT / "logs/verification.json", result)
                print(result, flush=True)


if __name__ == "__main__":
    main()
