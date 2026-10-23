# Final manuscript and code alignment

The references are `GWFM_last.docx`, `README3_last.docx`, and the supplied homework specification. This check preserves the final experiment and its saved numerical results. It corrects the earlier incomplete check of standalone figure exports.

## Corrections

- Restored Figure 1's attribute descriptions and both health-burden maps from the final plotting layout.
- Preserved the exact final manuscript artwork for Figures 2 and 3. Figure 2 remains a schematic, as its caption states.
- Kept the heat-burden error maps in Figure 4 and removed the obsolete respiratory-map export.
- Kept Figure 5's ablation table and regional PCA, and numbered every file as in the manuscript.
- Replaced the two competing export scripts with `build_figures.py`. Intermediate panels exist only while composing the final images.
- Removed duplicate figures, old preview directories, the empty `figures/performance/source_data` directory, and the obsolete heterogeneity plot.
- Renamed the color/label helper to `gwfm/visualization/palette.py`, and removed unused reconstruction-loss and donor-selection alternatives. Saved parameter names and initialization order remain compatible with the supplied checkpoints.

There are six delivered image files: Figures 1–5 and Table 1. No other image versions or standalone component panels are retained in `figures/`.

## Scientific correspondence

| Report / README content | Implementation and evidence |
| --- | --- |
| Hierarchy, spatial fields, population, targets, measurement error | `gwfm/synthetic/`, `configs/dataset.json`; deterministic data regeneration previously matched every array in eight archives |
| 600 regions, 4,800 subregions; 420/60/120 split | `gwfm/data.py`, `gwfm/synthetic/common.py`; fixed seed 1729, ten stratification groups, inherited child splits |
| Structured modality missingness | `gwfm/synthetic/missingness.py`; natural gaps are explicitly masked and excluded from reconstruction supervision |
| Dual GWFM branches and geographic weighting | `gwfm/models/visible_graph.py`; separate attribute and weight branches, `z = beta * h_k`, visible neighbors, mean/max/sum aggregation |
| Shared and modality-specific fusion | `gwfm/models/modality_structure.py`; three 16D blocks, 48D output, 106,297 parameters |
| Prototype objective and EMA teacher | `gwfm/models/prototype_beta.py`, `branch_beta.py`, `training.py`; 16 prototypes, detached teacher, EMA 0.99, two masked student views |
| Task-adapted PDFM | `gwfm/models/pdfm_neighborhood.py`; 16,764 parameters, normalized weighted mean, one masked view |
| Pretraining | `gwfm/training.py`; seeds 11/23/37, Adam 0.001, 300 epochs, cosine schedule, clipping at 5, validation Huber selection |
| Inference and target-excluded interpolation | `gwfm/current.py`; split-isolated encoding, no random downstream views, every observed regional target included |
| MLPs and modality-block comparisons | `gwfm/fit_mlp.py`, `decoders.py`, `fit_figure_probes.py`; train-only scaling, six validation candidates, 48D and 16D predictors |
| Coordinate baselines | `gwfm/baselines/`; training-region donors only; selected health settings match the report: KNN 8/8, IDW 32 with power 2 / 16 with power 1, GPR Matérn-3/2 / RBF |
| Three ablations | `gwfm/ablation/`; remove geographic multiplication, modality awareness, or the prototype auxiliary module, respectively |
| Tables, error maps, uncertainty | `build_figures.py`, `gwfm/visualization/report_statistics.py`; per-seed R², sample SD, and mean per-seed absolute error |
| Figures 1–5 and Table 1 | `README.md` provides a one-to-one file map; `report/figure_reference.json` records manuscript artwork hashes and printed numerical references |

## Verification

- All 54 Python files parse; undefined-name and unused-import checks pass. Source code, comments, README, and textual artifacts contain no Chinese text.
- All 29 unit tests pass, including observed-only gradients, split boundaries, target exclusion, train-only statistics, stale-cache rejection, ablation gradients, manuscript numerical references, and the final image inventory.
- The complete `python -m gwfm all` command passes using the supplied encoder and MLP caches.
- All 15 encoder checkpoints and every saved MLP weight file remain byte-for-byte unchanged. Dataset files and saved predictions also remain unchanged; provenance logs were refreshed.
- The 75 full-embedding heads and 90 modality-block heads replay successfully. All 504 R² entries equal the supplied final run, with maximum absolute difference 0.
- All 176 Table 1 cells, 64 ablation cells, 20 map statistics, six PCA panels, and four relative changes in the Results section match the final manuscript at its displayed precision.
- Every delivered figure was visually compared with the embedded manuscript image. Composition checks reject missing required labels, text outside the canvas, and overlapping text. Figure 1 includes all four panels; Figure 4 includes all six panels.
- The figure command also succeeds with a separate, initially empty figure-output directory using the supplied numerical artifacts. It requires no old figure folders or loose component panels.
- A separate smoke run trains GWFM and PDFM for two epochs and downstream heads for three epochs. Its scores are execution checks, not manuscript results.

Machine-readable checks are in `cleanup_validation.json` and `../figures/source_data/verification.json`. Execution logs are in this directory.

## Limits

This correction does not repeat 300-epoch pretraining or reinstall the environment. Original full-training records remain in `original_run/`. Quantitative figures are regenerated raster exports; Figures 2 and 3 retain the exact supplied image bytes. The report PDF preserves the author's final content, including the GitHub URL/commit placeholder that still needs repository information before submission. The historical comparison columns in the supplied README remain historical references; the code reproduces its current-run columns.
