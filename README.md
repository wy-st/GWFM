# GWFM Description and Reproducibility Guide

This repository implements a small-scale geographically weighted foundation model. The main model, GWFM, combines dual-branch encoding, explicit modality fusion, and prototype-assisted self-supervised learning. Downstream prediction uses a two-layer MLP trained on frozen regional embeddings. The implementation retains the report’s main baseline, the Population Dynamics Foundation Model (PDFM), the KNN, IDW, and GPR baselines, three ablation experiments, and result visualizations.

The code follows the experimental setup in the GWFM report. Attribute interpolation includes every region with an observed target modality within each split. The `data/`, `checkpoints/`, and `results/` directories contain the supplied final run, originally generated, trained, and evaluated from an empty directory. This cleaned package reuses those numerical artifacts and regenerates the report figures; it does not claim a new full training run. Comparisons with the original results are provided below.

This reproduction used an Apple M4 Max with 128 GB of memory, macOS 26.6.2 (ARM64), and an existing dedicated Conda environment named GWFM: Python 3.12.2, CPU execution with 2 compute threads, PyTorch 2.6.0, and PyTorch Geometric 2.6.1.

This work gratefully acknowledges the research ideas introduced by PDFM.

## Repository guide

This submission contains only the experiments described in the final report and this README: GWFM, PDFM, KNN/IDW/GPR, three GWFM component ablations, 16D modality-block comparisons, and the associated figures. The PDF report is [GWFM_report.pdf](GWFM_report.pdf).

```text
gwfm/                 Data generation, encoders, training, and evaluation
  synthetic/          Spatial hierarchy, fields, attributes, targets, and missingness
  models/             GWFM branches, modality fusion, prototypes, and PDFM
  ablation/           The three reported ablations and their paired training
  baselines/          Coordinate-only KNN, IDW, and GPR
  visualization/      Report plotting helpers
configs/              Exact reported settings and fixed interpolation cohorts
tests/                Supervision, split, masking, caching, and metric checks
data/                 Supplied final synthetic dataset
checkpoints/          Fifteen trained encoders and exported embeddings
results/              Selected prediction heads, tuning records, and predictions
figures/              One final image per manuscript figure/table, plus source_data/
audit/                Cleanup checks and original training records
```

Read the core code in this order:

| File | What to follow |
| --- | --- |
| `gwfm/__main__.py` | The pipeline stages and command-line options |
| `gwfm/synthetic/data.py` | Seeded generation from spatial fields to observations and targets |
| `gwfm/data.py` | Observation masks, isolated graph splits, training statistics, and SSL masking |
| `gwfm/models/modality_structure.py` | Three modality-specific tokens and explicit shared fusion |
| `gwfm/models/visible_graph.py` | Visible neighbors, learned distance weights, and `z = beta * h_k` |
| `gwfm/models/prototype_beta.py` | Observed-modality prototype similarity and teacher consistency |
| `gwfm/training.py` | Main-model and PDFM training, checkpoint selection, and embedding export |
| `gwfm/ablation/model.py` | The precise component removed in each ablation |
| `gwfm/current.py` | Final target-excluded interpolation, health inference, and numerical replay |
| `gwfm/fit_mlp.py` | Validation-only MLP selection and cache validation |
| `build_figures.py` | The single export entry point for Figures 1–5 and Table 1 |

The model modules are building blocks of the same final architecture, not separate experimental candidates. Their parameter names are retained so the supplied checkpoints load exactly. Health-target labels and filenames are English throughout this package.

## Environment setup

Run the following commands from this directory to create a CPU environment with Python 3.12.2. Dependencies are listed in requirements-lock.txt.

```bash
conda create -n GWFM --override-channels -c conda-forge python=3.12.2 pip -y
conda activate GWFM
python -m pip install -r requirements-lock.txt
python -m pip check
```

Alternatively, run conda env create -f environment.yml, followed by conda activate GWFM. The current reproduction used an existing dedicated environment without reinstalling dependencies.

## Reproduction workflow

Run the pipeline in a new directory, such as fresh_run, without changing the prescribed seeds or training epochs:

```bash
python -m gwfm all --work-dir ./fresh_run
```

The experiment includes GWFM, PDFM, and three ablations. Each setting is trained for 300 epochs with seeds 11, 23, and 37. For each downstream two-layer MLP, hidden width and weight decay are selected on the validation set before test-set evaluation.

The pipeline proceeds through data generation and fixed splitting, main-model and baseline pretraining, ablation pretraining, embedding extraction, health-burden inference and target-excluded interpolation, 16-dimensional modality-block comparisons, figures, and verification. Splits are fixed before pretraining; logs identify the model, seed, and epoch.

The exact commands for running individual stages are listed below. Use the same output directory for every stage:

```bash
python -m gwfm generate  --work-dir ./fresh_run
python -m gwfm train     --work-dir ./fresh_run
python -m gwfm ablations --work-dir ./fresh_run
python -m gwfm extract   --work-dir ./fresh_run
python -m gwfm contexts  --work-dir ./fresh_run
python -m gwfm decode    --work-dir ./fresh_run
python -m gwfm baselines --work-dir ./fresh_run
python -m gwfm evaluate  --work-dir ./fresh_run
python -m gwfm probes    --work-dir ./fresh_run
python -m gwfm figures   --work-dir ./fresh_run
python -m gwfm verify    --work-dir ./fresh_run
```

generate rejects incompatible configurations and incomplete caches. Before reusing outputs, train and ablations verify configuration, input, and weight hashes; downstream MLPs verify hashes of training and validation features and labels. Evaluation links labels, splits, inputs, tuning records, selected prediction heads, and predictions. Use a new output directory whenever parameters or data change. A new directory is also recommended after an interrupted run. Do not mix checkpoints or results from different runs.

## Verify the supplied results

```bash
python -m gwfm verify
python -m gwfm evaluate
python -m gwfm figures
python -m unittest discover -s tests -v
python -m gwfm.smoke --work-dir ./smoke_run
```

verify recomputes embeddings from saved weights using naturally observed inputs and target-excluded inputs. It replays 75 full-embedding MLPs and checks target IDs, training statistics, and 252 metrics. probes separately fits and verifies 90 modality-block MLPs and 252 metrics. extract exports embeddings for all nodes and for regions.

smoke checks the pipeline with 200 regions, 1,600 subregions, 2 pretraining epochs, and 3 prediction-head epochs. Its R² values do not represent the performance reported in the paper. Unit tests cover missing-data supervision, single-view PDFM masking, hierarchical splits, training statistics, exclusion of test labels, complete coverage of observed interpolation samples, rejection of duplicate or omitted IDs, and rejection of stale caches.

## Experimental settings

### Data and coordinates

The data and geometry seed is 20262001. The hierarchy contains 600 regions, 4,800 subregions, and 14,400 atomic Voronoi polygons. Coordinates in [0,1]² represent a 200×200 km plane with Euclidean distances.

The modalities are climate, pollution, and sociodemographic, with 4 attributes per modality. Natural missingness affects an entire modality at once. The mechanism is sigmoid(am-bm*u), with b = 0.8/1.8/1.1 and expected missingness rates of 15%/35%/25%. The realized missingness rates are 12.83%/35.50%/24.17% at the regional level and 14.94%/35.46%/25.46% at the subregional level, respectively, in the same modality order.

### Data splits

Seed 1729 partitions 10 groups of 60 regions each. Within each group, the training/validation/test allocation is 42/6/12, giving totals of 420/60/120 regions. Subregions inherit their parent regions’ split assignments.

### GWFM encoder

The encoder has 106,297 parameters. Its inputs are 12 attribute features and 3 modality-visibility flags. Hidden width is 64, and the output is a 48-dimensional embedding arranged as 3×16 modality blocks.

The dual branches each use a single GraphSAGE layer implemented in PyG, with mean, max, and sum spatial aggregation. Each node uses up to 8 visible same-scale neighbors (k=8). Distance bandwidths are learned separately for each branch, scale, and modality, and parent–child relations are bidirectional. Prototype learning uses 16 prototypes of dimension 48. Training seeds are 11/23/37.

### GWFM self supervised pretraining

Inputs comprise 12 attributes and 3 modality-visibility flags. Each epoch uses two independently mixed-mask views at each node. The output has 48 dimensions, with 16 dimensions per modality.

The main loss is Huber loss with δ=1. Prototype-consistency and prototype-similarity variance-regularization weights are each 0.05. Training uses 300 epochs, Adam with learning rate 0.001, cosine annealing, gradient clipping at 5, and teacher EMA decay 0.99.

With equal probability, mixed masking samples according to empirical missingness rates or randomly masks 1–3 modalities. Only originally visible values are masked. Fixed validation masks with a 30% masking probability select the encoder with the lowest reconstruction error.

### GWFM downstream predictors

Predictors take either a 16-dimensional modality block or the full 48-dimensional embedding. Each is a two-layer MLP with linear layers and GELU activation. Candidate hidden widths are 32/64/128. Hyperparameters are selected by validation MSE after standardizing targets using training-label statistics.

Training uses at most 500 epochs, patience 40, min_delta 1e-6, weight decay 1e-4/0.01, and AdamW with learning rate 0.003. The decoder seed is fixed at 11.

### PDFM baseline

The baseline has 16,764 parameters, takes 12 attributes and 3 visibility flags, and outputs 48-dimensional embeddings. It uses the same mixed-mask sampling rule as GWFM but only one masked view per epoch. Reconstruction includes only artificially masked observations and uses Huber loss with δ=1. Training uses Adam with learning rate 0.001, cosine annealing, 300 epochs, gradient clipping at 5, and fixed validation masks for checkpoint selection.

The network is a single-layer GraphSAGE model with normalized distance-weighted mean aggregation. Regional Gaussian bandwidth is 14.402076958647303 km, and edges with weights ≥0.1 are retained. For each modality, subregions connect to ≤8 visible neighbors, excluding themselves, with relative inverse-square distance weights. Bidirectional parent–child edges have weight 1.

Training seeds are 11/23/37. This is an unofficial, task-adapted implementation trained from scratch; it does not use official PDFM embeddings.

### KNN baseline

Euclidean K-nearest-neighbor regression averages the k nearest available observations with equal weights. Candidate values are k ∈ {1, 3, 5, 8, 16, 32}, selected by validation mean squared error. If fewer than k neighbors are available, all available neighbors are used.

### IDW baseline

Inverse-distance weighting averages the k nearest available observations using normalized weights proportional to the inverse p-th power of distance. Candidate neighbor counts are k ∈ {8, 16, 32}, and candidate distance-decay powers are p ∈ {1, 2, 3}. The combination is selected jointly by validation mean squared error.

### GPR baseline

The Gaussian process kernel is a constant kernel × spatial kernel + white-noise kernel. The spatial kernel is selected from RBF and Matérn (ν = 1.5). The constant-kernel coefficient starts at 1 with bounds [0.001, 1000]; the length scale starts at 0.1 with bounds [0.005, 2]; and the noise variance starts at 0.05 with bounds [10⁻⁵, 10].

### Inference and interpolation

For neural methods (GWFM, PDFM, and the ablations), health-burden inference uses embeddings of naturally observed inputs. For interpolation, the target modality is excluded from every regional node within each split before encoding. Other modalities and naturally visible subregional attributes remain available; no additional random downstream views are introduced.

Interpolation includes every region with an observed target modality in each split. Training/validation/test counts are 360/53/110 for climate, 271/36/80 for pollution, and 320/46/89 for sociodemographic attributes. Naturally missing targets are excluded from supervision and scoring. For health inference task, the 600 regions follow a fixed training/validation/test split of 420/60/120 regions, respectively.

Spatial methods (KNN, IDW, and GPR) use training-region labels for health-burden inference and observed attributes of the corresponding training regions for interpolation. Hyperparameters are selected on the validation set. All methods use the same target samples. Modal-1/2/3 denote 16-dimensional blocks of the same encoder; Multi-Modal denotes the full 48-dimensional embedding.

### Evaluation

R² is calculated separately for each attribute or health target and then averaged with equal weights. Neural methods report the mean and sample standard deviation over seeds 11/23/37; spatial methods use one fixed run. Error maps first calculate absolute error for each seed and then average across seeds for each region.

### Supervision boundaries

Pretraining gradients for GWFM and PDFM come only from observed attributes of the 420 training regions and their subregions. GWFM students receive two masked views. The teacher receives the original observations before artificial masking and cannot access naturally missing ground truth. PDFM uses one masked view. Validation masks select the encoder but do not update parameters.

Health-burden inference retains only natural missingness. Interpolation additionally applies deterministic exclusion of the regional target modality. Each split is encoded independently, with cross-split edges removed. Spatial baselines use training regions as donors only. Random masks do not determine training, validation, or test membership.

All fitted statistics use training observations or labels only. Self-supervised pretraining does not load health labels or the ground truth of naturally missing attributes. Originally observed values may supervise reconstruction after artificial masking. For both health-burden inference and attribute interpolation, spatial baselines use training regions as donors and never use validation or test targets as donors.

`configs/gwfm.json` and `configs/pdfm.json` control settings such as hidden width, spatial neighbors, and learning rate. Validation checks enforce the fixed 16-dimensional modality blocks, decoder depth, retention of the hierarchy, and GWFM auxiliary-loss settings.

configs/ablations.json records the paired ablation protocol. The full reproduction entry point fixes the three seeds and 300 epochs. For development, an individual encoder can be trained with python -m gwfm train --work-dir ./development --model GWFM --seeds 1 after generating the data.

## Data formats

This section describes the synthetic data files, fields, and supervision boundaries. The dataset contains 600 regional nodes and 4800 subregional nodes, for a total of 5400 nodes. All data come from the project’s deterministic generator; no external records are used. Coordinates in [0,1]² map to a plane with sides of 200 km, and distances are Euclidean. These coordinates are not real latitude or longitude.

| File | Contents |
| --- | --- |
| complementary.npz | x[5400,12]; observed[5400,3] Boolean natural-observation flags; pos[5400,2]; node_id/node_type/parent/group; edge_index[2,E], edge_type; modality_slices[[0,4],[4,8],[8,12]] |
| geometry.npz | region_xy, child_xy, child_area, parent; region_rook and subregion_rook; auxiliary arrays used to construct the geometry |
| split.npz | region_id[600], group[600], split[600]; 0=train, 1=validation, 2=test |
| targets.npz | region_id[600], y[600,2], ordered as Respiratory burden / Heat-related burden |
| schema.json | Names and physical units for all 12 attributes, in the exact order of the x columns |
| truth.npz | Complete attributes before missingness, latent fields, signals, noise, varying coefficients, and related values; must not be read during model training |
| missingness.npz | Urbanicity and specified missingness probabilities, for generator diagnostics only |
| population.npz | Synthetic population mass, area, density, and urbanicity |
| regions.geojson / subregions.geojson | Synthetic polygons and node IDs for mapping |
| metadata.json | Original generator configuration, random seeds, and data description |
| ../configs/interpolation_cohorts.json | Fixed interpolation IDs covering all observed regions in each split, together with visibility, hierarchy, and split hashes; checked for omitted, duplicate, or extra IDs |
| standardizer.npz | Per-scale, per-attribute means and standard deviations in original units, computed only from training observations; used for baseline fallback and NRMSE |

Node IDs 0–599 identify regions; IDs 600–5399 identify subregions. A region has parent=-1, while each subregion has a parent region with an ID in 0–599. The number of subregions varies across parents; each region does not necessarily contain exactly eight subregions.

In the raw data, edge_type uses 0 for same-scale regional spatial edges, 1 for child→parent edges, and 2 for parent→child edges. In the distance graphs rebuilt by the model, the codes are 0 for regional distance edges, 1 for subregional distance edges, 4 for child→parent edges, and 5 for parent→child edges. Do not mix the codes used at these two stages. GWFM’s actual visible-neighbor graphs explicitly distinguish spatial, up, and down relations using separate tensors for each modality.

gwfm.data.load_graph converts NumPy arrays into a dictionary of torch.Tensor objects so that hierarchical metadata can be carried alongside the features. PyG SAGEConv and aggregation components directly use x and edge indices of shape [2,E]. These fields map directly to torch_geometric.data.Data(x=..., edge_index=..., pos=...). subset_graph remaps parent-node and edge indices and completely removes cross-split edges.

## Synthetic data generation

Both the data and geometry seeds are 20262001. The following sections summarize the current generator. Full-precision parameters, attribute-specific perturbation coefficients, and the order of random sampling are specified in configs/dataset.json and gwfm/synthetic/.

### Spatial hierarchy

On a 200 km × 200 km plane, settlement density d(s) combines a background of 0.24 with two Gaussian components. Their centers are (58,76) and (152,144) km, their amplitudes are 1.8 and 1.25, and their scales along the two coordinate axes are (32,42) and (26,34) km, respectively. Uniformly sampled candidate points are accepted with probability d(s)/3.29 to generate 14,400 atomic Voronoi polygons. Adjacent polygons are then merged into 4,800 connected subregions and 600 connected regions. Each subregion has exactly one parent region.

### Random fields and modality mixing

Let normalized coordinates be s=(x,y)∈[0,1]². Each field uses the same 97×97 kernel centers over [−0.35,1.35]², giving 9,409 Gaussian kernels. Independent standard-normal coefficients weight the kernels. Kernel weights are L2-normalized at each location so that the theoretical pointwise variance is 1. The seed determines the positions and number of field peaks and troughs; they are not prescribed.

$$
g(s)=\mathrm{sigmoid}[(x-0.50-0.17\sin(2\pi y+\phi ))/0.10],   \phi \sim \mathrm{Uniform}(-\pi ,\pi )
$$

The three base fields F₁, F₂, and F₃ have bandwidths (0.128,0.064,0.192)×[0.65+0.70g(s)]. The positive multiplier ranges from 0.65–1.35, producing smoother fields where g is larger. Independent background fields C, I, and U each have bandwidth 0.176 and represent shared background, service-related attributes, and health targets, respectively.

Climate, pollution, and sociodemographic factors L_c, L_p, and L_s combine F₁–F₃ using weights (1,0.15,0.10), (0.65−1.10g,1,0.10), and (−0.35+0.70g,0.15,1), respectively. These weights determine the strength and sign of each contribution; the contribution of F₁ to pollution can reverse sign across space. Each row is L2-normalized to prevent the weight-vector magnitude from further inflating variance. The factors are then multiplied by amplitudes (0.85+0.30g,1.15−0.30g,0.80+0.40g), with means (2g−1)(0.15,−0.12,0.105) and the shared term 0.18C added.

### Population and urbanicity

Parent-region-shared and subregion-independent perturbations are added to log d(s), then exponentiated to obtain d*. A total population of 19.2 million is allocated across subregions in proportion to area A×d*. Regional population and area are the sums of their subregional values. Density is ρ=N/A. The urbanicity proxy u is standardized log ρ; both scales use the mean and standard deviation of the true log densities of all subregions. This proxy represents population concentration and may be positive or negative.

### Twelve attributes and measurement

The initial link-scale values η are defined in the table below. Attribute-specific Gaussian perturbations, shared within parent regions and independent across subregions, are added and rescaled to retain the original mean and standard deviation. The resulting η* values are then transformed. Population density is already perturbed during population allocation and is not perturbed again here. σ denotes the sigmoid function.

| Modality | Attribute | Initial η | Transformation |
| --- | --- | --- | --- |
| Climate | Temperature (°C) | $L_{c}+0.12u$ | $18+4\eta ^{*}$ |
| Climate | Relative humidity | $0.5-0.35L_{c}+0.70C$ | $\sigma (\eta ^{*})$ |
| Climate | Annual precipitation (mm) | $6.6-0.15L_{c}+0.35C$ | $\exp(\eta ^{*})$ |
| Climate | Hot-day proportion | $-3+0.70L_{c}+0.15u$ | $\sigma (\eta ^{*})$ |
| Pollution | PM₂.₅ (µg/m³) | $2.4+0.40L_{p}+0.15u$ | $\exp(\eta ^{*})$ |
| Pollution | NO₂ (µg/m³) | $2.6+0.45L_{p}+0.30u$ | $\exp(\eta ^{*})$ |
| Pollution | Ozone (µg/m³) | $3.5+0.25L_{c}-0.20L_{p}+0.20C$ | $\exp(\eta ^{*})$ |
| Pollution | Emissions index | $0.50L_{p}+0.20u$ | $\exp(\eta ^{*})$ |
| Sociodemographic | Population density (people/km²) | $\log \rho $ | $\rho =N/A$ |
| Sociodemographic | Deprivation index | $-0.4+0.75L_{s}+0.10C$ | $\sigma (\eta ^{*})$ |
| Sociodemographic | Older-adult proportion | $-0.7+0.45L_{s}-0.12u+0.25I$ | $0.05+0.40\sigma (\eta ^{*})$ |
| Sociodemographic | Healthcare accessibility | $0.5-0.50L_{s}+0.60u+0.25I$ | $\sigma (\eta ^{*})$ |

Before measurement error is added, regional climate and pollution attributes are area-weighted subregional means. Deprivation, the older-adult proportion, and healthcare accessibility are population-weighted means. Regional density equals total population divided by total area. Gaussian measurement errors on the link scale are then added independently at the two scales. Standard deviations are 0.06/0.08 for regions/subregions, halved for precipitation and pollution; density uses 0.10. Log-scale errors for positive attributes are adjusted by subtracting half the variance. Temperature error is applied to (T−18)/4.

### Two health targets

Using measurement-error-free subregional attributes, define h=(T−18)/4, p=(log PM₂.₅−2.4)/0.4, and v=(deprivation−0.4)/0.2+0.4(older-adult proportion−0.18)/0.08. Overbars denote population-weighted regional means. q is the weighted mean of subregional hv, and s_P is the weighted within-region standard deviation of p:

$$
y^{resp}=\beta _{1}\overline{h}+\beta _{2}\overline{p}+\beta _{3}\overline{v}+0.20W\overline{p}+0.15U+\epsilon 
$$

$$
y^{heat}=\beta _{4}q+\beta _{2}\overline{p}+\beta _{5}s_{P}+0.15W\overline{h}+0.15U+\epsilon 
$$

The coefficients follow β_k=b_k[1+0.65tanh(f_k)], with b=(0.60,0.60,0.40,0.50,0.35). The five independent coefficient fields f_k have bandwidths (0.28,0.13,0.21,0.10,0.18). W contains regional Gaussian distance weights with bandwidth 0.09 (18 km), excludes self-links, and is row-normalized. Noise is sampled independently for the two targets: ε∼N(0,{0.35[0.75+0.50g]}²). Respiratory burden is primarily additive, whereas heat-related burden includes an interaction and within-region variation. Both targets are synthetic continuous scores.

### Natural missingness and fixed splits

At each scale, natural missingness is sampled for entire modalities using sigmoid(a_m−b_m×u), with b=(0.8,1.8,1.1). The intercepts a are calibrated to expected missingness rates of 15%/35%/25%, and realized rates are checked to lie within 10%–40%. Regions and subregions are sampled separately; zero placeholders and visibility flags are stored separately.

Coordinates define 10 spatial strata. Within each stratum, seed 1729 assigns 42/6/12 regions to training/validation/test sets, yielding totals of 420/60/120 regions. Subregions inherit their parent regions’ split assignments. Splits are fixed before SSL masking, and model standardization uses training observations only.

This is a controlled, simplified simulation. It does not fit the empirical distribution of each real-world attribute or prescribe a population power law. The same generation command saves the complete data, labels, natural-visibility masks, and splits. Distributional realism and generalization to real-world data require further validation.

## Outputs and report mapping

| Location | Purpose |
| --- | --- |
| data/ | Synthetic graph, coordinates, IDs, natural masks, labels, splits, and data diagnostics |
| checkpoints/{GWFM,PDFM}/seed*/ | Models, training records, and 48-dimensional embeddings for all nodes and for regions |
| checkpoints/ablations/ | Weights, training records, and embeddings for the three ablations |
| results/current/inference/ | Health-inference heads, tuning records, predictions, and test_summary.csv |
| results/current/runs/ | Interpolation heads, tuning records, and frozen predictions |
| results/current/fixed_contexts/ | Embeddings from naturally observed inputs and from inputs with the regional target modality excluded |
| results/figure_probes/ | Downstream results for the three 16-dimensional modality blocks |
| figures/ | Exactly six final images: Figures 1–5 and Table 1, numbered as in the report |
| figures/source_data/ | Plotted values, PCA arrays, manuscript comparisons, and source/output hashes |
| gwfm/visualization/ | Data, performance, ablation, and PCA plotting helpers; `palette.py` contains shared figure colors and table labels |
| report/figure_reference.json | Final manuscript artwork hashes and independently transcribed printed values |
| audit/ | Cleanup verification and original full-training records in `original_run/` |

The directory is organized around the report’s experiments and retains the original model architecture and all research components reported in the paper.

### Final manuscript figures

Run `python -m gwfm figures` to rebuild the quantitative figures and verify their values against the final manuscript. The export keeps one image for each figure or table. Component panels are composed in a temporary directory and are not retained as duplicate figures. There is no separate `performance`, `current_version`, or preview directory.

| Manuscript item | Final file | Content and source |
| --- | --- | --- |
| Figure 1 | [Figure1_data.png](figures/Figure1_data.png) | Three modality maps, all 12 attribute names, urbanicity sampling, mask counts, and both health-target maps; regenerated from the final dataset |
| Figure 2 | [Figure2_heterogeneity.jpg](figures/Figure2_heterogeneity.jpg) | Exact schematic artwork extracted from the final Word manuscript; its reported correlations and Moran statistics are checked against the data |
| Figure 3 | [Figure3_framework.jpeg](figures/Figure3_framework.jpeg) | Exact framework artwork extracted from the final Word manuscript |
| Figure 4 | [Figure4_performance.png](figures/Figure4_performance.png) | Inference and interpolation results; panel c is heat-related burden, and panel f is temperature; regenerated from saved predictions |
| Figure 5 | [Figure5_ablation.png](figures/Figure5_ablation.png) | Three ablations and the first three regional PCs of both GWFM branches; regenerated from saved predictions and the seed-11 checkpoint |
| Table 1 | [Table1_performance.png](figures/Table1_performance.png) | All 176 printed R² values for full embeddings, modality blocks, and spatial baselines |

The Figure 2 maps and scatter locations are schematic, as stated in its caption. Figure 3 is supplied artwork. These two files are copied without image regeneration, including when using a separate `--work-dir`; their hashes must match the final manuscript. Figures 1, 4, and 5 and Table 1 are numerical plots, rebuilt from data and checked at the manuscript's displayed precision. The Figure 1 artificial mask is illustrative and never selects downstream samples.

The `natural_graph_partition0.npz`, `natural_graph_partition1.npz`, and `natural_graph_partition2.npz` files under the main-model checkpoints record the final training, validation, and test graphs, respectively. They contain node IDs, visible-neighbor edges, hierarchy edges, and weights. They are reproducibility records for the reported models. Candidate MLP files belong to the reported six-setting validation search; the selected head and all tuning scores are recorded in each `*_selection.json`.

## Reproduction verification record

The original final run supplied with this package regenerated data from an empty directory, trained all 15 encoders for 300 epochs each, selected 165 downstream MLPs from 990 candidate fits, and completed the spatial baselines and figures. All methods used the full set of observed interpolation samples. Compared item by item with the previous run using the same all-observed-sample protocol, encoder parameters, the existing regional embeddings, and all 504 R² metrics were identical. Original full-training logs are in `audit/original_run/`; cleanup validation is recorded separately in `audit/cleanup_validation.json`.

### Health inference comparison

The reference values are the originally supplied results, which used 521 regions for climate interpolation. The current run uses all 523 observed climate regions. Health-task samples are unchanged. Neural results are means over seeds 11, 23, and 37; spatial baselines use fixed runs. Differences are defined as Δ = current run − reference. Values are shown to nine decimal places; a displayed zero may still conceal a smaller floating-point difference.

| Method | Target | Reference R² | Current run R² | ΔR² |
| --- | --- | --- | --- | --- |
| GWFM | Respiratory burden | 0.833738219 | 0.833892752 | +0.000154533 |
| GWFM | Heat-related burden | 0.455628649 | 0.455520608 | -0.000108041 |
| PDFM | Respiratory burden | 0.797892613 | 0.797892613 | 0.000000000 |
| PDFM | Heat-related burden | 0.304129146 | 0.304129146 | 0.000000000 |
| KNN | Respiratory burden | 0.534164412 | 0.534164412 | 0.000000000 |
| KNN | Heat-related burden | 0.240270078 | 0.240270078 | 0.000000000 |
| IDW | Respiratory burden | 0.544952602 | 0.544952602 | 0.000000000 |
| IDW | Heat-related burden | 0.273857256 | 0.273857256 | 0.000000000 |
| GPR | Respiratory burden | 0.564446766 | 0.564446766 | 0.000000000 |
| GPR | Heat-related burden | 0.271082045 | 0.271082045 | 0.000000000 |
| No geographic weighting | Respiratory burden | 0.815182306 | 0.815182647 | +0.000000342 |
| No geographic weighting | Heat-related burden | 0.446952507 | 0.446840257 | -0.000112250 |
| No modality awareness | Respiratory burden | 0.801398017 | 0.801442560 | +0.000044543 |
| No modality awareness | Heat-related burden | 0.450424038 | 0.450815419 | +0.000391380 |
| No prototype regularization | Respiratory burden | 0.829094092 | 0.829098276 | +0.000004183 |
| No prototype regularization | Heat-related burden | 0.436515879 | 0.438788343 | +0.002272464 |

### Mean attribute interpolation comparison

The table below averages R² across the 12 attributes. Neural methods use the full 48-dimensional embedding and are averaged over three seeds.

| Method | Target | Reference R² | Current run R² | ΔR² |
| --- | --- | --- | --- | --- |
| GWFM | Attribute interpolation | 0.907482409 | 0.907722772 | +0.000240363 |
| PDFM | Attribute interpolation | 0.719308392 | 0.719361571 | +0.000053179 |
| KNN | Attribute interpolation | 0.339018500 | 0.339072930 | +0.000054430 |
| IDW | Attribute interpolation | 0.360849261 | 0.360809461 | -0.000039800 |
| GPR | Attribute interpolation | 0.387572543 | 0.387501348 | -0.000071195 |
| No geographic weighting | Attribute interpolation | 0.889685981 | 0.889658475 | -0.000027505 |
| No modality awareness | Attribute interpolation | 0.880145074 | 0.880168198 | +0.000023124 |
| No prototype regularization | Attribute interpolation | 0.898625336 | 0.898720279 | +0.000094943 |

The current and previous runs using the all-observed-sample protocol match elementwise or metric by metric in their encoder parameters, embeddings, and all R² values, with a maximum absolute difference of 0. The nonzero differences in the tables instead compare the current run with the original results.

## Cleanup validation

The cleaned package passed 29 unit tests and the complete `python -m gwfm all` entry point using the supplied encoder and MLP caches. A separate fresh generation reproduced every array in all eight data archives. The 15 encoder checkpoint files remain byte-for-byte unchanged; 165 selected MLPs were replayed, and all 504 R² entries match the supplied final run exactly. The smoke run trained GWFM and PDFM for two epochs and prediction heads for three epochs; the tests also checked gradients for all three ablations.

The final figure audit checks all 176 Table 1 cells, 64 ablation cells, 20 map statistics, six PCA panels, and the four relative changes quoted in the Results section. All match the final manuscript at its printed precision. Earlier standalone exports with missing panels or obsolete layouts have been removed, along with their export paths and empty directories. Details are in [audit/ALIGNMENT.md](audit/ALIGNMENT.md) and `figures/source_data/verification.json`.

This cleanup did not repeat the full 300-epoch training or reinstall the environment. Original full-training records remain in `audit/original_run/`; the checks above are recorded in `audit/cleanup_validation.json`. The supplied report was exported to PDF with five body pages and one reference page. Its GitHub URL and commit placeholder still needs the final repository information before submission.

## AI use disclosure

This cleanup also used OpenAI Codex to remove unused experiment branches, convert the final Word README to Markdown (including equations), add English explanations, translate artifact labels, and check saved results.

OpenAI Codex and ChatGPT assisted with synthetic data generation, model implementation, training and evaluation scripts, figure generation, report editing, code refactoring, and consistency checks. AI image tools also assisted with text and image layout; Figure 2 is schematic. Numerical tables, prediction-error maps, and statistics in the report were computed from saved data and predictions.

The main workflow supplied task requirements, method equations, existing code, and run outputs, then asked Codex to check equations against the implementation, trace the provenance of weights and predictions, verify missingness masks and data splits, and redraw figures from saved results. Consistency was checked by recomputing outputs and running regression checks on computed statistics. I reviewed and revised the AI-generated content and automated checks and take responsibility for the correctness, interpretation, and reproducibility of the final submission.
