# Learn a shared morphology–expression space

Start with [shared_space_morphology_expression.ipynb](../shared_space_morphology_expression.ipynb).
The notebook opens the saved CPU pilot by default. Its short explanations distinguish
learning a representation, evaluating it, and drawing a 2D picture. Set `RUN_NEW=True`
to run a new experiment; each execution creates a new output directory.

## Prerequisites and execution

- Observed binary spatial expression with **nucleus IDs** and an explicit cell-to-nucleus bridge.
- Frozen morphology NPZ (`label_ids`, `features`) or historical NPY (cell ID in column zero).
- Nucleus anchors, bilateral groups, manual display labels and cell/nucleus bounding-box tables.
- NumPy, pandas, SciPy, scikit-learn, h5py, matplotlib, joblib, PyYAML and threadpoolctl.
- `mofapy2==0.7.3`; PyTorch for the optional neural pilot and umap-learn for optional displays.

The task environment is `/g/kreshuk/frazer/envs/morpho_shared_space/bin/python`.
It is a venv overlay that inherits the existing `MorphoFeats_dev` packages and
adds MOFA2 without changing that base environment. Select its Python interpreter
as the notebook kernel. `scripts/create_shared_space_environment.sh` documents
and recreates this approach with optional base-Python and target-directory arguments.
Each run records Python/package versions and `pip freeze`; the inherited base must
also remain available. This is not a self-contained conda environment.

```bash
/g/kreshuk/frazer/envs/morpho_shared_space/bin/python scripts/run_shared_space.py \
  --config configs/sites/shared_space_platynereis.yaml
```

The site YAML contains actual local inputs. The portable
[template](../configs/shared_space.example.yaml) uses `/path/to` placeholders.
An experiment can extend either file; only the specified top-level sections/keys
are replaced. For example, save this next to the portable template:

```yaml
extends: sites/shared_space_platynereis.yaml
analysis:
  max_cells: null       # All eligible matched cells; the default CPU pilot uses 4,000.
  factors: 20          # Compare with the prespecified ten-factor pilot.
  cca_ridge: 0.1
  methods: [rcca, pls, mofa2_ridge]
  run_umap: false       # All numerical evaluation remains available.
```

Do not tune several settings using the same reported test set and then call the
best score an unbiased final result. This workflow uses fixed settings and two
exploratory holdouts; it does not implement nested hyperparameter selection.

## What the models learn

All methods receive the same training cells. Morphology is centered on training
features, row-L2-normalized, projected to up to 40 PCs, and each retained PC is
scaled to unit training variance. Very small/zero-variance PCs are removed.
This final scaling is an explicit choice: it gives lower-variance PCs more weight
than the earlier notebook's unwhitened representation. Expression keeps its
binary values; training-constant genes are removed from alignment only. All 201
genes remain visible in predictive evaluation.

- **Ridge CCA:** centers both views, adds `cca_ridge × mean covariance diagonal`
  to each within-view covariance, and decomposes the whitened cross-covariance.
  Separate training score standard deviations put the two projections on comparable
  scales. This is a regularized correlation objective, not sklearn's unregularized CCA.
- **PLS:** `PLSCanonical(scale=False, algorithm='svd')` learns covarying directions.
  It receives the same preprocessing and uses training score scaling. This is a
  finite-rank baseline, without the covariance penalty used by ridge CCA.
- **MOFA2 + ridge projections:** native MOFA2 trains a Gaussian likelihood on
  morphology PCs and a Bernoulli likelihood on unscaled 0/1 expression. ARD and
  spike-and-slab weights are enabled, view weighting is disabled, and no factor
  dropping is requested. Its training factors are inferred jointly from both views.
  Two additional ridge regressions predict those factors from each view separately;
  `alpha = n_train × mofa_head_ridge`. The heads and a shared training-factor scale
  are frozen for test cells. This is an explicit inductive extension, **not native
  MOFA posterior inference**. Scores assess the composite method. Head training R²,
  native weights, native HDF5 and ELBO traces are saved to diagnose the approximation.
- **Paired neural pilot:** two CPU MLPs (64 hidden units, ten latent coordinates)
  use bidirectional InfoNCE with cosine similarity/temperature 0.2 and within-view
  reconstruction (morphology MSE, expression Bernoulli cross-entropy). AdamW uses
  learning rate 0.001, weight decay 0.0001, batches up to 256 and 60 fixed epochs.
  There is no test-based early stopping. Similar biological cells treated as
  contrastive negatives are a limitation; this is a small baseline, not a tuned
  biological foundation model.
- **Shuffled-pair CCA:** breaks training pairings and provides an alignment control.
  Its downstream gene probe is still trained on correctly paired training genes,
  so it is not a no-information gene-prediction control.

References: [PLS/CCA](https://scikit-learn.org/stable/modules/cross_decomposition.html),
[MOFA2 likelihoods and limitations](https://biofam.github.io/MOFA2/faq.html),
[MOFA2 tutorials](https://biofam.github.io/MOFA2/tutorials.html).

## What is actually held out

The default cohort is a uniform sample of 4,000 cells from the common nonempty
expression/morphology cohort. Both morphology representations use exactly this
sample. Source rows, exclusions, unmapped nuclei and both ID namespaces are saved.

Random holdout splits known bilateral groups. Spatial holdout splits 12
coordinate-derived regions and additionally removes training partners of test
cells. The fraction refers to groups, so the test cell counts differ between
schemes. The region construction uses coordinates from all selected cells,
without gene or manual labels. Spatial groups have unbuffered boundaries and
are not independent animals. Within each scheme every method uses the same
training/test cells and the same retrieval candidates.

No test morphology or expression fits a preprocessor, shared model, regression
head or predictive decoder. The frozen image encoders themselves may have seen
the animal during their upstream training; this experiment does not retrain or
validate those encoders end to end.

Three questions are evaluated separately:

1. **Alignment:** can a morphology-only test embedding retrieve its exact
   expression-only counterpart among up to 800 test candidates, in either
   direction? Recall@10 has chance level `10 / candidate_count`. Ties receive
   their analytical random-order expectation; repeated binary profiles make
   exact-cell retrieval intrinsically ambiguous. Mean paired latent correlation
   and a within-region shuffled-test correlation are secondary diagnostics.
2. **Gene prediction:** the same per-gene logistic probe is fitted on training
   morphology-only scores for every method. Test expression is only a target.
   Compare with direct morphology PCs, training prevalence, and a position-only
   RBF-feature model. AP/AUC macro averages require at least ten positives and ten
   negatives in the same test genes; all-gene log loss/Brier also retain rare and
   constant genes. The position model is an anatomical control, not a complete
   causal confound adjustment.
3. **Reverse prediction:** a ridge decoder predicts the standardized morphology
   PCs from expression-only scores. This evaluates the compressed morphology
   representation, not reconstruction of real images or meshes.

PCA display axes are fitted once to both **training** projections and applied to
both test projections. Optional UMAP fits the combined test points only for
visualization; no validation score uses its coordinates. Two separated panels
share the same fitted axes. Different methods/runs have different axes.

## From factors to cells and hypotheses

Five genes per factor are selected by absolute **training** Spearman correlation
with the morphology-only factor. Their test correlations are shown separately.
These are descriptive, supervised associations, not p-values or causal effects.
Native MOFA expression loadings and linear CCA/PLS projection weights are exported
separately and have different mathematical meanings.

The notebook also relates factors to positive-call burden, position, voxel count
and axis-aligned bounding-box aspect ratio. Voxel counts are not physical volume
without voxel-size provenance. Bounding-box aspect ratios depend on orientation
and are not mesh elongation. Cell/nucleus count ratios are deliberately not formed
across potentially different segmentation grids. No physical surface or image
texture measurements are invented from embedding dimensions.

`inspection_candidates.tsv` lists high/low held-out cells along each
morphology-only factor, with cell and nucleus IDs. Inspect these in the existing
object viewer/MoBIE workflow, then quantify shape/texture properties with known
units. Follow up within independently defined cell types and anatomical regions.
Gene calls describe thresholded registered signal: zero is not proven silencing.
Gene perturbation or independent biological evidence is needed for a mechanistic
claim. The scRNA reference matrices, inherited atlas UMAP and scLocator assignment
scores are not used to train these paired models; the H5AD contributes only gene names.

## Outputs and reuse

- `provenance.json`, `config.yaml`, `environment.freeze.txt`: input/code fingerprints,
  package versions, settings, fit scope and limitations.
- `tables/scores.tsv`, `splits.tsv`, cohort/retrieval audits: matched comparisons.
- `tables/factor_gene_associations.tsv`, `factor_measurement_associations.tsv`,
  `inspection_candidates.tsv`: interpretation and inspection starting points.
- `models/<morphology>/<scheme>/<method>/`: fitted `model.joblib`, gene probabilities,
  gene metrics, held-out modality-specific coordinates, and training diagnostics.
- `figures/`: PNG/SVG performance plots and paired displays.

`project_saved` in `shared_space.experiment` can project morphology and expression
independently. Keep the training feature/gene order and matching encoder recipe;
only load trusted joblib files. Vectors retain `label_ids` as cell IDs and separate
`nucleus_ids`. Choose the namespace matching the segmentation for MoBIE import.
The saved models remain training-split models, not refits on all cells.
