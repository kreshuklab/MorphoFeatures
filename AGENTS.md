# Working in MorphoFeatures

MorphoFeatures is a scientific Python package and Streamlit workspace for morphology
embeddings from segmented electron-microscopy volumes. It supports the published
shape/texture artifacts, MAE, frozen DINOv2/v3, physical fine/coarse views, comparisons,
label transfer, object inspection, and MoBIE export.

## Preserve the user's work and scientific options

- Inspect `git status` before editing. This workspace may contain extensive ongoing
  changes and untracked research notebooks; do not reset, overwrite, or clean them.
- Preserve existing UI controls, configuration fields, CLI options, and workflows.
  Prefer additive changes and backward-compatible defaults. If removing an option is
  necessary, explain the concrete reason and obtain the user's review first.
- Do not delete research data, results, notebooks, scripts, or documentation as
  cleanup without an itemized proposal and explicit approval. In particular preserve
  `outputs/notebooks/04_platyneris_mae/`.
- Keep temporary investigation scripts outside the repository (for example `/tmp`).
  Put generated scientific artifacts under the configured output root. Add maintained
  tests and useful examples, rather than collections of ad hoc scripts or status MD files.
- Do not change existing site configurations or private source paths incidentally.
  Reusable example configurations must clearly distinguish example inputs from real data.

## Scientific contracts

- Object identity is explicit: cell IDs and nucleus IDs are different namespaces.
  Join through an explicit mapping; never infer identity from row position, numeric
  overlap, or rounded floating-point IDs. Preserve native IDs for inspection and MoBIE.
- Preferred embeddings are NPZ arrays `label_ids` and `features`. Historical NPY files
  put the ID in column zero; that column is never a feature. Validate uniqueness,
  finite values, dimensionality, source dataset and representation provenance.
- Spatial arrays use z/y/x. Record voxel size, units, origin, resolution level and
  coordinate frame. Reads of large/local/remote volumes must be bounded and lazy.
- Distinguish feature-space neighbors, physical spatial neighbors, and bilateral
  candidate partners. `analysis/data/bilateral_neighbors.pkl` is an evaluation mapping,
  not the physical adjacency graph used to generate MorphoContextFeatures.
- The paper's context recipe concatenates the focal embedding with the mean of that
  embedding and its spatial neighbors. Generic mean/max aggregation is a separate
  operation. Label alternatives and historical reproduction limits explicitly.
- Compare representations on matched object cohorts/search populations. Report missing
  data, exclusions, neighborhood coverage, preprocessing, rank conventions and seeds.
- Keep learned preprocessing inside training folds for predictive validation. Never use
  target evaluation labels for adaptation or feature construction. Distinguish spatial,
  object-level and specimen-level validation and exploratory from held-out results.
- Preserve saved published projections/clusters when displaying historical results;
  clearly label a newly fitted projection as a new analysis, not an exact reproduction.

## Where to work

- `morphofeatures/workspace_jobs.py`: worker stage execution and planning;
  `workflows.py`: specialist command definitions.
- `morphofeatures/workspace_state.py`, `workflow_ui.py`: guided workflow drafts/pages.
- `morphofeatures/workflow_validation.py`, `ui_settings.py`: field-specific validation/help.
- `morphofeatures/representation_analysis.py`, `analysis/`: scientific analysis.
- `morphofeatures/analysis_ui.py`, `analysis_variants_ui.py`: results and interactive reruns.
- `morphofeatures/spatial_probe_ui.py`, `analysis/bilateral.py`, `analysis/spatial_context.py`,
  `analysis/published.py`: bilateral retrieval, physical context and saved published layouts.
- `morphofeatures/provenance.py`, `provenance_ui.py`: artifact lineage and presentation.
- `morphofeatures/dashboard.py`, `cli.py`: specialist/legacy tools and command entry points.
- `morphofeatures/data/`: container readers, physical metadata, caches and preparation.
- `analysis/`, `analysis/data/`, `data_mobie/`: historical scripts and published reference data.

Extend the existing Workflow/Analyze/Results structure instead of adding redundant
top-level tabs. Put a short prerequisites checklist at the top of new workflows.
Help text must explain what a setting means, units, consequences and sensible defaults;
never substitute a configuration-key path for an explanation. Validation should identify
the page, section and field, and provide a concrete remedy. Keep advanced controls visible
in appropriate expanders. Support local and SLURM workers through the same stage contract.

## Verification and documentation

- Use `rg` for source discovery; consult nearby maintained tests and documentation.
- Run focused scientific tests for affected behavior, including ID permutations/missing
  IDs, small deterministic fixtures and meaningful failure cases. Run relevant workflow
  and UI integration tests. Do not run costly training or volume scans as routine tests.
- Typical environment here: `/g/kreshuk/frazer/miniforge3/envs/MorphoFeats_dev/bin/python`.
  Standard commands: `python -m pytest <relevant tests>` and `ruff check <changed files>`.
  Dependencies/extras and Python compatibility are defined by `pyproject.toml`.
- Use bounded real-data checks when helpful; report their cohort and limitations. Never
  claim an experiment, test, or UI interaction ran unless it actually did.
- Update the maintained `docs/workspace_workflows.md`, `docs/analysis_and_mobie.md`, or
  other relevant guide with biologist-readable instructions and reusable YAML examples.
- In the final response, state what changed, validation performed, and material limits.
  Do not commit, publish, submit cluster jobs, or launch expensive experiments unless
  the user's task authorizes those actions.
