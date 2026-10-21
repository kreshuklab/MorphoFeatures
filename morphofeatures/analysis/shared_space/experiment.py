"""Reproducible CPU experiment runner and saved-model projection helpers."""

from __future__ import annotations

import contextlib
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.decomposition import PCA
from sklearn.kernel_approximation import RBFSampler
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .data import load_data, make_splits
from .evaluation import (
    GeneDecoder,
    association_matrix,
    column_correlations,
    gene_scores,
    probe_expression_to_morphology,
    retrieval_metrics,
)
from .models import (
    ExpressionFilter,
    LinearAlignment,
    MOFARidge,
    MorphologyPreprocessor,
    PairedNeural,
)

REPO = Path(__file__).resolve().parents[3]


def read_config(path):
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text())
    if "extends" in config:
        base = read_config(path.parent / config.pop("extends"))
        for key, value in config.items():
            if isinstance(value, dict) and isinstance(base.get(key), dict):
                base[key].update(value)
            else:
                base[key] = value
        config = base
    return config


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.generic):
        return clean_json(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(value, path):
    Path(path).write_text(json.dumps(clean_json(value), indent=2, allow_nan=False) + "\n")


def fingerprint(path):
    path = Path(path)
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            h.update(block)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": h.hexdigest()}


def versions():
    result = {"python": platform.python_version(), "executable": sys.executable}
    for package in (
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "mofapy2",
        "torch",
        "joblib",
        "h5py",
        "matplotlib",
    ):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


def _gene_names(config, genes):
    """Read only the small gene dictionary, never the multi-GB altExperiments."""
    path = config["inputs"].get("gene_metadata_h5ad")
    if not path:
        return dict(zip(genes, genes))
    import h5py

    def column(node):
        if isinstance(node, h5py.Dataset):
            return node.asstr()[:] if node.dtype.kind in "OSU" else node[:]
        if "values" in node:
            values = column(node["values"]).astype(object)
            values[node["mask"][:]] = None
            return values
        raise ValueError(f"Unsupported gene metadata encoding: {node.name}")

    with h5py.File(path, "r") as f:
        names = dict(zip(column(f["var/XLOC"]), column(f["var/vis_name"])))
    return {g: names.get(g) or g for g in genes}


def project_saved(bundle_path, *, morphology=None, expression=None):
    """Project rows in the original training feature/gene order, independently.

    Load only trusted joblib bundles. Caller owns the ID join; this function
    does not claim that equal row numbers identify the same biological cell.
    """
    bundle = joblib.load(bundle_path)
    result = {}
    if morphology is not None:
        x = bundle["morphology_preprocessor"].transform(morphology)
        result["morphology"] = bundle["alignment"].transform_morphology(x)
    if expression is not None:
        y = bundle["expression_filter"].transform(expression)
        result["expression"] = bundle["alignment"].transform_expression(y)
    return result


def _fit_model(name, settings):
    if name in {"rcca", "rcca_shuffled", "pls"}:
        return LinearAlignment(
            "rcca" if name.startswith("rcca") else "pls",
            factors=settings["factors"],
            ridge=settings["cca_ridge"],
        )
    if name == "mofa2_ridge":
        return MOFARidge(
            settings["factors"],
            settings["mofa_iterations"],
            settings["mofa_head_ridge"],
            settings["seed"],
        )
    if name == "paired_neural":
        return PairedNeural(
            settings["factors"],
            settings["neural_epochs"],
            settings["neural_hidden"],
            settings["seed"],
            settings["threads"],
        )
    raise ValueError(f"Unknown shared-space method: {name}")


def run(config, *, output=None):
    """Run identical holdouts/search populations for all methods, saving every fit."""
    settings = config["analysis"]
    run_id = datetime.now(ZoneInfo("Europe/Berlin")).strftime("%Y%m%d-%H%M%S-%f")
    output = Path(output) if output else Path(config["output_root"]) / run_id
    output.mkdir(parents=True, exist_ok=False)
    for folder in ("tables", "models", "figures"):
        (output / folder).mkdir()
    manifest = {
        "schema": "paired_shared_space.v1",
        "run_id": run_id,
        "status": "running",
        "created": datetime.now(ZoneInfo("Europe/Berlin")).isoformat(),
        "config": config,
        "versions": versions(),
        "evaluation": "one fixed held-out partition per scheme; fixed hyperparameters; exploratory",
        "mofa_projection": "native MOFA2 Gaussian/Bernoulli training + training-only ridge heads; not native posterior inference",
        "coordinate_frame": "PlatyBrowser source nucleus table, explicitly XYZ; no voxel conversion",
        "coordinate_units": "source table units, not independently verified here",
        "measurement_limits": "voxel-count and axis-aligned bounding-box proxies only; no mesh or texture measurement extraction",
        "limits": [
            "one EM animal",
            "registered thresholded calls, not paired sequencing",
            "spatial holdout has no physical buffer",
            "frozen encoder provenance may include this animal",
            "manual/scLocator labels never used in representation fitting",
            "no mechanistic or causal conclusion",
        ],
    }
    write_json(manifest, output / "provenance.json")
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    print(f"Run directory: {output}", flush=True)
    with threadpool_limits(settings["threads"]):
        data = load_data(config)
        splits, regions = make_splits(data, settings)
        gene_names = _gene_names(config, data.genes)
        data.audit.to_csv(output / "tables/cohort_audit.tsv", sep="\t", index=False)
        pd.DataFrame({"nucleus_id": data.unmatched_nuclei}).to_csv(
            output / "tables/unmapped_expression_nuclei.tsv", sep="\t", index=False
        )
        data.measurements.to_csv(output / "tables/measurement_proxies.tsv", sep="\t", index=False)
        pd.DataFrame(
            {
                "gene": data.genes,
                "name": [gene_names[g] for g in data.genes],
                "positive_cells": data.expression.sum(0),
            }
        ).to_csv(output / "tables/genes.tsv", sep="\t", index=False)
        print(f"Selected {len(data.cells):,} matched cells; {len(data.genes)} genes", flush=True)
        split_frames, results, latent_rows, proxy_rows, inspection_rows = [], [], [], [], []
        for scheme, (tr, te, purged) in splits.items():
            roles = np.full(len(data.cells), "purged", dtype=object)
            roles[tr], roles[te] = "train", "test"
            split_frames.append(
                pd.DataFrame(
                    {
                        "scheme": scheme,
                        "cell_id": data.cells,
                        "nucleus_id": data.nuclei,
                        "role": roles,
                        "spatial_region": regions,
                        "pair_group": data.pairs,
                        "anchor_x": data.xyz[:, 0],
                        "anchor_y": data.xyz[:, 1],
                        "anchor_z": data.xyz[:, 2],
                        "manual_label": data.labels,
                    }
                )
            )
            rng = np.random.default_rng(settings["seed"])
            retrieval_ix = np.sort(
                rng.choice(len(te), min(settings["retrieval_cells"], len(te)), replace=False)
            )
            pd.DataFrame(
                {
                    "cell_id": data.cells[te[retrieval_ix]],
                    "nucleus_id": data.nuclei[te[retrieval_ix]],
                }
            ).to_csv(output / "tables" / f"{scheme}_retrieval_cohort.tsv", sep="\t", index=False)
            y_train_full, y_test_full = data.expression[tr], data.expression[te]
            efilter = ExpressionFilter().fit(y_train_full)
            y_train, y_test = efilter.transform(y_train_full), efilter.transform(y_test_full)
            prior = (y_train_full.sum(0) + 1) / (len(tr) + 2)
            prior_frame, prior_scores = gene_scores(
                y_test_full,
                np.broadcast_to(prior, y_test_full.shape),
                prior,
                data.genes,
                settings["min_test_class"],
            )
            results.append(
                {
                    "scheme": scheme,
                    "morphology": "control",
                    "method": "training_prevalence",
                    "train_cells": len(tr),
                    "test_cells": len(te),
                    **prior_scores,
                }
            )
            prior_frame.to_csv(
                output / "tables" / f"{scheme}_prior_gene_scores.tsv", sep="\t", index=False
            )
            position = make_pipeline(
                StandardScaler(),
                RBFSampler(gamma=0.25, n_components=128, random_state=settings["seed"]),
            )
            pos_tr, pos_te = position.fit_transform(data.xyz[tr]), position.transform(data.xyz[te])
            position_decoder = GeneDecoder(settings["logistic_c"]).fit(pos_tr, y_train_full)
            position_p = position_decoder.predict_proba(pos_te)
            pos_frame, pos_scores = gene_scores(
                y_test_full, position_p, prior, data.genes, settings["min_test_class"]
            )
            results.append(
                {
                    "scheme": scheme,
                    "morphology": "control",
                    "method": "position_RBF",
                    "train_cells": len(tr),
                    "test_cells": len(te),
                    **pos_scores,
                }
            )
            pos_frame.to_csv(
                output / "tables" / f"{scheme}_position_gene_scores.tsv", sep="\t", index=False
            )
            np.savez_compressed(
                output / "models" / f"{scheme}_position_predictions.npz",
                label_ids=data.cells[te],
                nucleus_ids=data.nuclei[te],
                genes=data.genes,
                probabilities=position_p.astype(np.float32),
            )
            joblib.dump(
                {"position_transform": position, "decoder": position_decoder},
                output / "models" / f"{scheme}_position.joblib",
            )
            for morph_name, raw_morph in data.morphology.items():
                prep = MorphologyPreprocessor(settings["morphology_pcs"], settings["seed"]).fit(
                    raw_morph[tr]
                )
                x_train, x_test = prep.transform(raw_morph[tr]), prep.transform(raw_morph[te])
                for method in ["morphology_direct", *settings["methods"]]:
                    start = time.monotonic()
                    directory = output / "models" / morph_name / scheme / method
                    directory.mkdir(parents=True)
                    print(
                        f"{morph_name} / {scheme} / {method}: train={len(tr)}, test={len(te)}",
                        flush=True,
                    )
                    model = None
                    if method == "morphology_direct":
                        zx_train, zx_test = x_train, x_test
                    else:
                        model = _fit_model(method, settings)
                        y_fit = (
                            y_train[rng.permutation(len(tr))]
                            if method == "rcca_shuffled"
                            else y_train
                        )
                        with (
                            (directory / "training.log").open("w") as log,
                            contextlib.redirect_stdout(log),
                        ):
                            model.fit(
                                x_train,
                                y_fit,
                                output=directory,
                                cell_ids=data.cells[tr],
                                genes=data.genes[efilter.keep_],
                            )
                        zx_train, zx_test = (
                            model.transform_morphology(x_train),
                            model.transform_morphology(x_test),
                        )
                        zy_train, zy_test = (
                            model.transform_expression(y_train),
                            model.transform_expression(y_test),
                        )
                    decoder = GeneDecoder(settings["logistic_c"]).fit(zx_train, y_train_full)
                    probabilities = decoder.predict_proba(zx_test)
                    gene_frame, scores = gene_scores(
                        y_test_full, probabilities, prior, data.genes, settings["min_test_class"]
                    )
                    gene_frame["name"] = gene_frame.gene.map(gene_names)
                    gene_frame.to_csv(directory / "gene_scores.tsv", sep="\t", index=False)
                    np.savez_compressed(
                        directory / "gene_predictions.npz",
                        label_ids=data.cells[te],
                        nucleus_ids=data.nuclei[te],
                        genes=data.genes,
                        observed=y_test_full.astype(np.uint8),
                        probabilities=probabilities.astype(np.float32),
                    )
                    record = {
                        "scheme": scheme,
                        "morphology": morph_name,
                        "method": method,
                        "train_cells": len(tr),
                        "test_cells": len(te),
                        "purged_cells": len(purged),
                        "morphology_retained_variance": prep.pca_.explained_variance_ratio_[
                            prep.pc_keep_
                        ].sum(),
                        **scores,
                    }
                    bundle = {
                        "morphology_preprocessor": prep,
                        "expression_filter": efilter,
                        "genes": data.genes,
                        "morphology_name": morph_name,
                        "alignment": model,
                        "gene_decoder": decoder,
                        "train_cell_ids": data.cells[tr],
                        "test_cell_ids": data.cells[te],
                        "schema": "shared_space_model.v1",
                    }
                    if model is not None:
                        record.update(
                            retrieval_metrics(
                                zx_test[retrieval_ix],
                                zy_test[retrieval_ix],
                                settings["retrieval_k"],
                            )
                        )
                        record["mean_paired_correlation"] = np.nanmean(
                            column_correlations(zx_test, zy_test)
                        )
                        shuffled = np.arange(len(te))
                        for region in np.unique(regions[te]):
                            local = np.flatnonzero(regions[te] == region)
                            shuffled[local] = rng.permutation(local)
                        record["within_region_shuffle_correlation"] = np.nanmean(
                            column_correlations(zx_test, zy_test[shuffled])
                        )
                        reverse, reverse_scores = probe_expression_to_morphology(
                            zy_train, zy_test, x_train, x_test
                        )
                        record.update(reverse_scores)
                        bundle["expression_to_morphology_decoder"] = reverse
                        # A single train-fitted display projection for both modalities.
                        display_projection = PCA(n_components=2, random_state=settings["seed"]).fit(
                            np.vstack([zx_train, zy_train])
                        )
                        bundle["display_projection"] = display_projection
                        np.savez_compressed(
                            directory / "heldout_embeddings.npz",
                            label_ids=data.cells[te],
                            nucleus_ids=data.nuclei[te],
                            morphology_features=zx_test.astype(np.float32),
                            expression_features=zy_test.astype(np.float32),
                            morphology_xy=display_projection.transform(zx_test),
                            expression_xy=display_projection.transform(zy_test),
                            positive_calls=y_test_full.sum(1),
                            manual_labels=data.labels[te],
                            region=regions[te],
                        )
                        correlations_train = association_matrix(zx_train, y_train_full)
                        correlations_test = association_matrix(zx_test, y_test_full)
                        for factor in range(zx_train.shape[1]):
                            selected = np.argsort(
                                -np.nan_to_num(np.abs(correlations_train[factor]), nan=-1)
                            )[:5]
                            for gene in selected:
                                latent_rows.append(
                                    {
                                        "morphology": morph_name,
                                        "scheme": scheme,
                                        "method": method,
                                        "factor": factor + 1,
                                        "gene": data.genes[gene],
                                        "name": gene_names[data.genes[gene]],
                                        "train_spearman": correlations_train[factor, gene],
                                        "test_spearman": correlations_test[factor, gene],
                                    }
                                )
                            for side, ix in [
                                ("low", np.argsort(zx_test[:, factor])[:3]),
                                ("high", np.argsort(zx_test[:, factor])[-3:]),
                            ]:
                                for local in ix:
                                    inspection_rows.append(
                                        {
                                            "morphology": morph_name,
                                            "scheme": scheme,
                                            "method": method,
                                            "factor": factor + 1,
                                            "extreme": side,
                                            "cell_id": data.cells[te[local]],
                                            "nucleus_id": data.nuclei[te[local]],
                                            "factor_score": zx_test[local, factor],
                                            "manual_label": data.labels[te[local]],
                                            "spatial_region": regions[te[local]],
                                        }
                                    )
                        descriptors = data.measurements.drop(
                            columns=["cell_id", "nucleus_id"]
                        ).copy()
                        descriptors["positive_call_burden"] = data.expression.sum(1)
                        for j, axis in enumerate("xyz"):
                            descriptors[f"anchor_{axis}"] = data.xyz[:, j]
                        for descriptor in descriptors:
                            values = descriptors[descriptor].to_numpy(float)[te]
                            good = np.isfinite(values)
                            corr = association_matrix(zx_test[good], values[good, None])[:, 0]
                            for factor, value in enumerate(corr):
                                proxy_rows.append(
                                    {
                                        "morphology": morph_name,
                                        "scheme": scheme,
                                        "method": method,
                                        "factor": factor + 1,
                                        "descriptor": descriptor,
                                        "test_spearman": value,
                                    }
                                )
                        if hasattr(model, "expression_loadings"):
                            weights = model.expression_loadings()
                            loading = pd.DataFrame(
                                weights,
                                index=data.genes[efilter.keep_],
                                columns=[f"factor_{i + 1}" for i in range(weights.shape[1])],
                            )
                            loading.rename_axis("gene").to_csv(
                                directory / "expression_loadings.tsv", sep="\t"
                            )
                        if isinstance(model, MOFARidge):
                            pd.DataFrame(
                                {
                                    key: model.training_stats_[key]
                                    for key in ("time", "elbo", "number_factors")
                                }
                            ).to_csv(
                                directory / "mofa_training.tsv", sep="\t", index_label="iteration"
                            )
                            record["mofa_iterations_recorded"] = int(
                                np.isfinite(model.training_stats_["time"]).sum() - 1
                            )
                            record["mofa_stopped_before_cap"] = (
                                record["mofa_iterations_recorded"] < settings["mofa_iterations"] - 1
                            )
                            record["mofa_morphology_head_train_R2"] = r2_score(
                                model.training_factors_, model.x_head_.predict(x_train)
                            )
                            record["mofa_expression_head_train_R2"] = r2_score(
                                model.training_factors_, model.y_head_.predict(y_train)
                            )
                        if isinstance(model, PairedNeural):
                            pd.DataFrame(
                                model.history_,
                                columns=["epoch", "loss", "alignment", "reconstruction"],
                            ).to_csv(directory / "neural_training.tsv", sep="\t", index=False)
                    joblib.dump(bundle, directory / "model.joblib", compress=3)
                    record["elapsed_seconds"] = time.monotonic() - start
                    results.append(record)
                    write_json(record, directory / "metrics.json")
                    pd.DataFrame(results).to_csv(
                        output / "tables/scores.tsv", sep="\t", index=False
                    )
                    print(
                        f"  AP={record['macro_AP']:.3f}; log-loss gain={record['mean_log_loss_gain_bits']:.4f} bits; {record['elapsed_seconds']:.1f}s",
                        flush=True,
                    )
        pd.concat(split_frames).to_csv(output / "tables/splits.tsv", sep="\t", index=False)
        pd.DataFrame(latent_rows).to_csv(
            output / "tables/factor_gene_associations.tsv", sep="\t", index=False
        )
        pd.DataFrame(proxy_rows).to_csv(
            output / "tables/factor_measurement_associations.tsv", sep="\t", index=False
        )
        pd.DataFrame(inspection_rows).to_csv(
            output / "tables/inspection_candidates.tsv", sep="\t", index=False
        )
    sources = []
    for key, value in config["inputs"].items():
        if key == "gene_metadata_h5ad":
            if value:
                path = Path(value)
                sources.append(
                    {
                        "path": str(path),
                        "bytes": path.stat().st_size,
                        "mtime_ns": path.stat().st_mtime_ns,
                        "note": "Only var/XLOC and var/vis_name read; metadata fingerprint only",
                    }
                )
        else:
            sources.extend(
                fingerprint(path)
                for path in (value.values() if isinstance(value, dict) else [value])
            )
    manifest.update(
        status="complete",
        selected_cells=len(data.cells),
        eligible_cells=int(data.audit.eligible.sum()),
        sources=sources,
        code=[fingerprint(p) for p in Path(__file__).parent.glob("*.py")],
    )
    write_json(manifest, output / "provenance.json")
    subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        stdout=(output / "environment.freeze.txt").open("w"),
        check=True,
    )
    from .plots import make_figures

    make_figures(output)
    summary = pd.DataFrame(results)
    report = [
        "# Paired shared-space exploration",
        f"Run: {run_id}",
        f"{len(data.cells):,} cells sampled from {int(data.audit.eligible.sum()):,} eligible matched cells. Fixed CPU pilot settings; no test-based tuning.",
        "CCA/PLS learn two projections. MOFA2 is fitted with Gaussian morphology and Bernoulli expression; the evaluated mofa2_ridge model uses two training-only ridge heads into those factors. The neural pilot uses paired contrastive alignment and reconstruction.",
        "Every gene prediction uses morphology alone at test time. Alignment and preprocessing were fitted only on training cells. Random holdout groups bilateral pairs; spatial holdout purges training partners of held-out cells.",
        summary.to_string(index=False),
        "Interpretation: exact-cell retrieval is strict and uses the saved candidate population. Compare against chance and shuffled-pair CCA. AP applies to genes with enough positive and negative test calls; log-loss gain compares with the training prevalence prior. Position_RBF is a flexible anatomical baseline, not exhaustive confound adjustment.",
        "Gene associations select five genes per factor using training correlations and then report held-out correlations. Bounding-box aspect ratio and voxel counts are proxies, not mesh elongation or calibrated texture. Morphology-only factor extremes are saved with both ID namespaces for image/MoBIE inspection. No causal effect of gene activation/silencing is established.",
        "Outputs: tables/scores.tsv, tables/splits.tsv, tables/factor_*_associations.tsv, tables/inspection_candidates.tsv; models/<morphology>/<scheme>/<method> contains fitted bundles, held-out vectors and gene probabilities. MOFA folders also contain native_mofa.hdf5 and training traces. Only load trusted joblib files. Input gene/feature order must be preserved for reuse.",
        "Do not call any exploratory winner an unbiased final result after choosing among methods. Confirm dimensionality/regularization/seed sensitivity and then validate on independent material.",
    ]
    (output / "findings.md").write_text("\n\n".join(report) + "\n")
    print(f"Completed: {output}", flush=True)
    return output
