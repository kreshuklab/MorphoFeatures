"""Single command-line entrypoint for analysis, fixtures, training, and diagnostics."""

from __future__ import annotations

import argparse
import importlib.util
import json
import pickle
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from sklearn.preprocessing import StandardScaler

from morphofeatures.analysis.classification import (
    cross_validate_logistic,
    load_class_labels,
    select_labeled_embeddings,
)
from morphofeatures.analysis.context import aggregate_neighbors, agglomerate_features
from morphofeatures.analysis.projection import cluster_embeddings, compute_umap
from morphofeatures.analysis.validation import validate_bundled_artifacts, validate_mobie_tables
from morphofeatures.config import load_config
from morphofeatures.data.io import export_embeddings, load_embeddings, merge_embeddings
from morphofeatures.data.synthetic import save_synthetic_dataset


def _path(value: Optional[str], fallback: Path) -> Path:
    return Path(value).expanduser().resolve() if value else fallback


def _validate(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config) if args.config else None)
    arrays = validate_bundled_artifacts(config.paths.analysis_data)
    mobie = validate_mobie_tables(config.paths.mobie_data)
    payload = {
        "arrays": {name: result.to_dict() for name, result in arrays.items()},
        "mobie_tables": mobie,
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        for name, result in arrays.items():
            state = "PASS" if result.passed else "FAIL"
            print("{} {}: observed {}, expected {}".format(state, name, result.observed_shape, result.expected_shape))
        print("MoBIE tables with label_id: {}/{}".format(sum(mobie.values()), len(mobie)))
    return 0 if all(result.passed for result in arrays.values()) and all(mobie.values()) else 1


def _classify(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config) if args.config else None)
    embedding_path = _path(
        args.embedding, config.paths.analysis_data / "morphofeatures_all_cells.npy"
    )
    labels_path = _path(args.labels, config.paths.analysis_data / "class_labels.tsv")
    embeddings = load_embeddings(embedding_path)
    ids, labels, class_names = load_class_labels(labels_path, args.skip_type)
    features, labels = select_labeled_embeddings(embeddings, ids, labels)
    features = StandardScaler().fit_transform(features)
    result = cross_validate_logistic(
        features, labels, class_names, folds=args.folds, seed=args.seed, max_iter=args.max_iter
    )
    print("accuracy mean={:.4f} std={:.4f}".format(result.mean_accuracy, result.std_accuracy))
    print("classes={}".format(",".join(result.class_names)))
    print(result.confusion)
    return 0


def _project(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config) if args.config else None)
    embedding_path = _path(
        args.embedding, config.paths.analysis_data / "morphofeatures_all_cells.npy"
    )
    table = load_embeddings(embedding_path)
    rng = np.random.default_rng(args.seed)
    if args.subset and args.subset < len(table.label_ids):
        selected = np.sort(rng.choice(len(table.label_ids), size=args.subset, replace=False))
        ids = table.label_ids[selected]
        features = table.features[selected]
    else:
        ids = table.label_ids
        features = table.features
    features = StandardScaler().fit_transform(features)
    projection = compute_umap(
        features,
        n_neighbors=args.umap_neighbors,
        seed=args.seed,
        n_epochs=args.umap_epochs,
    )
    labels = cluster_embeddings(
        features,
        method=args.cluster_method,
        n_neighbors=args.cluster_neighbors,
        resolution=args.resolution,
        n_clusters=args.clusters,
        seed=args.seed,
    )
    output = _path(args.output, config.paths.output_root / "projection.tsv")
    export_embeddings(
        output,
        ids,
        np.column_stack((labels, projection)),
        ("cluster", "umap_1", "umap_2"),
    )
    print("Saved {} rows to {}".format(len(ids), output))
    return 0


def _synthetic(args: argparse.Namespace) -> int:
    output = save_synthetic_dataset(Path(args.output), seed=args.seed)
    print("Synthetic fixture written to {}".format(output.resolve()))
    return 0


def _doctor(_: argparse.Namespace) -> int:
    optional = (
        "umap",
        "igraph",
        "leidenalg",
        "torch",
        "torch_cluster",
        "trimesh",
        "zarr",
        "z5py",
        "pybdv",
        "wandb",
        "streamlit",
    )
    for module in optional:
        print("{:<16} {}".format(module, "available" if importlib.util.find_spec(module) else "missing"))
    return 0


def _combine(args: argparse.Namespace) -> int:
    table = merge_embeddings([Path(path) for path in args.embeddings], standardize=args.standardize)
    export_embeddings(Path(args.output), table.label_ids, table.features)
    print("Combined {} groups into {} features for {} cells".format(
        len(args.embeddings), table.features.shape[1], len(table.label_ids)))
    return 0


def _context(args: argparse.Namespace) -> int:
    table = load_embeddings(Path(args.embedding))
    with Path(args.neighbors).open("rb") as stream:
        neighbors = pickle.load(stream)
    context = aggregate_neighbors(table, neighbors, include_self=not args.exclude_self,
                                  reducer=args.reducer)
    if args.features:
        context = agglomerate_features(context, n_features=args.features)
    export_embeddings(Path(args.output), context.label_ids, context.features)
    print("Saved {} context rows with {} features".format(
        len(context.label_ids), context.features.shape[1]))
    return 0


def _ui(args: argparse.Namespace) -> int:
    from morphofeatures.ui import run_navigator

    return run_navigator(port=args.port, address=args.address, headless=args.headless)


def _mae_train(args: argparse.Namespace) -> int:
    from morphofeatures.mae3d import train_from_config

    train_from_config(Path(args.config), output=Path(args.output) if args.output else None)
    return 0


def _mae_encode(args: argparse.Namespace) -> int:
    from morphofeatures.mae3d import encode_from_config

    encode_from_config(Path(args.config), Path(args.checkpoint), Path(args.output))
    return 0


def _shape_train(args: argparse.Namespace) -> int:
    from morphofeatures.shape.train_shape_model import main as shape_main

    shape_main([str(args.config)])
    return 0


def _shape_encode(args: argparse.Namespace) -> int:
    from morphofeatures.shape.generate_shape_embeddings import main as shape_main

    shape_main(["--config", str(args.config), "--save-to", str(args.output)])
    return 0


def _texture_train(args: argparse.Namespace) -> int:
    from morphofeatures.texture.train import main as texture_main

    forwarded = [str(args.experiment), "--device", args.device]
    if args.from_checkpoint:
        forwarded.append("--from-checkpoint")
    texture_main(forwarded)
    return 0


def _texture_encode(args: argparse.Namespace) -> int:
    from morphofeatures.texture.predict import main as texture_main

    forwarded = [str(args.experiment), "--device", args.device]
    if args.patches:
        forwarded.append("--save-patches")
    if args.aggregate:
        forwarded.append("--aggregate-patches")
    texture_main(forwarded)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="morphofeatures", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="Validate bundled paper artifacts")
    validate.add_argument("--config")
    validate.add_argument("--json", action="store_true")
    validate.set_defaults(handler=_validate)

    classify = subparsers.add_parser("classify", help="Run deterministic logistic-regression CV")
    classify.add_argument("--embedding")
    classify.add_argument("--labels")
    classify.add_argument("--config")
    classify.add_argument("--folds", type=int, default=5)
    classify.add_argument("--seed", type=int, default=42)
    classify.add_argument("--max-iter", type=int, default=2000)
    classify.add_argument("--skip-type", action="append", default=[])
    classify.set_defaults(handler=_classify)

    project = subparsers.add_parser("project", help="Run UMAP and Leiden or K-means clustering")
    project.add_argument("--embedding")
    project.add_argument("--output")
    project.add_argument("--config")
    project.add_argument("--subset", type=int, default=0)
    project.add_argument("--seed", type=int, default=42)
    project.add_argument("--umap-neighbors", type=int, default=15)
    project.add_argument("--umap-epochs", type=int)
    project.add_argument("--cluster-method", choices=("leiden", "kmeans"), default="leiden")
    project.add_argument("--cluster-neighbors", type=int, default=20)
    project.add_argument("--clusters", type=int, default=8)
    project.add_argument("--resolution", type=float, default=0.004)
    project.set_defaults(handler=_project)

    synthetic = subparsers.add_parser("synthetic", help="Create a tiny fixture dataset")
    synthetic.add_argument("output")
    synthetic.add_argument("--seed", type=int, default=42)
    synthetic.set_defaults(handler=_synthetic)

    doctor = subparsers.add_parser("doctor", help="Report optional runtime capabilities")
    doctor.set_defaults(handler=_doctor)

    navigator = subparsers.add_parser("ui", help="Open the Streamlit workflow workspace")
    navigator.add_argument("--port", type=int, default=8501)
    navigator.add_argument("--address", default="localhost")
    navigator.add_argument("--headless", action="store_true")
    navigator.set_defaults(handler=_ui)

    combine = subparsers.add_parser("combine", help="Concatenate aligned embedding groups")
    combine.add_argument("embeddings", nargs="+")
    combine.add_argument("--output", required=True)
    combine.add_argument("--standardize", action="store_true")
    combine.set_defaults(handler=_combine)

    context = subparsers.add_parser("context", help="Aggregate neighbor morphology features")
    context.add_argument("--embedding", required=True)
    context.add_argument("--neighbors", required=True)
    context.add_argument("--output", required=True)
    context.add_argument("--features", type=int, default=200)
    context.add_argument("--reducer", choices=("mean", "max"), default="mean")
    context.add_argument("--exclude-self", action="store_true")
    context.set_defaults(handler=_context)

    mae_train = subparsers.add_parser("mae-train", help="Train the pragmatic 3D masked autoencoder")
    mae_train.add_argument("--config", required=True)
    mae_train.add_argument("--output")
    mae_train.set_defaults(handler=_mae_train)

    mae_encode = subparsers.add_parser("mae-encode", help="Encode crops with a trained 3D MAE")
    mae_encode.add_argument("--config", required=True)
    mae_encode.add_argument("--checkpoint", required=True)
    mae_encode.add_argument("--output", required=True)
    mae_encode.set_defaults(handler=_mae_encode)

    shape_train = subparsers.add_parser("shape-train", help="Train the legacy DeepGCN encoder")
    shape_train.add_argument("--config", type=Path, required=True)
    shape_train.set_defaults(handler=_shape_train)

    shape_encode = subparsers.add_parser("shape-encode", help="Export DeepGCN shape embeddings")
    shape_encode.add_argument("--config", type=Path, required=True)
    shape_encode.add_argument("--output", type=Path, required=True)
    shape_encode.set_defaults(handler=_shape_encode)

    texture_train = subparsers.add_parser("texture-train", help="Train legacy-style texture embeddings")
    texture_train.add_argument("experiment", type=Path)
    texture_train.add_argument("--device", default="auto")
    texture_train.add_argument("--from-checkpoint", action="store_true")
    texture_train.set_defaults(handler=_texture_train)

    texture_encode = subparsers.add_parser("texture-encode", help="Export texture embeddings")
    texture_encode.add_argument("experiment", type=Path)
    texture_encode.add_argument("--device", default="auto")
    texture_encode.add_argument("--patches", action="store_true")
    texture_encode.add_argument("--aggregate", action="store_true")
    texture_encode.set_defaults(handler=_texture_encode)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
