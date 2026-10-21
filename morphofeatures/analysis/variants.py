"""Saved, named projection/clustering variants over an immutable object cohort.

Worker batches and interactive Results use the same implementation. Models and
classifiers are never refitted here. Each revision owns its coordinates, labels,
figures and provenance; embedding files are linked (copied across filesystems).
"""

import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

from morphofeatures.analysis.projection import (
    PROJECTION_DEFAULTS,
    project_embeddings,
    validate_projection_settings,
)
from morphofeatures.artifacts import write_json_atomic
from morphofeatures.data.io import load_embeddings
from morphofeatures.metrics import utc_now
from morphofeatures.workflow_validation import FieldError

VARIANT_KEYS = frozenset(PROJECTION_DEFAULTS) | {"umap"}
REPORT_SCHEMAS = {
    "morphofeatures.analysis.v1",
    "morphofeatures.comparison.v1",
    "morphofeatures.joint_analysis.v1",
}
COORDINATES = {"cluster", "pca_1", "pca_2", "umap_1", "umap_2", "representation"}


@contextmanager
def record_lock(directory):
    """Serialize short metadata updates across processes and browser sessions."""
    with (Path(directory) / ".analysis-records.lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def read_json(path):
    return json.loads(Path(path).read_text())


def validate_variants(settings):
    variants = settings.get("analysis_variants", [])
    if not isinstance(variants, list):
        raise FieldError(
            "Analysis variants must be a list of named parameter combinations.", "analysis_variants"
        )
    names = set()
    for i, variant in enumerate(variants):
        prefix = ("analysis_variants", i)
        if not isinstance(variant, dict):
            raise FieldError("Each analysis variant needs a name and projection settings.", prefix)
        name = variant.get("name")
        if not isinstance(name, str) or not name.strip() or name.strip() in names:
            raise FieldError(
                "Give each analysis variant a unique, nonempty name.", prefix + ("name",)
            )
        names.add(name.strip())
        unknown = set(variant) - VARIANT_KEYS - {"name"}
        if unknown:
            raise FieldError(
                "Variants change plotting/clustering only. Unsupported settings: "
                + ", ".join(sorted(unknown)),
                *(prefix + (key,) for key in sorted(unknown)),
            )
        try:
            validate_projection_settings({**settings, **variant})
        except FieldError as error:
            raise FieldError(str(error), *(prefix + field for field in error.fields)) from error


def variant_history(parent):
    path = Path(parent).parent / "analysis_variants.json"
    return read_json(path).get("variants", []) if path.is_file() else []


def _update_history(parent, entry):
    parent = Path(parent).resolve()
    with record_lock(parent.parent):
        rows = variant_history(parent)
        rows = [entry if row["id"] == entry["id"] else row for row in rows]
        if not any(row["id"] == entry["id"] for row in rows):
            rows.append(entry)
        write_json_atomic(
            parent.parent / "analysis_variants.json",
            {
                "schema": "morphofeatures.analysis_variants.v1",
                "parent": str(parent),
                "variants": rows,
            },
        )


def _reserve(parent, variant, origin):
    parent = Path(parent).resolve()
    metadata = read_json(parent)
    if metadata.get("schema") not in REPORT_SCHEMAS:
        raise ValueError(
            "Open an analysis, representation comparison or shared-space result first."
        )
    validate_variants({**metadata.get("settings", {}), "analysis_variants": [variant]})
    identifier = uuid.uuid4().hex
    directory = parent.parent / "variants" / identifier
    directory.mkdir(parents=True, exist_ok=False)
    entry = {
        "id": identifier,
        "name": variant["name"].strip(),
        "state": "queued",
        "created_at": utc_now(),
        "origin": origin,
        "parent": str(parent),
        "overrides": {key: value for key, value in variant.items() if key != "name"},
        "directory": str(directory),
    }
    write_json_atomic(directory / "variant.json", entry)
    _update_history(parent, entry)
    return entry


def _link(source, destination):
    source, destination = Path(source), Path(destination)
    if not source.is_file():
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _copy_inputs(source, destination):
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("embeddings.npz", "embeddings.metadata.json", "splits.tsv", "predictions.tsv"):
        _link(source / name, destination / name)


def _frame(path):
    return pd.read_csv(path, sep="\t", dtype={"label_id": np.int64, "source_label_id": np.int64})


def _project(source, old_frame, settings):
    table = load_embeddings(source).sorted()
    ids = old_frame.label_id.to_numpy(dtype=np.int64)
    if not old_frame.label_id.is_unique or not np.isin(ids, table.label_ids).all():
        raise ValueError(
            "Saved coordinates and embeddings disagree on object IDs; cannot reuse this cohort."
        )
    rows = np.searchsorted(table.label_ids, ids)
    frame, diagnostics = project_embeddings(ids, table.features[rows], {**settings, "subset": 0})
    # Keep known/predicted labels, exact native IDs and dataset identities.
    annotations = old_frame.drop(columns=list(COORDINATES), errors="ignore")
    return frame.merge(annotations, on="label_id", validate="one_to_one"), diagnostics


def _labels(source, destination, frame):
    labels_path = source / "object_labels.tsv"
    labels = (
        _frame(labels_path)
        if labels_path.exists()
        else frame.drop(columns=[c for c in COORDINATES if c in frame], errors="ignore")
    )
    labels = labels.drop(columns="cluster", errors="ignore").merge(
        frame[["label_id", "cluster"]], on="label_id", how="left", validate="one_to_one"
    )
    labels.to_csv(destination / "object_labels.tsv", sep="\t", index=False)


def _save_projection(frame, destination, name, settings, metadata):
    from morphofeatures.analysis.visualization import category_colors, save_projection_figure

    frame.to_csv(destination / "coordinates.tsv", sep="\t", index=False)
    metadata["cluster_colors"] = category_colors(frame.cluster)
    frame.attrs.update(
        class_colors=metadata.get("class_colors", {}), cluster_colors=metadata["cluster_colors"]
    )
    save_projection_figure(frame, destination / "projection.svg", name, settings)


def _recompute(parent, destination, entry):
    original = read_json(parent)
    result = deepcopy(original)
    settings = {**PROJECTION_DEFAULTS, **original.get("settings", {}), **entry["overrides"]}
    settings.pop("analysis_variants", None)
    settings.pop("name", None)
    if settings.get("use_published_layout"):
        settings["use_published_layout"] = False
    if result.get("bilateral"):
        result["bilateral"] = str((parent.parent / result["bilateral"]).resolve())
    # Classification provenance remains the original fit even when plot scaling changes.
    result["classification_settings"] = original.get(
        "classification_settings", original.get("settings", {})
    )
    result.update(
        settings=settings,
        display_name=entry["name"],
        reanalysis={
            **entry,
            "cohort": "exact IDs from the parent coordinates",
            "classification_reused": True,
        },
    )
    result.pop("name_history", None)
    old_frame = _frame(parent.parent / original.get("coordinates", "coordinates.tsv"))
    if "representations" in original:
        frames = []
        for name, old in original["representations"].items():
            source = parent.parent / old["directory"]
            folder = destination / old["directory"]
            _copy_inputs(source, folder)
            previous = old_frame[old_frame.representation == name]
            frame, diagnostics = _project(source / "embeddings.npz", previous, settings)
            current = result["representations"][name]
            current["diagnostics"] = diagnostics
            _labels(source, folder, frame)
            _save_projection(frame, folder, entry["name"] + " · " + name, settings, current)
            frame["representation"] = name
            frames.append(frame)
        pd.concat(frames).to_csv(destination / "coordinates.tsv", sep="\t", index=False)
        _link(parent.parent / "splits.tsv", destination / "splits.tsv")
    else:
        source = parent.parent / original.get("embedding", "embeddings.npz")
        _copy_inputs(parent.parent, destination)
        if source.name != "embeddings.npz" or source.parent != parent.parent:
            if not (destination / "embeddings.npz").exists():
                _link(source, destination / "embeddings.npz")
        result["embedding"] = "embeddings.npz"
        frame, diagnostics = _project(source, old_frame, settings)
        result["diagnostics"] = diagnostics
        _save_projection(frame, destination, entry["name"], settings, result)
        if original["schema"] == "morphofeatures.joint_analysis.v1":
            _link(parent.parent / "object_index.tsv", destination / "object_index.tsv")
            for name, dataset in original["datasets"].items():
                source_report = parent.parent / dataset["report"]
                folder = destination / dataset["directory"]
                _copy_inputs(source_report.parent, folder)
                native = (
                    frame[frame.dataset == name]
                    .drop(columns=["label_id", "dataset", "projected"], errors="ignore")
                    .rename(columns={"source_label_id": "label_id"})
                )
                child = read_json(source_report)
                child.update(
                    settings={
                        **child.get("settings", {}),
                        **{k: settings[k] for k in VARIANT_KEYS if k in settings},
                    },
                    diagnostics=diagnostics,
                    reanalysis=result["reanalysis"],
                    display_name=entry["name"] + " · " + name,
                )
                child["shared_projection"] = {"report": "../joint_analysis.json", "dataset": name}
                _labels(source_report.parent, folder, native)
                _save_projection(native, folder, child["display_name"], settings, child)
                write_json_atomic(folder / "analysis.json", child)
        else:
            result.pop("shared_projection", None)  # A native report opened alone gets its own fit.
            _labels(parent.parent, destination, frame)
    result["interpretation"] = (
        "Saved analysis variant on the parent's exact plotted object cohort. "
        "Embedding vectors and existing known/predicted cell types are retained. "
        "Classification scores, if present, belong to the original classifier settings; "
        "only projection and clustering were recomputed."
    )
    result["reanalysis"]["state"] = "completed"
    path = write_json_atomic(destination / parent.name, result)
    # The archive is a completed scientific snapshot, not a copy of the
    # earlier running-state marker. The parent index stays running until export ends.
    write_json_atomic(
        destination / "variant.json",
        {
            **entry,
            "state": "completed",
            "result": str(path),
            "completed_at": utc_now(),
        },
    )
    from morphofeatures.representation_analysis import package_results

    package_results(destination)
    return path


def _register(parent, result):
    """Attach an interactive result to the original run, without a new submission."""
    import yaml

    from morphofeatures.registry import JobRegistry

    for directory in Path(parent).parents:
        config = directory / "job.yaml"
        if config.is_file():
            worker = (yaml.safe_load(config.read_text()) or {}).get("worker", {})
            if worker.get("registry") and worker.get("record_id"):
                JobRegistry(Path(worker["registry"])).append_artifacts(
                    worker["record_id"], [str(result)]
                )
                return


def execute_variant(task):
    task = Path(task)
    entry = read_json(task)
    parent = Path(entry["parent"])
    started = time.perf_counter()
    # Claim a task once, even if an interrupted browser reconnects.
    with record_lock(task.parent):
        entry = read_json(task)
        if entry["state"] != "queued":
            return entry
        entry.update(state="running", started_at=utc_now(), pid=os.getpid())
        write_json_atomic(task, entry)
    _update_history(parent, entry)
    try:
        result = _recompute(parent, task.parent, entry)
        entry.update(state="completed", result=str(result))
        if entry["origin"] == "results":
            try:
                _register(parent, result)
            except (OSError, KeyError) as error:
                entry["registration_warning"] = str(error)
    except Exception as error:
        entry.update(state="failed", error=f"{type(error).__name__}: {error}")
    entry.update(completed_at=utc_now(), seconds=time.perf_counter() - started)
    write_json_atomic(task, entry)
    _update_history(parent, entry)
    return entry


def run_configured_variants(parent, settings):
    """Run stacked variants once; a failed variant does not hide successful ones."""
    validate_variants(settings)
    for variant in settings.get("analysis_variants", []):
        entry = _reserve(parent, variant, "workflow")
        execute_variant(Path(entry["directory"]) / "variant.json")


def validate_saved_variant(stage):
    """Read-only review for a new run that reuses a saved analysis's cohort."""
    from morphofeatures.artifact_availability import probe
    from morphofeatures.result_catalog import vector_artifacts

    parent = Path(stage.get("analysis") or "")
    metadata = read_json(parent)
    if metadata.get("schema") not in REPORT_SCHEMAS:
        raise ValueError("Choose a saved analysis, comparison or joint-analysis report.")
    variant = stage.get("variant", {})
    validate_variants({**metadata.get("settings", {}), "analysis_variants": [variant]})
    paths = vector_artifacts(metadata, parent)
    if metadata.get("schema") == "morphofeatures.joint_analysis.v1":
        paths["Pooled vectors"] = parent.parent / metadata.get("embedding", "embeddings.npz")
    paths["Saved coordinates"] = parent.parent / metadata.get("coordinates", "coordinates.tsv")
    for name, path in paths.items():
        status = probe(path)
        expected = "coordinates" if name == "Saved coordinates" else "vectors"
        if status.state not in {"ready", "unchecked"} or status.kind != expected:
            raise ValueError(f"{name}: {status.reason or 'not a valid analysis input'}")


def run_saved_variant(stage, destination):
    """Use the existing variant engine in a new local/SLURM worker directory.

    The source report and its variant index remain untouched. The new report's
    reanalysis record links to the parent and retains classifier provenance.
    """
    validate_saved_variant(stage)
    parent, destination = Path(stage["analysis"]), Path(destination)
    variant = stage["variant"]
    entry = {
        "id": uuid.uuid4().hex, "name": variant["name"], "state": "running",
        "origin": "workspace", "parent": str(parent.resolve()),
        "directory": str(destination.resolve()),
        "overrides": {k: v for k, v in variant.items() if k != "name"},
    }
    return _recompute(parent, destination, entry)


def start_variants(parent, variants):
    """Launch a sequential batch on the UI host in a separate, single-threaded process."""
    from morphofeatures.run_deletion import run_write_guard

    with run_write_guard(parent):
        return _start_variants(parent, variants)


def _start_variants(parent, variants):
    parent = Path(parent).resolve()
    validate_variants({**read_json(parent).get("settings", {}), "analysis_variants": variants})
    if not variants:
        raise ValueError("Add at least one named analysis variant.")
    # Lock the check and reservation together to prevent double-submit across tabs.
    with (parent.parent / ".analysis-launch.lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        if any(row["state"] in {"queued", "running"} for row in variant_history(parent)):
            raise ValueError(
                "An analysis batch for this result is already active. Wait for it to finish."
            )
        entries = [_reserve(parent, variant, "results") for variant in variants]
        tasks = [str(Path(entry["directory"]) / "variant.json") for entry in entries]
        env = {
            **os.environ,
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        }
        repository = str(Path(__file__).resolve().parents[2])
        env["PYTHONPATH"] = repository + os.pathsep + env.get("PYTHONPATH", "")
        try:
            with (Path(entries[0]["directory"]) / "computation.log").open("ab") as log:
                subprocess.Popen(
                    [sys.executable, "-m", "morphofeatures.analysis.variants", *tasks],
                    env=env,
                    cwd=repository,
                    stdout=log,
                    stderr=log,
                    start_new_session=True,
                )
        except Exception as error:
            for entry in entries:
                entry.update(state="failed", error=str(error), completed_at=utc_now())
                write_json_atomic(Path(entry["directory"]) / "variant.json", entry)
                _update_history(parent, entry)
            raise
    return entries


def rename_result(path, name):
    from morphofeatures.run_deletion import run_write_guard

    with run_write_guard(path):
        return _rename_result(path, name)


def _rename_result(path, name):
    path = Path(path).resolve()
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Enter a nonempty display name.")
    with record_lock(path.parent):
        record = read_json(path)
        old = record.get("display_name", path.stem)
        record.setdefault("name_history", []).append(
            {"previous": old, "name": name.strip(), "at": utc_now()}
        )
        record["display_name"] = name.strip()
        write_json_atomic(path, record)
    revision = record.get("reanalysis", {})
    if revision.get("parent"):
        for entry in variant_history(revision["parent"]):
            if entry["id"] == revision["id"]:
                entry.update(name=name.strip(), name_history=record["name_history"])
                _update_history(revision["parent"], entry)
    return record


if __name__ == "__main__":
    for task_path in sys.argv[1:]:
        execute_variant(task_path)
