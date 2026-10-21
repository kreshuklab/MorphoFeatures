"""Bounded discovery and explicit content validation for saved scientific outputs.

Registry listing never loads arrays or scans directories. Selected small inputs
are validated directly; large inputs can be checked explicitly in the background.
Availability is distinct from cross-artifact scientific compatibility.
"""

import csv
import json
import zipfile
import zlib
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

from morphofeatures.artifact_locations import artifact_path, relocated_metadata

MAX_METADATA = 8 * 1024**2
INLINE_CHECK = 32 * 1024**2


@dataclass(frozen=True)
class Availability:
    path: str
    kind: str
    state: str
    reason: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def ready(self):
        return self.state == "ready"


def signature(path):
    stat = Path(path).stat()
    return (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino)


def read_report(path, *, relocate=True):
    path = artifact_path(path) if relocate else Path(path)
    if path.stat().st_size > MAX_METADATA:
        raise ValueError("Report exceeds the 8 MiB metadata limit. Open a smaller indexed report.")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("A result report must contain a JSON object.")
    for key in ("representations", "settings"):
        if key in value and not isinstance(value[key], dict) and not (
            key == "representations" and value.get("schema") == "morphofeatures.mobie_export.v1" and isinstance(value[key], list)
        ):
            raise ValueError(f"Invalid report: {key} must be a mapping.")
    schema = value.get("schema", "")
    for key in ("representations", "datasets"):
        entries = value.get(key)
        if schema and isinstance(entries, dict) and any(not isinstance(entry, dict) for entry in entries.values()):
            raise ValueError(f"Invalid report: {key} entries must be mappings.")
    if schema in {"morphofeatures.dino_experiment.v1", "morphofeatures.transfer.v1"}:
        for key in ("embeddings", "original_embeddings", "comparisons", "joint_analyses", "transfers", "targets"):
            if key in value and (not isinstance(value[key], list) or any(not isinstance(entry, dict) for entry in value[key])):
                raise ValueError(f"Invalid report: {key} must be a list of mappings.")
    return relocated_metadata(value) if relocate else value


def _header(stream):
    import numpy as np

    version = np.lib.format.read_magic(stream)
    if version == (1, 0):
        shape, order, dtype = np.lib.format.read_array_header_1_0(stream)
    elif version in {(2, 0), (3, 0)}:
        shape, order, dtype = np.lib.format.read_array_header_2_0(stream)
    else:
        raise ValueError("Unsupported NumPy array version")
    if dtype.hasobject or dtype.kind not in "biuf":
        raise ValueError("Embedding arrays must contain numeric values, without Python objects.")
    return shape, order, dtype


def _array_chunks(stream, shape, dtype):
    import math

    import numpy as np

    remaining = math.prod(shape)
    while remaining:
        count = min(remaining, max(1, 1024**2 // dtype.itemsize))
        data = stream.read(count * dtype.itemsize)
        if len(data) != count * dtype.itemsize:
            raise ValueError("Array is incomplete or truncated.")
        values = np.frombuffer(data, dtype=dtype)
        if not np.isfinite(values).all():
            raise ValueError("Array contains NaN or infinite values.")
        yield values
        remaining -= count


def _ids(values, seen):
    import numpy as np

    if values.dtype.kind == "f" and (
        not np.equal(values, np.rint(values)).all() or np.any(np.abs(values) > 2**53)
    ):
        raise ValueError("Object IDs must be exact integers; use NPZ integer IDs for large IDs.")
    if np.any(values > np.iinfo(np.int64).max) or np.any(values < np.iinfo(np.int64).min):
        raise ValueError("Object IDs exceed the supported integer range.")
    batch = [int(v) for v in values]
    unique = set(batch)
    if len(batch) != len(unique) or seen.intersection(unique):
        raise ValueError("Embedding contains duplicate object IDs.")
    seen.update(unique)


def validate_vectors(path):
    """Read feature payloads in bounded chunks; never deserialize weights/pickles."""
    import numpy as np

    path = Path(path)
    seen = set()
    if path.suffix.lower() == ".npz":
        with zipfile.ZipFile(path) as archive:
            with archive.open("label_ids.npy") as stream:
                ids_shape, _, dtype = _header(stream)
                if len(ids_shape) != 1 or not ids_shape[0]:
                    raise ValueError("label_ids must be a nonempty one-dimensional array.")
                for values in _array_chunks(stream, ids_shape, dtype):
                    _ids(values, seen)
            with archive.open("features.npy") as stream:
                shape, _, dtype = _header(stream)
                if len(shape) != 2 or shape[0] != ids_shape[0] or not shape[1]:
                    raise ValueError("features must have one nonempty vector per object ID.")
                for _ in _array_chunks(stream, shape, dtype):
                    pass
    elif path.suffix.lower() == ".npy":
        with path.open("rb") as stream:
            shape, fortran, dtype = _header(stream)
            if len(shape) != 2 or not shape[0] or shape[1] < 2:
                raise ValueError("Historical NPY needs IDs in column zero and at least one feature.")
            offset = 0
            for values in _array_chunks(stream, shape, dtype):
                if fortran:
                    ids = values[:max(0, min(len(values), shape[0] - offset))]
                else:
                    ids = values[(-offset) % shape[1]::shape[1]]
                _ids(ids, seen)
                offset += len(values)
    else:
        import pandas as pd

        from morphofeatures.analysis.relations import integer_ids

        count = 0
        for frame in pd.read_csv(path, sep="\t" if path.suffix.lower() == ".tsv" else ",",
                                 dtype={"label_id": str}, chunksize=10000):
            ids = integer_ids(frame.label_id)
            _ids(ids, seen)
            features = frame.drop(columns="label_id").to_numpy(dtype=float)
            if not features.shape[1] or not np.isfinite(features).all():
                raise ValueError("Feature columns must be numeric and finite.")
            count += len(frame)
        if not count:
            raise ValueError("Embedding is empty.")


def validate_coordinates(path):
    import numpy as np
    import pandas as pd

    from morphofeatures.analysis.relations import integer_ids

    count, seen = 0, set()
    for frame in pd.read_csv(path, sep="\t" if Path(path).suffix.lower() == ".tsv" else ",",
                             dtype={"label_id": str, "source_label_id": str}, chunksize=10000):
        ids = integer_ids(frame.label_id)
        coordinates = [c for method in ("pca", "umap") for c in (method + "_1", method + "_2") if c in frame]
        if not coordinates or not np.isfinite(frame[coordinates].to_numpy(dtype=float)).all():
            raise ValueError("Saved map coordinates must be numeric and finite.")
        names = [c for c in ("representation", "dataset") if c in frame]
        keys = list(zip(*[frame[c].astype(str) for c in names], ids))
        unique = set(keys)
        if len(unique) != len(keys) or seen.intersection(unique):
            raise ValueError("Saved map contains duplicate object identities.")
        seen.update(unique)
        count += len(frame)
    if not count:
        raise ValueError("Saved map is empty.")


def probe(path, *, validate=False):
    path = artifact_path(path).resolve()
    try:
        if not path.is_file():
            return Availability(str(path), "unknown", "missing", "File is missing or is not a regular file.")
        result = _probe(str(path), signature(path), validate)
        if result.metadata:
            from dataclasses import replace
            result = replace(result, metadata=relocated_metadata(result.metadata))
        return result
    except (OSError, ValueError) as error:
        return Availability(str(path), "unknown", "unreadable", str(error))


@lru_cache(maxsize=1024)
def _probe(path, stamp, validate):
    file = Path(path)
    kind = "file"
    try:
        if not stamp[0]:
            raise ValueError("File is empty or is still being written.")
        suffix = file.suffix.lower()
        if suffix in {".yaml", ".yml"}:
            if stamp[0] > MAX_METADATA:
                raise ValueError("Configuration exceeds the metadata inspection limit.")
            metadata = yaml.safe_load(file.read_text())
            if not isinstance(metadata, dict):
                raise ValueError("Configuration must be a mapping.")
            data = metadata.get("data", {})
            prepared = isinstance(data, dict) and bool(data.get("crops") or data.get("source"))
            return Availability(path, "prepared" if prepared else "configuration", "ready", metadata=metadata)
        if suffix in {".pt", ".pth"}:
            with file.open("rb") as stream:
                stream.read(1)
            return Availability(path, "checkpoint", "ready", "Model/data compatibility is checked when configuring extraction.")
        if suffix == ".json":
            metadata = read_report(file, relocate=False)
            from morphofeatures.workflow_outputs import report_action

            kind = "report" if report_action(metadata) else "metadata"
            return Availability(path, kind, "ready", metadata=metadata)
        size = stamp[0]
        if suffix == ".npz":
            kind = "vectors"
            with zipfile.ZipFile(file) as archive:
                members = set(archive.namelist())
                if not members.intersection({"label_ids.npy", "features.npy"}):
                    return Availability(path, "parameters", "ready")
                sizes = []
                for member in ("label_ids.npy", "features.npy"):
                    sizes.append(archive.getinfo(member).file_size)
                    with archive.open(member) as stream:
                        _header(stream)
                size = sum(sizes)
        elif suffix == ".npy":
            kind = "vectors"
            with file.open("rb") as stream:
                shape, _, _ = _header(stream)
            if len(shape) != 2 or shape[1] < 2:
                raise ValueError("Not an ID-preserving embedding array.")
        elif suffix in {".csv", ".tsv"}:
            with file.open(newline="") as stream:
                line = stream.readline(65537)
            if len(line) > 65536:
                raise ValueError("Table header exceeds the inspection limit.")
            columns = next(csv.reader([line], delimiter="\t" if suffix == ".tsv" else ","))
            if not columns or len(set(columns)) != len(columns):
                raise ValueError("Table needs a readable header with distinct column names.")
            paired = any({m + "_1", m + "_2"} <= set(columns) for m in ("pca", "umap"))
            if "label_id" in columns and paired:
                kind = "coordinates"
            elif "label_id" in columns and len(columns) > 1 and not set(columns).intersection({
                "cluster", "known_label", "predicted_label", "reason", "excluded", "label", "cell_type",
                "native_id", "cell_id", "nucleus_id", "source_label_id", "best_partner_id", "rank",
                "fold", "split", "role", "neighbors", "coverage", "dataset", "representation",
                "x", "y", "z", "x_um", "y_um", "z_um", "centroid_x", "centroid_y", "centroid_z",
            }):
                kind = "vectors"
            else:
                kind = "table"
        elif suffix == ".zip":
            kind = "archive"
            with zipfile.ZipFile(file) as archive:
                if not archive.namelist():
                    raise ValueError("Archive has no saved artifacts.")
        elif suffix in {".png", ".jpg", ".jpeg"}:
            from PIL import Image

            kind = "image"
            with Image.open(file) as picture:
                picture.verify()
        if kind in {"vectors", "coordinates"}:
            if not validate:
                return Availability(path, kind, "unchecked", "Content validation has not run.", {"bytes": size})
            (validate_vectors if kind == "vectors" else validate_coordinates)(file)
        else:
            with file.open("rb") as stream:
                stream.read(1)
        return Availability(path, kind, "ready")
    except (OSError, ValueError, TypeError, KeyError, EOFError, OverflowError, zipfile.BadZipFile, zlib.error, yaml.YAMLError) as error:
        return Availability(path, kind, "invalid", str(error))


def selected_artifact(path):
    """Validate small selected inputs; keep large inputs explicitly unchecked."""
    path = artifact_path(path)
    status = probe(path)
    if status.state == "unchecked" and status.metadata.get("bytes", 0) <= INLINE_CHECK:
        return probe(path, validate=True)
    # A background check uses the same content-addressed cache key.
    if status.state == "unchecked":
        return _validated.get((str(Path(path).resolve()), signature(path)), status)
    return status


_validated = {}


def check_selected(path):
    path = artifact_path(path)
    before = signature(path)
    result = probe(path, validate=True)
    if signature(path) != before:
        return Availability(str(path), result.kind, "unchecked", "File changed during validation; check it again.")
    _validated[(str(Path(path).resolve()), before)] = result
    return result


def available_vectors(choices):
    return {name: str(artifact_path(path).resolve()) for name, path in choices.items()
            if (status := selected_artifact(path)).ready and status.kind == "vectors"}


def availability_details(choices, *, key):
    """Display unavailable artifacts and explicitly check large vectors in a thread."""
    import streamlit as st

    unavailable = {name: selected_artifact(path) for name, path in choices.items()
                   if not selected_artifact(path).ready}
    if not unavailable:
        return
    with st.expander(f"Artifact availability · {len(unavailable)} need attention"):
        for name, status in unavailable.items():
            st.caption(f"{name}: {status.state} — {status.reason}")
            st.caption(status.path)
            if status.state == "unchecked":
                task_key = key + ":check:" + status.path
                future = st.session_state.get(task_key)
                if future and future.done():
                    try:
                        result = future.result()
                        if not result.ready:
                            st.warning(result.reason)
                    except Exception as error:
                        st.warning(str(error))
                    st.session_state.pop(task_key, None)
                    st.rerun()
                elif future:
                    st.info("Checking exact IDs and values in bounded chunks. Refresh to see the outcome.")
                    st.button("Refresh artifact check", key=task_key + ":refresh")
                elif st.button("Check " + name, key=task_key + ":start"):
                    from concurrent.futures import ThreadPoolExecutor

                    pool = ThreadPoolExecutor(max_workers=1)
                    st.session_state[task_key] = pool.submit(check_selected, status.path)
                    pool.shutdown(wait=False)
                    st.rerun()
        missing = {name: status for name, status in unavailable.items() if status.state in {"missing", "unreadable", "invalid"}}
        if missing:
            st.markdown("**Locate a moved artifact**")
            name = st.selectbox("Artifact to relink", list(missing), key=key + ":relink-name")
            replacement = st.text_input("New location on the app host", key=key + ":relink-path")
            confirmed = st.checkbox("This is the same saved artifact, at a new location", key=key + ":relink-confirm")
            st.caption("The relocation is recorded separately from the original report. Scientific compatibility still needs to be checked.")
            if st.button("Check and record new location", key=key + ":relink", disabled=not replacement or not confirmed):
                from morphofeatures.artifact_locations import record_location
                try:
                    record_location(st.session_state["workspace_root"], missing[name].path, replacement,
                                    reason="User relinked " + name + " from Results")
                    st.rerun()
                except (OSError, ValueError) as error:
                    st.error(str(error))
