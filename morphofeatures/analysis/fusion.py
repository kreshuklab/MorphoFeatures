"""ID-aligned, stateless feature-block concatenation with explicit provenance."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from morphofeatures.analysis.joint import _signature, extraction_metadata
from morphofeatures.artifacts import write_json_atomic
from morphofeatures.data.io import export_embeddings, load_embeddings
from morphofeatures.representations import fingerprint_file
from morphofeatures.workflow_validation import FieldError


def dataset_namespace(metadata):
    if metadata.get("dataset_identity"):
        return metadata["dataset_identity"]
    sources = {
        k: v
        for k, v in metadata.get("sources", {}).items()
        if k in {"crops", "label_ids", "patches_container", "positions_container"}
    }
    return json.dumps(sources, sort_keys=True) if sources else None


def freeze_component_order(settings):
    """Make positional block weights stable across sorted JSON/YAML serialization."""
    names = list(
        dict.fromkeys([*settings.get("embeddings", {}), *settings.get("from_embeddings", [])])
    )
    if not settings.get("component_order"):
        settings["component_order"] = names
    return names


def validate_fusion_settings(settings):
    names = freeze_component_order(settings)
    order = settings["component_order"]
    if len(order) != len(names) or set(order) != set(names):
        raise FieldError(
            "List every feature block exactly once in component_order", "component_order"
        )
    if len(names) < 2:
        raise FieldError(
            "Combine at least two named feature blocks", "embeddings", "from_embeddings"
        )
    weights = np.asarray(settings.get("weights", [1 / len(names)] * len(names)), dtype=float)
    if (
        weights.shape != (len(names),)
        or not np.isfinite(weights).all()
        or np.any(weights < 0)
        or weights.sum() <= 0
    ):
        raise FieldError(
            "Supply one finite nonnegative weight per block, with positive total weight", "weights"
        )
    if settings.get("combination", "balanced") not in {"balanced", "raw"}:
        raise FieldError("Combination must be balanced or raw", "combination")
    declaration = settings.get("identity_override") or {}
    if not isinstance(declaration, dict):
        raise FieldError("Object identity across source files must be a settings mapping.", "identity_override")
    if declaration.get("confirmed"):
        for key in ("dataset", "reason"):
            if not str(declaration.get(key, "")).strip():
                raise FieldError(
                    "Cross-source fusion requires a specimen/dataset name and an explanation of "
                    "how you verified that matching IDs refer to the same objects.",
                    ("identity_override", key),
                )
        if declaration.get("id_namespace") not in {"cell", "nucleus", "object"}:
            raise FieldError(
                "Choose the common cell, nucleus or object ID namespace. Translate IDs before "
                "fusion if the files use different namespaces.", "identity_override.id_namespace"
            )
    return weights / weights.sum()


def fusion_identity(settings, metadata):
    """Separate source fingerprints from an explicit, auditable object correspondence.

    Historical crop containers and newer volume extractions can depict the same
    objects while carrying different source signatures. Never infer that equivalence
    from numeric overlap; a cross-source declaration records the user's evidence.
    The declaration does not change IDs, feature recipes, or source sidecars.
    """
    namespaces = {name: dataset_namespace(m) for name, m in metadata.items()}
    declaration = settings.get("identity_override") or {}
    if declaration.get("confirmed"):
        identity = {k: str(declaration[k]).strip() for k in ("dataset", "id_namespace")}
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        return "declared:" + digest, {
            "method": "user-declared correspondence across source files",
            "declaration": declaration, "source_namespaces": namespaces,
            "limitations": "Matching IDs are joined exactly; this declaration does not independently verify anatomy or translate cell/nucleus IDs.",
        }
    if len({n for n in namespaces.values() if n}) > 1:
        details = "; ".join(
            f"{name}: {str(value)[:180]}" for name, value in namespaces.items()
        )
        raise FieldError(
            "Feature blocks have source records that identify different datasets/object namespaces. "
            "This can also happen when the same specimen was prepared from different containers "
            "or by an older extraction workflow. In Combine embeddings, open 'Object identity "
            "across source files' and confirm correspondence only after checking the specimen "
            "and ID namespace; record your evidence there. Different view recipes are allowed "
            "for fusion. Source identities: " + details,
            "embeddings", "identity_override.confirmed",
        )
    if any(n is None for n in namespaces.values()) and not settings.get("assume_same_dataset", False):
        raise FieldError(
            "Missing dataset provenance: confirm matching object identities when metadata is "
            "missing, or provide a documented correspondence in 'Object identity across source files'. "
            "Check specimen and cell/nucleus IDs first; equal numbers alone do not establish identity.",
            "assume_same_dataset", "embeddings",
        )
    return next((n for n in namespaces.values() if n), None), {
        "method": "source provenance" if all(namespaces.values()) else "user-confirmed missing provenance",
        "source_namespaces": namespaces,
    }


def fuse_embeddings(settings, destination):
    settings = json.loads(json.dumps(settings, default=str))
    weights = validate_fusion_settings(settings)
    inputs = {name: settings["embeddings"][name] for name in settings["component_order"]}
    metadata = {name: extraction_metadata(path) for name, path in inputs.items()}
    namespace, identity_check = fusion_identity(settings, metadata)
    tables = {
        name: load_embeddings(path, require_finite=False).sorted() for name, path in inputs.items()
    }
    ids = np.array(
        sorted(set.intersection(*(set(t.label_ids) for t in tables.values()))), dtype=np.int64
    )
    if not len(ids):
        raise ValueError("Feature blocks have no matching object IDs")
    mode = settings.get("combination", "balanced")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    blocks, components, excluded = [], [], []
    all_ids = set.union(*(set(t.label_ids) for t in tables.values()))
    valid = np.ones(len(ids), dtype=bool)
    offset = 0
    for (name, table), weight in zip(tables.items(), weights):
        values = np.asarray(table.features[np.searchsorted(table.label_ids, ids)], dtype=np.float64)
        norm = np.linalg.norm(values, axis=1)
        good = np.isfinite(values).all(1) & (norm > 0)
        valid &= good
        excluded.extend(
            {"label_id": int(i), "component": name, "reason": "zero or nonfinite component"}
            for i in ids[~good]
        )
        excluded.extend(
            {"label_id": int(i), "component": name, "reason": "missing component"}
            for i in sorted(all_ids - set(table.label_ids))
        )
        normalized = np.zeros_like(values)
        np.divide(values, norm[:, None], out=normalized, where=good[:, None])
        blocks.append(values if mode == "raw" else normalized * np.sqrt(weight))
        components.append(
            {
                "name": name,
                "start": offset,
                "stop": offset + values.shape[1],
                "weight": float(weight),
                "recipe": _signature(metadata[name]),
                "source": fingerprint_file(inputs[name]),
            }
        )
        offset += values.shape[1]
        component_path = export_embeddings(
            destination / f"component-{len(components) - 1}.npz", table.label_ids, table.features
        )
        write_json_atomic(component_path.with_suffix(".metadata.json"), metadata[name])
    pd.DataFrame(excluded, columns=["label_id", "component", "reason"]).to_csv(
        destination / "excluded.tsv", sep="\t", index=False
    )
    if not valid.any():
        raise ValueError("No complete nonzero feature rows remain for combination")
    path = export_embeddings(
        destination / "embeddings.npz",
        ids[valid],
        np.concatenate(blocks, axis=1)[valid].astype(np.float32),
    )
    if settings.get("export_tsv", False):
        export_embeddings(
            path.with_suffix(".tsv"),
            ids[valid],
            np.concatenate(blocks, axis=1)[valid].astype(np.float32),
        )
    recipe = {
        "combination": mode,
        "blocks": [{k: c[k] for k in ("start", "stop", "weight", "recipe")} for c in components],
    }
    digest = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()
    first = next(iter(metadata.values()))
    report = {
        "schema": "morphofeatures.fusion.v1",
        "model": "fusion",
        "sources": {"checkpoint": {"sha256": digest}},
        "dataset_identity": namespace,
        "identity_check": identity_check,
        "components": components,
        "fusion_recipe": recipe,
        "settings": {**settings, "config": first.get("settings", {}).get("config", {})},
        "recommended_normalization": "none",
        "rows": int(valid.sum()),
        "dimensions": offset,
        "embedding": path.name,
    }
    write_json_atomic(path.with_suffix(".metadata.json"), report)
    write_json_atomic(destination / "fusion.json", report)
    return path
