"""Bounded, read-only lineage from workflow snapshots and artifact metadata.

No arrays or remote data are loaded. Recorded dependencies are distinguished
from execution order, and historical failures are never rewritten by inspection.
"""

import json
import re
from pathlib import Path

import yaml

from morphofeatures.workflow_validation import STAGES

INPUT_KEYS = {
    "candidates",
    "sides",
    "adjacency_table",
    "positions",
    "embedding",
    "checkpoint",
    "reference_model",
    "annotations",
    "input_config",
    "config_path",
    "raw",
    "segmentation",
    "crops",
    "label_ids",
    "loss_masks",
    "patches_container",
    "positions_container",
    "patch_label_ids",
    "patch_offsets",
    "positions_um",
    "object_index",
    "id_mapping",
    "analysis",
    "preprocessing",
    "raw_container",
    "foreground_mask_container",
    "qc_raw_container",
}


def read_record(path):
    path = Path(path)
    if not path.is_file() or path.stat().st_size > 8 * 1024 * 1024:
        return {}
    value = (
        yaml.safe_load(path.read_text())
        if path.suffix in {".yaml", ".yml"}
        else json.loads(path.read_text())
    )
    return value if isinstance(value, dict) else {}


def input_paths(value, prefix=()):
    if isinstance(value, dict):
        for key, child in value.items():
            path = prefix + (str(key),)
            if key in {"embeddings", "comparison_reports"} and isinstance(child, dict):
                for name, source in child.items():
                    if isinstance(source, str) and source:
                        yield path + (name,), source
            elif isinstance(child, str) and child and (key in INPUT_KEYS or key == "url"):
                yield path, child
            elif isinstance(child, (dict, list)):
                yield from input_paths(child, path)
    elif isinstance(value, list):
        for i, child in enumerate(value):
            yield from input_paths(child, prefix + (str(i),))


def build_provenance(*, document=None, base=None, artifact=None, max_nodes=120, max_depth=6):
    """Build a graph usable by the UI or exported as JSON/DOT."""
    nodes, edges, notices, workflows = {}, [], [], {}
    base = Path(base or Path.cwd()).absolute()

    def node(identifier, **info):
        if identifier not in nodes:
            if len(nodes) >= max_nodes:
                if (
                    "Graph truncated at the node limit; inspect an upstream artifact separately for more detail."
                    not in notices
                ):
                    notices.append(
                        "Graph truncated at the node limit; inspect an upstream artifact separately for more detail."
                    )
                return None
            nodes[identifier] = {"id": identifier, **info}
        return identifier

    def edge(source, target, label, kind="data"):
        if source and target and source != target:
            row = {"source": source, "target": target, "label": label, "kind": kind}
            if row not in edges:
                edges.append(row)

    def read(path):
        try:
            return read_record(path)
        except (OSError, ValueError, yaml.YAMLError) as error:
            notices.append(f"Cannot read provenance at {path}: {error}")
            return {}

    def local(value, directory):
        path = Path(value).expanduser()
        return path.absolute() if path.is_absolute() else (directory / path).absolute()

    def workflow(doc, directory, depth, planned=False):
        key = str(directory)
        if key in workflows:
            return workflows[key]
        stages = workflows[key] = []
        status = {} if planned else read(directory / "status.json")
        naming = {} if planned else read(directory / "display_name.json")
        states = {s.get("index"): s for s in status.get("stages", [])}
        named, previous = {}, {}
        for i, stage in enumerate(doc.get("stages", [])):
            action = stage.get("action", "unknown")
            state = states.get(i, {})
            identifier = node(
                f"stage:{key}:{i}",
                kind="stage",
                label=f"{naming.get('display_name') or (directory.parent.name if directory.name == 'workspace' else 'Workflow')} · {i + 1}. {stage.get('name') or STAGES.get(action, action)}",
                state=state.get("state", "planned" if planned else "not recorded"),
                settings=stage,
                execution=state,
                path=str(directory / "job.yaml"),
                resources=doc.get("slurm", {}),
                naming=naming,
            )
            stages.append(identifier)
            parent_report = doc.get("followup_of", {}).get("analysis")
            if parent_report and i == 0:
                edge(artifact_node(parent_report, directory, depth + 1, "analysis"), identifier, "follow-up of")
            if i:
                edge(stages[i - 1], identifier, "runs after", "order")
            for flag, producer in (
                ("from_preprocessing", "preprocess"),
                ("from_training", "train"),
                ("from_extraction", "extract"),
                ("from_analysis", "analyze"),
                ("from_bilateral", "bilateral"),
            ):
                if stage.get(flag):
                    edge(previous.get(producer), identifier, flag.replace("from_", ""))

            def links(value, identifier=identifier):
                if isinstance(value, dict):
                    for k, v in value.items():
                        if k == "from_embedding":
                            edge(named.get(v), identifier, str(v))
                        elif k == "from_embeddings":
                            for name in v:
                                edge(named.get(name), identifier, str(name))
                        elif isinstance(v, (dict, list)):
                            links(v)
                elif isinstance(value, list):
                    for v in value:
                        links(v)

            links(stage)
            for field, source in input_paths(stage):
                role = "analysis" if "comparison_reports" in field else "embedding" if "embeddings" in field else field[-1]
                edge(artifact_node(source, directory, depth + 1, role), identifier, ".".join(field))
            previous[action] = identifier
            if action in {"joint_analyze", "compare"}:
                previous["analyze"] = identifier
            if stage.get("name"):
                named[stage["name"]] = identifier
                if action == "distill":
                    for method in ("base", "pca", "adapter"):
                        named[stage["name"] + "." + method] = identifier
            output = state.get("result")
            if isinstance(output, str):
                edge(identifier, artifact_node(output, directory, depth + 1, "output"), "produced")
        return stages

    def artifact_node(value, directory, depth, role="embedding"):
        remote = "://" in value
        path = None if remote else local(value, directory)
        location = value if remote else str(path)
        identifier = "artifact:" + location
        if identifier in nodes:
            return identifier
        identifier = node(
            identifier,
            kind="artifact",
            label=f"{role}: {value.rstrip('/').rsplit('/', 1)[-1]}",
            state="recorded input",
            path=location,
            settings={},
        )
        if not identifier or remote:
            return identifier
        if depth > max_depth:
            notices.append(f"Upstream trace stopped at depth {max_depth}: {location}")
            return identifier
        # Only artifact/config records are inspected; raw images and weights are
        # represented by path without probing their contents or remote servers.
        if role not in {
            "embedding",
            "output",
            "analysis",
            "reference_model",
            "preprocessing",
        } and path.suffix not in {".yaml", ".yml"}:
            return identifier
        for ancestor in list(path.parents)[:6]:
            if ancestor.name != "workspace":
                continue
            doc = read(ancestor / "job.yaml")
            if doc:
                stages = workflow(doc, ancestor, depth)
                part = path.relative_to(ancestor).parts[0]
                match = re.match(r"(\d+)-", part)
                if match and int(match[1]) < len(stages):
                    edge(stages[int(match[1])], identifier, "produced")
            break
        record = {}
        if path.suffix in {".json", ".yaml", ".yml"}:
            record = read(path)
        elif role == "embedding":
            record = read(path.with_suffix(".metadata.json"))
            if not record:
                analysis = read(path.parent / "analysis.json")
                if analysis.get("embedding") == path.name:
                    record = analysis.get("extraction", {})
        nodes[identifier]["settings"] = record
        if record.get("display_name"):
            nodes[identifier]["label"] = record["display_name"]
        revision = record.get("reanalysis", {})
        if revision.get("parent"):
            edge(
                artifact_node(revision["parent"], path.parent, depth + 1, "analysis"),
                identifier, "recomputed projection / clustering",
            )
        if role in {"analysis", "output"} and path.suffix == ".json":
            for variant in read(path.parent / "analysis_variants.json").get("variants", []):
                if variant.get("parent") != str(path):
                    continue
                target = variant.get("result") or str(Path(variant["directory"]) / "variant.json")
                child = artifact_node(target, path.parent, depth + 1, "analysis")
                if child:
                    nodes[child]["kind"] = "revision"
                    nodes[child]["label"] = variant["name"]
                    nodes[child]["state"] = variant["state"]
                    nodes[child]["execution"] = variant
                    edge(identifier, child, "analysis combination")
        if "stages" in record:
            workflow(record, path.parent, depth)
        extraction = record.get("extraction", record)
        if extraction.get("settings"):
            recipe = node(
                "recipe:" + location,
                kind="process",
                label=f"Recorded {extraction.get('model') or 'analysis'} settings",
                state="recorded",
                settings=extraction,
                path=location,
            )
            edge(recipe, identifier, "representation / analysis")
            for field, source in input_paths(extraction["settings"]):
                role = "analysis" if "comparison_reports" in field else "embedding" if "embeddings" in field else field[-1]
                edge(artifact_node(source, path.parent, depth + 1, role), recipe, ".".join(field))
        if record.get("schema") == "morphofeatures.dino_experiment.v1":
            for item in record.get("embeddings", []):
                if item.get("embedding"):
                    edge(
                        identifier,
                        artifact_node(item["embedding"], path.parent, depth + 1),
                        str(item.get("representation", item.get("name", "embedding output"))),
                    )
        preparation = extraction.get("preprocessing", {})
        if isinstance(preparation, dict) and preparation:
            prep = node(
                "preparation:" + location,
                kind="process",
                label="Recorded crop preparation",
                state="recorded",
                settings=preparation,
                path=location,
            )
            edge(prep, identifier, "input preparation")
            for field, source in input_paths(preparation.get("settings", {})):
                edge(
                    artifact_node(source, path.parent, depth + 1, field[-1]), prep, ".".join(field)
                )
        return identifier

    if document is not None:
        workflow(document, base, 0, planned=True)
    if artifact is not None:
        artifact_path = Path(artifact)
        if artifact_path.is_dir():
            workflow(read(artifact_path / "job.yaml"), artifact_path.absolute(), 0)
        else:
            artifact_node(
                str(artifact),
                base,
                0,
                "embedding" if artifact_path.suffix != ".json" else "analysis",
            )
    return {
        "schema": "morphofeatures.provenance.v1",
        "nodes": list(nodes.values()),
        "edges": edges,
        "notices": list(dict.fromkeys(notices)),
    }


def workflow_overview(graph):
    """Collapse file/metadata paths into stage dependencies, never execution edges."""
    stages = {node["id"] for node in graph["nodes"] if node["kind"] in {"stage", "revision"}}
    if not any(node["kind"] == "stage" for node in graph["nodes"]):
        return graph
    adjacent = {}
    edges = [
        e
        for e in graph["edges"]
        if e["kind"] == "order" and e["source"] in stages and e["target"] in stages
    ]
    for edge in graph["edges"]:
        if edge["kind"] == "data":
            adjacent.setdefault(edge["source"], []).append(edge["target"])
    for source in sorted(stages):
        pending, seen = list(adjacent.get(source, [])), {source}
        while pending:
            target = pending.pop()
            if target in seen:
                continue
            seen.add(target)
            if target in stages:
                edges.append(
                    {"source": source, "target": target, "label": "supplies data", "kind": "data"}
                )
            else:
                pending.extend(adjacent.get(target, []))
    return {**graph, "nodes": [n for n in graph["nodes"] if n["id"] in stages], "edges": edges}


def provenance_dot(graph):
    lines = [
        "digraph lineage {",
        'rankdir=LR; bgcolor="transparent";',
        'node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, fontcolor="#17212b"];',
        'edge [color="#808080", fontcolor="#808080", fontsize=9];',
    ]
    colors = {
        "failed": "#ffd7d7",
        "completed": "#d5eddf",
        "succeeded": "#d5eddf",
        "running": "#fff0bf",
    }
    for node in graph["nodes"]:
        label = node["label"] + "\n" + node["state"]
        color = colors.get(node["state"], "#e3ecf5" if node["kind"] == "stage" else "#eeeeee")
        lines.append(f'{json.dumps(node["id"])} [label={json.dumps(label)}, fillcolor="{color}"];')
    for edge in graph["edges"]:
        label = edge["label"] if len(edge["label"]) < 44 else edge["label"][-44:]
        style = "dashed" if edge["kind"] == "order" else "solid"
        lines.append(
            f"{json.dumps(edge['source'])} -> {json.dumps(edge['target'])} [label={json.dumps(label)}, style={style}];"
        )
    return "\n".join(lines + ["}"])
