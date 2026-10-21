"""Lineage remains inspectable after failures and across saved workflow runs."""

import json

import yaml

from morphofeatures.provenance import build_provenance, provenance_dot, workflow_overview


def recorded_runs(tmp_path):
    upstream = tmp_path / "upstream" / "workspace"
    downstream = tmp_path / "transfer" / "workspace"
    (upstream / "00-extract").mkdir(parents=True)
    downstream.mkdir(parents=True)
    embedding = upstream / "00-extract" / "embeddings.npz"
    # Graph inspection must use metadata, never try to read an embedding array.
    embedding.write_bytes(b"not an array")
    (upstream / "job.yaml").write_text(
        yaml.safe_dump(
            {
                "stages": [
                    {
                        "action": "extract",
                        "name": "Fine",
                        "config": {"data": {"raw": "https://example.invalid/raw.zarr"}},
                        "views": {"min_foreground_fraction": 0.5},
                    }
                ]
            }
        )
    )
    (upstream / "status.json").write_text(
        json.dumps({"stages": [{"index": 0, "state": "completed", "result": str(embedding)}]})
    )
    document = {
        "stages": [
            {
                "action": "transfer",
                "reference": {"embedding": str(embedding)},
                "targets": [{"name": "Target", "embedding": str(tmp_path / "external.npz")}],
            }
        ]
    }
    (downstream / "job.yaml").write_text(yaml.safe_dump(document))
    (downstream / "status.json").write_text(
        json.dumps(
            {"stages": [{"index": 0, "state": "failed", "error": "Historical recipe mismatch"}]}
        )
    )
    return downstream, embedding


def test_saved_graph_links_runs_preserves_failure_and_does_not_open_arrays(tmp_path):
    downstream, embedding = recorded_runs(tmp_path)
    graph = build_provenance(artifact=downstream)
    nodes = {n["id"]: n for n in graph["nodes"]}
    stages = [n for n in nodes.values() if n["kind"] == "stage"]
    assert {n["state"] for n in stages} == {"completed", "failed"}
    assert (
        next(n for n in stages if n["state"] == "failed")["execution"]["error"]
        == "Historical recipe mismatch"
    )
    assert any(n["path"] == "https://example.invalid/raw.zarr" for n in nodes.values())
    assert any(
        e["target"] == "artifact:" + str(embedding) and e["label"] == "produced"
        for e in graph["edges"]
    )
    assert all(e["source"] in nodes and e["target"] in nodes for e in graph["edges"])
    assert 'fillcolor="#ffd7d7"' in provenance_dot(graph)
    overview = workflow_overview(graph)
    assert len(overview["nodes"]) == 2
    assert len(overview["edges"]) == 1 and overview["edges"][0]["kind"] == "data"
    limited = build_provenance(artifact=downstream, max_nodes=2)
    assert len(limited["nodes"]) <= 2 and limited["notices"]


def test_planned_links_distinguish_data_from_execution_order(tmp_path):
    graph = build_provenance(
        document={
            "stages": [
                {"action": "extract", "name": "Fine"},
                {"action": "extract", "name": "Coarse"},
                {"action": "fuse", "name": "Combined", "from_embeddings": ["Fine", "Coarse"]},
                {"action": "analyze", "from_embedding": "Combined"},
            ]
        },
        base=tmp_path,
    )
    data = [e for e in graph["edges"] if e["kind"] == "data"]
    order = [e for e in graph["edges"] if e["kind"] == "order"]
    assert len(data) == 3 and len(order) == 3
    assert all(n["state"] == "planned" for n in graph["nodes"])
    overview = workflow_overview(graph)
    assert len([e for e in overview["edges"] if e["kind"] == "data"]) == 3


def test_saved_bilateral_comparison_and_export_keep_lineage(tmp_path):
    report = tmp_path / "old-bilateral.json"
    report.write_text(json.dumps({"schema": "morphofeatures.bilateral.v1", "settings": {"candidates": "candidates.tsv", "variants": [{"name": "Old settings"}]}}))
    graph = build_provenance(document={"stages": [
        {"action": "bilateral", "compare_saved_runs": True, "comparison_reports": {"Custom MAE": str(report)}},
        {"action": "export_bilateral", "from_bilateral": True},
    ]}, base=tmp_path)
    saved = next(n for n in graph["nodes"] if n.get("path") == str(report) and n["kind"] == "artifact")
    assert saved["settings"]["settings"]["variants"][0]["name"] == "Old settings"
    assert any(e["label"] == "comparison_reports.Custom MAE" for e in graph["edges"])
    assert any(e["label"] == "bilateral" and e["kind"] == "data" for e in graph["edges"])


def test_provenance_ui_reveals_saved_failure_and_settings(tmp_path):
    from streamlit.testing.v1 import AppTest

    downstream, _ = recorded_runs(tmp_path)

    def app(directory):
        from morphofeatures.provenance_ui import provenance_view

        provenance_view(key="test", artifact=directory)

    ui = AppTest.from_function(app, args=(str(downstream),), default_timeout=30).run()
    assert not ui.exception
    ui.checkbox[0].check().run()
    assert not ui.exception
    assert len(ui.selectbox) == 1
    assert any("Historical recipe mismatch" in v.value for v in ui.error)
    assert len(ui.get("graphviz_chart")) == 1


def test_sampling_controls_expose_both_cutoffs_and_anchor():
    from streamlit.testing.v1 import AppTest

    def app():
        import streamlit as st

        from morphofeatures.dino_experiment_ui import sampling_controls
        from morphofeatures.dino_experiments import experiment_defaults

        draft = st.session_state.setdefault(
            "draft",
            {"id": "sampling", "revision": 0, "document": {"stages": [experiment_defaults()]}},
        )
        sampling_controls(draft, ["document", "stages", 0])

    ui = AppTest.from_function(app, default_timeout=30).run()
    assert not ui.exception
    controls = {widget.label: widget for widget in ui.number_input}
    controls["Minimum nucleus content per 3D fine tile (0–1)"].set_value(0.5)
    controls["Minimum nucleus content per 2D fine image (0–1)"].set_value(0.25)
    ui.selectbox[0].select("centroid").run()
    assert not ui.exception
    views = ui.session_state["draft"]["document"]["stages"][0]["views"]
    assert views["min_foreground_fraction"] == 0.5
    assert views["min_slice_foreground_fraction"] == 0.25
    assert views["tile_anchor"] == "centroid"
