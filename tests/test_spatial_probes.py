"""Scientific and worker contracts for bilateral retrieval and spatial context."""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from morphofeatures.analysis.bilateral import defaults, rank_candidates, run_bilateral
from morphofeatures.analysis.published import references
from morphofeatures.analysis.relations import cell_identity, read_relations
from morphofeatures.analysis.spatial_context import build_context, contact_graph, spatial_graph
from morphofeatures.data.contracts import EmbeddingTable
from morphofeatures.data.io import export_embeddings, load_embeddings
from morphofeatures.data.volumes import SpatialVolume
from morphofeatures.workflow_validation import FieldError, WorkflowValidationError
from morphofeatures.workspace_jobs import plan_workspace_job, resolve_job, run_job
from morphofeatures.workspace_state import new_draft


def table_file(tmp_path, name, data):
    path = tmp_path / name
    pd.DataFrame(data).to_csv(path, sep="\t", index=False)
    return str(path)


def test_mapping_preserves_large_ids_and_rejects_missing_ambiguous(tmp_path):
    high = 2**60 + 11
    mapping = table_file(
        tmp_path, "mapping.tsv", {"label_id": [high, high + 1], "nucleus_id": [7, 9]}
    )
    values = EmbeddingTable(np.array([9, 7]), np.array([[9.0, 1], [7, 1]]))
    cell, bridge = cell_identity(values, {"id_namespace": "nucleus", "id_mapping": mapping})
    assert cell.label_ids.tolist() == [high + 1, high]
    assert bridge.native_id.tolist() == [9, 7]
    assert np.array_equal(cell.features, values.features)
    with pytest.raises(FieldError, match="no cell ID"):
        cell_identity(
            EmbeddingTable(np.array([8]), np.ones((1, 2))),
            {"id_namespace": "nucleus", "id_mapping": mapping},
        )
    table_file(tmp_path, "mapping.tsv", {"label_id": [high, high + 1], "nucleus_id": [7, 7]})
    with pytest.raises(FieldError, match="one-to-one"):
        cell_identity(values, {"id_namespace": "nucleus", "id_mapping": mapping})


def test_candidate_import_is_numeric_only_and_not_spatial(tmp_path):
    path = tmp_path / "partners.pkl"
    with path.open("wb") as stream:
        pickle.dump(
            {np.int64(2**60 + 1): np.array([2**60 + 2], dtype=np.int64)}, stream, protocol=4
        )
    assert read_relations(path, bilateral=True).tolist() == [[2**60 + 1, 2**60 + 2]]
    with pytest.raises(FieldError, match="Spatial adjacency"):
        read_relations(path)
    with path.open("wb") as stream:
        pickle.dump(Path("/tmp/anything"), stream)
    with pytest.raises(ValueError, match="Only numeric"):
        read_relations(path, bilateral=True)


def test_exact_ranks_ties_exclusions_opposite_and_block_invariance():
    ids = np.array([10, 11, 12, 13, 14])
    x = np.array([[0.0, 0], [1, 0], [1, 0], [3, 0], [9, 0]])
    candidates = {10: [12], 11: [13], 12: [999], 13: [11, 14]}
    variant = {"normalization": "none", "search": "all"}
    a = rank_candidates(x, ids, candidates, variant, batch_size=1)
    pd.testing.assert_frame_equal(a, rank_candidates(x, ids, candidates, variant, batch_size=4))
    assert a.loc[0, "rank"] == 1 and a.loc[0, "tied_at_best_distance"] == 2
    assert a.loc[1, "rank"] == 3
    assert a.loc[2, "reason"] == "No candidate partners remain in the shared cohort"
    assert a.loc[4, "reason"] == "No candidate partners listed"
    b = rank_candidates(
        x,
        ids,
        candidates,
        {**variant, "search": "opposite"},
        sides=np.array([False, False, True, True, True]),
    )
    assert b.loc[0, "rank"] == 1
    assert "same-side" in b.loc[3, "reason"]


def test_probe_matches_search_population_and_keeps_native_ids(tmp_path):
    a = export_embeddings(tmp_path / "a.npz", [1, 2, 3, 4], np.arange(8).reshape(4, 2))
    b = export_embeddings(tmp_path / "b.npz", [44, 22, 33], [[6.0, 7], [2, 3], [4, 5]])
    candidates = table_file(
        tmp_path, "pairs.tsv", {"source_id": [1, 2, 3, 4], "target_id": [2, 3, 4, 3]}
    )
    mapping = table_file(tmp_path, "map.tsv", {"label_id": [2, 3, 4], "nucleus_id": [22, 33, 44]})
    settings = {
        "candidates": candidates,
        "same_specimen": True,
        "identities": {"B": {"id_namespace": "nucleus", "id_mapping": mapping}},
    }
    report_path = run_bilateral({"A": a, "B": b}, tmp_path / "probe", settings)
    report = json.loads(report_path.read_text())
    assert report["matched_objects"] == 3
    assert report["representations"]["A"]["excluded_from_matching"] == [1]
    first = pd.read_csv(report_path.parent / "representation-0/ranks-0.tsv", sep="\t")
    second = pd.read_csv(report_path.parent / "representation-1/ranks-0.tsv", sep="\t")
    assert first["rank"].tolist() == second["rank"].tolist()
    assert second.native_id.tolist() == [22, 33, 44]
    assert report_path.with_name("bilateral.svg").is_file()
    assert report_path.with_name("export.zip").is_file()
    with pytest.raises(FieldError, match="same specimen"):
        run_bilateral({"A": a, "B": b}, tmp_path / "invalid", {**settings, "same_specimen": False})


def test_comparison_checks_bilateral_before_reading_features(tmp_path, monkeypatch):
    from morphofeatures import representation_analysis as analysis

    def unexpected_read(*args, **kwargs):
        pytest.fail("An invalid bilateral setting must fail before feature reads or projections")

    monkeypatch.setattr(analysis, "load_embeddings", unexpected_read)
    with pytest.raises(FieldError, match="same specimen") as caught:
        analysis.compare({"Raw": "raw.npz", "Balanced": "balanced.npz"}, tmp_path / "result", {
            "analysis_normalizations": ["none", "l2", "standardize"],
            "bilateral": {"enabled": True, "same_specimen": False},
        })
    assert caught.value.fields == (("bilateral", "same_specimen"),)
    assert not (tmp_path / "result").exists()


def test_fusion_comparison_with_bilateral_and_multiple_scalings(tmp_path):
    from morphofeatures.workspace_state import with_fusion_comparison

    rng = np.random.default_rng(42)
    inputs = {
        name: str(export_embeddings(tmp_path / f"{name}.npz", [1, 2, 3, 4], rng.normal(size=(4, 3))))
        for name in ("Fine", "Coarse")
    }
    candidates = table_file(tmp_path, "pairs.tsv", {"source_id": [1, 2], "target_id": [2, 1]})
    document = with_fusion_comparison({"stages": [
        {"action": "fuse", "name": "Combined", "embeddings": inputs, "assume_same_dataset": True},
        {"action": "analyze", "from_embedding": "Combined", "umap": False, "clusters": 2,
         "analysis_normalizations": ["none", "l2", "standardize"],
         "bilateral": {"enabled": True, "same_specimen": True, "candidates": candidates}},
    ]}, 0)
    job = tmp_path / "job.yaml"
    job.write_text(yaml.safe_dump(document))
    run_job(job)
    status = json.loads((tmp_path / "results/status.json").read_text())
    assert status["state"] == "completed"
    path = tmp_path / "results/02-compare/comparison.json"
    report = json.loads(path.read_text())
    assert len(report["representations"]) == 6
    bilateral = json.loads((path.parent / report["bilateral"]).read_text())
    assert bilateral["matched_objects"] == 4
    assert set(bilateral["representations"]) == {"Combined", "Combined · raw"}


def test_bilateral_ui_warns_next_to_unchecked_confirmation():
    from streamlit.testing.v1 import AppTest

    def render():
        import streamlit as st

        from morphofeatures.spatial_probe_ui import bilateral_fields
        from morphofeatures.workspace_state import new_draft

        if "draft" not in st.session_state:
            draft = new_draft("embeddings")
            draft["document"]["stages"] = [{
                "action": "compare", "from_embeddings": ["Raw", "Balanced"],
                "bilateral": {"enabled": True},
            }]
            st.session_state["draft"] = draft
        draft = st.session_state["draft"]
        bilateral_fields(draft, ["document", "stages", 0, "bilateral"],
                         draft["document"]["stages"][0]["bilateral"], inline=True)

    app = AppTest.from_function(render).run()
    assert not app.exception
    assert any("confirmation above" in warning.value for warning in app.warning)
    next(c for c in app.checkbox if c.label == "These representations describe the same specimen and cells").check().run()
    assert not app.exception
    assert not any("confirmation above" in warning.value for warning in app.warning)
    assert app.session_state["draft"]["document"]["stages"][0]["bilateral"]["same_specimen"]


@pytest.mark.parametrize("connectivity, expected", [(1, []), (2, [[1, 2]]), (3, [[1, 2], [2, 3]])])
def test_contact_connectivity_and_block_seams(connectivity, expected):
    volume = np.zeros((7, 7, 7), dtype=np.uint64)
    volume[2, 2, 2], volume[3, 3, 2], volume[4, 4, 3] = 1, 2, 3
    a, nodes, boundary = contact_graph(
        SpatialVolume(volume, "zyx"), {"connectivity": connectivity, "block_shape": [3, 3, 3]}
    )
    b, _, _ = contact_graph(
        SpatialVolume(volume, "zyx"), {"connectivity": connectivity, "block_shape": [7, 7, 7]}
    )
    assert a.tolist() == b.tolist() == expected
    assert nodes == {1, 2, 3} and not boundary


def test_contacts_large_ids_boundary_and_work_budget():
    x = np.zeros((5, 5, 5), dtype=np.uint64)
    x[1:4, 1:4, 1:3] = 2**60 + 1
    x[1:4, 1:4, 3:] = 2**60 + 2
    edges, nodes, boundary = contact_graph(SpatialVolume(x, "zyx"), {"block_shape": [2, 2, 2]})
    assert edges.tolist() == [[2**60 + 1, 2**60 + 2]]
    assert nodes == {2**60 + 1, 2**60 + 2} and boundary == {2**60 + 2}
    with pytest.raises(FieldError, match="125 segmentation voxels"):
        contact_graph(SpatialVolume(x, "zyx"), {"max_scan_voxels": 10})


def context_inputs(tmp_path):
    embedding = export_embeddings(tmp_path / "features.npz", [3, 1, 2], [[6.0, 8], [2, 4], [4, 6]])
    positions = table_file(
        tmp_path,
        "centers.tsv",
        {"label_id": [1, 2, 3], "z": [0, 0, 0], "y": [0, 0, 0], "x": [0, 1000, 10000]},
    )
    return {
        "embedding": str(embedding),
        "graph_mode": "radius",
        "positions": positions,
        "unit": "nm",
        "radius_um": 1.5,
    }


def test_context_formula_is_self_plus_mean_and_exported_reducer(tmp_path):
    settings = context_inputs(tmp_path)
    output = build_context(settings, tmp_path / "raw")
    table = load_embeddings(output)
    assert table.label_ids.tolist() == [1, 2, 3]
    np.testing.assert_allclose(table.features, [[2, 4, 3, 5], [4, 6, 3, 5], [6, 8, 6, 8]])
    reduced = build_context(
        {**settings, "reduction": "agglomeration", "n_features": 2}, tmp_path / "reduced"
    )
    with np.load(reduced.parent / "agglomeration.npz") as archive:
        groups = archive["feature_groups"]
    reproduced = np.column_stack([table.features[:, groups == i].mean(axis=1) for i in range(2)])
    np.testing.assert_allclose(load_embeddings(reduced).features, reproduced)
    balanced = load_embeddings(
        build_context(
            {**settings, "combination": "balanced", "self_weight": 0.25}, tmp_path / "balanced"
        )
    )
    np.testing.assert_allclose(np.sum(balanced.features[:, :2] ** 2, axis=1), 0.25)
    np.testing.assert_allclose(np.sum(balanced.features[:, 2:] ** 2, axis=1), 0.75)


def test_context_recipe_keeps_encoder_and_neighborhood_identity(tmp_path):
    from morphofeatures.analysis.joint import _signature
    from morphofeatures.analysis.recipe_compatibility import _verified, differences

    settings = context_inputs(tmp_path)
    Path(settings["embedding"]).with_suffix(".metadata.json").write_text(
        json.dumps(
            {
                "model": "dinov2",
                "sources": {"checkpoint": {"sha256": "example-checkpoint"}},
                "settings": {"variant": "dinov2_vitb14_reg", "views": {"feature": "cls"}},
            }
        )
    )
    a = build_context(settings, tmp_path / "a")
    b = build_context({**settings, "radius_um": 2.5}, tmp_path / "b")
    c = build_context({**settings, "self_weight": 0.8}, tmp_path / "c")
    signature_a, signature_b, signature_c = [
        _signature(json.loads(p.with_suffix(".metadata.json").read_text())) for p in (a, b, c)
    ]
    assert _verified(signature_a)
    assert signature_a == signature_c  # Weight is inactive for raw concatenation.
    assert any("radius_um" in d["parameter"] for d in differences(signature_a, signature_b))
    assert signature_a["context"]["input_recipe"]["checkpoint"] == "example-checkpoint"


def test_missing_neighbors_and_boundary_do_not_silently_change_context(tmp_path):
    settings = context_inputs(tmp_path)
    settings.update(
        graph_mode="adjacency_table",
        adjacency_table=table_file(
            tmp_path, "edges.tsv", {"source_id": [1, 2, 3], "target_id": [2, 4, 4]}
        ),
    )
    with pytest.raises(FieldError, match="without embeddings"):
        build_context(settings, tmp_path / "fail")
    output = build_context(
        {**settings, "missing_neighbors": "available", "minimum_coverage": 0.5},
        tmp_path / "partial",
    )
    assert load_embeddings(output).label_ids.tolist() == [1, 2]
    coverage = pd.read_csv(output.parent / "coverage.tsv", sep="\t")
    assert coverage.coverage.tolist() == [1.0, 0.5, 0.0]
    table_file(
        tmp_path,
        "graph_nodes.tsv",
        {"label_id": [1, 2, 3, 4], "boundary": [True, False, False, False]},
    )
    output = build_context(
        {**settings, "missing_neighbors": "available", "minimum_coverage": 0.5},
        tmp_path / "boundary",
    )
    assert load_embeddings(output).label_ids.tolist() == [2]


def test_segmentation_cache_and_nucleus_mapping(tmp_path):
    x = np.zeros((7, 7, 7), dtype=np.int64)
    x[1:4, 1:4, 1:3], x[1:4, 1:4, 3:5] = 11, 22
    np.save(tmp_path / "seg.npy", x)
    mapping = table_file(tmp_path, "map.tsv", {"label_id": [1, 2], "nucleus_id": [11, 22]})
    embedding = export_embeddings(tmp_path / "x.npz", [11, 22], [[1.0, 2], [3, 4]])
    settings = {
        "graph_mode": "segmentation",
        "segmentation": str(tmp_path / "seg.npy"),
        "graph_cache": str(tmp_path / "cache"),
        "embedding": str(embedding),
        "id_namespace": "nucleus",
        "graph_id_namespace": "nucleus",
        "id_mapping": mapping,
    }
    output = build_context(settings, tmp_path / "first")
    assert load_embeddings(output).label_ids.tolist() == [1, 2]
    assert not json.loads(output.with_name("context.json").read_text())["graph"]["cache_hit"]
    other = build_context(settings, tmp_path / "second")
    assert json.loads(other.with_name("context.json").read_text())["graph"]["cache_hit"]
    pd.testing.assert_frame_equal(
        pd.read_csv(other.parent / "graph_id_mapping.tsv", sep="\t"),
        pd.DataFrame({"native_id": [11, 22], "label_id": [1, 2]}),
    )


def test_knn_and_imported_graph_consistency(tmp_path):
    settings = context_inputs(tmp_path)
    edges, _, _, _ = spatial_graph({**settings, "graph_mode": "knn", "neighbors": 1})
    assert edges.tolist() == [[1, 2], [2, 3]]
    source = table_file(tmp_path, "adjacency.tsv", {"source_id": [1, 2], "target_id": [2, 3]})
    with pytest.raises(FieldError, match="exceeds max_edges"):
        spatial_graph({"graph_mode": "adjacency_table", "adjacency_table": source, "max_edges": 1})
    table_file(tmp_path, "graph_nodes.tsv", {"label_id": [1, 2], "boundary": [False, False]})
    with pytest.raises(FieldError, match="every endpoint"):
        spatial_graph({"graph_mode": "adjacency_table", "adjacency_table": source})


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_remote_zarr_contact_graph_matches_local(tmp_path, monkeypatch, zarr_format):
    zarr = pytest.importorskip("zarr")
    from morphofeatures.data.remote_n5 import HttpCache

    major = int(zarr.__version__.split(".")[0])
    if major < 3 and zarr_format == 3:
        pytest.skip("Fixture writes Zarr v2/v3 with Zarr-Python 3")
    x = np.zeros((8, 8, 8), dtype=np.uint16)
    x[1:6, 1:6, 1:4], x[1:6, 1:6, 4:7] = 1, 2
    store = zarr.open_group(
        str(tmp_path / "source"), mode="w", **({"zarr_format": zarr_format} if major >= 3 else {})
    )
    if major >= 3:
        store.create_array("labels", data=x, chunks=(3, 3, 3))
    else:
        store.create_dataset("labels", data=x, chunks=(3, 3, 3))
    calls = []

    def download(self, url):
        calls.append(url)
        path = tmp_path / "source" / url.removeprefix("https://example.test/")
        if path.is_file():
            return path.read_bytes()
        raise FileNotFoundError(url)

    monkeypatch.setattr(HttpCache, "_download", download)
    edges, _, _, _ = spatial_graph(
        {
            "graph_mode": "segmentation",
            "segmentation": "https://example.test",
            "segmentation_key": "labels",
            "remote_options": {
                "format": "ome_zarr",
                "cache_directory": str(tmp_path / "http_cache"),
            },
            "block_shape": [3, 3, 3],
        }
    )
    assert edges.tolist() == [[1, 2]] and len(calls) > 1


def test_worker_context_probe_named_handoffs_and_slurm_plan(tmp_path):
    context = context_inputs(tmp_path)
    candidates = table_file(tmp_path, "pairs.tsv", {"source_id": [1, 2, 3], "target_id": [2, 1, 2]})
    document = {
        "stages": [
            {"action": "context", "name": "Context", **context},
            {"action": "analyze", "from_embedding": "Context", "umap": False, "clusters": 2},
            {
                "action": "bilateral",
                "from_embeddings": ["Context"],
                "embeddings": {"Original": context["embedding"]},
                "candidates": candidates,
                "same_specimen": True,
            },
        ],
        "slurm": {"cpus": 2, "gpus": 0, "partition": "compute", "memory": "4G", "time": "00:10:00"},
    }
    job = tmp_path / "job.yaml"
    job.write_text(yaml.safe_dump(document))
    run_job(job)
    status = json.loads((tmp_path / "results/status.json").read_text())
    assert status["state"] == "completed"
    assert any(p.endswith("context.json") for p in status["artifacts"])
    assert any(p.endswith("bilateral.json") for p in status["artifacts"])
    report = json.loads((tmp_path / "results/02-bilateral/bilateral.json").read_text())
    assert report["matched_objects"] == 3
    plan = plan_workspace_job(
        document,
        output_root=tmp_path / "plans",
        run_id="context-test",
        execution="slurm",
        base=tmp_path,
    )
    assert "context" in plan.resolved_yaml and "#SBATCH" in plan.script
    bad = {"stages": [{"action": "context", **context, "positions": "missing.tsv"}]}
    with pytest.raises(WorkflowValidationError, match="Spatial source"):
        resolve_job(bad, tmp_path)


@pytest.mark.parametrize(
    "reference", ["MorphoFeatures (480 dimensions)", "MorphoContextFeatures (200 dimensions)"]
)
def test_published_layout_is_preserved_and_custom_features_rejected(tmp_path, reference):
    from morphofeatures.representation_analysis import analyze

    draft = new_draft("published")
    draft["document"]["stages"][0].update(
        published_reference=reference, embedding=references()[reference]["embedding"]
    )
    stage = resolve_job(draft["document"], tmp_path)["stages"][0]
    report = analyze(stage["embedding"], tmp_path / "published", stage)
    coordinates = pd.read_csv(report.parent / "coordinates.tsv", sep="\t")
    original = pd.read_csv(
        references()[stage["published_reference"]]["layout"], sep="\t"
    ).sort_values("label_id")
    np.testing.assert_allclose(coordinates[["umap_1", "umap_2"]], original[["umap_1", "umap_2"]])
    assert coordinates.cluster.tolist() == original.cluster.tolist()
    diagnostics = json.loads(report.read_text())["diagnostics"]
    assert diagnostics["feature_objects"] - diagnostics["projected_objects"] == len(
        diagnostics["objects_without_published_coordinates"]
    )
    stage["embedding"] = str(export_embeddings(tmp_path / "custom.npz", [1, 2, 3], np.ones((3, 2))))
    with pytest.raises(WorkflowValidationError, match="matching published"):
        resolve_job({"stages": [stage]}, tmp_path)


def test_bundled_candidate_data_readable():
    edges = read_relations(defaults()["candidates"], bilateral=True)
    assert edges.shape == (107088, 2)
    assert np.all(edges > 0)


@pytest.mark.parametrize("kind", ["context", "bilateral"])
def test_spatial_results_and_native_pair_inspection(tmp_path, kind):
    from streamlit.testing.v1 import AppTest

    settings = context_inputs(tmp_path)
    if kind == "context":
        output = build_context(settings, tmp_path / "context").with_name("context.json")
    else:
        pairs = table_file(tmp_path, "pairs.tsv", {"source_id": [1, 2, 3], "target_id": [2, 1, 2]})
        output = run_bilateral(
            {"Test": settings["embedding"]}, tmp_path / "bilateral", {"candidates": pairs}
        )
    code = """
import json
from pathlib import Path
import streamlit as st
from morphofeatures.config import load_config
from morphofeatures.spatial_probe_ui import probe_results
path = Path(st.session_state["report"])
probe_results(load_config(), json.loads(path.read_text()), path)
"""
    app = AppTest.from_string(code, default_timeout=30)
    app.session_state["report"] = str(output)
    app.run()
    assert not app.exception
    if kind == "bilateral":
        next(
            b for b in app.button if b.label == "Inspect this cell and its best candidate"
        ).click().run()
        assert not app.exception
        assert next(r for r in app.radio if r.label == "Comparison mode").value == "Chosen objects"
        next(b for b in app.button if b.label == "Export bilateral pairs in 3D").click().run()
        assert not app.exception
        assert app.session_state["navigate_to"] == "Workflow"
        draft = next(v for k, v in app.session_state.filtered_state.items() if k.startswith("workflow:"))
        assert draft["document"]["stages"][0]["analysis"] == str(output)
        assert draft["document"]["stages"][0]["action"] == "export_bilateral"
