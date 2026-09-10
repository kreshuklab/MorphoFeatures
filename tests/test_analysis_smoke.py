import importlib.util
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from morphofeatures.analysis.classification import (
    cross_validate_logistic,
    load_class_labels,
    select_labeled_embeddings,
)
from morphofeatures.analysis.projection import cluster_embeddings, compute_umap
from morphofeatures.config import load_config
from morphofeatures.data.io import load_embeddings


@pytest.mark.parametrize("model", ["logistic", "mlp"])
def test_workflow_classifier_matches_tools_by_id(tmp_path, model):
    from morphofeatures.analysis.classification import evaluate_embedding_classifier
    from morphofeatures.data.io import export_embeddings
    from morphofeatures.representation_analysis import classify_embedding

    ids = np.arange(40, dtype=np.int64) + 2**53 + 17
    features = np.random.default_rng(8).normal(size=(40, 8))
    names = np.repeat(["neuron", "epithelial", "custom"], 12)
    features[:36, :3] += np.repeat(np.eye(3), 12, axis=0) * 3
    path = export_embeddings(tmp_path / "embeddings.npz", ids, features)
    annotations = pd.DataFrame(
        {"label_id": ids, "cell_type": list(names) + ["rare", None, None, None]}
    )
    annotations.sample(frac=1, random_state=5).to_csv(
        tmp_path / "labels.tsv", sep="\t", index=False
    )
    tool = evaluate_embedding_classifier(
        path, tmp_path / "labels.tsv", model=model, folds=3, seed=7, max_iter=100
    )
    destination = tmp_path / "central"
    destination.mkdir()
    report = classify_embedding(
        load_embeddings(path).sorted(),
        destination,
        {
            "annotations": str(tmp_path / "labels.tsv"),
            "classifier_models": [model],
            "prediction_model": model,
            "folds": 3,
            "seed": 7,
            "max_iter": 100,
        },
    )
    predictions = pd.read_csv(destination / "predictions.tsv", sep="\t").set_index("label_id")
    expected = [tool.class_names[value] for value in tool.predictions]
    assert predictions.loc[tool.label_ids, "predicted"].tolist() == expected
    score = "linear_accuracy" if model == "logistic" else "mlp_accuracy"
    np.testing.assert_array_equal([row[score] for row in report["fold_metrics"]], tool.scores)
    labels = pd.read_csv(destination / "object_labels.tsv", sep="\t").set_index("label_id")
    assert labels.loc[ids[:36], "prediction_source"].eq("held_out_fold").all()
    assert labels.loc[ids[36:], "prediction_source"].eq("fit_on_labeled_objects").all()
    assert labels.loc[ids[36:], "predicted_label"].notna().all()
    assert labels.loc[ids[36], "known_label"] == "rare"


def test_annotations_preserve_large_and_decimal_ids_and_missing_types(tmp_path):
    from morphofeatures.analysis.annotations import read_annotations

    path = tmp_path / "labels.tsv"
    path.write_text("label_id\tcell_label\n3.0\tNone\n9007199254741011\tneuron\n4\tunknown\n")
    labels = read_annotations(path)
    assert labels.label_id.tolist() == [3, 9007199254741011, 4]
    assert labels.known_label.isna().tolist() == [True, False, True]
    path.write_text("label_id\tcell_type\n3.5\tneuron\n")
    with pytest.raises(ValueError, match="exact"):
        read_annotations(path)


def test_projection_subset_is_id_stable_and_uses_shared_tool_defaults():
    pytest.importorskip("umap")
    from morphofeatures.analysis.projection import project_embeddings
    from morphofeatures.representation_analysis import project_table

    ids = np.arange(36, dtype=np.int64) + 101
    features = np.random.default_rng(5).normal(size=(36, 6))
    settings = {"subset": 24, "umap_epochs": 20, "seed": 5}
    tool, diagnostics = project_embeddings(ids, features, settings)
    order = np.random.default_rng(11).permutation(len(ids))
    central, _ = project_table(ids[order], features[order], settings)
    pd.testing.assert_frame_equal(tool, central)
    assert len(tool) == 24 and {"umap_1", "pca_1", "cluster"} <= set(tool)
    assert diagnostics["projection_settings"]["min_dist"] == 0.0
    assert diagnostics["clusters_observed"] == 8


def test_known_label_overlay_does_not_require_classification(tmp_path):
    from morphofeatures.data.io import export_embeddings
    from morphofeatures.representation_analysis import analyze

    ids = np.arange(6)
    embedding = export_embeddings(
        tmp_path / "embeddings.npz", ids, np.random.default_rng(8).normal(size=(6, 3))
    )
    pd.DataFrame({"label_id": [2], "cell_type": ["rare"]}).to_csv(
        tmp_path / "labels.tsv", sep="\t", index=False
    )
    result = analyze(
        embedding,
        tmp_path / "analysis",
        {"annotations": str(tmp_path / "labels.tsv"), "classify": False, "umap": False},
    )
    frame = pd.read_csv(result.parent / "coordinates.tsv", sep="\t")
    assert frame.known_label.notna().sum() == 1
    assert "classification" not in json.loads(result.read_text())


def test_volume_id_mapping_rejects_ambiguous_objects(tmp_path):
    from morphofeatures.analysis.label_volumes import validate_volume_export

    pytest.importorskip("h5py")
    np.save(tmp_path / "seg.npy", np.ones((2, 2, 2), dtype=np.int64))
    pd.DataFrame({"label_id": [5, 6], "cluster": [0, 1]}).to_csv(
        tmp_path / "labels.tsv", sep="\t", index=False
    )
    mapping = tmp_path / "mapping.tsv"
    mapping.write_text("label_id\tsegmentation_id\n5\t9007199254741011\n6\t9007199254741012\n")
    settings = {
        "labels": str(tmp_path / "labels.tsv"),
        "segmentation": str(tmp_path / "seg.npy"),
        "layers": ["cluster"],
        "id_mapping": str(mapping),
    }
    labels = validate_volume_export(settings)
    assert labels.segmentation_id.tolist() == [9007199254741011, 9007199254741012]
    mapping.write_text("label_id\tsegmentation_id\n5\t11\n6\t11\n")
    with pytest.raises(ValueError, match="unique|ambig"):
        validate_volume_export(settings)


def test_projection_panels_share_coordinates_and_select_exact_ids():
    pytest.importorskip("plotly")
    from morphofeatures.analysis.visualization import (
        interactive_projection,
        label_palette,
        selected_object_ids,
    )

    ids = [2**60 + 1, 2**60 + 2, 3]
    frame = pd.DataFrame(
        {
            "label_id": ids,
            "cluster": [0, 1, 0],
            "known_label": ["a", None, "b"],
            "predicted_label": ["a", "b", "b"],
            "umap_1": [1.0, 2.0, 3.0],
            "umap_2": [4.0, 5.0, 6.0],
        }
    )
    chart = interactive_projection(frame, unlabeled_opacity=0.07)
    panels = {}
    for trace in chart.data:
        for custom, x, y, opacity in zip(trace.customdata, trace.x, trace.y, trace.marker.opacity):
            panels.setdefault(trace.xaxis, {})[custom[0]] = (x, y)
            if custom[0] == str(ids[1]):
                assert opacity == 0.07
    assert panels["x"] == panels["x2"] == panels["x3"]
    assert {trace.legend for trace in chart.data} == {"legend", "legend2", "legend3"}
    assert chart.layout.legend.x < chart.layout.legend2.x < chart.layout.legend3.x
    for legend in (chart.layout.legend, chart.layout.legend2, chart.layout.legend3):
        assert legend.bgcolor == "rgba(0,0,0,0)"
        assert legend.font.color is None  # Inherits light/dark/custom UI text color.
    assert all("known_label:" not in trace.name for trace in chart.data)
    assert selected_object_ids(
        {"selection": {"points": [{"customdata": [str(ids[0]), "a"]}]}}, ids
    ) == [ids[0]]
    # Subsampling must not reassign a remaining class to another palette color.
    frame.attrs["class_colors"] = label_palette(frame)
    subset = frame.iloc[[2]].copy()
    subset_chart = interactive_projection(subset)
    assert all(
        trace.marker.color == frame.attrs["class_colors"]["b"]
        for trace in subset_chart.data
        if trace.xaxis in {"x2", "x3"}
    )


def test_neighbor_sheet_reads_selected_crops_and_exports_both_formats(tmp_path):
    from morphofeatures.analysis.object_inspection import neighbor_sheet, read_object_crops

    ids = np.array([2**60 + 1, 2**60 + 2, 2**60 + 3])
    crops = np.arange(3 * 16**3, dtype=np.float32).reshape(3, 16, 16, 16)
    np.save(tmp_path / "ids.npy", ids)
    np.save(tmp_path / "crops.npy", crops)
    config = {
        "data": {"crops": str(tmp_path / "crops.npy"), "label_ids": str(tmp_path / "ids.npy")}
    }
    neighbors = pd.DataFrame({"label_id": ids[[2, 0]], "distance": [0.0, 1.2]})
    values = list(read_object_crops(config, neighbors.label_id, max_side=8))
    np.testing.assert_array_equal(values[0][1], crops[2, 4:12, 4:12, 4:12])
    output = neighbor_sheet(config, neighbors, max_side=8)
    assert b"<svg" in output["svg"] and str(ids[2]).encode() in output["svg"]
    assert b"<text" in output["svg"]  # Labels remain editable in a vector editor.
    assert output["png"].startswith(b"\x89PNG")


@pytest.mark.parametrize("count", [1, 3, 7, 25])
def test_neighbor_figure_keeps_long_labels_clear_of_images_and_other_text(count):
    import matplotlib.pyplot as plt

    from morphofeatures.analysis.object_inspection import neighbor_figure

    ids = np.arange(count, dtype=np.int64) + 2**60 + 1
    neighbors = pd.DataFrame({"label_id": ids, "distance": np.arange(count) / 31})
    labels = pd.DataFrame(
        {
            "label_id": ids,
            "known_label": ["cholinergic sensory neuron of the anterior apical organ"] * count,
            "predicted_label": ["neuron_subtype_" * 7] * count,
            "cluster": np.arange(count),
        }
    )
    images = [
        (
            int(value),
            np.zeros((8, 16, 12)),
            "Raw volume around mean patch position (bounded field of view)",
        )
        for value in ids
    ]
    figure = neighbor_figure(images, neighbors, labels)
    try:
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        texts = [(text.get_text(), text.get_window_extent(renderer)) for text in figure.texts]
        image_bounds = [axis.get_window_extent(renderer) for axis in figure.axes]
        assert len(image_bounds) == 3 * count
        for i, (value, bounds) in enumerate(texts):
            assert figure.bbox.contains(bounds.x0, bounds.y0), value
            assert figure.bbox.contains(bounds.x1, bounds.y1), value
            assert not any(bounds.overlaps(image) for image in image_bounds), value
            assert not any(bounds.overlaps(other) for _, other in texts[i + 1 :]), value
    finally:
        plt.close(figure)


def test_inspection_id_checks_counts_actual_ids_bounds_and_mapping(tmp_path, monkeypatch):
    import sqlite3

    from morphofeatures.analysis.inspection_validation import (
        check_inspection_ids,
        compare_object_ids,
    )
    from morphofeatures.analysis.mesh_inspection import load_mesh_batch

    connect = sqlite3.connect
    connections = []

    def track_connection(*args, **kwargs):
        connection = connect(*args, **kwargs)
        connections.append(connection)
        return connection

    monkeypatch.setattr(sqlite3, "connect", track_connection)
    ids = [2**60 + 1, 2**60 + 2]
    segmentation = np.zeros((8, 10, 12), dtype=np.uint64)
    segmentation[1:3, 2:5, 3:7] = ids[0]
    segmentation[4:6, 5:8, 7:9] = ids[1]
    np.save(tmp_path / "seg.npy", segmentation)
    settings = {
        "source": "segmentation",
        "segmentation": str(tmp_path / "seg.npy"),
        "check_block_shape": [3, 3, 3],
    }
    report = check_inspection_ids([ids[0], ids[1] + 1], {}, settings, tmp_path / "check")
    assert report["count_equal"] and report["state"] == "mismatch"
    assert report["missing_embedding_ids"] == [ids[1] + 1]
    assert report["extra_source_ids"] == [ids[1]]
    bounds = pd.read_csv(report["object_index"], sep="\t").set_index("label_id")
    assert bounds.loc[ids[0], ["bbox_min_z", "bbox_min_y", "bbox_min_x"]].tolist() == [1, 2, 3]
    assert bounds.loc[ids[1], ["bbox_max_z", "bbox_max_y", "bbox_max_x"]].tolist() == [6, 8, 9]
    mapping = tmp_path / "mapping.tsv"
    mapping.write_text(f"label_id\tsegmentation_id\n10\t{ids[0]}\n20\t{ids[1]}\n")
    mapped_settings = {**settings, "id_mapping": str(mapping)}
    mapped = check_inspection_ids([10, 20], {}, mapped_settings, tmp_path / "mapped")
    assert mapped["state"] == "match"
    meshes = load_mesh_batch(
        {},
        [10],
        {**mapped_settings, "object_index": mapped["object_index"], "index_ids": "segmentation"},
    )
    assert not meshes["failures"] and meshes["meshes"][0]["segmentation_id"] == str(ids[0])
    # The site's native mapping stores decimal-form cell/nucleus IDs and repeats
    # zero for cells without nuclei. Those rows must not become background meshes.
    mapping.write_text(
        f"label_id\tnucleus_id\n10.0\t{ids[0]}.0\n20.0\t{ids[1]}.0\n30.0\t0.0\n40.0\t0.0\n"
    )
    native = check_inspection_ids([10, 20, 30], {}, mapped_settings, tmp_path / "native")
    assert native["matched_objects"] == 2 and native["missing_embedding_ids"] == [30]
    meshes = load_mesh_batch(
        {},
        [20],
        {**mapped_settings, "object_index": native["object_index"], "index_ids": "segmentation"},
    )
    assert not meshes["failures"] and meshes["meshes"][0]["segmentation_id"] == str(ids[1])
    mapping.write_text(f"label_id\tnucleus_id\n10\t{ids[0]}\n20\t{ids[0]}\n")
    with pytest.raises(ValueError, match="unique positive"):
        check_inspection_ids([10, 20], {}, mapped_settings, tmp_path / "duplicate_mapping")
    assert compare_object_ids([0, 2], [0, 2, 3])["state"] == "subset"
    assert compare_object_ids([2], [2], {})["state"] == "mismatch"
    segmentation[:] = 0
    np.save(tmp_path / "seg.npy", segmentation)
    empty = check_inspection_ids([1, 255], {}, settings, tmp_path / "empty")
    assert empty["state"] == "mismatch" and empty["source_objects"] == 0
    segmentation[1:3, 2:5, 3:7] = 1
    segmentation[4:6, 5:8, 7:9] = 255
    np.save(tmp_path / "seg.npy", segmentation)
    two_objects = check_inspection_ids([1, 255], {}, settings, tmp_path / "two_objects")
    assert two_objects["state"] == "match"
    for connection in connections:
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            connection.execute("SELECT 1")


def test_foreground_scores_and_whole_volume_masks_are_not_object_ids(tmp_path, monkeypatch):
    from morphofeatures.analysis.inspection_validation import check_inspection_ids
    from morphofeatures.data.volumes import SpatialVolume

    np.save(tmp_path / "scores.npy", np.arange(32**3, dtype=np.uint8).reshape(32, 32, 32))
    read = SpatialVolume.read

    def bounded(self, slices):
        assert all(value.stop - value.start <= 16 for value in slices)
        return read(self, slices)

    monkeypatch.setattr(SpatialVolume, "read", bounded)
    report = check_inspection_ids(
        [11, 12],
        {},
        {
            "source": "segmentation",
            "segmentation": str(tmp_path / "scores.npy"),
            "label_kind": "foreground_scores",
        },
        tmp_path / "scores",
    )
    assert report["state"] == "not_comparable" and report["source_objects"] is None
    np.save(tmp_path / "ids.npy", [11, 12])
    with pytest.raises(ValueError, match="one.*entry per row"):
        check_inspection_ids(
            [11, 12],
            {
                "data": {
                    "label_ids": str(tmp_path / "ids.npy"),
                    "loss_masks": str(tmp_path / "scores.npy"),
                }
            },
            {"source": "prepared"},
            tmp_path / "prepared",
        )


def test_inspection_check_runs_in_background_and_reuses_cache(tmp_path, monkeypatch):
    import threading

    from morphofeatures.analysis.inspection_validation import start_inspection_check
    from morphofeatures.data.volumes import SpatialVolume

    np.save(tmp_path / "seg.npy", np.full((4, 4, 4), 12, dtype=np.int64))
    started, release = threading.Event(), threading.Event()
    original = SpatialVolume.read
    reads = []

    def blocked(self, slices):
        reads.append(threading.current_thread().name)
        started.set()
        assert release.wait(timeout=10)
        return original(self, slices)

    monkeypatch.setattr(SpatialVolume, "read", blocked)
    settings = {"source": "segmentation", "segmentation": str(tmp_path / "seg.npy")}
    task = start_inspection_check([12], {}, settings, tmp_path / "cache")
    try:
        assert started.wait(timeout=10) and not task.future.done()
        assert start_inspection_check([12], {}, settings, tmp_path / "cache") is task
    finally:
        release.set()
    assert task.future.result(timeout=10)["state"] == "match"
    cached = start_inspection_check([12], {}, settings, tmp_path / "cache").future.result(
        timeout=10
    )
    assert cached["cached"] and len(reads) == 1
    assert all(name.startswith("inspection-check") for name in reads)
    (tmp_path / "cache" / task.key / "check.json").write_text("{incomplete")
    rebuilt = start_inspection_check([12], {}, settings, tmp_path / "cache").future.result(
        timeout=10
    )
    assert rebuilt["state"] == "match" and len(reads) == 2


def test_cancelled_inspection_scan_has_no_successful_report(tmp_path):
    import threading

    from morphofeatures.analysis.inspection_validation import CheckCancelled, check_inspection_ids

    np.save(tmp_path / "seg.npy", np.full((6, 6, 6), 12, dtype=np.int64))
    cancel = threading.Event()

    def progress(**values):
        if values.get("blocks") == 1:
            cancel.set()

    destination = tmp_path / "cancelled"
    with pytest.raises(CheckCancelled):
        check_inspection_ids(
            [12],
            {},
            {
                "source": "segmentation",
                "segmentation": str(tmp_path / "seg.npy"),
                "check_block_shape": [2, 2, 2],
            },
            destination,
            progress=progress,
            cancel=cancel,
        )
    assert not (destination / "objects.tsv").exists()


def test_projection_exports_individual_panels_with_separate_clear_legends():
    import matplotlib.pyplot as plt

    from morphofeatures.analysis.visualization import projection_figure

    names = [f"Long descriptive sensory neuron class number {i}" for i in range(10)]
    frame = pd.DataFrame(
        {
            "label_id": np.arange(10),
            "cluster": np.arange(10),
            "known_label": names,
            "predicted_label": names[::-1],
            "pca_1": np.arange(10),
            "pca_2": np.arange(10) ** 2,
        }
    )
    for panels in (None, ["cluster"], ["known_label"], ["predicted_label"]):
        figure = projection_figure(frame, methods=["pca"], panels=panels)
        try:
            figure.canvas.draw()
            renderer = figure.canvas.get_renderer()
            plots = [axis for axis in figure.axes if axis.collections]
            legends = [axis.get_legend() for axis in figure.axes if axis.get_legend()]
            assert len(plots) == len(legends) == (3 if panels is None else 1)
            for i, legend in enumerate(legends):
                bounds = legend.get_window_extent(renderer)
                assert not any(bounds.overlaps(axis.get_window_extent(renderer)) for axis in plots)
                assert not any(
                    bounds.overlaps(other.get_window_extent(renderer)) for other in legends[i + 1 :]
                )
        finally:
            plt.close(figure)


def test_mesh_masks_and_indexed_segmentation_preserve_geometry_and_limit_reads(
    tmp_path, monkeypatch
):
    from morphofeatures.analysis.mesh_inspection import load_mesh_batch
    from morphofeatures.data.volumes import SpatialVolume

    ids = [2**60 + 1, 2**60 + 2]
    segmentation = np.zeros((24, 24, 24), dtype=np.uint64)
    segmentation[2:6, 3:9, 4:10] = ids[0]
    segmentation[12:19, 13:20, 14:21] = ids[1]
    np.save(tmp_path / "seg.npy", segmentation)
    index = pd.DataFrame(
        {
            "label_id": ids,
            "bbox_min_z": [2, 12],
            "bbox_min_y": [3, 13],
            "bbox_min_x": [4, 14],
            "bbox_max_z": [6, 19],
            "bbox_max_y": [9, 20],
            "bbox_max_x": [10, 21],
        }
    )
    index.to_csv(tmp_path / "objects.tsv", sep="\t", index=False)
    reads = []
    original = SpatialVolume.read

    def bounded(self, slices):
        reads.append(slices)
        assert all(item.stop - item.start <= 9 for item in slices)
        return original(self, slices)

    monkeypatch.setattr(SpatialVolume, "read", bounded)
    settings = {
        "source": "segmentation",
        "segmentation": str(tmp_path / "seg.npy"),
        "object_index": str(tmp_path / "objects.tsv"),
        "spacing_zyx": [2, 3, 4],
        "origin_zyx": [10, 20, 30],
        "unit": "nm",
    }
    batch = load_mesh_batch({}, ids, settings, limit=1)
    assert len(batch["meshes"]) == len(reads) == 1
    assert batch["omitted_ids"] == ids[1:] and not batch["failures"]
    mesh = batch["meshes"][0]
    np.testing.assert_allclose(mesh["vertices"].min(0), [44, 27.5, 13])
    np.testing.assert_allclose(mesh["vertices"].max(0), [68, 45.5, 21])
    assert not mesh["boundary_contact"]
    # Equivalent prepared mask uses the same physical frame when start/origin are known.
    np.save(tmp_path / "ids.npy", np.array(ids))
    np.save(tmp_path / "masks.npy", np.stack([segmentation == value for value in ids]))
    index[["crop_start_voxel_z", "crop_start_voxel_y", "crop_start_voxel_x"]] = 0
    index.to_csv(tmp_path / "objects.tsv", sep="\t", index=False)
    (tmp_path / "preprocessing.json").write_text(json.dumps(settings))
    config = {
        "data": {
            "loss_masks": str(tmp_path / "masks.npy"),
            "label_ids": str(tmp_path / "ids.npy"),
            "preprocessing": str(tmp_path / "preprocessing.json"),
        }
    }
    prepared = load_mesh_batch(config, [ids[0]], {"source": "prepared"})
    assert not prepared["failures"]
    np.testing.assert_allclose(prepared["meshes"][0]["vertices"].min(0), mesh["vertices"].min(0))
    np.testing.assert_allclose(prepared["meshes"][0]["vertices"].max(0), mesh["vertices"].max(0))
    too_large = load_mesh_batch({}, ids, {**settings, "max_voxels": 8}, limit=1)
    assert "budget" in too_large["failures"][0]["error"] and len(reads) == 1


@pytest.mark.parametrize("format", ["ply", "obj", "glb"])
def test_mesh_file_previews_and_archives_keep_original_geometry(tmp_path, format):
    import io
    import zipfile

    import trimesh

    from morphofeatures.analysis.mesh_inspection import (
        interactive_meshes,
        load_mesh_batch,
        mesh_archive,
        preview_mesh,
    )

    object_id = 2**60 + 7
    sphere = trimesh.creation.icosphere(subdivisions=3)
    sphere.apply_translation([100, 200, 300])
    sphere.export(tmp_path / f"{object_id}.ply")
    batch = load_mesh_batch(
        {}, [object_id], {"source": "files", "mesh_directory": str(tmp_path), "unit": "um"}
    )
    assert not batch["failures"]
    mesh = batch["meshes"][0]
    vertices, faces = mesh["vertices"].copy(), mesh["faces"].copy()
    preview = preview_mesh(mesh, max_faces=200)
    assert 0 < len(preview["faces"]) <= 200 < len(faces)
    chart = interactive_meshes([preview])
    assert len(chart.data) == 1 and chart.data[0].type == "mesh3d"
    assert np.max(np.abs(chart.data[0].x)) < 2  # Centering is display-only.
    with zipfile.ZipFile(io.BytesIO(mesh_archive([mesh], format=format))) as archive:
        assert f"{object_id}.{format}" in archive.namelist()
        manifest = json.loads(archive.read("manifest.json"))
        assert manifest["objects"][0]["label_id"] == str(object_id)
        with np.load(io.BytesIO(archive.read(f"{object_id}.npz"))) as arrays:
            np.testing.assert_array_equal(arrays["vertices"], vertices)
            np.testing.assert_array_equal(arrays["faces"], faces)
            assert int(arrays["label_id"]) == object_id
    np.testing.assert_array_equal(mesh["vertices"], vertices)


@pytest.mark.parametrize("format", ["h5", "zarr2", "zarr3"])
def test_label_volume_export_is_bounded_and_preserves_grid_and_mapping(
    tmp_path, monkeypatch, format
):
    pytest.importorskip("h5py" if format == "h5" else "zarr")
    if format == "zarr3":
        pytest.importorskip("z5py")
    from morphofeatures.analysis.label_volumes import export_label_volumes
    from morphofeatures.data.volumes import SpatialVolume

    ids = np.array([2**60 + 1, 2**60 + 2, 2**60 + 3])
    segmentation = np.zeros((8, 10, 12), dtype=np.uint64)
    segmentation[1:4, 2:5, 3:6] = ids[0]
    segmentation[4:7, 5:8, 6:9] = ids[1]
    segmentation[0, 0, 0] = 77  # Not in the embedding.
    np.save(tmp_path / "seg.npy", segmentation.transpose(2, 0, 1))
    pd.DataFrame(
        {
            "label_id": ids,
            "known_label": ["neuron", None, "epithelial"],
            "predicted_label": ["neuron", "epithelial", "epithelial"],
            "cluster": [0, 1, 0],
            "prediction_source": ["held_out_fold", "fit_on_labeled_objects", "held_out_fold"],
        }
    ).to_csv(tmp_path / "labels.tsv", sep="\t", index=False)
    original = SpatialVolume.read

    def bounded(self, slices):
        assert all(s.stop - s.start <= 3 for s in slices)
        return original(self, slices)

    monkeypatch.setattr(SpatialVolume, "read", bounded)
    report_path = export_label_volumes(
        {
            "labels": str(tmp_path / "labels.tsv"),
            "segmentation": str(tmp_path / "seg.npy"),
            "segmentation_axes": "xzy",
            "output_format": format,
            "block_shape": [3, 3, 3],
            "spacing_zyx": [2, 3, 4],
            "origin_zyx": [5, 6, 7],
            "unit": "nm",
            "include_rgb": True,
        },
        tmp_path / "export",
    )
    report = json.loads(report_path.read_text())
    assert report["matched_objects"] == 2 and report["absent_object_ids"] == [int(ids[2])]
    if format == "h5":
        import h5py

        store = h5py.File(report["container"], "r")
    elif format == "zarr2":
        import zarr

        store = zarr.open_group(report["container"], mode="r")
    else:
        import z5py

        store = z5py.File(report["container"], "r")
        assert (
            json.loads((tmp_path / "export/labels.zarr/zarr.json").read_text())["zarr_format"] == 3
        )
    try:
        assert tuple(store["cluster"].shape) == segmentation.shape
        np.testing.assert_array_equal(store.attrs["spacing_zyx"], [2, 3, 4])
        expected = np.zeros(segmentation.shape, dtype=np.uint32)
        expected[segmentation == ids[0]] = 1
        expected[segmentation == ids[1]] = 2
        np.testing.assert_array_equal(store["cluster"][:], expected)
        assert np.all(store["known_label"][:][segmentation == ids[1]] == 0)
        assert np.all(store["predicted_label"][:][segmentation == ids[1]] > 0)
        assert store["cluster_rgb"].shape == segmentation.shape + (3,)
        assert np.any(store["cluster_rgb"][:][segmentation == ids[0]] > 0)
        assert np.all(store["cluster_rgb"][:][segmentation == 0] == 0)
    finally:
        if hasattr(store, "close"):
            store.close()
    mapped = pd.read_csv(tmp_path / "export/object_mapping.tsv", sep="\t")
    from morphofeatures.analysis.visualization import label_palette

    lookup = pd.read_csv(tmp_path / "export/label_lookup.tsv", sep="\t")
    class_colors = label_palette(mapped)
    for row in lookup.itertuples():
        if row.layer in {"known_label", "predicted_label"} and row.value:
            assert row.color == class_colors[row.name]
    np.testing.assert_array_equal(mapped.label_id, ids)
    assert "prediction_source" in mapped
    np.testing.assert_array_equal(np.load(tmp_path / "seg.npy"), segmentation.transpose(2, 0, 1))


def test_logistic_regression_on_bundled_labels():
    config = load_config()
    table = load_embeddings(config.paths.analysis_data / "morphofeatures_all_cells.npy")
    ids, labels, class_names = load_class_labels(config.paths.analysis_data / "class_labels.tsv")
    features, labels = select_labeled_embeddings(table, ids, labels)
    features = StandardScaler().fit_transform(features[:, :24])
    result = cross_validate_logistic(features, labels, class_names, folds=3, seed=7, max_iter=500)
    assert result.scores.shape == (3,)
    assert result.confusion.sum() == len(labels)
    assert 0.0 <= result.mean_accuracy <= 1.0


@pytest.mark.optional
def test_umap_and_clustering_small_bundled_subset():
    if importlib.util.find_spec("umap") is None:
        pytest.skip("umap-learn is not installed")
    try:
        import umap  # noqa: F401
    except (ImportError, OSError) as error:
        pytest.skip(f"umap-learn runtime is unavailable: {error}")
    config = load_config()
    table = load_embeddings(config.paths.analysis_data / "morphofeatures_all_cells.npy")
    features = StandardScaler().fit_transform(table.features[:64, :16])
    projection = compute_umap(features, n_neighbors=8, seed=7, n_epochs=20)
    labels = cluster_embeddings(features, method="kmeans", n_clusters=4, seed=7)
    assert projection.shape == (64, 2)
    assert labels.shape == (64,)
    assert len(np.unique(labels)) == 4
