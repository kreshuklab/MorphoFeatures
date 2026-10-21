"""MoBIE spatial identity, portable data, streaming references and worker handoffs."""

import json
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from morphofeatures.analysis.annotations import read_id_table
from morphofeatures.analysis.mobie_export import export_mobie, resolve_mobie_export
from morphofeatures.data.io import export_embeddings
from morphofeatures.workspace_jobs import resolve_job, run_job


def saved_result(tmp_path, ids, *, comparison=False):
    root = tmp_path / "analysis"
    root.mkdir()
    metadata = {
        "schema": "morphofeatures.comparison.v1" if comparison else "morphofeatures.analysis.v1",
        "settings": {},
        "interpretation": "Synthetic test data",
    }
    entries = (
        {"MAE": "representation-0", "DINO": "representation-1"}
        if comparison
        else {"Embedding": "."}
    )
    if comparison:
        metadata["representations"] = {}
    for i, (name, sub) in enumerate(entries.items()):
        folder = root / sub
        folder.mkdir(exist_ok=True)
        frame = pd.DataFrame(
            {
                "label_id": ids,
                "pca_1": np.arange(len(ids)) + 10 * i,
                "pca_2": np.arange(len(ids)) * -2,
                "umap_1": np.arange(len(ids)) / 2,
                "umap_2": np.arange(len(ids)) + 1,
                "cluster": np.arange(len(ids)) % 2,
                "known_label": ["neuron"] + [None] * (len(ids) - 1),
                "predicted_label": ["neuron"] * len(ids),
                "prediction_source": ["held_out_fold"]
                + ["fit_on_labeled_objects"] * (len(ids) - 1),
            }
        )
        frame.iloc[::-1].to_csv(folder / "coordinates.tsv", sep="\t", index=False)
        frame.drop(columns=["pca_1", "pca_2", "umap_1", "umap_2"]).to_csv(
            folder / "object_labels.tsv", sep="\t", index=False
        )
        export_embeddings(
            folder / "embeddings.npz", np.asarray(ids), np.arange(len(ids) * 3).reshape(-1, 3)
        )
        (folder / "projection.svg").write_text('<svg xmlns="http://www.w3.org/2000/svg"/>')
        if comparison:
            metadata["representations"][name] = {"directory": sub}
    path = root / ("comparison.json" if comparison else "analysis.json")
    path.write_text(json.dumps(metadata))
    return path


def portable_sources(tmp_path, ids):
    h5py = pytest.importorskip("h5py")
    seg = np.zeros((10, 12, 14), dtype=np.int64)
    seg[3:5, 4:6, 5:7] = ids[0]
    seg[6:8, 7:9, 8:10] = ids[1]
    seg[8:9, 9:10, 10:11] = 31  # an object not embedded
    raw = np.arange(seg.size, dtype=np.float32).reshape(seg.shape)
    with h5py.File(tmp_path / "data.h5", "w") as f:
        f["seg"] = seg
        # Exercise non-ZYX axes with channel/timepoint selection.
        stored = np.zeros((2, 2, 14, 12, 10), dtype=np.float32)
        stored[1, 1] = raw.transpose(2, 1, 0)
        f["raw"] = stored
    return (
        {
            "mode": "portable",
            "segmentation": str(tmp_path / "data.h5"),
            "segmentation_key": "seg",
            "raw": str(tmp_path / "data.h5"),
            "raw_key": "raw",
            "raw_axes": "tcxyz",
            "raw_channel": 1,
            "raw_timepoint": 1,
            "roi": [[2, 3, 4], [9, 10, 11]],
            "spacing_zyx": [2.0, 0.5, 0.25],
            "origin_zyx": [10.0, 20.0, 30.0],
            "unit": "um",
            "block_shape": [3, 3, 3],
        },
        raw,
        seg,
    )


def test_portable_project_preserves_ids_world_coordinates_and_evidence(tmp_path):
    zarr = pytest.importorskip("zarr")
    ids = [2**60 + 1, 2**60 + 3, 999]
    analysis = saved_result(tmp_path, ids, comparison=True)
    source, raw, seg = portable_sources(tmp_path, ids)
    report_path = export_mobie(
        {**source, "analysis": str(analysis), "include_feature_table": True}, tmp_path / "export"
    )
    report = json.loads(report_path.read_text())
    assert report["segmentation_objects"] == 3
    assert [r["matched_objects"] for r in report["representations"]] == [2, 2]
    assert [r["excluded_objects"] for r in report["representations"]] == [1, 1]
    root = report_path.parent / "project"
    dataset = root / "morphofeatures"
    table_root = dataset / "tables/objects"
    default = read_id_table(table_root / "default.tsv").set_index("label_id")
    assert set(default.index) == {ids[0], ids[1], 31}
    np.testing.assert_allclose(
        default.loc[ids[0], ["anchor_z", "anchor_y", "anchor_x"]].to_numpy(float),
        [17, 22.25, 31.375],
    )
    np.testing.assert_allclose(
        default.loc[ids[0], ["bb_min_z", "bb_max_z"]].to_numpy(float), [16, 18]
    )
    assert default.loc[31, "touches_export_boundary"]
    data = zarr.open_group(str(dataset / "images/objects.ome.zarr"), mode="r")
    np.testing.assert_array_equal(data["0"][:], seg[2:9, 3:10, 4:11])
    assert data.attrs["multiscales"][0]["name"] == "objects"
    transforms = data.attrs["multiscales"][0]["datasets"][0]["coordinateTransformations"]
    assert transforms == [
        {"type": "scale", "scale": [2, 0.5, 0.25]},
        {"type": "translation", "translation": [14, 21.5, 31]},
    ]
    raw_data = zarr.open_group(str(dataset / "images/raw.ome.zarr"), mode="r")
    np.testing.assert_array_equal(raw_data["0"][:], raw[2:9, 3:10, 4:11])
    first = read_id_table(table_root / "mf0.tsv").set_index("label_id")
    second = read_id_table(table_root / "mf1.tsv").set_index("label_id")
    assert first.loc[ids[0], "mf0_known_label"] == "neuron"
    assert first.loc[ids[1], "mf0_pca_1"] == 1
    assert second.loc[ids[1], "mf1_pca_1"] == 11
    # A left join with absent IDs must never round an original ID via float64.
    exact = pd.read_csv(table_root / "mf0.tsv", sep="\t", dtype=str).set_index("label_id")
    assert exact.loc[str(ids[0]), "mf0_embedding_label_id"] == str(ids[0])
    assert exact.loc[str(ids[1]), "mf0_embedding_label_id"] == str(ids[1])
    features = read_id_table(table_root / "mf0-features.tsv").set_index("label_id")
    np.testing.assert_array_equal(features.loc[ids[1]].to_numpy(), [3, 4, 5])
    metadata = json.loads((dataset / "dataset.json").read_text())
    assert metadata["is2D"] is False
    display = metadata["views"]["mf1-known_label-umap"]["sourceDisplays"][-1]["segmentationDisplay"]
    assert display["scatterPlotAxes"] == ["mf1_umap_1", "mf1_umap_2"]
    assert display["additionalTables"] == ["mf1.tsv"]
    assert display["colorByColumn"] == "mf1_known_label"
    assert (dataset / "misc/analysis/representation-0/embeddings.npz").is_file()
    assert report["evidence"] and len(report["evidence"][0]["sha256"]) == 64
    with zipfile.ZipFile(report_path.parent / "project.zip") as archive:
        assert "project/project.json" in archive.namelist()
        assert "project/morphofeatures/tables/objects/mf0.tsv" in archive.namelist()
    with pytest.raises(FileExistsError):
        export_mobie({**source, "analysis": str(analysis)}, tmp_path / "export")


def test_mapping_partial_coverage_and_guards(tmp_path):
    pytest.importorskip("zarr")
    analysis = saved_result(tmp_path, [100, 200, 300])
    source, _, _ = portable_sources(tmp_path, [7, 11])
    mapping = tmp_path / "map.tsv"
    mapping.write_text("label_id\tnucleus_id\n100\t7\n200\t11\n300\t0\n")
    settings = {**source, "analysis": str(analysis), "id_mapping": str(mapping), "make_zip": False}
    report = json.loads(export_mobie(settings, tmp_path / "mapped").read_text())
    assert report["representations"][0]["matched_objects"] == 2
    with pytest.raises(ValueError, match="missing"):
        export_mobie({**settings, "require_all_ids": True}, tmp_path / "strict")
    assert not (tmp_path / "strict/mobie_export.json").exists()
    with pytest.raises(ValueError, match="max_voxels"):
        resolve_mobie_export({**settings, "max_voxels": 3})
    with pytest.raises(ValueError, match="ROI"):
        resolve_mobie_export({**settings, "roi": [[0, 0, 0], [1.5, 2, 3]]})
    with pytest.raises(ValueError, match="spatial metadata"):
        export_mobie({**settings, "raw_spacing_zyx": [1, 1, 1]}, tmp_path / "misaligned")
    with pytest.raises(ValueError, match="outside the saved analysis"):
        export_mobie(settings, analysis.parent / "export")
    mapping.write_text("label_id\tnucleus_id\n100\t7\n200\t7\n")
    with pytest.raises(ValueError, match="unique positive"):
        resolve_mobie_export(settings)


def test_published_export_uses_metadata_not_volume_reads(tmp_path, monkeypatch):
    from morphofeatures.analysis import mobie_export, platybrowser
    from morphofeatures.data.remote_n5 import HttpCache

    analysis = saved_result(tmp_path, [100, 200])
    catalog = tmp_path / "catalog.tsv"
    pd.DataFrame(
        {"label_id": [100, 200, 900], **{c: [1.0, 2.0, 3.0] for c in mobie_export.SPATIAL_COLUMNS}}
    ).to_csv(catalog, sep="\t", index=False)
    xml = b'<SpimData><SequenceDescription><ViewSetups><ViewSetup><name>published-name</name></ViewSetup></ViewSetups><ImageLoader format="bdv.n5.s3"><ServiceEndpoint>https://example.test</ServiceEndpoint><BucketName>data</BucketName><Key>raw.n5</Key></ImageLoader></SequenceDescription></SpimData>'
    provenance = {
        "raw": {"xml_url": "https://example.test/raw.xml"},
        "segmentation": {"xml_url": "https://example.test/seg.xml"},
        "revision": "a" * 40,
    }
    resolved = {
        "data": {"object_table": str(catalog)},
        "inspection": {
            "platybrowser": {"revision": "a" * 40, "cache_directory": str(tmp_path / "cache")},
            "mesh": {"remote_provenance": provenance},
        },
    }
    monkeypatch.setattr(platybrowser, "resolve_platybrowser", lambda config: resolved)
    monkeypatch.setattr(HttpCache, "get", lambda self, url: xml)

    def no_pixels(*args, **kwargs):
        pytest.fail("Published export must not download image chunks")

    monkeypatch.setattr(mobie_export, "open_volume", no_pixels)
    result = export_mobie(
        {
            "analysis": str(analysis),
            "mode": "platybrowser",
            "platybrowser": {"object_ids": "segmentation"},
        },
        tmp_path / "published",
    )
    report = json.loads(result.read_text())
    assert "no whole-volume" in report["spatial"]["coverage_basis"]
    root = result.parent / "project/morphofeatures"
    assert (
        ET.parse(root / "images/objects.xml").findtext(
            "./SequenceDescription/ViewSetups/ViewSetup/name"
        )
        == "objects"
    )
    sources = json.loads((root / "dataset.json").read_text())["sources"]
    assert sources["objects"]["segmentation"]["imageData"] == {
        "bdv.n5.s3": {"relativePath": "images/objects.xml"}
    }
    assert not list(root.rglob("*.ome.zarr"))


def test_worker_handoff_and_relative_export_paths(tmp_path, monkeypatch):
    pytest.importorskip("zarr")
    from morphofeatures import representation_analysis

    result = saved_result(tmp_path, [10, 20])
    source, _, _ = portable_sources(tmp_path, [10, 20])
    embedding = result.parent / "embeddings.npz"
    stages = [
        {"action": "analyze", "embedding": str(embedding)},
        {**source, "action": "export_mobie", "from_analysis": True, "make_zip": False},
    ]
    monkeypatch.setattr(representation_analysis, "analyze", lambda *args: result)
    path = tmp_path / "job.yaml"
    path.write_text(yaml.safe_dump({"stages": stages}))
    status = json.loads(run_job(path).read_text())
    assert status["state"] == "completed"
    report = json.loads(Path(status["stages"][-1]["result"]).read_text())
    assert report["representations"][0]["matched_objects"] == 2
    with pytest.raises(ValueError, match="earlier analyze"):
        resolve_job({"stages": [stages[-1]]}, tmp_path)
    resolved = resolve_job(
        {"stages": [{**source, "action": "export_mobie", "analysis": "analysis/analysis.json"}]},
        tmp_path,
    )
    assert resolved["stages"][0]["analysis"] == str(result)


def test_export_form_defaults_and_source_recovery(tmp_path):
    pytest.importorskip("streamlit.testing.v1")
    from streamlit.testing.v1 import AppTest

    source, _, _ = portable_sources(tmp_path, [10, 20])
    result = saved_result(tmp_path, [10, 20])
    input_config = tmp_path / "input.yaml"
    input_config.write_text(
        yaml.safe_dump(
            {"data": {k: v for k, v in source.items() if k not in {"mode", "block_shape"}}}
        )
    )

    def app_main(analysis, input_path):
        import streamlit as st

        from morphofeatures.workflow_ui import _export_mobie
        from morphofeatures.workspace_state import new_draft

        if "draft" not in st.session_state:
            st.session_state["draft"] = new_draft(
                "export_mobie",
                document={
                    "stages": [
                        {"action": "export_mobie", "analysis": analysis, "input_config": input_path}
                    ]
                },
            )
        _export_mobie(st.session_state["draft"], 0)

    app = AppTest.from_function(
        app_main, args=(str(result), str(input_config)), default_timeout=30
    ).run()
    assert not app.exception
    assert app.session_state["draft"]["document"]["stages"][0]["roi"] == [[0, 0, 0], [64, 64, 64]]
    next(
        b for b in app.button if b.label == "Check export settings and recover source metadata"
    ).click().run()
    assert not app.exception and not app.error
    stage = app.session_state["draft"]["document"]["stages"][0]
    assert stage["roi"] == source["roi"] and stage["raw"] == source["raw"]
    app.run()
    assert (
        next(w for w in app.text_input if w.label == "Raw container or URL (optional)").value
        == source["raw"]
    )


def test_linked_export_inherits_preparation_without_mutating_draft(tmp_path):
    from copy import deepcopy

    from morphofeatures.workspace_state import preprocessing_defaults

    source, _, _ = portable_sources(tmp_path, [10, 20])
    result = saved_result(tmp_path, [10, 20])
    document = {
        "stages": [
            {
                **preprocessing_defaults(),
                **source,
                "action": "preprocess",
                "segmentation_axes": "zyx",
                "normalization": "percentile",
            },
            {"action": "analyze", "embedding": str(result.parent / "embeddings.npz")},
            {"action": "export_mobie", "from_analysis": True, "mode": "portable"},
        ]
    }
    original = deepcopy(document)
    resolved = resolve_job(document, tmp_path)
    assert resolved["stages"][-1]["roi"] == source["roi"]
    assert resolved["stages"][-1]["raw_timepoint"] == 1
    assert document == original


def test_portable_physical_raw_resamples_to_segmentation_grid(tmp_path):
    zarr = pytest.importorskip('zarr')
    ids = [11, 21]
    analysis = saved_result(tmp_path, ids)
    source, _, _ = portable_sources(tmp_path, ids)
    spacing = np.array(source['spacing_zyx'])
    native_spacing = spacing / 2
    native_origin = np.array(source['origin_zyx']) - spacing / 4
    points = np.indices((20, 24, 28)).transpose(1, 2, 3, 0) * native_spacing + native_origin
    np.save(tmp_path / 'fine-raw.npy', points.sum(3).astype(np.float32))
    source.update(raw=str(tmp_path / 'fine-raw.npy'), raw_key=None, raw_axes='zyx', raw_channel=0, raw_timepoint=0, raw_spacing_zyx=native_spacing.tolist(), raw_origin_zyx=native_origin.tolist(), resample_raw=True)
    report = export_mobie({**source, 'analysis': str(analysis)}, tmp_path / 'export')
    raw = zarr.open_group(str(report.parent / 'project/morphofeatures/images/raw.ome.zarr'), mode='r')['0'][:]
    coordinates = np.indices(raw.shape).transpose(1, 2, 3, 0) + np.array(source['roi'][0])
    expected = (coordinates * spacing + np.array(source['origin_zyx'])).sum(3)
    np.testing.assert_allclose(raw, expected, atol=1e-4)
