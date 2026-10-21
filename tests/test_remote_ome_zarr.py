"""OME-NGFF axes, codec, resolution, bounded I/O and Workflow contracts."""

import json
from copy import deepcopy

import numpy as np
import pytest

from morphofeatures.data.ome_zarr import discover_ome_zarr, resolve_ome_zarr_preprocessing
from morphofeatures.data.remote_n5 import HttpCache


@pytest.fixture(params=[2, 3])
def ome_source(tmp_path, monkeypatch, request):
    zarr = pytest.importorskip("zarr")
    major = int(zarr.__version__.split(".")[0])
    if request.param == 3 and major < 3:
        pytest.skip("Zarr v3 reading requires Zarr-Python 3")
    axes = [{"name": "t", "type": "time", "unit": "second"}, {"name": "c", "type": "channel"}] + [
        {"name": a, "type": "space", "unit": "nanometer"} for a in "xyz"
    ]
    # Stored T,C,X,Y,Z axes differ from the Z,Y,X coordinates of the UI.
    raw = np.arange(2 * 2 * 13 * 11 * 9, dtype=np.float32).reshape(2, 2, 13, 11, 9)
    labels = np.zeros(raw.shape, dtype=np.uint64)
    labels[1, 1, 3:7, 2:6, 1:4] = 2**53 + 1
    labels[1, 1, 8:12, 7:10, 6:8] = 2**53 + 3
    for kind, array in (("raw", raw), ("segmentation", labels)):
        kwargs = {"zarr_format": request.param} if major >= 3 else {}
        store = zarr.open_group(str(tmp_path / kind), mode="w", **kwargs)
        levels = []
        for level, step in enumerate((1, 2)):
            data = array[:, :, ::step, ::step, ::step]
            key = "fine" if level == 0 else "coarse"
            if major >= 3:
                store.create_array(
                    key,
                    data=data,
                    chunks=(1, 1, 4, 5, 6),
                    **({"shards": (1, 1, 8, 10, 12)} if request.param == 3 else {}),
                )
            else:
                store.create_dataset(key, data=data, chunks=(1, 1, 4, 5, 6), order="F")
            levels.append(
                {
                    "path": key,
                    "coordinateTransformations": [
                        {"type": "scale", "scale": [1, 1, 100 * step, 200 * step, 300 * step]},
                        {"type": "translation", "translation": [0, 0, 500, 600, 700]},
                    ],
                }
            )
        ngff = {"multiscales": [{"version": "0.4", "axes": axes, "datasets": levels}]}
        store.attrs.update(ngff if request.param == 2 else {"ome": {"version": "0.5", **ngff}})
    requests = []

    def download(self, url):
        requests.append(url)
        path = tmp_path / url.removeprefix("https://images.example.test/")
        if path.is_file():
            return path.read_bytes()
        raise FileNotFoundError(url)

    monkeypatch.setattr(HttpCache, "_download", download)
    settings = {
        "action": "preprocess",
        "source": "ome_zarr",
        "ome_zarr": {
            kind: {
                "url": "https://images.example.test/" + kind,
                "level": 0,
                "channel": 1,
                "timepoint": 1,
            }
            for kind in ("raw", "segmentation")
        },
        "remote_options": {"cache_directory": str(tmp_path / "cache"), "cache_size_mb": 16},
        "roi": [[0, 0, 0], [9, 11, 13]],
        "crop_shape": [8, 8, 8],
        "output_format": "h5",
        "normalization": "percentile",
        "max_objects": 2,
        "roi_boundary": "allow",
        "oversized": "clip",
    }
    return settings, raw, labels, requests


def test_context_graph_streams_selected_ome_labels_and_preserves_geometry(ome_source, tmp_path):
    from morphofeatures.analysis.context_sources import resolve_context_source
    from morphofeatures.analysis.spatial_context import build_context
    from morphofeatures.data.io import load_embeddings

    source, _, _, requests = ome_source
    embedding = tmp_path / "features.npz"
    np.savez(embedding, label_ids=np.array([2**53 + 1, 2**53 + 3]), features=[[1., 2.], [3., 4.]])
    settings = {
        "action": "context", "embedding": str(embedding),
        "graph_mode": "segmentation", "spatial_source": "ome_zarr",
        "ome_zarr": source["ome_zarr"]["segmentation"],
        "remote_options": source["remote_options"],
        "roi": source["roi"], "block_shape": [4, 5, 6],
    }
    resolved = resolve_context_source(settings)
    assert resolved["segmentation_axes"] == "tcxyz"
    np.testing.assert_allclose(resolved["segmentation_spacing_zyx"], [.3, .2, .1])
    np.testing.assert_allclose(resolved["segmentation_origin_zyx"], [.7, .6, .5])
    assert all(url.endswith((".json", ".zattrs", ".zarray")) for url in requests)
    build_context(settings, tmp_path / "context")
    vectors = load_embeddings(tmp_path / "context/embeddings.npz")
    np.testing.assert_array_equal(vectors.label_ids, [2**53 + 1, 2**53 + 3])
    np.testing.assert_allclose(vectors.features, [[1, 2, 1, 2], [3, 4, 3, 4]])
    report = json.loads((tmp_path / "context/context.json").read_text())
    assert report["spatial_source_metadata"]["format"] == "ome_zarr"
    assert not any("/raw/" in url for url in requests)
    with pytest.raises(ValueError, match="integer instance"):
        resolve_context_source({**settings, "ome_zarr": source["ome_zarr"]["raw"]})
    with pytest.raises(ValueError, match="choose segmentation graph"):
        resolve_context_source({**settings, "graph_mode": "radius"})


def test_context_catalog_accepts_published_ome_source(ome_source, monkeypatch):
    from morphofeatures.analysis import context_sources

    settings, _, _, _ = ome_source
    options = {**settings["remote_options"], "revision": "test-version", "segmentation_source": "cells", "segmentation_level": 1}
    catalog = {"cells": {"segmentation": {"imageData": {"ome.zarr.s3": {
        "s3Address": settings["ome_zarr"]["segmentation"]["url"], "channel": 1,
    }}}}}
    monkeypatch.setattr(context_sources, "_project_sources", lambda _: (options, None, "https://catalog.example.test/", "https://catalog.example.test/dataset.json", catalog))
    assert context_sources.context_catalog(options)["segmentation"] == ["cells"]
    resolved = context_sources.resolve_context_source({"spatial_source": "platybrowser", "graph_mode": "segmentation", "platybrowser": options})
    assert resolved["segmentation_key"] == "coarse"
    assert resolved["spatial_source_metadata"]["format"] == "ome_zarr"
    np.testing.assert_allclose(resolved["segmentation_spacing_zyx"], [.6, .4, .2])


def test_context_ome_ui_connects_resolutions_and_previews(ome_source, tmp_path):
    from test_guided_workspace import _app, _button, _widget

    from morphofeatures.workspace_state import workflow_document

    source, _, _, _ = ome_source
    app = _app(tmp_path / "workspace")
    _widget(app, "selectbox", "Starting point").set_value("context").run()
    _button(app, "Create workflow").click().run()
    _widget(app, "selectbox", "How to define physical neighbors").set_value("segmentation").run()
    _widget(app, "selectbox", "Spatial context segmentation source").set_value("ome_zarr").run()
    _widget(app, "text_input", "Context segmentation OME-Zarr URL").set_value(source["ome_zarr"]["segmentation"]["url"]).run()
    _widget(app, "number_input", "OME-Zarr segmentation channel").set_value(1).run()
    _widget(app, "number_input", "OME-Zarr segmentation time point").set_value(1).run()
    _widget(app, "text_input", "Remote reader settings (optional YAML)").set_value(json.dumps(source["remote_options"])).run()
    _button(app, "Discover context segmentation resolutions").click().run()
    assert not app.exception and not app.error, [entry.value for entry in app.error]
    _widget(app, "selectbox", "Context segmentation resolution").set_value(1).run()
    _button(app, "Connect context segmentation").click().run()
    assert not app.exception and not app.error, [entry.value for entry in app.error]
    _button(app, "Preview context segmentation").click().run()
    assert not app.exception and not app.error, [entry.value for entry in app.error]
    draft = next(v for k, v in app.session_state.filtered_state.items() if k.startswith("workflow:"))
    stage = workflow_document(draft)["stages"][0]
    assert stage["segmentation_key"] == "coarse"
    assert stage["segmentation_channel"] == stage["segmentation_timepoint"] == 1
    assert stage["spatial_source_metadata"]["format"] == "ome_zarr"


def test_streamed_distributed_patch_preparation(ome_source, tmp_path):
    from morphofeatures.configuration_editor import load_document
    from morphofeatures.data.preprocessing import preprocess
    from morphofeatures.dino import object_readers

    settings, _, _, requests = ome_source
    settings.update(
        layout="distributed_patches",
        patches={"tile_size_um": 0.2, "min_foreground_fraction": 0.25, "max_tiles": 3},
    )
    config = load_document(preprocess(settings, tmp_path / "distributed"))
    seen = []
    for object_id, read in object_readers(config):
        patches = list(read())
        assert 1 <= len(patches) <= 3
        assert all(raw.shape == mask.shape == (8, 8, 8) for raw, mask in patches)
        seen.append(object_id)
    assert seen == [2**53 + 1, 2**53 + 3]
    assert requests
    assert config["data"]["volume"]["raw_timepoint"] == 1
    assert config["data"]["volume"]["raw_channel"] == 1


def test_ome_axes_resolution_preview_and_cache(ome_source):
    from morphofeatures.data.volumes import open_volume, preview_volume_pair, preview_volume_region

    settings, raw, labels, requests = ome_source
    metadata = discover_ome_zarr(settings["ome_zarr"]["raw"]["url"], settings["remote_options"])
    assert metadata["axes"] == "tcxyz"
    assert [v["key"] for v in metadata["levels"]] == ["fine", "coarse"]
    resolved = resolve_ome_zarr_preprocessing(settings)
    np.testing.assert_allclose(resolved["spacing_zyx"], [0.3, 0.2, 0.1])
    np.testing.assert_allclose(resolved["origin_zyx"], [0.7, 0.6, 0.5])
    assert resolved["remote_provenance"]["raw"]["spatial_shape"] == (9, 11, 13)
    assert all(url.endswith((".json", ".zattrs", ".zarray")) for url in requests)
    preview = preview_volume_pair(resolved, axis=0, index=2)
    np.testing.assert_array_equal(preview["raw"], raw[1, 1, :, :, 2].T)
    np.testing.assert_array_equal(preview["segmentation"], labels[1, 1, :, :, 2].T)
    assert 2**53 + 1 in preview["segmentation"]
    downloaded = len(requests)
    preview_volume_pair(resolved, axis=0, index=2)
    assert len(requests) == downloaded
    with open_volume(
        resolved["raw"],
        resolved["raw_key"],
        "tcxyz",
        1,
        timepoint=1,
        remote_options=resolved["remote_options"],
    ) as volume:
        with pytest.raises(ValueError, match="contiguous"):
            volume.read((slice(None, None, 2), slice(None), slice(None)))
    coarse = deepcopy(settings)
    coarse["ome_zarr"]["raw"]["level"] = 1
    coarse["ome_zarr"]["segmentation"]["url"] = None
    coarse["roi"] = None
    resolved = resolve_ome_zarr_preprocessing(coarse, require_pair=False)
    assert "segmentation" not in resolved
    preview = preview_volume_region(resolved, axis=0, index=1)
    np.testing.assert_array_equal(preview["raw"], raw[1, 1, ::2, ::2, 2].T)
    with pytest.raises(ValueError, match="both raw"):
        resolve_ome_zarr_preprocessing(coarse)


def test_ome_preparation_requires_aligned_bounded_roi_and_keeps_ids(ome_source, tmp_path):
    from morphofeatures.configuration_editor import load_document
    from morphofeatures.data.crop_storage import load_crop_array
    from morphofeatures.data.preprocessing import preprocess, validate_preprocessing
    from morphofeatures.workspace_jobs import resolve_job

    settings, _, labels, requests = ome_source
    bad = deepcopy(settings)
    bad["ome_zarr"]["segmentation"]["level"] = 1
    with pytest.raises(ValueError, match="share dimensions"):
        resolve_ome_zarr_preprocessing(bad)
    resolved = resolve_job({"stages": [settings]}, tmp_path)["stages"][0]
    assert all(url.endswith((".json", ".zattrs", ".zarray")) for url in requests)
    with pytest.raises(ValueError, match="explicit bounded ROI"):
        validate_preprocessing({**resolved, "roi": None})
    with pytest.raises(ValueError, match="max_scan_voxels"):
        validate_preprocessing({**resolved, "max_scan_voxels": 10})
    output = preprocess(settings, tmp_path / "prepared")
    data = load_document(output)["data"]
    np.testing.assert_array_equal(load_crop_array(data, "label_ids"), [2**53 + 1, 2**53 + 3])
    assert load_crop_array(data, "loss_masks").sum() == np.count_nonzero(labels)
    report = json.loads((output.parent / "preprocessing.json").read_text())
    assert report["summary"]["processed"] == 2 and report["summary"]["failed"] == 0
    assert report["settings"]["raw_timepoint"] == 1
    from morphofeatures.analysis.mesh_inspection import mesh_source_defaults
    from morphofeatures.analysis.object_inspection import read_object_crops

    config = load_document(output)
    defaults = mesh_source_defaults(config)
    assert defaults["remote_options"]["format"] == "ome_zarr"
    assert defaults["segmentation_timepoint"] == 1
    assert list(read_object_crops(config, [2**53 + 1], source="original"))[0][1].ndim == 3


def test_ome_ui_raw_only_resolution_selection_and_preview(ome_source):
    from streamlit.testing.v1 import AppTest

    settings, _, _, requests = ome_source
    settings = deepcopy(settings)
    settings["ome_zarr"]["segmentation"]["url"] = None
    settings["roi"] = None

    def app_main(settings):
        import streamlit as st

        from morphofeatures.workflow_ui import _prepare
        from morphofeatures.workspace_state import new_draft

        if "draft" not in st.session_state:
            st.session_state["draft"] = new_draft("ome_zarr", document={"stages": [settings]})
        _prepare(st.session_state["draft"], 0)

    app = AppTest.from_function(app_main, args=(settings,), default_timeout=30).run()

    def widget(kind, label):
        return next(w for w in getattr(app, kind) if w.label == label)

    assert not app.exception and not requests
    widget("button", "Find raw resolutions").click().run()
    assert not app.exception and not app.error
    widget("selectbox", "Raw OME-Zarr resolution").set_value(1).run()
    widget("button", "Connect OME-Zarr sources").click().run()
    assert not app.exception and not app.error
    widget("button", "Inspect data dimensions").click().run()
    assert not app.exception and not app.error
    assert all(url.endswith((".json", ".zattrs", ".zarray")) for url in requests)
    widget("checkbox", "Show ROI image preview").check().run()
    assert not app.exception and not app.error
    assert len(app.get("image")) == 1
    widget("number_input", "Raw OME-Zarr timepoint").set_value(0).run()
    assert not app.exception
    assert not any(w.label == "Inspect data dimensions" for w in app.button)


def test_physical_dino_views_stream_selected_channel_and_exact_ids(ome_source, tmp_path):
    from morphofeatures.data.dino_views import PhysicalObjects, index_source, validate_source

    settings, raw, labels, requests = ome_source
    source = validate_source(settings)
    objects = index_source(source, tmp_path / 'physical-index')
    assert set(objects.label_id) == {2**53 + 1, 2**53 + 3}
    with PhysicalObjects(source, objects, {'size': 28, 'axes': [0], 'tile_size_um': 1., 'max_tiles': 2}) as reader:
        obj = reader.prepare(objects.iloc[0].to_dict())
        fine = list(reader.planes(obj, 'fixed_tiles'))
        coarse = list(reader.planes(obj, 'whole_label'))
    assert fine and coarse
    assert all(mask.any() for _, mask, _ in fine + coarse)
    assert all(info['label_id'] == 2**53 + 1 for _, _, info in fine + coarse)
    count = len(requests)
    with PhysicalObjects(source, objects, {'size': 28, 'axes': [0]}) as reader:
        reader.prepare(objects.iloc[0].to_dict())
    assert len(requests) == count
