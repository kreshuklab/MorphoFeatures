"""Streaming contracts: real N5 encodings, bounded I/O, catalogs and UI selection."""

import json
from urllib.error import HTTPError

import numpy as np
import pytest
import yaml

from morphofeatures.analysis.platybrowser import DEFAULTS, resolve_platybrowser
from morphofeatures.data.remote_n5 import HttpCache, HttpN5Array


@pytest.fixture
def published_source(tmp_path, monkeypatch):
    z5py = pytest.importorskip("z5py")
    shape = (9, 11, 13)
    labels = np.zeros(shape, dtype=np.uint16)
    labels[1:4, 2:6, 3:7] = 101
    labels[6:8, 7:10, 8:12] = 202
    raw = (np.arange(np.prod(shape)).reshape(shape) % 200).astype(np.uint8)
    for name, array in (("nuclei", labels), ("raw", raw)):
        with z5py.File(str(tmp_path / (name + ".n5")), "w") as store:
            dataset = store.create_dataset(
                "setup0/timepoint0/s0", data=array, chunks=(4, 5, 6), compression="gzip"
            )
            dataset.attrs["downsamplingFactors"] = [1, 1, 1]
            if name == "raw":
                coarse = store.create_dataset(
                    "setup0/timepoint0/s1",
                    data=array[::2, ::2, ::2],
                    chunks=(4, 5, 6),
                    compression="gzip",
                )
                coarse.attrs["downsamplingFactors"] = [2, 2, 2]

    base = f"https://raw.githubusercontent.com/mobie/platybrowser-project/{DEFAULTS['revision']}/data/1.0.1/"
    sources = {}
    files = {}
    for name in ("raw", "nuclei"):
        sources[name] = {
            "image" if name == "raw" else "segmentation": {
                "imageData": {"bdv.n5.s3": {"relativePath": f"images/{name}.xml"}},
                "tableData": {"tsv": {"relativePath": "tables/nuclei"}},
            }
        }
        files[base + f"images/{name}.xml"] = f"""<SpimData>
        <SequenceDescription><ViewSetups><ViewSetup><id>0</id>
        <voxelSize><unit>micrometer</unit><size>0.08 0.08 0.1</size></voxelSize>
        </ViewSetup></ViewSetups><ImageLoader format="bdv.n5.s3">
        <ServiceEndpoint>https://images.example.test</ServiceEndpoint><BucketName>volumes</BucketName>
        <Key>{name}.n5</Key></ImageLoader></SequenceDescription>
        <ViewRegistrations><ViewRegistration><ViewTransform>
        <affine>0.08 0 0 0 0 0.08 0 0 0 0 0.1 0</affine>
        </ViewTransform></ViewRegistration></ViewRegistrations></SpimData>""".encode()
    sources["cells"] = {"segmentation": {"tableData": {"tsv": {"relativePath": "tables/cells"}}}}
    files[base + "dataset.json"] = json.dumps({"sources": sources}).encode()
    files[base + "tables/nuclei/default.tsv"] = b"../../../1.0.0/tables/nuclei/default.tsv"
    files[base.replace("/1.0.1/", "/1.0.0/") + "tables/nuclei/default.tsv"] = (
        b"label_id\tbb_min_z\tbb_min_y\tbb_min_x\tbb_max_z\tbb_max_y\tbb_max_x\tn_pixels\n"
        b"101.0\t0.1\t0.16\t0.24\t0.3\t0.4\t0.48\t48\n"
        b"202.0\t0.6\t0.56\t0.64\t0.7\t0.72\t0.88\t24\n"
    )
    files[base + "tables/cells/cells_to_nuclei.tsv"] = (
        b"label_id\tnucleus_id\n3.0\t101.0\n9.0\t202.0\n7.0\t0.0\n8.0\t0.0\n"
    )
    requests = []

    def download(self, url):
        requests.append(url)
        if url in files:
            return files[url]
        prefix = "https://images.example.test/volumes/"
        if url.startswith(prefix):
            path = tmp_path / url[len(prefix) :]
            if path.is_file():
                return path.read_bytes()
        raise FileNotFoundError(url)

    monkeypatch.setattr(HttpCache, "_download", download)
    document = {
        "schema": "morphofeatures.inspection.v1",
        "inspection": {
            "platybrowser": {
                **DEFAULTS,
                "raw_level": 0,
                "segmentation_level": 0,
                "cache_directory": str(tmp_path / "cache"),
            }
        },
    }
    return document, requests, labels, raw, tmp_path


def test_context_catalog_connects_n5_without_reading_raw(published_source):
    from morphofeatures.analysis.context_sources import context_catalog, resolve_context_source
    from morphofeatures.analysis.spatial_context import build_context

    document, requests, _, _, root = published_source
    options = {**document["inspection"]["platybrowser"], "segmentation_source": "nuclei"}
    assert context_catalog(options)["segmentation"] == ["nuclei"]
    embedding = root / "embedding.npz"
    np.savez(embedding, label_ids=[101, 202], features=[[1., 2.], [3., 4.]])
    settings = {"spatial_source": "platybrowser", "graph_mode": "segmentation", "platybrowser": options, "embedding": str(embedding), "roi": [[0, 0, 0], [9, 11, 13]], "block_shape": [4, 5, 6]}
    resolved = resolve_context_source(settings)
    assert resolved["spatial_source_metadata"]["format"] == "n5"
    assert resolved["source_version"] == options["revision"]
    np.testing.assert_allclose(resolved["segmentation_spacing_zyx"], [.1, .08, .08])
    assert all(url.endswith((".xml", ".json", ".tsv")) for url in requests)
    build_context(settings, root / "context")
    assert (root / "context/embeddings.npz").is_file()
    assert not any("/raw.n5/" in url for url in requests)


def test_context_published_source_ui_connects_catalogue(published_source):
    from test_guided_workspace import _app, _button, _widget

    from morphofeatures.workspace_state import workflow_document

    _, requests, _, _, root = published_source
    app = _app(root / "workspace")
    _widget(app, "selectbox", "Starting point").set_value("context").run()
    _button(app, "Create workflow").click().run()
    _widget(app, "selectbox", "How to define physical neighbors").set_value("segmentation").run()
    _widget(app, "selectbox", "Spatial context segmentation source").set_value("platybrowser").run()
    _widget(app, "text_input", "Context streaming cache directory").set_value(str(root / "context-cache")).run()
    _button(app, "List published context segmentations").click().run()
    assert not app.exception and not app.error, [entry.value for entry in app.error]
    _widget(app, "selectbox", "Published context segmentation").set_value("nuclei").run()
    _button(app, "Connect context segmentation").click().run()
    assert not app.exception and not app.error, [entry.value for entry in app.error]
    draft = next(v for k, v in app.session_state.filtered_state.items() if k.startswith("workflow:"))
    assert workflow_document(draft)["stages"][0]["spatial_source_metadata"]["format"] == "n5"
    assert all(url.endswith((".xml", ".json", ".tsv")) for url in requests)


def test_remote_n5_matches_native_partial_chunks_and_cache(published_source):
    document, requests, labels, _, _ = published_source
    options = document["inspection"]["platybrowser"]
    array = HttpN5Array(
        "https://images.example.test/volumes/nuclei.n5", "setup0/timepoint0/s0", options
    )
    roi = (slice(2, 9), slice(3, 11), slice(4, 13))
    np.testing.assert_array_equal(array[roi], labels[roi])
    downloaded = len(requests)
    np.testing.assert_array_equal(array[roi], labels[roi])
    assert len(requests) == downloaded
    assert all("/nuclei.n5/" in url for url in requests)
    with pytest.raises(ValueError, match="contiguous"):
        array[(slice(None, None, 2), slice(None), slice(None))]


def test_catalog_check_never_reads_voxels_and_mapped_meshes_verify_labels(
    published_source, monkeypatch
):
    from morphofeatures.analysis.inspection_validation import check_inspection_ids
    from morphofeatures.analysis.mesh_inspection import load_mesh_batch
    from morphofeatures.analysis.object_inspection import read_object_crops

    document, requests, _, raw, root = published_source
    resolved = resolve_platybrowser(document)
    settings = resolved["inspection"]["mesh"]
    assert "/1.0.0/" in settings["remote_provenance"]["nucleus_table_url"]
    assert all(url.endswith((".xml", ".json", ".tsv")) for url in requests)
    report = check_inspection_ids([3, 9, 7], resolved, settings, root / "check")
    assert report["validation_basis"] == "object_table" and not report["voxel_scan"]
    assert report["matched_objects"] == 2 and report["missing_embedding_ids"] == [7]
    assert all(url.endswith((".xml", ".json", ".tsv")) for url in requests)
    batch = load_mesh_batch(resolved, [3, 9], settings)
    assert not batch["failures"]
    assert [m["segmentation_id"] for m in batch["meshes"]] == ["101", "202"]
    assert [m["selected_label_voxels"] for m in batch["meshes"]] == [48, 24]
    crops = list(read_object_crops(resolved, [3], source="original"))
    assert crops[0][1].dtype == raw.dtype and "Streamed" in crops[0][2]
    downloads = len(requests)
    assert not load_mesh_batch(resolved, [3], settings)["failures"]
    assert len(requests) == downloads
    with pytest.raises(ValueError, match="different source/grid"):
        check_inspection_ids(
            [3], resolved, {**settings, "segmentation_key": "setup0/timepoint0/s1"}, root / "wrong"
        )
    with pytest.raises(ValueError, match="object table"):
        check_inspection_ids([3], resolved, {**settings, "object_index": None}, root / "no_index")
    assert load_mesh_batch(resolved, [7], settings)["failures"]


def test_nucleus_id_mode_and_uri_config_loading(published_source):
    from morphofeatures.configuration_editor import load_document

    document, _, _, _, root = published_source
    document["inspection"]["platybrowser"]["object_ids"] = "nucleus"
    resolved = resolve_platybrowser(document)
    assert resolved["data"]["id_mapping"] is None
    resolved["inspection"]["platybrowser"]["cache_directory"] = "cache-relative"
    file = root / "inspection.yaml"
    file.write_text(yaml.safe_dump(resolved))
    loaded = load_document(file)
    assert loaded["data"]["raw"] == resolved["data"]["raw"]
    assert loaded["inspection"]["mesh"]["segmentation"].startswith("https://")
    assert loaded["inspection"]["platybrowser"]["cache_directory"] == str(root / "cache-relative")


def test_http_failures_are_not_silently_empty_voxels(tmp_path, monkeypatch):
    from morphofeatures.data import remote_n5

    def unavailable(request, **kwargs):
        code = 404 if request.full_url.endswith("missing") else 503
        raise HTTPError(request.full_url, code, "test", {}, None)

    monkeypatch.setattr(remote_n5, "urlopen", unavailable)
    cache = HttpCache({"cache_directory": str(tmp_path)})
    assert cache.get("https://example.test/missing", missing_ok=True) is None
    with pytest.raises(OSError, match="503"):
        cache.get("https://example.test/unavailable", missing_ok=True)
    with pytest.raises(FileNotFoundError):
        cache.get("https://example.test/missing")


def test_cache_evicts_downloads_and_versions_invalidate(tmp_path, monkeypatch):
    calls = []

    def download(self, url):
        calls.append(url)
        return b"123456"

    monkeypatch.setattr(HttpCache, "_download", download)
    settings = {"cache_directory": str(tmp_path), "cache_size_mb": 12 / 1024**2}
    cache = HttpCache(settings)
    for url in ("https://example.test/a", "https://example.test/b"):
        assert cache.get(url) == b"123456"
    assert sum(path.stat().st_size for path in (tmp_path / "downloads").iterdir()) <= 12
    cache.get("https://example.test/b")
    assert len(calls) == 2
    HttpCache({**settings, "cache_version": "2"}).get("https://example.test/b")
    assert len(calls) == 3


def test_decoded_cache_avoids_repeated_decompression(published_source, monkeypatch):
    document, _, labels, _, _ = published_source
    array = HttpN5Array("https://images.example.test/volumes/nuclei.n5", "setup0/timepoint0/s0", document["inspection"]["platybrowser"])
    region = (slice(1, 8), slice(1, 9), slice(1, 10))
    np.testing.assert_array_equal(array[region], labels[region])
    monkeypatch.setattr(array, "_decode", lambda *_: pytest.fail("decoded chunks should be reused"))
    np.testing.assert_array_equal(array[region], labels[region])
    assert array.decoded.hits > 0
    assert array.decoded.bytes <= array.decoded.limit


def test_http_download_locks_allow_independent_requests(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    barrier = Barrier(2)

    def download(self, url):
        barrier.wait(timeout=5)
        return url.encode()

    monkeypatch.setattr(HttpCache, "_download", download)
    cache = HttpCache({"cache_directory": str(tmp_path)})
    urls = ["https://example.test/a", "https://example.test/b"]
    with ThreadPoolExecutor(2) as pool:
        assert list(pool.map(cache.get, urls)) == [url.encode() for url in urls]


def test_parallel_physical_extraction_over_controlled_http(tmp_path):
    import threading
    from functools import partial
    from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

    import torch

    from morphofeatures.data.io import load_embeddings
    from morphofeatures.dino_physical import extract_physical

    z5py = pytest.importorskip("z5py")
    raw = np.random.default_rng(3).integers(1, 255, (20, 20, 20), dtype=np.uint8)
    labels = np.zeros(raw.shape, dtype=np.uint64)
    labels[3:8, 3:8, 3:8] = 2**60 + 1
    labels[11:16, 11:16, 11:16] = 2**60 + 2
    for name, data in (("raw", raw), ("labels", labels)):
        with z5py.File(str(tmp_path / (name + ".n5")), "w") as store:
            store.create_dataset("volume", data=data, chunks=(8, 8, 8), compression="gzip")

    class QuietHandler(SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(QuietHandler, directory=str(tmp_path)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    class Model(torch.nn.Module):
        def forward_features(self, images):
            tokens = torch.nn.functional.avg_pool2d(images, 14, 14).flatten(2).transpose(1, 2)
            return {"x_norm_patchtokens": tokens, "x_norm_clstoken": tokens.mean(1)}

    checkpoint = tmp_path / "weights.pth"
    checkpoint.write_bytes(b"http-fixture-model")
    url = f"http://127.0.0.1:{server.server_port}"
    settings = {
        "model": "dinov2", "checkpoint": str(checkpoint),
        "views": {"mode": "both", "size": 28, "axes": [0], "fine_fractions": [.5], "coarse_fractions": [.5], "tile_size_um": .3, "max_tiles": 2, "features": ["cls", "foreground_mean"]},
        "config": {"device": "cpu", "data": {"volume": {
            "raw": url + "/raw.n5", "raw_key": "volume", "segmentation": url + "/labels.n5", "segmentation_key": "volume",
            "unit": "um", "spacing_zyx": [.1, .1, .1], "roi": [[0, 0, 0], [20, 20, 20]],
            "remote_options": {"cache_directory": str(tmp_path / "downloads"), "cache_size_mb": 4}}}},
    }
    try:
        serial = extract_physical(settings, tmp_path / "serial", backbone=Model())
        settings["runtime"] = {"prepare_workers": 2, "prefetch_mb": 1, "decoded_cache_mb": 1}
        parallel = extract_physical(settings, tmp_path / "parallel", backbone=Model())
        for name in serial:
            left, right = load_embeddings(serial[name]), load_embeddings(parallel[name])
            np.testing.assert_array_equal(left.label_ids, right.label_ids)
            np.testing.assert_allclose(left.features, right.features, atol=1e-6)
        assert load_embeddings(parallel["fixed_tiles.cls"]).label_ids.tolist() == [2**60 + 1, 2**60 + 2]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.parametrize("id_namespace", ["cell", "segmentation"])
def test_streaming_ui_connect_render_export_and_edit_invalidation(published_source, id_namespace):
    from streamlit.testing.v1 import AppTest

    document, requests, _, _, root = published_source
    document["inspection"]["platybrowser"]["object_ids"] = id_namespace
    ids = [3, 9] if id_namespace == "cell" else [101, 202]

    def app_main(document, root, ids):
        from pathlib import Path

        from morphofeatures.analysis_ui import _inspection_source_controls, _mesh_comparison

        resolved = _inspection_source_controls(document, "test", Path(root) / "cache")
        if resolved:
            _mesh_comparison(
                resolved, ids[:1], "test", embedding_ids=ids, check_root=Path(root) / "checks"
            )

    app = AppTest.from_function(app_main, args=(document, str(root), ids), default_timeout=30).run()
    assert not app.exception and not requests
    app.button(key="test:remote:connect").click().run()
    assert not app.exception and not app.error
    assert all(url.endswith((".xml", ".json", ".tsv")) for url in requests)
    task = app.session_state["test:meshes:check_task"]
    assert task.future.result(timeout=10)["validation_basis"] == "object_table"
    app.run()
    app.button(key="test:meshes:render").click().run()
    assert not app.exception and not app.error
    assert app.session_state["test:meshes:batch"]["meshes"][0]["selected_label_voxels"] == 48
    assert any(item.label == "Download inspection YAML" for item in app.get("download_button"))
    app.text_input(key="test:remote:dataset").set_value("1.0.2").run()
    assert not app.exception
    assert not any(button.label == "Render selected meshes" for button in app.button)


@pytest.mark.parametrize("compression", ["raw", "gzip"])
def test_remote_n5_preserves_large_uint64_ids(published_source, compression):
    import z5py

    document, _, _, _, root = published_source
    values = np.arange(5 * 7 * 9, dtype=np.uint64).reshape(5, 7, 9) + 2**60
    with z5py.File(str(root / "large.n5"), "w") as store:
        store.create_dataset("data", data=values, chunks=(3, 4, 5), compression=compression)
    array = HttpN5Array(
        "https://images.example.test/volumes/large.n5",
        "data",
        document["inspection"]["platybrowser"],
    )
    roi = (slice(1, 5), slice(2, 7), slice(3, 9))
    np.testing.assert_array_equal(array[roi], values[roi])


def test_workflow_remote_catalog_preparation_and_id_selection(published_source):
    from morphofeatures.analysis.platybrowser import (
        list_platybrowser_sources,
        resolve_platybrowser_preprocessing,
    )
    from morphofeatures.configuration_editor import load_document
    from morphofeatures.data.crop_storage import load_crop_array
    from morphofeatures.data.preprocessing import preprocess, validate_preprocessing
    from morphofeatures.data.volumes import preview_volume_pair
    from morphofeatures.workspace_jobs import resolve_job

    document, requests, labels, raw, root = published_source
    options = document["inspection"]["platybrowser"]
    catalog = list_platybrowser_sources(options)
    assert catalog["image"] == ["raw"] and catalog["segmentation"] == ["nuclei"]
    stage = {
        "action": "preprocess",
        "source": "platybrowser",
        "platybrowser": options,
        "crop_shape": [8, 8, 8],
        "output_format": "h5",
        "object_ids": [202],
        "max_objects": 1,
        "roi_boundary": "allow",
        "oversized": "clip",
    }
    # Imports preserve HTTP URLs and cache paths; review reads metadata only.
    imported = resolve_job(
        {"stages": [{**stage, "raw": "https://example.test/raw.n5"}]}, root, validate=False
    )
    assert imported["stages"][0]["raw"] == "https://example.test/raw.n5"
    resolved = resolve_job({"stages": [stage]}, root)["stages"][0]
    assert resolved["platybrowser"]["object_ids"] == "segmentation"
    assert resolved["remote_provenance"]["raw_source"] == "raw"
    assert all(url.endswith((".xml", ".json", ".tsv")) for url in requests)
    assert not any("cells_to_nuclei" in url for url in requests)
    with pytest.raises(ValueError, match="different source/grid"):
        validate_preprocessing({**resolved, "segmentation_key": "setup0/timepoint0/s1"})
    with pytest.raises(ValueError, match="different grids"):
        resolve_platybrowser_preprocessing({**stage, "platybrowser": {**options, "raw_level": 1}})

    output = preprocess(stage, root / "prepared")
    config = load_document(output)
    data = config["data"]
    ids = load_crop_array(data, "label_ids")
    masks = load_crop_array(data, "loss_masks")
    np.testing.assert_array_equal(ids, [202])
    assert np.count_nonzero(masks) == np.count_nonzero(labels == 202)
    metadata = json.loads((output.parent / "preprocessing.json").read_text())
    assert metadata["summary"]["processed"] == 1
    assert metadata["summary"]["not_selected"] == 1
    assert metadata["raw"]["url"].startswith("https://")
    assert metadata["settings"]["platybrowser"]["revision"] == DEFAULTS["revision"]
    assert config["inspection"]["mesh"]["id_mapping"] is None
    # Only chunks covering the chosen object are read; nucleus 101's origin block stays unread.
    assert not any(url.endswith("/nuclei.n5/setup0/timepoint0/s0/0/0/0") for url in requests)
    preview = preview_volume_pair({**resolved, "roi": [[5, 6, 7], [9, 11, 13]]}, axis=0, index=6)
    np.testing.assert_array_equal(preview["raw"], raw[6, 6:11, 7:13])
    np.testing.assert_array_equal(preview["segmentation"], labels[6, 6:11, 7:13])


def test_remote_workflow_trains_embeds_and_compares(published_source, repo_root, monkeypatch):
    from morphofeatures.configuration_editor import load_document
    from morphofeatures.data.io import load_embeddings
    from morphofeatures.workspace_jobs import plan_workspace_job, run_job, submit_workspace_plan
    from morphofeatures.workspace_state import stages_after_preparation

    torch = pytest.importorskip("torch")
    torch.set_num_threads(1)

    class TinyBackbone(torch.nn.Module):
        def forward_features(self, batch):
            return {"x_norm_clstoken": batch.mean((2, 3))}

    monkeypatch.setattr("morphofeatures.dino.load_backbone", lambda settings: TinyBackbone())
    document, requests, _, _, root = published_source
    import z5py

    # Analysis needs at least three objects; add a third instance to this fixture.
    with z5py.File(str(root / "nuclei.n5"), "a") as store:
        store["setup0/timepoint0/s0"][1:3, 8:10, 1:3] = np.uint16(303)
    original_download = HttpCache._download

    def download(self, url):
        content = original_download(self, url)
        if "/1.0.0/tables/nuclei/default.tsv" in url:
            content += b"303\t0.1\t0.64\t0.08\t0.2\t0.72\t0.16\t8\n"
        return content

    monkeypatch.setattr(HttpCache, "_download", download)
    stage = {
        "action": "preprocess",
        "source": "platybrowser",
        "platybrowser": document["inspection"]["platybrowser"],
        "crop_shape": [8, 8, 8],
        "max_objects": 3,
        "output_format": "h5",
        "roi_boundary": "allow",
        "oversized": "clip",
    }
    stages = stages_after_preparation(stage, mae="train", dino=True, compare=True)
    stages[1]["config"] = load_document(repo_root / "configs/smoke.yaml")
    repository = root / "backbone"
    repository.mkdir()
    (repository / "hubconf.py").write_text("# Tiny test backbone injected locally\n")
    weights = root / "dino.pth"
    weights.write_bytes(b"test backbone")
    for item in stages:
        if item.get("model") == "dinov2":
            item.update(
                checkpoint=str(weights),
                model_repository=str(repository),
                config={"device": "cpu"},
                views={"size": 28, "axes": [0], "fractions": [0.5], "normalization": "unit"},
            )
        if item["action"] in {"analyze", "compare"}:
            item.update(clusters=2, umap=False)
    plan = plan_workspace_job(
        {"stages": stages}, output_root=root, run_id="remote-models", execution="slurm"
    )
    assert all(url.endswith((".xml", ".json", ".tsv")) for url in requests)
    record = submit_workspace_plan(plan, expected_fingerprint=plan.fingerprint, dry_run=True)
    status = json.loads(run_job(record.config_snapshot).read_text())
    assert status["state"] == "completed"
    assert [s["action"] for s in status["stages"]] == [
        "preprocess",
        "train",
        "extract",
        "analyze",
        "extract",
        "analyze",
        "compare",
    ]
    for item in status["stages"]:
        if item["action"] == "extract":
            table = load_embeddings(item["result"])
            np.testing.assert_array_equal(table.label_ids, [101, 202, 303])
            assert np.isfinite(table.features).all()
    # An existing crop MAE checkpoint uses the same prepared data handoff.
    existing = stages_after_preparation(stage, mae="checkpoint")
    existing[1].update(checkpoint=status["stages"][1]["result"], config=stages[1]["config"])
    existing[2].update(clusters=2, umap=False)
    plan = plan_workspace_job({"stages": existing}, output_root=root, run_id="remote-checkpoint")
    record = submit_workspace_plan(plan, expected_fingerprint=plan.fingerprint, dry_run=True)
    status = json.loads(run_job(record.config_snapshot).read_text())
    np.testing.assert_array_equal(
        load_embeddings(status["stages"][1]["result"]).label_ids, [101, 202, 303]
    )


def test_workflow_streaming_ui_source_selection_and_model_routes(published_source):
    from streamlit.testing.v1 import AppTest

    document, requests, _, _, root = published_source

    def app_main(options):
        import streamlit as st

        from morphofeatures.workflow_ui import _prepare
        from morphofeatures.workspace_state import new_draft

        if "draft" not in st.session_state:
            draft = new_draft("platybrowser")
            draft["document"]["stages"][0]["platybrowser"] = options
            st.session_state["draft"] = draft
        _prepare(st.session_state["draft"], 0)

    app = AppTest.from_function(
        app_main, args=(document["inspection"]["platybrowser"],), default_timeout=30
    ).run()
    assert not app.exception and not requests

    def widget(kind, label):
        return next(item for item in getattr(app, kind) if item.label == label)

    widget("button", "Load available PlatyBrowser datasets").click().run()
    assert not app.exception
    assert widget("selectbox", "Published raw image source").options == ["raw"]
    widget("button", "Connect Workflow to PlatyBrowser").click().run()
    assert not app.exception and not app.error
    assert all(url.endswith((".xml", ".json", ".tsv")) for url in requests)
    widget("selectbox", "MAE workflow").set_value("checkpoint").run()
    widget("checkbox", "Include DINO embeddings").check().run()
    widget("button", "Apply training and embedding choices").click().run()
    assert not app.exception
    stages = app.session_state["draft"]["document"]["stages"]
    assert [s["action"] for s in stages] == [
        "preprocess",
        "extract",
        "analyze",
        "extract",
        "analyze",
        "compare",
    ]
    assert stages[1]["from_preprocessing"] and stages[3]["from_preprocessing"]
    # Editing a source invalidates its UI connection and hides stale previews.
    widget("number_input", "Raw resolution level").set_value(1).run()
    assert not app.exception
    assert not any(b.label == "Inspect data dimensions" for b in app.button)


def test_workflow_supports_named_sources_and_retains_source_failures(published_source, monkeypatch):
    from morphofeatures.analysis.platybrowser import (
        list_platybrowser_sources,
        resolve_platybrowser_preprocessing,
    )
    from morphofeatures.configuration_editor import load_document
    from morphofeatures.data.crop_storage import load_crop_array
    from morphofeatures.data.preprocessing import preprocess

    document, _, _, _, root = published_source
    download = HttpCache._download

    def alias_sources(self, url):
        content = download(self, url)
        if url.endswith("/dataset.json"):
            metadata = json.loads(content)
            metadata["sources"]["signal"] = metadata["sources"].pop("raw")
            metadata["sources"]["instances"] = metadata["sources"].pop("nuclei")
            return json.dumps(metadata).encode()
        return content

    monkeypatch.setattr(HttpCache, "_download", alias_sources)
    options = {
        **document["inspection"]["platybrowser"],
        "raw_source": "signal",
        "segmentation_source": "instances",
    }
    catalog = list_platybrowser_sources(options)
    assert catalog["image"] == ["signal"] and catalog["segmentation"] == ["instances"]
    stage = resolve_platybrowser_preprocessing(
        {
            "source": "platybrowser",
            "platybrowser": options,
            "crop_shape": [8, 8, 8],
            "max_objects": 1,
            "roi_boundary": "allow",
            "oversized": "clip",
        }
    )
    assert stage["remote_provenance"]["segmentation_source"] == "instances"
    # A stale published instance must be reported, not turned into a background crop.
    import z5py

    with z5py.File(str(root / "nuclei.n5"), "a") as store:
        store["setup0/timepoint0/s0"][1:4, 2:6, 3:7] = np.uint16(0)
    result = preprocess(stage, root / "source-failures")
    report = json.loads((result.parent / "preprocessing.json").read_text())
    assert report["summary"] == {"processed": 1, "failed": 1, "not_selected": 0, "skipped": 0}
    assert (
        "absent from its streamed segmentation bounds"
        in (result.parent / "objects.tsv").read_text()
    )
    np.testing.assert_array_equal(
        load_crop_array(load_document(result)["data"], "label_ids"), [202]
    )


def test_distributed_platybrowser_patches_keep_native_mapped_ids(published_source):
    from morphofeatures.configuration_editor import load_document
    from morphofeatures.data.preprocessing import preprocess
    from morphofeatures.dino import object_readers
    from morphofeatures.workspace_state import preprocessing_defaults

    document, _, _, _, root = published_source
    settings = {
        **preprocessing_defaults(),
        "layout": "distributed_patches",
        "source": "platybrowser",
        "platybrowser": document["inspection"]["platybrowser"],
        "roi": None,
        "output_format": "h5",
        "crop_shape": [8, 8, 8],
        "patches": {"tile_size_um": 0.15, "max_tiles": 2, "min_foreground_fraction": 0.25},
    }
    config = load_document(preprocess(settings, root / "distributed"))
    # Replace Prepare's ZYXC defaults with the published array's actual ZYX axes.
    assert config["data"]["volume"]["raw_axes"] == "zyx"
    ids = []
    for object_id, read in object_readers(config):
        assert len(list(read())) == 2
        ids.append(object_id)
    assert ids == [3, 9]
    table = config["inspection"]["mesh"]["object_index"]
    import pandas as pd

    objects = pd.read_csv(table, sep="\t")
    assert objects.segmentation_id.tolist() == [101, 202]


def test_physical_platybrowser_mapping_and_catalog_identity_are_stable(published_source):
    from morphofeatures.data.dino_views import (
        PhysicalObjects,
        index_source,
        source_identity,
        validate_source,
    )
    from morphofeatures.representations import fingerprint_file

    document, _, _, _, root = published_source
    settings = {'source': 'platybrowser', 'platybrowser': document['inspection']['platybrowser']}
    first = validate_source(settings)
    fingerprint = fingerprint_file(first['object_index'])
    second = validate_source(settings)
    assert source_identity(first) == source_identity(second)
    assert fingerprint == fingerprint_file(second['object_index'])
    objects = index_source(second, root / 'physical-index')
    assert set(objects.label_id) == {3, 9}
    assert set(objects.segmentation_id) == {101, 202}
    with PhysicalObjects(second, objects, {'size': 28, 'axes': [0], 'tile_size_um': .2, 'max_tiles': 2}) as reader:
        obj = reader.prepare(objects.iloc[0].to_dict())
        images = list(reader.planes(obj, 'fixed_tiles')) + list(reader.planes(obj, 'whole_label'))
    assert images and all(info['label_id'] == 3 for _, _, info in images)
