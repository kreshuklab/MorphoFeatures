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


def test_streaming_ui_connect_render_export_and_edit_invalidation(published_source):
    from streamlit.testing.v1 import AppTest

    document, requests, _, _, root = published_source

    def app_main(document, root):
        from pathlib import Path

        from morphofeatures.analysis_ui import _inspection_source_controls, _mesh_comparison

        resolved = _inspection_source_controls(document, "test", Path(root) / "cache")
        if resolved:
            _mesh_comparison(
                resolved, [3], "test", embedding_ids=[3, 9], check_root=Path(root) / "checks"
            )

    app = AppTest.from_function(app_main, args=(document, str(root)), default_timeout=30).run()
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
