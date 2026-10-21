"""Saved-run fairness, physical pair geometry, and MoBIE/native viewer exports."""

import json

import numpy as np
import pandas as pd
import pytest
import yaml

from morphofeatures.analysis.bilateral import run_bilateral
from morphofeatures.analysis.bilateral_comparison import read_ranks
from morphofeatures.analysis.bilateral_export import export_bilateral
from morphofeatures.data.io import export_embeddings, load_embeddings
from morphofeatures.workflow_validation import FieldError
from morphofeatures.workspace_jobs import resolve_job, run_job


def fixture_report(tmp_path, ids=None):
    ids = ids or [1, 2, 3, 4]
    features = np.array([[0.0, 1], [1, 0], [10, 5], [9, 4]])
    embedding = export_embeddings(tmp_path / "input.npz", ids, features)
    candidates = tmp_path / "candidates.tsv"
    pd.DataFrame({"source_id": ids, "target_id": [ids[1], ids[0], ids[3], ids[2]]}).to_csv(
        candidates, sep="\t", index=False
    )
    report = run_bilateral(
        {"Custom DINO": str(embedding)}, tmp_path / "original", {"candidates": str(candidates)}
    )
    return report, embedding, candidates


def positions(tmp_path, ids):
    path = tmp_path / "positions.tsv"
    pd.DataFrame(
        {
            "label_id": ids,
            "anchor_z": np.arange(len(ids)) * 1000,
            "anchor_y": np.arange(len(ids)) * 2000,
            "anchor_x": np.arange(len(ids)) * 3000,
        }
    ).to_csv(path, sep="\t", index=False)
    return str(path)


def test_saved_run_recomputed_on_shared_cells_without_modifying_original(tmp_path):
    old, _, candidates = fixture_report(tmp_path)
    before = old.read_bytes()
    new = export_embeddings(tmp_path / "new.npz", [2, 1, 3], [[10.0, 0], [0, 1], [1, 1]])
    settings = {
        "candidates": str(candidates),
        "compare_saved_runs": True,
        "comparison_reports": {"Previous custom run": str(old)},
        "same_specimen": True,
    }
    result = run_bilateral({"New MAE": str(new)}, tmp_path / "comparison", settings)
    report = json.loads(result.read_text())
    assert report["matched_objects"] == 3
    previous = report["representations"]["Previous custom run / Custom DINO"]
    assert previous["saved_run"]["original_matched_objects"] == 4
    assert load_embeddings(
        result.parent / previous["directory"] / "embeddings.npz"
    ).label_ids.tolist() == [1, 2, 3]
    ranks = read_ranks(result.parent / previous["directory"] / "ranks-0.tsv")
    assert (
        ranks.loc[ranks.label_id == 3, "reason"].iloc[0]
        == "No candidate partners remain in the shared cohort"
    )
    assert old.read_bytes() == before
    assert (result.parent / "bilateral.svg").is_file()
    broken = (
        old.parent
        / json.loads(before)["representations"]["Custom DINO"]["directory"]
        / "embeddings.npz"
    )
    broken.unlink()
    with pytest.raises(FieldError, match="saved vectors"):
        run_bilateral({"New": str(new)}, tmp_path / "fail", settings)


def test_physical_geometry_preserves_large_ids_and_zyx_units(tmp_path):
    ids = [2**60 + i for i in (1, 3, 7, 9)]
    source, _, _ = fixture_report(tmp_path, ids)
    report = export_bilateral(
        {
            "analysis": str(source),
            "make_mobie": False,
            "positions": positions(tmp_path, ids),
            "positions_unit": "nm",
            "query_ids": ids[:2],
            "max_pairs": 2,
        },
        tmp_path / "export",
    )
    metadata = json.loads(report.read_text())
    folder = report.parent / metadata["groups"][0]["directory"]
    with np.load(folder / "pairs.npz", allow_pickle=False) as data:
        assert data["query_ids"].tolist() == ids[:2]
        assert data["partner_ids"].tolist() == ids[1::-1]
        np.testing.assert_allclose(data["endpoints_zyx_um"][0], [[0, 0, 0], [1, 2, 3]])
        np.testing.assert_allclose(data["vectors_zyx_um"][1], [[1, 2, 3], [-1, -2, -3]])
    assert "3 2 1" in (folder / "pairs.vtk").read_text()
    with pytest.raises(FieldError, match="More query IDs"):
        export_bilateral(
            {
                "analysis": str(source),
                "make_mobie": False,
                "positions": positions(tmp_path, ids),
                "query_ids": ids,
                "max_pairs": 1,
            },
            tmp_path / "oversized",
        )


def test_mobie_pair_views_and_overlapping_hdf5_layers(tmp_path):
    from test_mobie_export import portable_sources

    source, _, _ = fixture_report(tmp_path)
    segment_ids = [2**60 + 1, 2**60 + 3]
    spatial, _, _ = portable_sources(tmp_path, segment_ids)
    mapping = tmp_path / "map.tsv"
    pd.DataFrame({"label_id": [1, 2], "segmentation_id": segment_ids}).to_csv(
        mapping, sep="\t", index=False
    )
    result = export_bilateral(
        {
            **spatial,
            "analysis": str(source),
            "make_mobie": True,
            "export_pair_masks": True,
            "id_mapping": str(mapping),
            "query_ids": [1, 2],
            "max_pairs": 2,
            "include_artifacts": False,
        },
        tmp_path / "export",
    )
    report = json.loads(result.read_text())
    assert report["pairs_missing_mask_endpoint"] == 0
    dataset = result.parent / "mobie/project/morphofeatures"
    meta = json.loads((dataset / "dataset.json").read_text())
    display = meta["views"]["mf0-pair-1"]["sourceDisplays"][-1]["segmentationDisplay"]
    assert display["selectedSegmentIds"] == [f"objects;0;{v}" for v in segment_ids]
    assert display["showSelectedSegmentsIn3d"]
    assert (
        meta["views"]["mf0-bilateral-rank"]["sourceDisplays"][-1]["segmentationDisplay"]["lut"]
        == "viridis"
    )
    import h5py

    with h5py.File(result.parent / "pair_masks.h5") as store:
        masks = store["roles"][:]
        assert masks.shape[0] == 2
        assert (masks[0] == 1).any() and (masks[0] == 2).any()
        np.testing.assert_array_equal(masks[0] == 1, masks[1] == 2)
        np.testing.assert_array_equal(masks[0] == 2, masks[1] == 1)
        assert store.attrs["axes"] == "pzyx"
        np.testing.assert_allclose(store.attrs["spacing_zyx"], spatial["spacing_zyx"])
    table = pd.read_csv(dataset / "tables/objects/mf0.tsv", sep="\t", dtype=str)
    assert table.loc[
        table.label_id == str(segment_ids[0]), "mf0_best_partner_segmentation_id"
    ].iloc[0] == str(segment_ids[1])
    budget_output = tmp_path / "over-budget"
    with pytest.raises(FieldError, match="total mask budget before exporting"):
        export_bilateral(
            {
                **spatial,
                "analysis": str(source),
                "make_mobie": True,
                "export_pair_masks": True,
                "id_mapping": str(mapping),
                "query_ids": [1, 2],
                "max_mask_voxels": 1,
            },
            budget_output,
        )
    assert not budget_output.exists()
    with pytest.raises(FieldError, match="cannot calibrate a mask volume"):
        export_bilateral(
            {
                **spatial,
                "analysis": str(source),
                "make_mobie": False,
                "export_pair_masks": True,
                "unit": "voxel",
                "positions": positions(tmp_path, [1, 2, 3, 4]),
            },
            tmp_path / "uncalibrated",
        )


def test_export_worker_handoff_and_comparison_relative_paths(tmp_path):
    old, embedding, candidates = fixture_report(tmp_path)
    spec = {
        "stages": [
            {
                "action": "bilateral",
                "embeddings": {"New": "input.npz"},
                "candidates": "candidates.tsv",
                "compare_saved_runs": True,
                "comparison_reports": {"Old": "original/bilateral.json"},
                "same_specimen": True,
            },
            {
                "action": "export_bilateral",
                "from_bilateral": True,
                "make_mobie": False,
                "positions": positions(tmp_path, [1, 2, 3, 4]),
                "positions_unit": "nm",
                "max_pairs": 2,
            },
        ]
    }
    resolved = resolve_job(spec, tmp_path)
    assert resolved["stages"][0]["comparison_reports"]["Old"] == str(old)
    config = tmp_path / "job.yaml"
    config.write_text(yaml.safe_dump(spec))
    run_job(config)
    status = json.loads((tmp_path / "results/status.json").read_text())
    assert status["state"] == "completed"
    assert any(p.endswith("bilateral_export.json") for p in status["artifacts"])


def test_saved_run_picker_and_pair_export_results(tmp_path):
    from streamlit.testing.v1 import AppTest
    from test_guided_workspace import _app, _button, _widget

    from morphofeatures.registry import JobRecord, JobRegistry
    from morphofeatures.workspace_state import workflow_document

    old, _, _ = fixture_report(tmp_path)
    root = tmp_path / "workspace"
    registry = JobRegistry.under_output_root(root)
    registry.create(
        JobRecord(
            run_id="previous-custom-run",
            workflow="workspace_run",
            stage="workspace",
            command=(),
            submission_key="saved-picker",
            working_directory=str(root),
            config_snapshot="",
            stdout_path="",
            stderr_path="",
            metrics_path="",
            application_state="completed",
            artifacts=(str(old),),
            provenance={"display_name": "Earlier custom DINO"},
        )
    )
    app = _app(root)
    _widget(app, "selectbox", "Starting point").set_value("bilateral").run()
    _button(app, "Create workflow").click().run()
    _widget(app, "checkbox", "Compare with previous bilateral runs").check().run()
    _widget(app, "multiselect", "Completed bilateral runs in this workspace").set_value(
        [str(old)]
    ).run()
    _button(app, "Add selected runs to this comparison").click().run()
    assert not app.exception
    draft = next(
        v for k, v in app.session_state.filtered_state.items() if k.startswith("workflow:")
    )
    assert list(workflow_document(draft)["stages"][0]["comparison_reports"].values()) == [str(old)]
    report = export_bilateral(
        {
            "analysis": str(old),
            "make_mobie": False,
            "positions": positions(tmp_path, [1, 2, 3, 4]),
            "max_pairs": 2,
        },
        tmp_path / "export",
    )

    def results(path):
        import json
        from pathlib import Path

        from morphofeatures.spatial_probe_ui import bilateral_export_results

        bilateral_export_results(None, json.loads(Path(path).read_text()), Path(path))

    app = AppTest.from_function(results, args=(str(report),), default_timeout=30).run()
    assert not app.exception
    assert any("pairs.npz" in widget.label for widget in app.get("download_button"))
