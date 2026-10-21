"""Explicit cohorts must filter before reads, preserve IDs and survive workflow handoffs."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from morphofeatures.data.object_selection import resolve_selection, select_objects, write_selection
from morphofeatures.workflow_validation import FieldError


def test_table_filter_large_ids_mapping_and_missing_audit(tmp_path):
    large = 2**60 + 7
    table = tmp_path / "labels.csv"
    table.write_text(f"cell,type\n{large},neuron\n{large},neuron\n20,muscle\n21,\n22,neuron\n")
    mapping = tmp_path / "mapping.tsv"
    mapping.write_text(f"cell\tnucleus\n{large}\t{large + 1}\n20\t200\n")
    options = dict(
        mode="table",
        ids_file=str(table),
        id_column="cell",
        filter_column="type",
        filter_values=["neuron"],
        mapping_file=str(mapping),
        mapping_from="cell",
        mapping_to="nucleus",
        missing="skip",
    )
    selected, report = select_objects(pd.DataFrame({"label_id": [200, large + 1]}), options)
    assert selected.label_id.tolist() == [large + 1]
    assert report["duplicate_ids_removed"] == 1
    assert report["missing"] == 1 and report["requested"] == 2
    write_selection(tmp_path / "out", report)
    rows = pd.read_csv(tmp_path / "out/selection.tsv", sep="\t", dtype=str).fillna("")
    row = rows[rows.requested_id == str(large)].iloc[0]
    assert row.target_id == str(large + 1) and row.label_id == str(large + 1)
    assert "missing mapping" in rows.status.values
    options["missing"] = "error"
    with pytest.raises(FieldError, match="1 of 2 requested IDs") as error:
        select_objects(
            pd.DataFrame({"label_id": [200, large + 1]}), options, destination=tmp_path / "failed"
        )
    assert ("object_selection", "missing") in error.value.fields
    assert (tmp_path / "failed/selection.tsv").exists()


@pytest.mark.parametrize("ids", [[], [1.5], [float(2**60)], [True], ["abc"]])
def test_invalid_or_empty_targets_never_mean_all(ids):
    with pytest.raises(FieldError):
        resolve_selection({"mode": "ids", "ids": ids})


def test_filters_namespace_and_no_match(tmp_path):
    table = tmp_path / "labels.tsv"
    table.write_text("label_id\tclass\n11\tA\n12\t\n13\tB\n")
    options = dict(mode="table", ids_file=str(table), filter_column="class")
    targets, _ = resolve_selection(options)
    assert list(targets) == [11, 13]
    with pytest.raises(FieldError, match="No IDs remain"):
        resolve_selection({**options, "filter_values": ["not a class"]})
    available = pd.DataFrame({"label_id": [101, 103], "segmentation_id": [11, 13]})
    with pytest.raises(FieldError, match="unavailable"):
        select_objects(available, options)
    assert select_objects(available, {**options, "id_space": "segmentation"})[
        0
    ].label_id.tolist() == [101, 103]
    with pytest.raises(FieldError, match="matched no input"):
        select_objects(available, {**options, "missing": "skip"})
    assert select_objects(available, {"mode": "all"})[0] is available


def test_ambiguous_mapping_is_rejected(tmp_path):
    mapping = tmp_path / "mapping.tsv"
    mapping.write_text("label_id\tnucleus_id\n1\t11\n2\t11\n")
    with pytest.raises(FieldError, match="one-to-one") as error:
        resolve_selection(dict(mode="ids", ids=[1], mapping_file=str(mapping)))
    assert ("object_selection", "mapping_file") in error.value.fields


def test_switching_model_cannot_silently_drop_target_selection(tmp_path):
    from morphofeatures.representations import extract_representation

    with pytest.raises(FieldError, match="requires DINO"):
        extract_representation(
            {"model": "mae", "object_selection": {"mode": "ids", "ids": [1]}}, tmp_path
        )
    assert not list(tmp_path.iterdir())


class Backbone(torch.nn.Module):
    def forward_features(self, images):
        tokens = torch.nn.functional.avg_pool2d(images, 14, 14).flatten(2).transpose(1, 2)
        return {"x_norm_clstoken": tokens.mean(1), "x_norm_patchtokens": tokens}


def volume(tmp_path):
    raw = np.random.default_rng(7).uniform(1, 100, (20, 20, 20)).astype(np.float32)
    seg = np.zeros(raw.shape, dtype=np.int64)
    for i, center in enumerate(np.ndindex(2, 2, 2)):
        start = np.array(center) * 9 + 2
        seg[tuple(slice(a, a + 5) for a in start)] = i + 1
    np.save(tmp_path / "raw.npy", raw)
    np.save(tmp_path / "seg.npy", seg)
    return dict(
        raw=str(tmp_path / "raw.npy"),
        segmentation=str(tmp_path / "seg.npy"),
        unit="um",
        spacing_zyx=[0.1] * 3,
    )


def test_physical_targets_before_cap_and_identity_unchanged(tmp_path):
    from morphofeatures.data.dino_views import index_source, source_identity, validate_source

    source = validate_source(volume(tmp_path))
    targeted = {**source, "object_selection": {"mode": "ids", "ids": [3, 7, 8]}, "max_objects": 2}
    for run in ("one", "two"):
        objects = index_source(targeted, tmp_path / run)
        assert set(objects.label_id) <= {3, 7, 8} and len(objects) == 2
    assert (tmp_path / "one/objects.tsv").read_bytes() == (
        tmp_path / "two/objects.tsv"
    ).read_bytes()
    audit = json.loads((tmp_path / "one/selection.json").read_text())
    assert audit["matched"] == 3 and audit["selected"] == 2
    assert source_identity(source) == source_identity(targeted)


@pytest.mark.parametrize("layout", ["single_crop", "distributed_patches"])
def test_preparation_filters_and_saved_groups_can_be_retargeted(tmp_path, layout):
    from morphofeatures.configuration_editor import load_document
    from morphofeatures.data.preprocessing import preprocess
    from morphofeatures.dino import extract_dino

    config = load_document(
        preprocess(
            {
                **volume(tmp_path),
                "layout": layout,
                "crop_shape": [8] * 3,
                "normalization": "percentile",
                "object_selection": {"mode": "ids", "ids": [3, 7]},
                "patches": {"tile_size_um": 0.3, "max_tiles": 2},
            },
            tmp_path / "prep",
        )
    )
    assert np.load(config["data"]["label_ids"]).tolist() == [3, 7]
    settings = dict(
        model="dinov2",
        config=config,
        object_selection={"mode": "ids", "ids": [7]},
        views=dict(size=28, axes=[0], fractions=[0.5], normalization="unit"),
    )
    audit = {}
    ids, vectors, excluded = extract_dino(settings, backbone=Backbone(), selection_report=audit)
    assert ids.tolist() == [7] and vectors.shape == (1, 3) and not excluded
    assert audit["requested"] == 1
    assert (tmp_path / "prep/selection.tsv").exists()


def test_prepared_reads_skip_unselected_pixels_and_cache_tracks_table(tmp_path, monkeypatch):
    from morphofeatures import dino
    from morphofeatures.representations import extract_representation, extraction_identity

    large = 2**60 + 3
    crops = np.random.default_rng(3).uniform(1, 10, (3, 8, 8, 8)).astype(np.float32)
    crops[0] = np.nan  # Unselected invalid pixels must not produce an exclusion.
    np.save(tmp_path / "crops.npy", crops)
    np.save(tmp_path / "ids.npy", [1, large, large + 1])
    checkpoint = tmp_path / "weights.pth"
    checkpoint.write_bytes(b"test identity")
    table = tmp_path / "targets.tsv"
    table.write_text(f"label_id\n{large}\n")
    settings = dict(
        model="dinov2",
        checkpoint=str(checkpoint),
        cache=str(tmp_path / "cache"),
        object_selection=dict(mode="table", ids_file=str(table)),
        config=dict(
            device="cpu",
            data=dict(crops=str(tmp_path / "crops.npy"), label_ids=str(tmp_path / "ids.npy")),
        ),
        views=dict(size=28, axes=[0], fractions=[0.5]),
    )
    monkeypatch.setattr(dino, "load_backbone", lambda settings: Backbone())
    output = extract_representation(settings, tmp_path / "run")
    assert np.load(output)["label_ids"].tolist() == [large]
    metadata = json.loads(output.with_suffix(".metadata.json").read_text())
    assert metadata["excluded_objects"] == []
    assert extract_representation(settings, tmp_path / "cached") == output
    assert (tmp_path / "cached/selection.tsv").read_bytes() == (
        tmp_path / "run/selection.tsv"
    ).read_bytes()
    before = extraction_identity(settings)[0]
    table.write_text(f"label_id\n{large + 1}\n")
    assert extraction_identity(settings)[0] != before
    table.write_text("label_id\n999\n")
    with pytest.raises(FieldError, match="unavailable"):
        extract_representation(settings, tmp_path / "missing")
    assert (tmp_path / "missing/selection.tsv").exists()


def test_relative_paths_in_workflow_and_loaded_config(tmp_path):
    from morphofeatures.configuration_editor import load_document
    from morphofeatures.workspace_jobs import resolve_job

    document = {
        "stages": [
            {
                "action": "preprocess",
                "object_selection": {
                    "mode": "table",
                    "ids_file": "targets.tsv",
                    "mapping_file": "mapping.tsv",
                },
            }
        ]
    }
    resolved = resolve_job(document, tmp_path, validate=False)
    options = resolved["stages"][0]["object_selection"]
    assert options["ids_file"] == str(tmp_path / "targets.tsv")
    assert options["mapping_file"] == str(tmp_path / "mapping.tsv")
    config = tmp_path / "source.yaml"
    config.write_text(
        yaml.safe_dump(
            {"data": {"volume": {"object_selection": document["stages"][0]["object_selection"]}}}
        )
    )
    assert load_document(config)["data"]["volume"]["object_selection"] == options


def test_targeted_experiment_reuses_one_cohort_for_every_representation(tmp_path):
    from morphofeatures.dino_experiments import experiment_defaults, run_experiment

    checkpoint = tmp_path / "weights.pth"
    checkpoint.write_bytes(b"test identity")
    options = experiment_defaults()
    options.update(
        datasets=[
            {
                "name": "test",
                "source": {
                    **volume(tmp_path),
                    "object_selection": {"mode": "ids", "ids": [2, 4, 6, 8]},
                },
            }
        ],
        backbones=[dict(model="dinov2", variant="dinov2_vits14", checkpoint=str(checkpoint))],
        max_objects=0,
        classify=False,
        umap=False,
        clusters=2,
        device="cpu",
        tile_sizes_um=[0.3],
    )
    options["views"].update(
        size=28, axes=[0], fine_fractions=[0.5], coarse_fractions=[0.5], max_tiles=2
    )
    report_path = run_experiment(options, tmp_path / "experiment", backbones=[Backbone()])
    report = json.loads(report_path.read_text())
    assert report["object_selections"][0]["dataset"] == "test"
    assert Path(report["object_selections"][0]["table"]).exists()
    for entry in report["embeddings"]:
        assert np.load(entry["path"])["label_ids"].tolist() == [2, 4, 6, 8]


@pytest.mark.parametrize("workflow", ["preprocess", "dino", "dino_experiment"])
def test_ui_target_controls_roundtrip_and_ground_truth_shortcut(tmp_path, workflow):
    from test_guided_workspace import _app, _button, _widget

    app = _app(tmp_path)
    _widget(app, "selectbox", "Starting point").set_value(workflow).run()
    _button(app, "Create workflow").click().run()
    _button(app, "Use bundled Platynereis ground-truth IDs").click().run()
    assert not app.exception
    _button(app, "Check target table / IDs").click().run()
    assert not app.error and not app.exception
    assert any("unique target IDs" in entry.value for entry in app.success)
    _widget(app, "selectbox", "Objects to process").set_value("ids").run()
    _widget(app, "text_input", "Target object IDs").set_value("[3, 7]").run()
    if workflow == "dino_experiment":
        _button(app, "Use every targeted object (set object cap to 0)").click().run()
    _widget(app, "checkbox", "Edit pipeline YAML").check().run()
    stage = yaml.safe_load(_widget(app, "text_area", "Pipeline YAML").value)["stages"][0]
    selection = (
        stage["datasets"][0]["source"]["object_selection"]
        if workflow == "dino_experiment"
        else stage["object_selection"]
    )
    assert selection["mode"] == "ids" and selection["ids"] == [3, 7]
    if workflow == "dino_experiment":
        assert stage["max_objects"] == 0
    if workflow == "dino":
        _widget(app, "selectbox", "Embedding model").set_value("mae").run()
        assert any("DINO target selection" in entry.value for entry in app.warning)
        assert _widget(app, "selectbox", "Objects to process").value == "ids"
        _widget(app, "selectbox", "Objects to process").set_value("all").run()
        assert not any("DINO target selection" in entry.value for entry in app.warning)
    assert not app.exception
