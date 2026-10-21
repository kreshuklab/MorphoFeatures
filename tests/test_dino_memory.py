"""Allocation-aware planning and recovery must never change the scientific cohort."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from morphofeatures import dino_memory
from morphofeatures.dino_runtime import BatchEncoder, identity_settings
from morphofeatures.workflow_validation import FieldError


def object_box(side=500):
    source = {"segmentation_shape_zyx": [2000] * 3, "max_object_voxels": 500_000_000}
    objects = pd.DataFrame(
        [
            {
                "label_id": 10067,
                **{f"bbox_min_{a}": 10 for a in "zyx"},
                **{f"bbox_max_{a}": 10 + side for a in "zyx"},
            }
        ]
    )
    return source, objects


@pytest.mark.parametrize("prepare", [4096, 20000, 40000])
@pytest.mark.parametrize("mode", ["auto", "manual"])
def test_repairs_observed_100g_job_conflicts_and_keeps_inputs(prepare, mode):
    settings = {
        "runtime": {
            "memory_mode": mode,
            "allocation_memory_mb": 102400,
            "allocation_cpus": 20,
            "prepare_workers": 2,
            "prepare_memory_mb": prepare,
            "decoded_cache_mb": 10000,
            "prefetch_mb": 10000,
        }
    }
    original = deepcopy(settings)
    source, objects = object_box()
    plan = dino_memory.plan_memory(settings, source=source, objects=objects, limits={}, rss_mb=0)
    assert plan["budget_mb"] == 102400
    assert plan["estimated_total_mb"] <= 102400
    assert plan["largest_object_mb"] > 13000
    assert (
        plan["preparation_mb"]
        > plan["decoded_caches_mb"] + plan["worker_overhead_mb"] + plan["largest_object_mb"]
    )
    assert plan["adjustments"]
    assert settings == original
    assert source["max_object_voxels"] == 500_000_000


def test_worker_limits_tighten_allocation_and_local_budget_is_not_all_host_ram():
    plan = dino_memory.plan_memory(
        {
            "runtime": {
                "allocation_memory_mb": 102400,
                "job_memory_mb": 200000,
                "prepare_workers": 32,
            }
        },
        limits={"container": 16000},
        rss_mb=500,
    )
    assert plan["budget_mb"] == 16000
    assert plan["estimated_total_mb"] <= 16000
    assert plan["runtime"]["prepare_workers"] < 32
    assert dino_memory.plan_memory({}, limits={"host": 1000000})["budget_mb"] == 8192
    assert (
        dino_memory.plan_memory({"runtime": {"job_memory_mb": 12000}}, limits={"host": 1000000})[
            "budget_mb"
        ]
        == 12000
    )


def test_manual_strict_checks_both_cache_conflicts_and_full_job_sum():
    for runtime in ({"prepare_memory_mb": 1, "prepare_workers": 2}, {"prepare_memory_mb": 9000}):
        with pytest.raises(FieldError, match="RAM budget"):
            dino_memory.plan_memory(
                {"runtime": {"memory_mode": "manual", "auto_adjust_memory": False, **runtime}},
                limits={},
            )


def test_large_objects_cannot_bypass_budget_in_serial_or_parallel():
    source, objects = object_box()
    for workers in (0, 4):
        with pytest.raises(FieldError, match="Object 10067.*Even serial"):
            dino_memory.plan_memory(
                {"runtime": {"job_memory_mb": 8192, "prepare_workers": workers}},
                source=source,
                objects=objects,
                limits={},
            )
    source["max_object_voxels"] = 100  # Existing explicit exclusion policy is retained.
    plan = dino_memory.plan_memory({}, source=source, objects=objects, limits={})
    assert plan["largest_object_mb"] == 0


def test_allocation_units_and_detected_parent_cgroups(tmp_path, monkeypatch):
    assert dino_memory.memory_mib("100G") == 102400
    assert dino_memory.memory_mib("100Gi") == 102400
    assert dino_memory.memory_mib("1TiB") == 1024**2
    assert dino_memory.memory_mib("1000") == 1000
    proc, cg = tmp_path / "proc", tmp_path / "cgroup"
    (proc / "self").mkdir(parents=True)
    (cg / "job/step").mkdir(parents=True)
    (proc / "meminfo").write_text("MemAvailable: 102400000 kB\n")
    (proc / "self/cgroup").write_text("0::/job/step\n")
    for directory, maximum, current in [
        (cg, "max", 0),
        (cg / "job", 6000, 5000),
        (cg / "job/step", 8000, 2000),
    ]:
        (directory / "memory.max").write_text(
            str(maximum * 1024**2) if isinstance(maximum, int) else maximum
        )
        (directory / "memory.current").write_text(str(current * 1024**2))
    monkeypatch.setattr(dino_memory, "resident_mib", lambda: 256)
    limits = dino_memory.system_limits(
        environ={"SLURM_MEM_PER_CPU": "2000", "SLURM_CPUS_PER_TASK": "4"}, proc=proc, cgroup=cg
    )
    assert limits["SLURM_MEM_PER_CPU"] == 8000
    assert min(limits.values()) == 1256  # Parent can be more restrictive than leaf.
    (cg / "job/memory.stat").write_text(f"inactive_file {2000 * 1024**2}\n")
    refreshed = dino_memory.system_limits(environ={}, proc=proc, cgroup=cg)
    assert min(refreshed.values()) == 3256  # Reclaimable volume file cache is not an object heap.


def test_serial_admission_refuses_before_opening_pixels(monkeypatch):
    from morphofeatures.data import dino_stream
    from morphofeatures.dino_runtime import runtime_options

    monkeypatch.setattr(
        dino_stream, "PhysicalObjects", lambda *a, **k: pytest.fail("No volume should be opened")
    )
    source, objects = object_box()
    runtime = runtime_options({"runtime": {"prepare_workers": 0, "prepare_memory_mb": 1000}})
    with pytest.raises(FieldError, match="Object 10067"):
        next(dino_stream.prepared_events(source, objects, {"size": 224}, runtime, ["fixed_tiles"]))


def test_gpu_batch_retry_preserves_order_and_every_image(monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    from morphofeatures.metrics import MetricWriter, read_metric_events

    model = torch.nn.Linear(1, 1)
    writer = MetricWriter(tmp_path / "metrics.jsonl")
    encoder = BatchEncoder(model, torch.device("cpu"), {"views": {"batch_size": 8}}, writer=writer)
    calls = []

    def constrained(images, masks, features, stride):
        calls.append(len(images))
        if len(images) > 2:
            raise torch.cuda.OutOfMemoryError("simulated capacity")
        return {"cls": np.asarray(images)[:, None]}

    monkeypatch.setattr(encoder, "_encode", constrained)
    values = encoder.encode(list(range(11)), [True] * 11, ["cls"], 14)
    np.testing.assert_array_equal(values["cls"][:, 0], np.arange(11))
    assert encoder.batch_limit == 2
    assert calls[:3] == [8, 4, 2]
    assert len(read_metric_events(writer.path)) == 2
    encoder.options["auto_adjust_memory"] = False
    encoder.batch_limit = 8
    with pytest.raises(FieldError, match="recovery is disabled"):
        encoder.encode([0, 1, 2], [True] * 3, ["cls"], 14)
    encoder.options["auto_adjust_memory"] = True
    monkeypatch.setattr(
        encoder, "_encode", lambda *a: (_ for _ in ()).throw(torch.cuda.OutOfMemoryError())
    )
    with pytest.raises(FieldError, match="single image"):
        encoder.encode([0], [True], ["cls"], 14)


def test_memory_changes_preserve_extraction_identity():
    numerical = dict.fromkeys(
        ["runtime_version", "precision", "torch", "device", "matmul_precision", "xformers"], "same"
    )
    original = {"views": {"size": 224, "batch_size": 32}, "runtime": {"prepare_memory_mb": 1}}
    changed = {
        "views": {"size": 224, "batch_size": 2},
        "runtime": {
            "prepare_memory_mb": 70000,
            "allocation_memory_mb": 102400,
            "memory_mode": "auto",
        },
    }
    assert identity_settings(original, numerical) == identity_settings(changed, numerical)


def test_review_freezes_allocation_and_keeps_requested_settings(tmp_path):
    from morphofeatures.dino_experiments import experiment_defaults
    from morphofeatures.workspace_jobs import plan_workspace_job

    stage = experiment_defaults()
    raw = tmp_path / "raw.npy"
    seg = tmp_path / "seg.npy"
    np.save(raw, np.ones((8, 8, 8), np.uint8))
    np.save(seg, np.ones((8, 8, 8), np.uint32))
    stage["datasets"][0]["source"].update(raw=str(raw), segmentation=str(seg))
    weights = tmp_path / "weights.pth"
    weights.write_bytes(b"not loaded during review")
    (tmp_path / "hubconf.py").write_text("")
    stage["backbones"][0].update(checkpoint=str(weights), model_repository=str(tmp_path))
    stage["runtime"] = {"prepare_workers": 2, "prepare_memory_mb": 20000, "decoded_cache_mb": 10000}
    document = {"stages": [stage], "slurm": {"memory": "100G", "cpus": 20}}
    plan = plan_workspace_job(document, output_root=tmp_path, run_id="memory", execution="slurm")
    saved = yaml.safe_load(plan.resolved_yaml)["stages"][0]["runtime"]
    assert saved["allocation_memory_mb"] == 102400
    assert saved["prepare_memory_mb"] == 20000
    assert plan.memory_plans[0]["runtime"]["decoded_cache_mb"] == 128
    assert not (tmp_path / "experiments").exists()
    document["stages"][0]["runtime"].update(memory_mode="manual", auto_adjust_memory=False)
    with pytest.raises(ValueError, match="RAM budget"):
        plan_workspace_job(document, output_root=tmp_path, run_id="strict", execution="slurm")


def test_ui_exposes_manual_controls_and_reviews_allocation(tmp_path):
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(
        str(Path(__file__).parents[1] / "morphofeatures/dashboard.py"), default_timeout=30
    )
    app.session_state["workspace_root"] = str(tmp_path)
    app.run()

    def widget(kind, label):
        return next(w for w in getattr(app, kind) if w.label == label)

    widget("selectbox", "Starting point").set_value("dino_experiment").run()
    widget("button", "Create workflow").click().run()
    app.radio(key="step_widget").set_value("Analyze").run()
    assert widget("selectbox", "DINO memory planning").value == "auto"
    widget("number_input", "CPU preparation workers").set_value(2)
    widget("number_input", "Decoded chunk cache per volume / worker (MiB)").set_value(10000).run()
    app.radio(key="step_widget").set_value("Review & run").run()
    widget("selectbox", "Run on").set_value("slurm").run()
    widget("text_input", "Memory").set_value("100G").run()
    plans = [frame.value for frame in app.dataframe if "Job RAM (MiB)" in frame.value.columns]
    assert plans[0]["Job RAM (MiB)"].iloc[0] == 102400
    assert plans[0]["Caches within preparation (MiB)"].iloc[0] == 512
    app.radio(key="step_widget").set_value("Analyze").run()
    assert widget("number_input", "Decoded chunk cache per volume / worker (MiB)").value == 10000
    widget("selectbox", "DINO memory planning").set_value("manual").run()
    widget("checkbox", "Repair manual RAM conflicts and retry smaller GPU batches").uncheck().run()
    app.radio(key="step_widget").set_value("Review & run").run()
    assert any("DINO RAM budget" in error.value for error in app.error)
    assert not app.exception
