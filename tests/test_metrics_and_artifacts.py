import numpy as np
import pytest

from morphofeatures.artifacts import (
    inspect_embedding,
    read_run_metadata,
    tail_text,
    write_json_atomic,
)
from morphofeatures.data.io import export_embeddings
from morphofeatures.metrics import MetricWriter, metric_series, read_metric_events


def test_metrics_jsonl_ignores_partially_written_final_line(tmp_path):
    path = tmp_path / "metrics.jsonl"
    writer = MetricWriter(path)
    writer.write("epoch", epoch=1, train_loss=2.0, validation_loss=3.0)
    with path.open("ab") as stream:
        stream.write(b'{"timestamp":"unfinished"')
    events = read_metric_events(path)
    assert len(events) == 1
    assert metric_series(events)[0]["train_loss"] == 2.0


def test_invalid_complete_metrics_line_is_rejected(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text("not-json\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_metric_events(path)


def test_embedding_and_run_metadata_inspection(tmp_path):
    path = export_embeddings(
        tmp_path / "embedding.npy",
        [9, 3, 6],
        np.arange(12, dtype=np.float32).reshape(3, 4),
    )
    summary = inspect_embedding(path)
    assert (summary.rows, summary.features) == (3, 4)
    assert summary.finite and summary.unique_labels
    metadata_path = write_json_atomic(
        tmp_path / "run.json",
        {"run_id": "test", "artifacts": {"embedding": str(path)}},
    )
    assert read_run_metadata(metadata_path)["run_id"] == "test"
    log = tmp_path / "stdout.log"
    log.write_text("\n".join("line {}".format(index) for index in range(500)), encoding="utf-8")
    bounded = tail_text(log, max_bytes=256, max_lines=5)
    assert "line 499" in bounded
    assert len(bounded.splitlines()) <= 6
