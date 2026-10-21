"""Scientific contracts for the paired morphology/expression exploration."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
from threadpoolctl import threadpool_limits

from morphofeatures.analysis.shared_space.data import (
    exact_ids,
    load_data,
    make_splits,
    map_expression,
)
from morphofeatures.analysis.shared_space.evaluation import column_correlations, retrieval_metrics
from morphofeatures.analysis.shared_space.experiment import read_config, run
from morphofeatures.analysis.shared_space.models import (
    ExpressionFilter,
    LinearAlignment,
    MOFARidge,
    MorphologyPreprocessor,
    PairedNeural,
)


@pytest.fixture(autouse=True)
def bounded_threads():
    with threadpool_limits(2):
        yield


def test_identity_mapping_is_order_independent_and_reports_missing():
    expression = pd.DataFrame([[1, 0], [0, 1], [1, 1]], index=[91, 53, 77])
    mapping = pd.DataFrame({"nucleus_id": [53, 91, 999]}, index=[7, 8, 9])
    actual = map_expression(expression, mapping)
    reordered = map_expression(expression.iloc[::-1], mapping.iloc[::-1])
    pd.testing.assert_frame_equal(actual.sort_index(), reordered.sort_index())
    assert actual.index.tolist() == [7, 8]
    assert actual.loc[7].tolist() == [0, 1]
    with pytest.raises(ValueError, match="unique"):
        map_expression(expression, pd.DataFrame({"nucleus_id": [53, 53]}, index=[1, 2]))


@pytest.mark.parametrize("value", [1.1, "nan", -1, float(2**53)])
def test_unsafe_ids_rejected(value):
    with pytest.raises(ValueError):
        exact_ids([value])


def test_preprocessing_learns_only_training_features():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(60, 9))
    x[:, -1] = 5
    prep = MorphologyPreprocessor(5).fit(x)
    query = x[:3].copy()
    before = prep.transform(query)
    query[:, -1] = 100000
    np.testing.assert_allclose(before, prep.transform(query))
    y = rng.binomial(1, 0.3, size=(60, 7))
    y[:, -1] = 0
    filt = ExpressionFilter().fit(y)
    query = y[:3].copy()
    query[:, -1] = 1
    np.testing.assert_array_equal(filt.transform(y[:3]), filt.transform(query))
    with pytest.raises(ValueError, match="No variable"):
        ExpressionFilter().fit(np.ones((20, 5)))


@pytest.mark.parametrize("kind", ["rcca", "pls"])
def test_linear_alignment_recovers_heldout_shared_signal(kind, tmp_path):
    rng = np.random.default_rng(2)
    z = rng.normal(size=(300, 3))
    x = z @ rng.normal(size=(3, 15)) + 0.1 * rng.normal(size=(300, 15))
    y = z @ rng.normal(size=(3, 20)) + 0.1 * rng.normal(size=(300, 20))
    model = LinearAlignment(kind, factors=3).fit(x[:200], y[:200])
    xm, ye = model.transform_morphology(x[200:]), model.transform_expression(y[200:])
    assert np.mean(column_correlations(xm, ye)) > 0.85
    assert retrieval_metrics(xm, ye)["morph_to_expr_recall_at_k"] > 0.85
    joblib.dump(model, tmp_path / "model.joblib")
    restored = joblib.load(tmp_path / "model.joblib")
    np.testing.assert_allclose(xm, restored.transform_morphology(x[200:]))
    frozen = model.x_mean_.copy()
    model.transform_expression(y[200:] * 100)
    np.testing.assert_array_equal(model.x_mean_, frozen)


def test_retrieval_ties_have_chance_expectation():
    all_identical = np.ones((100, 3))
    scores = retrieval_metrics(all_identical, all_identical, top_k=10)
    assert scores["morph_to_expr_recall_at_k"] == pytest.approx(0.1)
    assert scores["morph_to_expr_median_rank"] == pytest.approx(50.5)
    exact = retrieval_metrics(np.eye(20), np.eye(20), top_k=1)
    assert exact["morph_to_expr_recall_at_k"] == 1


@pytest.mark.optional
def test_mofa_native_training_and_independent_ridge_projection(tmp_path):
    pytest.importorskip("mofapy2")
    rng = np.random.default_rng(3)
    z = rng.normal(size=(240, 3))
    x = z @ rng.normal(size=(3, 20)) + 0.2 * rng.normal(size=(240, 20))
    y = rng.binomial(1, expit(z @ rng.normal(size=(3, 30)) - 1)).astype(float)
    model = MOFARidge(factors=3, iterations=40).fit(x[:180], y[:180], output=tmp_path)
    assert (tmp_path / "native_mofa.hdf5").exists()
    xm = model.transform_morphology(x[180:])
    ye = model.transform_expression(y[180:])
    assert xm.shape == ye.shape == (60, 3)
    assert np.isfinite(xm).all() and np.isfinite(ye).all()
    assert model.training_factors_.shape[0] == 180
    assert retrieval_metrics(xm, ye)["morph_to_expr_recall_at_k"] > 10 / 60
    before = model.x_head_.coef_.copy()
    model.transform_expression(1 - y[180:])
    np.testing.assert_array_equal(before, model.x_head_.coef_)


@pytest.mark.optional
def test_neural_cpu_outputs_are_independent_and_reloadable(tmp_path):
    pytest.importorskip("torch")
    rng = np.random.default_rng(4)
    x, y = rng.normal(size=(80, 15)), rng.binomial(1, 0.3, size=(80, 20)).astype(float)
    model = PairedNeural(factors=4, epochs=3).fit(x[:60], y[:60])
    before = model.transform_morphology(x[60:])
    assert before.shape == (20, 4)
    assert np.isfinite(model.transform_expression(y[60:])).all()
    model.transform_expression(1 - y[60:])
    np.testing.assert_array_equal(before, model.transform_morphology(x[60:]))
    joblib.dump(model, tmp_path / "neural.joblib")
    np.testing.assert_allclose(
        before, joblib.load(tmp_path / "neural.joblib").transform_morphology(x[60:])
    )


def fixture_config(tmp_path):
    rng = np.random.default_rng(5)
    n = 160
    cells, nuclei = np.arange(100, 100 + n), np.arange(2000, 2000 + n)
    z = rng.normal(size=(n, 3))
    x = z @ rng.normal(size=(3, 20)) + 0.2 * rng.normal(size=(n, 20))
    y = rng.binomial(1, expit(z @ rng.normal(size=(3, 20)))).astype(float)
    y[:, 0] = 1  # Keep every profile nonempty.
    np.savez(tmp_path / "morph.npz", label_ids=cells[::-1], features=x[::-1])
    pd.DataFrame(
        y, index=pd.Index(nuclei, name="label_id"), columns=[f"gene_{i}" for i in range(20)]
    ).to_csv(tmp_path / "expression.tsv", sep="\t")
    pd.DataFrame({"label_id": cells, "nucleus_id": nuclei}).to_csv(
        tmp_path / "mapping.tsv", sep="\t", index=False
    )
    pd.DataFrame({"label_id": cells, "pair_index": np.arange(n) // 2 + 1}).to_csv(
        tmp_path / "bilateral.tsv", sep="\t", index=False
    )
    pd.DataFrame({"label_id": cells, "cell_type": np.where(z[:, 0] > 0, "a", "b")}).to_csv(
        tmp_path / "manual.tsv", sep="\t", index=False
    )
    for name, ids in [("cells", cells), ("nuclei", nuclei)]:
        frame = pd.DataFrame({"label_id": ids, "n_pixels": np.exp(z[:, 0] + 8)})
        for j, axis in enumerate("xyz"):
            frame[f"anchor_{axis}"] = z[:, j]
            frame[f"bb_min_{axis}"] = z[:, j] - 1
            frame[f"bb_max_{axis}"] = z[:, j] + 1
        frame.to_csv(tmp_path / f"{name}.tsv", sep="\t", index=False)
    config = read_config(Path(__file__).parents[1] / "configs/shared_space.example.yaml")
    config["inputs"] = {
        "expression": str(tmp_path / "expression.tsv"),
        "mapping": str(tmp_path / "mapping.tsv"),
        "morphology": {"fixture": str(tmp_path / "morph.npz")},
        "bilateral": str(tmp_path / "bilateral.tsv"),
        "nucleus_table": str(tmp_path / "nuclei.tsv"),
        "cell_table": str(tmp_path / "cells.tsv"),
        "manual_labels": str(tmp_path / "manual.tsv"),
    }
    config["analysis"].update(
        factors=3,
        morphology_pcs=10,
        methods=["rcca", "pls"],
        run_umap=False,
        spatial_regions=4,
        retrieval_cells=20,
        plot_cells=20,
        min_test_class=2,
    )
    config["output_root"] = str(tmp_path / "results")
    return config


def test_split_pair_purge_and_end_to_end_outputs(tmp_path):
    config = fixture_config(tmp_path)
    data = load_data(config)
    splits, regions = make_splits(data, config["analysis"])
    for scheme, (tr, te, purged) in splits.items():
        assert not set(data.pairs[tr]) & set(data.pairs[te])
        assert len(tr) + len(te) + len(purged) == len(data.cells)
        if scheme == "spatial":
            assert not set(regions[tr]) & set(regions[te])
    config["analysis"]["schemes"] = ["random_pair_grouped"]
    out = run(config)
    frame = pd.read_csv(out / "tables/scores.tsv", sep="\t")
    assert set(frame.method) == {
        "training_prevalence",
        "position_RBF",
        "morphology_direct",
        "rcca",
        "pls",
    }
    assert np.isfinite(frame.macro_AP).all()
    fit = out / "models/fixture/random_pair_grouped/rcca"
    with np.load(fit / "heldout_embeddings.npz") as arrays:
        np.testing.assert_array_equal(arrays["label_ids"] + 1900, arrays["nucleus_ids"])
    assert (out / "figures/fixture_heldout_performance.svg").exists()
