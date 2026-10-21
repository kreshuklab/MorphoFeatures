"""Scientific contracts for the full binary-expression benchmark."""

import joblib
import numpy as np
import pytest

from morphofeatures.analysis.expression_prediction import (
    BinaryExpressionPredictor,
    embedding_preprocessor,
    profile_scores,
)


@pytest.mark.parametrize("kind", ["logistic", "random_forest", "extra_trees", "hist_gradient_boosting"])
def test_full_profile_constant_targets_column_order_and_reload(kind, tmp_path):
    rng = np.random.default_rng(42)
    x = rng.normal(size=(160, 5))
    y = np.column_stack([np.zeros(160), x[:, 0] > 0, np.ones(160), x[:, 1] > 0])
    model = BinaryExpressionPredictor(
        kind, trees=12, min_leaf=2, boosting_iterations=12, workers=1
    ).fit(x, y)
    p = model.predict_proba(x)
    assert p.shape == y.shape
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    np.testing.assert_array_equal(model.constant_targets_, [True, False, True, False])
    np.testing.assert_allclose(p[:, 0], 1 / 162)
    np.testing.assert_allclose(p[:, 2], 161 / 162)
    assert ((p[:, [1, 3]] >= .5) == y[:, [1, 3]]).mean() > .85
    model_file = tmp_path / "profile.joblib"
    joblib.dump(model, model_file)
    np.testing.assert_allclose(joblib.load(model_file).predict_proba(x), p)
    # Changing the requested target order must also change the exported order.
    swapped = BinaryExpressionPredictor(
        kind, trees=12, min_leaf=2, boosting_iterations=12, workers=1
    ).fit(x, y[:, [3, 2, 1, 0]])
    np.testing.assert_array_equal(swapped.constant_targets_, [False, True, False, True])


@pytest.mark.parametrize("kind", ["random_forest", "extra_trees"])
def test_native_forest_single_gene(kind):
    x = np.arange(80).reshape(-1, 1)
    model = BinaryExpressionPredictor(kind, trees=8, min_leaf=2, workers=1)
    model.fit(x, (x > 40).astype(int))
    assert model.predict_proba(x).shape == (80, 1)


def test_preprocessing_uses_training_cells_only():
    rng = np.random.default_rng(7)
    train = rng.normal(size=(100, 12))
    prep = embedding_preprocessor(5).fit(train)
    training_mean = prep.steps[0][1].mean_.copy()
    components = prep.steps[-1][1].components_.copy()
    prep.transform(rng.normal(10000, 100, size=(20, 12)))
    np.testing.assert_allclose(training_mean, train.mean(axis=0))
    np.testing.assert_array_equal(components, prep.steps[-1][1].components_)
    np.testing.assert_array_equal(training_mean, prep.steps[0][1].mean_)


def test_metrics_keep_full_panel_and_exclude_undefined_discrimination():
    y = np.array([[0, 0, 1], [0, 1, 1], [0, 0, 0], [0, 1, 0]])
    p = .1 + .8 * y
    summary, genes = profile_scores(y, p, ["zero", "a", "b"], minimum_class_size=2)
    assert summary["genes"] == 3 and summary["variable_genes"] == 2
    assert summary["macro_AP"] == pytest.approx(1)
    assert summary["hamming_loss"] == 0
    assert summary["exact_profile_accuracy"] == 1
    assert summary["brier"] == pytest.approx(.01)
    assert np.isnan(genes.loc[0, "average_precision"])
    assert genes.gene_id.tolist() == ["zero", "a", "b"]
    with pytest.raises(ValueError, match="finite"):
        profile_scores(y, np.full(y.shape, np.nan), ["zero", "a", "b"])


def test_rejects_nonbinary_targets_and_incorrect_feature_width():
    model = BinaryExpressionPredictor("logistic", workers=1)
    with pytest.raises(ValueError, match="binary"):
        model.fit(np.ones((5, 2)), np.full((5, 2), .5))
    model.fit(np.arange(10).reshape(5, 2), np.array([[0], [1], [0], [1], [0]]))
    with pytest.raises(ValueError, match="Expected 2"):
        model.predict_proba(np.ones((3, 4)))
