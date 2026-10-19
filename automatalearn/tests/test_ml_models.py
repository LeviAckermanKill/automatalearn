"""Unit tests for ML classifiers (Decision Tree, Random Forest, Logistic Regression)."""

import pytest
from automatalearn.core.languages import get_language
from automatalearn.data.generator import DataGenerator
from automatalearn.ml.features import StringFeatureExtractor
from automatalearn.ml.models import train_and_evaluate_ml_model


@pytest.mark.parametrize("model_type", ["decision_tree", "random_forest", "logistic_regression"])
def test_ml_models_training_and_evaluation(model_type: str):
    """Verify ML models train and output valid metrics on generated data."""
    lang = get_language("ending_ab")
    gen = DataGenerator(alphabet=lang.alphabet, oracle=lang.predicate, dfa=lang.dfa, seed=42)
    splits = gen.generate_dataset(min_length=0, max_length=5, num_samples=80, gen_samples=30)

    extractor = StringFeatureExtractor(alphabet=lang.alphabet)
    X_train = extractor.fit_transform(splits.train_strings)
    y_train = splits.train_df["label"].to_numpy()
    X_test = extractor.transform(splits.test_strings)
    y_test = splits.test_df["label"].to_numpy()
    X_gen = extractor.transform(splits.gen_strings)
    y_gen = splits.generalization_df["label"].to_numpy()

    res = train_and_evaluate_ml_model(
        model_type=model_type,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        X_gen=X_gen,
        y_gen=y_gen,
        feature_names=extractor.get_feature_names(),
        run_cv=True,
        seed=42,
    )

    assert 0.0 <= res.in_dist_accuracy <= 1.0
    assert 0.0 <= res.gen_accuracy <= 1.0
    assert res.train_time_ms > 0
    assert res.inference_time_us_per_sample >= 0
    assert "primary_metric" in res.model_complexity
    assert len(res.feature_importances) > 0


def test_ml_model_end_to_end_inference_and_extractor():
    """Verify end-to-end inference measurement and extractor attachment."""
    lang = get_language("ending_ab")
    gen = DataGenerator(alphabet=lang.alphabet, oracle=lang.predicate, dfa=lang.dfa, seed=42)
    splits = gen.generate_dataset(min_length=0, max_length=5, num_samples=60, gen_samples=20)

    extractor = StringFeatureExtractor(alphabet=lang.alphabet)
    X_train = extractor.fit_transform(splits.train_strings)
    y_train = splits.train_df["label"].to_numpy()
    X_test = extractor.transform(splits.test_strings)
    y_test = splits.test_df["label"].to_numpy()
    X_gen = extractor.transform(splits.gen_strings)
    y_gen = splits.generalization_df["label"].to_numpy()

    res = train_and_evaluate_ml_model(
        model_type="decision_tree",
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        X_gen=X_gen,
        y_gen=y_gen,
        feature_names=extractor.get_feature_names(),
        seed=42,
        raw_test_strings=splits.test_strings,
        feature_extractor=extractor,
    )

    assert res.feature_extractor is extractor
    assert res.extractor is extractor
    assert res.inference_time_us_per_sample > 0
    # Test that model's extractor can transform a single string and predict
    X_sample = res.extractor.transform(["ab"])
    pred = res.raw_model.predict(X_sample)
    assert pred[0] in [0, 1]

