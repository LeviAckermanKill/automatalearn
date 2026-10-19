"""Machine learning model training, evaluation, and profiling.

Wraps Decision Tree, Random Forest, and Logistic Regression models with
standardized performance metrics, training and per-sample inference profiling,
model complexity tracking, and feature importance analysis.
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


@dataclass
class ModelEvaluationResult:
    """Comprehensive evaluation record for a trained classifier."""

    model_name: str
    model_type: str
    in_dist_accuracy: float
    in_dist_precision: float
    in_dist_recall: float
    in_dist_f1: float
    in_dist_confusion: np.ndarray
    gen_accuracy: float
    gen_precision: float
    gen_recall: float
    gen_f1: float
    gen_confusion: np.ndarray
    train_time_ms: float
    inference_time_us_per_sample: float
    model_complexity: Dict[str, Union[int, float]]
    feature_importances: Dict[str, float] = field(default_factory=dict)
    cv_accuracy_mean: Optional[float] = None
    cv_accuracy_std: Optional[float] = None
    raw_model: Optional[Any] = None
    feature_extractor: Optional[Any] = None

    @property
    def extractor(self) -> Optional[Any]:
        """Convenience alias for feature_extractor."""
        return self.feature_extractor

    @property
    def complexity_metric(self) -> str:
        """Convenience string description of model complexity."""
        return str(self.model_complexity.get("primary_metric", "-"))

    @property
    def cv_mean_accuracy(self) -> Optional[float]:
        """Alias for cv_accuracy_mean."""
        return self.cv_accuracy_mean

    @property
    def cv_std_accuracy(self) -> Optional[float]:
        """Alias for cv_accuracy_std."""
        return self.cv_accuracy_std

    def summary_dict(self) -> Dict[str, Any]:
        """Return key scalar metrics for tabular comparison."""
        return {
            "Model": self.model_name,
            "Type": self.model_type,
            "In-Dist Acc": f"{self.in_dist_accuracy * 100:.2f}%",
            "In-Dist F1": f"{self.in_dist_f1:.3f}",
            "Length-Gen Acc": f"{self.gen_accuracy * 100:.2f}%",
            "Length-Gen F1": f"{self.gen_f1:.3f}",
            "Train Time (ms)": f"{self.train_time_ms:.2f}",
            "Infer Time (µs/str)": f"{self.inference_time_us_per_sample:.1f}",
            "Complexity": self.complexity_metric,
        }


def get_model_complexity(model: Any, model_type: str) -> Dict[str, Union[int, float]]:
    """Compute structural complexity parameters for the model."""
    if model_type == "decision_tree":
        depth = model.get_depth()
        node_count = model.tree_.node_count
        leaf_count = model.get_n_leaves()
        return {
            "depth": depth,
            "node_count": node_count,
            "leaf_count": leaf_count,
            "primary_metric": f"Depth {depth}, {node_count} nodes",
        }
    elif model_type == "random_forest":
        n_trees = len(model.estimators_)
        avg_depth = float(np.mean([t.get_depth() for t in model.estimators_]))
        total_nodes = int(np.sum([t.tree_.node_count for t in model.estimators_]))
        return {
            "n_trees": n_trees,
            "avg_depth": round(avg_depth, 1),
            "total_nodes": total_nodes,
            "primary_metric": f"{n_trees} trees, avg depth {avg_depth:.1f}",
        }
    elif model_type == "logistic_regression":
        coef = model.coef_[0] if hasattr(model, "coef_") else np.array([])
        non_zero = int(np.sum(np.abs(coef) > 1e-4))
        total = len(coef)
        return {
            "non_zero_coefs": non_zero,
            "total_coefs": total,
            "primary_metric": f"{non_zero}/{total} non-zero weights",
        }
    return {"primary_metric": "Unknown"}


def extract_feature_importances(
    model: Any,
    model_type: str,
    feature_names: List[str],
) -> Dict[str, float]:
    """Extract normalized feature importance scores."""
    importances: Dict[str, float] = {}
    if not feature_names:
        return importances

    if model_type in ("decision_tree", "random_forest") and hasattr(model, "feature_importances_"):
        raw = model.feature_importances_
        for name, val in zip(feature_names, raw):
            importances[name] = float(val)
    elif model_type == "logistic_regression" and hasattr(model, "coef_"):
        raw = np.abs(model.coef_[0])
        total = np.sum(raw)
        norm = (raw / total) if total > 0 else raw
        for name, val in zip(feature_names, norm):
            importances[name] = float(val)

    # Sort descending
    return dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))


def build_classifier(
    model_type: str,
    hyperparams: Optional[Dict[str, Any]] = None,
    seed: int = 42,
) -> Any:
    """Factory creating an unfitted scikit-learn classifier."""
    hp = hyperparams or {}
    if model_type == "decision_tree":
        return DecisionTreeClassifier(
            max_depth=hp.get("max_depth", None),
            criterion=hp.get("criterion", "gini"),
            min_samples_split=hp.get("min_samples_split", 2),
            random_state=seed,
        )
    elif model_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=hp.get("n_estimators", 100),
            max_depth=hp.get("max_depth", None),
            criterion=hp.get("criterion", "gini"),
            random_state=seed,
            n_jobs=-1,
        )
    elif model_type == "logistic_regression":
        return LogisticRegression(
            C=hp.get("C", 1.0),
            penalty=hp.get("penalty", "l2"),
            solver=hp.get("solver", "lbfgs"),
            max_iter=hp.get("max_iter", 1000),
            random_state=seed,
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def train_and_evaluate_ml_model(
    model_type: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    X_gen: np.ndarray,
    y_gen: np.ndarray,
    feature_names: List[str],
    hyperparams: Optional[Dict[str, Any]] = None,
    run_cv: bool = True,
    seed: int = 42,
    raw_test_strings: Optional[List[str]] = None,
    feature_extractor: Optional[Any] = None,
) -> ModelEvaluationResult:
    """Train classifier and evaluate on in-distribution and length-generalization test sets."""
    model = build_classifier(model_type, hyperparams, seed)

    # Measure Training Time
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    t1 = time.perf_counter()
    train_time_ms = (t1 - t0) * 1000.0

    # Measure End-to-End Inference Time (raw string -> extractor -> predict) if raw strings & extractor provided
    if feature_extractor is not None and raw_test_strings is not None and len(raw_test_strings) > 0:
        repeat_factor = max(1, (1000 // max(1, len(raw_test_strings))) + 1)
        timing_strings = (raw_test_strings * repeat_factor)[:max(1000, len(raw_test_strings))]
        # Warm-up phase: raw string -> feature extraction -> model prediction
        warmup_strings = timing_strings[:min(200, len(timing_strings))]
        X_warmup = feature_extractor.transform(warmup_strings)
        _ = model.predict(X_warmup)
        # Timed measurement phase: end-to-end from raw strings through feature extraction and prediction
        t_inf_0 = time.perf_counter()
        X_timed = feature_extractor.transform(timing_strings)
        _ = model.predict(X_timed)
        t_inf_1 = time.perf_counter()
        inf_time_us = ((t_inf_1 - t_inf_0) * 1e6) / len(timing_strings)
    else:
        # Fallback to model-only prediction timing on feature matrix
        repeat_factor = max(1, (1000 // max(1, len(X_test))) + 1)
        X_timing = np.tile(X_test, (repeat_factor, 1))[:max(1000, len(X_test))]
        _ = model.predict(X_timing[:min(200, len(X_timing))])
        t_inf_0 = time.perf_counter()
        _ = model.predict(X_timing)
        t_inf_1 = time.perf_counter()
        inf_time_us = ((t_inf_1 - t_inf_0) * 1e6) / len(X_timing)

    # In-Distribution Predictions
    y_pred_test = model.predict(X_test)

    # In-Distribution Metrics
    acc_test = float(accuracy_score(y_test, y_pred_test))
    prec_test = float(precision_score(y_test, y_pred_test, zero_division=0))
    rec_test = float(recall_score(y_test, y_pred_test, zero_division=0))
    f1_test = float(f1_score(y_test, y_pred_test, zero_division=0))
    cm_test = confusion_matrix(y_test, y_pred_test, labels=[0, 1])

    # Length-Generalization Metrics
    y_pred_gen = model.predict(X_gen)
    acc_gen = float(accuracy_score(y_gen, y_pred_gen))
    prec_gen = float(precision_score(y_gen, y_pred_gen, zero_division=0))
    rec_gen = float(recall_score(y_gen, y_pred_gen, zero_division=0))
    f1_gen = float(f1_score(y_gen, y_pred_gen, zero_division=0))
    cm_gen = confusion_matrix(y_gen, y_pred_gen, labels=[0, 1])

    # Complexity and Feature Importance
    complexity = get_model_complexity(model, model_type)
    feat_imp = extract_feature_importances(model, model_type, feature_names)

    # Cross-validation on training data
    cv_mean, cv_std = None, None
    if run_cv and len(X_train) >= 10:
        try:
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
            cv_mean = float(np.mean(cv_scores))
            cv_std = float(np.std(cv_scores))
        except Exception:
            pass

    display_names = {
        "decision_tree": "Decision Tree",
        "random_forest": "Random Forest",
        "logistic_regression": "Logistic Regression",
    }

    return ModelEvaluationResult(
        model_name=display_names.get(model_type, model_type),
        model_type=model_type,
        in_dist_accuracy=acc_test,
        in_dist_precision=prec_test,
        in_dist_recall=rec_test,
        in_dist_f1=f1_test,
        in_dist_confusion=cm_test,
        gen_accuracy=acc_gen,
        gen_precision=prec_gen,
        gen_recall=rec_gen,
        gen_f1=f1_gen,
        gen_confusion=cm_gen,
        train_time_ms=train_time_ms,
        inference_time_us_per_sample=inf_time_us,
        model_complexity=complexity,
        feature_importances=feat_imp,
        cv_accuracy_mean=cv_mean,
        cv_accuracy_std=cv_std,
        raw_model=model,
        feature_extractor=feature_extractor,
    )
