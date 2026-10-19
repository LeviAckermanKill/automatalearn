"""Automata and Machine Learning Comparison Benchmark Engine.

Standardizes evaluation of Ground-Truth DFA, RPNI-learned DFA, L*-learned DFA,
Decision Tree, Random Forest, and Logistic Regression on identical data splits.
Measures in-distribution accuracy, length generalization, complexity, execution time,
exact DFA equivalence with counterexamples, learning curves, and noise robustness.
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

from automatalearn.core.dfa import DFA
from automatalearn.core.languages import LanguageDefinition, get_language
from automatalearn.data.generator import DataGenerator, DatasetSplits
from automatalearn.learning.lstar import LStarLearner, LStarResult
from automatalearn.learning.rpni import RPNILearner, RPNIResult
from automatalearn.ml.features import (
    ALL_FEATURE_GROUPS,
    BASIC_FEATURE_GROUPS,
    GENERIC_FEATURE_GROUPS,
    StringFeatureExtractor,
)
from automatalearn.ml.models import (
    ModelEvaluationResult,
    extract_feature_importances,
    get_model_complexity,
    train_and_evaluate_ml_model,
)


@dataclass
class BenchmarkEntry:
    """Benchmark outcome for a single model on a dataset."""
    model_name: str
    model_category: str  # "Ground Truth", "Automata Learning", "Machine Learning"
    in_dist_acc: float
    in_dist_f1: float
    gen_acc: float
    gen_f1: float
    train_time_ms: float
    infer_time_us: float
    complexity_desc: str
    state_count: Optional[int] = None
    is_equivalent_to_target: Optional[bool] = None
    counterexample: Optional[str] = None
    raw_object: Optional[Any] = None


@dataclass
class BenchmarkSuiteResult:
    """Complete collection of benchmark comparisons for a language."""
    language_name: str
    language_key: str
    entries: List[BenchmarkEntry]
    dataset_splits: DatasetSplits
    feature_extractor: StringFeatureExtractor
    rpni_result: Optional[RPNIResult] = None
    lstar_result: Optional[LStarResult] = None
    ml_eval_results: Dict[str, ModelEvaluationResult] = field(default_factory=dict)

    @property
    def alphabet(self) -> List[str]:
        """Alphabet of the evaluated language."""
        if hasattr(self, "dataset_splits") and hasattr(self.dataset_splits, "alphabet"):
            return sorted(list(self.dataset_splits.alphabet))
        if hasattr(self, "feature_extractor") and hasattr(self.feature_extractor, "alphabet"):
            return sorted(list(self.feature_extractor.alphabet))
        try:
            lang = get_language(self.language_key)
            return sorted(list(lang.alphabet))
        except Exception:
            return []

    @property
    def num_samples(self) -> int:
        """Total in-distribution sample count across train, val, and test."""
        if hasattr(self, "dataset_splits") and self.dataset_splits is not None:
            return len(self.dataset_splits.train_df) + len(self.dataset_splits.val_df) + len(self.dataset_splits.test_df)
        return 0

    @property
    def gen_samples(self) -> int:
        """Total out-of-distribution length-generalization sample count."""
        if hasattr(self, "dataset_splits") and self.dataset_splits is not None:
            return len(self.dataset_splits.generalization_df)
        return 0

    def to_dataframe(self) -> pd.DataFrame:
        """Convert benchmark entries to a presentation DataFrame."""
        rows = []
        for e in self.entries:
            if e.is_equivalent_to_target is True:
                equiv_str = "Exact (True)"
            elif e.is_equivalent_to_target is False:
                if e.counterexample:
                    equiv_str = f"Not equivalent (counterexample: '{e.counterexample}')"
                else:
                    equiv_str = "Not equivalent"
            else:
                equiv_str = "N/A"

            rows.append({
                "Model": e.model_name,
                "Category": e.model_category,
                "In-Dist Acc": f"{e.in_dist_acc * 100:.1f}%",
                "In-Dist F1": f"{e.in_dist_f1:.3f}",
                "Length-Gen Acc": f"{e.gen_acc * 100:.1f}%",
                "Length-Gen F1": f"{e.gen_f1:.3f}",
                "Train Time (ms)": f"{e.train_time_ms:.2f}",
                "Infer (µs/str)": f"{e.infer_time_us:.1f}",
                "Complexity": e.complexity_desc,
                "Target Equiv?": equiv_str,
            })
        return pd.DataFrame(rows)

    def generate_narrative_summary(self) -> str:
        """Generate automatic pedagogical insights from the benchmark run."""
        df = pd.DataFrame([
            {
                "name": e.model_name,
                "in_acc": e.in_dist_acc,
                "gen_acc": e.gen_acc,
                "cat": e.model_category,
                "equiv": e.is_equivalent_to_target,
            }
            for e in self.entries
        ])

        target_row = df[df["name"] == "Ground-Truth DFA"].iloc[0]
        rpni_row = df[df["name"] == "RPNI Learned DFA"].iloc[0]
        lstar_row = df[df["name"] == "L* Learned DFA"].iloc[0]
        ml_rows = df[df["cat"].str.contains("Machine Learning", na=False)]

        if not ml_rows.empty:
            drops = []
            for _, row in ml_rows.iterrows():
                d_pts = (row["in_acc"] - row["gen_acc"]) * 100.0
                drops.append((row["name"], row["in_acc"] * 100.0, row["gen_acc"] * 100.0, d_pts))
            drops.sort(key=lambda x: x[3], reverse=True)
            biggest_model, b_in, b_gen, biggest_drop = drops[0]

            if all(d[3] < 1.0 for d in drops):
                ml_narrative = (
                    f"While ML models achieved solid in-distribution performance (up to {ml_rows['in_acc'].max()*100:.1f}%), "
                    f"no model showed a generalization drop on this language."
                )
            else:
                ml_narrative = (
                    f"While ML models achieved solid in-distribution performance (up to {ml_rows['in_acc'].max()*100:.1f}%), "
                    f"their length generalization degraded significantly. The largest drop occurred in **{biggest_model}**, "
                    f"which dropped by **{biggest_drop:.1f} percentage points** (from **{b_in:.1f}%** in-distribution down to **{b_gen:.1f}%** length-generalization)."
                )
        else:
            ml_narrative = "No statistical ML models were evaluated in this benchmark run."

        lines = [
            f"### Empirical Findings for '{self.language_name}'",
            "",
            f"1. **Length Generalization Discrepancy:**",
            f"   - **Automata Learning:** L* achieved **{lstar_row['gen_acc']*100:.1f}%** generalization accuracy "
            f"(Equivalence to target: **{'Confirmed' if lstar_row['equiv'] else 'Approximate'}**). "
            f"RPNI achieved **{rpni_row['gen_acc']*100:.1f}%** generalization.",
            f"   - **Machine Learning Models:** {ml_narrative}",
            "",
            "2. **Theoretical Root Cause:**",
            "   - Regular languages are fundamentally governed by state transitions rather than independent feature frequencies. "
            "When strings exceed the training length distribution, classical ML classifiers must extrapolate beyond their trained "
            "feature bounds (e.g. unseen lengths, shifted n-gram indices, or unbounded counts).",
            "   - In contrast, DFA learning induces cyclic transition structures (e.g., modulo rings, loops, and absorbing sinks). "
            "Once the correct recurrent cycle is discovered, the automaton guarantees **100% exact generalization across strings of infinite length**.",
            "",
            "3. **Inference Latency:**",
            "   - Table-based DFA transitions execute in pure $O(L)$ time with minimal state lookups, "
            "yielding microsecond-level latency per string without vectorization or feature generation overhead.",
        ]
        return "\n".join(lines)


class BenchmarkEngine:
    """Orchestrates comprehensive multi-model experiments."""

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed

    def _profile_dfa(
        self,
        dfa: DFA,
        test_strings: List[str],
        test_labels: List[int],
        gen_strings: List[str],
        gen_labels: List[int],
    ) -> Tuple[float, float, float, float, float]:
        """Profile DFA accuracy, F1, and per-sample inference latency with warm-up and >= 1000 strings."""
        # Standardized timing measurement with warm-up and averaging over >= 1000 strings
        repeat_factor = max(1, (1000 // max(1, len(test_strings))) + 1)
        timing_strings = (test_strings * repeat_factor)[:max(1000, len(test_strings))]

        # Warm-up phase
        for s in timing_strings[:min(200, len(timing_strings))]:
            dfa.accepts(s)

        # Timed measurement phase
        t0 = time.perf_counter()
        for s in timing_strings:
            dfa.accepts(s)
        t1 = time.perf_counter()
        infer_us = ((t1 - t0) * 1e6) / len(timing_strings)

        # In-distribution evaluation
        preds_test = [1 if dfa.accepts(s) else 0 for s in test_strings]
        acc_test = float(accuracy_score(test_labels, preds_test))
        f1_test = float(f1_score(test_labels, preds_test, zero_division=0))

        # Length-generalization evaluation
        preds_gen = [1 if dfa.accepts(s) else 0 for s in gen_strings]
        acc_gen = float(accuracy_score(gen_labels, preds_gen))
        f1_gen = float(f1_score(gen_labels, preds_gen, zero_division=0))

        return acc_test, f1_test, acc_gen, f1_gen, infer_us

    def run_benchmark(
        self,
        language: Union[str, LanguageDefinition],
        num_samples: int = 1500,
        max_length: int = 12,
        gen_samples: int = 500,
        noise_rate: float = 0.0,
        enabled_features: Optional[List[str]] = None,
        include_oracle: bool = False,
        include_basic: bool = False,
        feature_mode: str = "generic",
        dt_params: Optional[Dict[str, Any]] = None,
        rf_params: Optional[Dict[str, Any]] = None,
        lr_params: Optional[Dict[str, Any]] = None,
    ) -> BenchmarkSuiteResult:
        """Run all classifiers on identical data splits."""
        if isinstance(language, str):
            lang_def = get_language(language)
        else:
            lang_def = language

        # 1. Generate partitioned dataset
        generator = DataGenerator(
            alphabet=lang_def.alphabet,
            oracle=lang_def.predicate,
            dfa=lang_def.dfa,
            seed=self.seed,
        )
        splits = generator.generate_dataset(
            min_length=0,
            max_length=max_length,
            num_samples=num_samples,
            class_balance=0.5,
            noise_rate=noise_rate,
            gen_min_length=max_length + 1,
            gen_max_length=max_length + 12,
            gen_samples=gen_samples,
        )

        entries: List[BenchmarkEntry] = []

        # 2. Evaluate Ground-Truth DFA
        gt_dfa = lang_def.dfa
        acc_t, f1_t, acc_g, f1_g, infer_us = self._profile_dfa(
            gt_dfa,
            splits.test_strings,
            splits.test_labels,
            splits.gen_strings,
            splits.gen_labels,
        )
        entries.append(BenchmarkEntry(
            model_name="Ground-Truth DFA",
            model_category="Ground Truth",
            in_dist_acc=acc_t,
            in_dist_f1=f1_t,
            gen_acc=acc_g,
            gen_f1=f1_g,
            train_time_ms=0.0,
            infer_time_us=infer_us,
            complexity_desc=f"{len(gt_dfa.states)} states (minimal)",
            state_count=len(gt_dfa.states),
            is_equivalent_to_target=True,
            counterexample=None,
            raw_object=gt_dfa,
        ))

        # 3. Learn via RPNI (State Merging)
        pos_train = [s for s, y in zip(splits.train_strings, splits.train_labels) if y == 1]
        neg_train = [s for s, y in zip(splits.train_strings, splits.train_labels) if y == 0]

        rpni_learner = RPNILearner(alphabet=lang_def.alphabet)
        rpni_res = rpni_learner.learn(pos_train, neg_train)
        rpni_dfa = rpni_res.minimized_dfa

        acc_t, f1_t, acc_g, f1_g, infer_us = self._profile_dfa(
            rpni_dfa,
            splits.test_strings,
            splits.test_labels,
            splits.gen_strings,
            splits.gen_labels,
        )
        equiv_rpni, ce_rpni = rpni_dfa.equivalent_to(gt_dfa)

        entries.append(BenchmarkEntry(
            model_name="RPNI Learned DFA",
            model_category="Automata Learning",
            in_dist_acc=acc_t,
            in_dist_f1=f1_t,
            gen_acc=acc_g,
            gen_f1=f1_g,
            train_time_ms=rpni_res.execution_time_ms,
            infer_time_us=infer_us,
            complexity_desc=f"{len(rpni_dfa.states)} states (from {rpni_res.pta_state_count} PTA)",
            state_count=len(rpni_dfa.states),
            is_equivalent_to_target=equiv_rpni,
            counterexample=ce_rpni,
            raw_object=rpni_dfa,
        ))

        # 4. Learn via Angluin's L*
        lstar_learner = LStarLearner(
            alphabet=lang_def.alphabet,
            membership_oracle=lang_def.predicate,
            target_dfa=gt_dfa,
        )
        lstar_res = lstar_learner.learn()
        lstar_dfa = lstar_res.minimized_dfa

        acc_t, f1_t, acc_g, f1_g, infer_us = self._profile_dfa(
            lstar_dfa,
            splits.test_strings,
            splits.test_labels,
            splits.gen_strings,
            splits.gen_labels,
        )
        equiv_lstar, ce_lstar = lstar_dfa.equivalent_to(gt_dfa)

        entries.append(BenchmarkEntry(
            model_name="L* Learned DFA",
            model_category="Automata Learning",
            in_dist_acc=acc_t,
            in_dist_f1=f1_t,
            gen_acc=acc_g,
            gen_f1=f1_g,
            train_time_ms=lstar_res.execution_time_ms,
            infer_time_us=infer_us,
            complexity_desc=f"{len(lstar_dfa.states)} states ({lstar_res.num_membership_queries} MQs, {lstar_res.num_equivalence_queries} EQs)",
            state_count=len(lstar_dfa.states),
            is_equivalent_to_target=equiv_lstar,
            counterexample=ce_lstar,
            raw_object=lstar_dfa,
        ))

        # 5. Determine active ML feature variants
        ml_configs = [
            ("decision_tree", "Decision Tree", dt_params),
            ("random_forest", "Random Forest", rf_params),
            ("logistic_regression", "Logistic Regression", lr_params),
        ]

        variants: List[Tuple[str, List[str], str]] = []
        mode = feature_mode.lower().strip()
        if mode == "basic":
            variants.append(("Basic", BASIC_FEATURE_GROUPS, "Machine Learning (Basic)"))
        elif mode == "oracle":
            variants.append(("Oracle Leaky", ALL_FEATURE_GROUPS, "Machine Learning (Oracle)"))
        elif mode == "all":
            variants.append(("Basic", BASIC_FEATURE_GROUPS, "Machine Learning (Basic)"))
            gen_feats = enabled_features if enabled_features is not None else GENERIC_FEATURE_GROUPS
            variants.append(("Generic", gen_feats, "Machine Learning (Generic)"))
            variants.append(("Oracle Leaky", ALL_FEATURE_GROUPS, "Machine Learning (Oracle)"))
        else:  # default "generic"
            if include_basic:
                variants.append(("Basic", BASIC_FEATURE_GROUPS, "Machine Learning (Basic)"))
            gen_feats = enabled_features if enabled_features is not None else GENERIC_FEATURE_GROUPS
            variants.append(("Generic", gen_feats, "Machine Learning (Generic)"))
            if include_oracle:
                variants.append(("Oracle Leaky", ALL_FEATURE_GROUPS, "Machine Learning (Oracle)"))

        ml_evals: Dict[str, ModelEvaluationResult] = {}
        primary_extractor: Optional[StringFeatureExtractor] = None

        y_train = splits.train_df["label"].to_numpy()
        y_test = splits.test_df["label"].to_numpy()
        y_gen = splits.generalization_df["label"].to_numpy()

        for v_label, v_groups, v_cat in variants:
            var_extractor = StringFeatureExtractor(
                alphabet=lang_def.alphabet,
                enabled_groups=v_groups,
            )
            X_train_v = var_extractor.fit_transform(splits.train_strings)
            X_test_v = var_extractor.transform(splits.test_strings)
            X_gen_v = var_extractor.transform(splits.gen_strings)

            if primary_extractor is None:
                primary_extractor = var_extractor

            for m_type, m_base, params in ml_configs:
                m_name = f"{m_base} ({v_label})"
                eval_res = train_and_evaluate_ml_model(
                    model_type=m_type,
                    X_train=X_train_v,
                    y_train=y_train,
                    X_test=X_test_v,
                    y_test=y_test,
                    X_gen=X_gen_v,
                    y_gen=y_gen,
                    feature_names=var_extractor.get_feature_names(),
                    hyperparams=params,
                    run_cv=False,
                    seed=self.seed,
                    raw_test_strings=splits.test_strings,
                    feature_extractor=var_extractor,
                )
                eval_res.model_name = m_name
                eval_res.feature_extractor = var_extractor
                ml_evals[m_name] = eval_res
                entries.append(BenchmarkEntry(
                    model_name=m_name,
                    model_category=v_cat,
                    in_dist_acc=eval_res.in_dist_accuracy,
                    in_dist_f1=eval_res.in_dist_f1,
                    gen_acc=eval_res.gen_accuracy,
                    gen_f1=eval_res.gen_f1,
                    train_time_ms=eval_res.train_time_ms,
                    infer_time_us=eval_res.inference_time_us_per_sample,
                    complexity_desc=str(eval_res.model_complexity.get("primary_metric", "-")),
                    is_equivalent_to_target=None,
                    counterexample=None,
                    raw_object=eval_res.raw_model,
                ))

        return BenchmarkSuiteResult(
            language_name=lang_def.name,
            language_key=lang_def.key,
            entries=entries,
            dataset_splits=splits,
            feature_extractor=primary_extractor,
            rpni_result=rpni_res,
            lstar_result=lstar_res,
            ml_eval_results=ml_evals,
        )

    def run_learning_curve(
        self,
        language: Union[str, LanguageDefinition],
        sample_sizes: Optional[List[int]] = None,
        max_length: int = 10,
    ) -> pd.DataFrame:
        """Compute learning curve: performance vs number of training samples."""
        if isinstance(language, str):
            lang_def = get_language(language)
        else:
            lang_def = language

        sizes = sample_sizes or [40, 80, 160, 320, 500]
        records = []

        for sz in sizes:
            suite = self.run_benchmark(
                language=lang_def,
                num_samples=sz,
                max_length=max_length,
                gen_samples=100,
            )
            for entry in suite.entries:
                records.append({
                    "sample_size": sz,
                    "model": entry.model_name,
                    "category": entry.model_category,
                    "in_dist_acc": entry.in_dist_acc,
                    "gen_acc": entry.gen_acc,
                    "train_time_ms": entry.train_time_ms,
                })

        return pd.DataFrame(records)

    def run_noise_robustness(
        self,
        language: Union[str, LanguageDefinition],
        noise_rates: Optional[List[float]] = None,
        num_samples: int = 300,
        max_length: int = 8,
    ) -> pd.DataFrame:
        """Evaluate model robustness against label noise."""
        if isinstance(language, str):
            lang_def = get_language(language)
        else:
            lang_def = language

        rates = noise_rates or [0.0, 0.05, 0.1, 0.2]
        records = []

        for rate in rates:
            suite = self.run_benchmark(
                language=lang_def,
                num_samples=num_samples,
                max_length=max_length,
                noise_rate=rate,
                gen_samples=100,
            )
            for entry in suite.entries:
                records.append({
                    "noise_rate": rate,
                    "noise_pct": f"{int(rate * 100)}%",
                    "model": entry.model_name,
                    "category": entry.model_category,
                    "in_dist_acc": entry.in_dist_acc,
                    "gen_acc": entry.gen_acc,
                })

        return pd.DataFrame(records)
