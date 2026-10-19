"""CLI Benchmark Runner for AutomataLearn.

Runs comparative benchmarks across Ground-Truth DFA, RPNI Learned DFA,
Angluin's L* Learned DFA, Decision Tree, Random Forest, and Logistic Regression.
Evaluates both in-distribution accuracy and out-of-distribution length generalization.
Saves consolidated metrics to results/benchmark.csv and logs progress to console.
"""

from __future__ import annotations
import argparse
import os
import sys

# Ensure project's parent directory is on sys.path so 'automatalearn' package is importable
_script_dir = os.path.abspath(os.path.dirname(__file__))
_project_root = _script_dir if os.path.isdir(os.path.join(_script_dir, "automatalearn")) else os.path.abspath(os.path.join(_script_dir, ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import time
import pandas as pd

from automatalearn.core.languages import PREDEFINED_LANGUAGES, get_language
from automatalearn.ml.benchmark import BenchmarkEngine


def main():
    parser = argparse.ArgumentParser(description="AutomataLearn Benchmark Runner")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run on all 7 predefined regular languages (default runs the 3 canonical benchmark languages)",
    )
    parser.add_argument(
        "--oracle",
        action="store_true",
        help="Include oracle features (default: generic features only)",
    )
    parser.add_argument(
        "--feature-mode",
        type=str,
        default="basic",
        choices=["basic", "generic", "oracle", "all"],
        help="Feature set for ML models: 'basic' (honest baseline without parity/affixes/oracle), 'generic', 'oracle', or 'all' (default: basic)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1500,
        help="In-distribution total sample size (default: 1500, yielding >=500 test strings)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=12,
        help="Max string length for in-distribution set (default: 12)",
    )
    parser.add_argument(
        "--gen-samples",
        type=int,
        default=500,
        help="Length-generalization sample size (default: 500 strings with lengths > max_length)",
    )
    args = parser.parse_args()

    target_keys = (
        list(PREDEFINED_LANGUAGES.keys())
        if args.all
        else ["binary_div_3", "even_0_odd_1", "count_a_div_3", "length_mod_4"]
    )

    print("=" * 80, flush=True)
    print(" AutomataLearn: Intelligent Automata Learning & Regular Language Benchmark", flush=True)
    print(f" Languages to evaluate: {target_keys}", flush=True)
    print(f" Feature Mode: {args.feature_mode.upper()} {'(+ Leaky Oracle)' if args.oracle else ''}", flush=True)
    print(f" Dataset sizes: In-Dist={args.num_samples} (Test >= 500), Length-Gen={args.gen_samples}", flush=True)
    print("=" * 80, flush=True)

    engine = BenchmarkEngine(seed=42)
    os.makedirs("results", exist_ok=True)

    all_rows = []
    start_time = time.time()

    for idx, lang_key in enumerate(target_keys, 1):
        lang = get_language(lang_key)
        print(f"\n[{idx}/{len(target_keys)}] Running benchmark for: '{lang.name}' (alphabet {lang.alphabet})...", flush=True)
        t_lang0 = time.time()

        suite = engine.run_benchmark(
            language=lang,
            num_samples=args.num_samples,
            max_length=args.max_length,
            gen_samples=args.gen_samples,
            include_oracle=args.oracle,
            feature_mode=args.feature_mode,
        )
        t_lang_dur = time.time() - t_lang0

        for entry in suite.entries:
            all_rows.append({
                "language_key": lang_key,
                "language_name": lang.name,
                "alphabet": "".join(lang.alphabet),
                "model_name": entry.model_name,
                "model_category": entry.model_category,
                "in_dist_accuracy": round(entry.in_dist_acc, 4),
                "in_dist_f1": round(entry.in_dist_f1, 4),
                "gen_accuracy": round(entry.gen_acc, 4),
                "gen_f1": round(entry.gen_f1, 4),
                "train_time_ms": round(entry.train_time_ms, 2),
                "inference_time_us_per_str": round(entry.infer_time_us, 1),
                "complexity": entry.complexity_desc,
                "target_equivalent": entry.is_equivalent_to_target,
                "counterexample": entry.counterexample or "",
            })

        # Print language summary table
        print(f"    Completed in {t_lang_dur:.2f}s:", flush=True)
        for entry in suite.entries:
            print(f"      - {entry.model_name:<28} | In-Dist: {entry.in_dist_acc*100:5.1f}% | Gen: {entry.gen_acc*100:5.1f}% | Train: {entry.train_time_ms:6.2f} ms | Infer: {entry.infer_time_us:5.1f} us/str", flush=True)

    total_duration = time.time() - start_time
    df_all = pd.DataFrame(all_rows)
    out_csv = os.path.join("results", "benchmark.csv")
    df_all.to_csv(out_csv, index=False)

    print("\n" + "=" * 80, flush=True)
    print(f" Benchmark completed in {total_duration:.2f}s!", flush=True)
    print(f" Total entries recorded: {len(df_all)}", flush=True)
    print(f" Results written to: {os.path.abspath(out_csv)}", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
