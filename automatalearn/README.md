# AutomataLearn: Intelligent Automata Learning & Regular Language Classification

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-1.40+-FF4B4B.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests: Pytest](https://img.shields.io/badge/tests-32%20passed-brightgreen.svg)](https://pytest.org)

**AutomataLearn** is an interactive computer-science education and research platform designed to investigate a central question in theoretical computer science and artificial intelligence:

> *Can statistical machine learning models learn formal regular grammars, and how does their out-of-distribution length generalization compare to grammatical inference (RPNI, Angluin's $L^*$) and minimal Deterministic Finite Automata (DFAs)?*

Built entirely from scratch in pure Python without heavy external automata libraries, AutomataLearn provides complete implementations of DFAs, NFAs with $\varepsilon$-transitions, Thompson's regex compiler, Myhill-Nerode Table-Filling and Hopcroft minimization, passive state-merging (RPNI), active teacher learning ($L^*$), and comparative classical ML baselines (Decision Trees, Random Forests, Logistic Regression).

---

## 🌟 Key Features

1. **Pure Python Automata Engine from Scratch (`core/`)**
   - **DFA:** State path tracing, completeness checking, sink-state completion, unreachable state pruning, product-construction equivalence testing ($L_1 \oplus L_2 = \emptyset$) with counterexample generation.
   - **NFA:** $\varepsilon$-closure calculation, non-deterministic branching, and subset (powerset) construction to DFA.
   - **Minimization:** Both **Hopcroft's $O(k n \log n)$ partition refinement** and the **Myhill-Nerode Table-Filling algorithm** with step-by-step distinguishability pair matrices.
   - **Regex-to-NFA:** Complete Thompson construction with Dijkstra's Shunting-Yard parser supporting `*`, `+`, `?`, `|`, and parentheses.

2. **Benchmark Regular Languages (`core/languages.py`)**
   - Strings ending with `"ab"`
   - Even number of 0s AND odd number of 1s (Z2 x Z2 parity)
   - Contains substring `"101"`
   - Number of a's divisible by 3 (Z3 counting)
   - Binary numbers divisible by 3 (Horner's rule modulo automaton)
   - No two consecutive 1s (Fibonacci language)
   - Length mod 4 == 0 (periodic cycle)
   - Custom regular expression / transition table input

3. **Grammatical Inference & Learning Algorithms (`learning/`)**
   - **RPNI (Oncina & García, 1992):** Builds a Prefix Tree Acceptor (PTA), maintains Red/Blue state partitions, performs deterministic folding, rejects merges violating negative samples, and outputs minimal DFAs.
   - **Angluin's $L^*$ (1987):** Implements an active Minimal Adequate Teacher (MAT) with an Observation Table $(S, E, T)$, resolving closedness and consistency violations and handling counterexamples.

4. **Machine Learning Pipeline (`ml/`)**
   - Feature extractor extracting >60 dimensions across 6 modular groups (basic lengths/counts, parities mod 2/3/4, prefix/suffix affixes, n-grams, run-length dynamics, position-weighted sums).
   - Scikit-learn Decision Tree, Random Forest, and Logistic Regression with full profiling (accuracy, F1, latency, complexity, and feature importances).

5. **Comparison & Benchmark Engine (`ml/benchmark.py`)**
   - Evaluates all 6 classifiers on identical splits.
   - Probes the **Length-Generalization Gap** by evaluating models on strings strictly longer than any seen during training.
   - Profiles microsecond-level per-string inference latency and millisecond-level training times.

6. **Interactive 7-Tab Streamlit Web Application (`app.py`)**
   - Graphviz interactive rendering and live execution path animations.

---

## 📂 Project Structure

```
automatalearn/
├── app.py                      # Streamlit interactive web application
├── requirements.txt            # Python dependencies
├── core/
│   ├── dfa.py                  # DFA implementation from scratch
│   ├── nfa.py                  # NFA with epsilon transitions + subset construction
│   ├── minimize.py             # Hopcroft and Table-Filling minimization
│   ├── languages.py            # 7 predefined languages + ground-truth DFAs
│   └── regex_to_nfa.py         # Thompson construction regex compiler
├── data/
│   └── generator.py            # Dataset generator, balanced sampling, noise injection
├── learning/
│   ├── prefix_tree.py          # Prefix Tree Acceptor (PTA) builder
│   ├── rpni.py                 # RPNI passive state-merging learning
│   └── lstar.py                # Angluin's L* active learning algorithm
├── ml/
│   ├── features.py             # Numerical string feature extraction pipeline
│   ├── models.py               # Decision Tree, Random Forest, Logistic Regression
│   └── benchmark.py            # Multi-model comparison suite & learning curves
├── viz/
│   ├── draw.py                 # Graphviz and DOT diagram generators
│   └── plots.py                # Plotly interactive charts and heatmaps
├── tests/
│   ├── test_dfa.py             # DFA acceptance, trace, equivalence tests
│   ├── test_nfa.py             # NFA epsilon-closure, subset construction tests
│   ├── test_minimize.py        # Hopcroft & Table-Filling minimization tests
│   ├── test_learning.py        # PTA, RPNI consistency, and L* exact convergence
│   ├── test_features.py        # Feature extractor group verification
│   ├── test_generator.py       # Dataset generator and CSV serialization tests
│   └── test_ml_models.py       # ML classifier training and profiling tests
docs/
└── REPORT.md                   # Detailed experimental report and findings
results/
└── benchmark.csv               # Empirical benchmark results across all 7 languages
run_benchmark.py                # Standalone CLI benchmark script
README.md                       # Documentation and usage guide
```

---

## 🚀 Quick Start Guide

### 1. Installation
Clone or navigate to the repository directory and install the dependencies:

```bash
pip install -r requirements.txt
```

### 2. Run the Interactive Streamlit Web UI
Launch the interactive web application:

```bash
streamlit run automatalearn/app.py
```
Open your browser at `http://localhost:8501` (or the port displayed in your terminal).

### 3. Run the CLI Benchmark
Run the automated benchmark across all 7 predefined languages and generate `results/benchmark.csv`:

```bash
python run_benchmark.py
```

### 4. Run the Full Test Suite
Execute the 32 unit tests verifying core algorithms and learning convergence:

```bash
python -m pytest automatalearn/tests -v
```

---

## 🖥️ Streamlit Application Walkthrough (7 Tabs)

| Tab | Name | Description |
| :--- | :--- | :--- |
| **Tab 1** | **Language & Data** | Select a target language or define a custom regex. View the ground-truth DFA diagram. Configure dataset generation (alphabet, lengths, sample sizes, class balance, noise rate) and download/upload CSVs. |
| **Tab 2** | **DFA / NFA Lab** | Convert NFAs to DFAs via subset construction. Run Hopcroft or Table-Filling minimization. Inspect the Myhill-Nerode distinguishability matrix and step through interactive state path traces with active node highlighting. |
| **Tab 3** | **Learn Automaton** | Run RPNI (passive state-merging) or Angluin's $L^*$ (active teacher queries). View detailed step-by-step logs, compare initial PTA size vs. final state count, and check exact equivalence with counterexamples. |
| **Tab 4** | **ML Classifiers** | Select numerical feature groups (Basic, Parity, Affixes, N-grams, Runs, Position-weighted). Train Decision Trees, Random Forests, or Logistic Regression, and inspect confusion matrices, metrics, and feature importances. |
| **Tab 5** | **Compare** | View side-by-side performance tables comparing all 6 approaches. Explore Plotly charts for in-distribution vs. length-generalization accuracy, empirical learning curves, noise robustness, and auto-generated pedagogical summaries. |
| **Tab 6** | **Classify** | Enter any custom string to view live predictions across all models simultaneously. Automatically flags disagreements and animates the exact DFA trajectory. |
| **Tab 7** | **Theory & Foundations**| Comprehensive educational guide covering formal automata theory, the Chomsky hierarchy, the Myhill-Nerode theorem, Gold's theorem, RPNI, Angluin's $L^*$, and the inductive bias paradox. |

---

## 📊 Summary of Experimental Results

Empirical results from running `run_benchmark.py` (300 in-distribution samples, lengths 0–8; 100 length-generalization samples, lengths 9–20):

```
========================================================================================================
Language                      Ground-Truth DFA      RPNI DFA       L* DFA        DT        RF       LR
========================================================================================================
Strings ending with "ab"      100.0% | 100.0%    100% | 100%   100% | 100%   100|100   100|100  100|98%
Even 0s AND Odd 1s (Parity)   100.0% | 100.0%    100% | 100%   100% | 100%   100|100   100|82%  100|98%
Contains "101"                100.0% | 100.0%    100% | 100%   100% | 100%   100|100   100|100  100|94%
Number of a's div by 3 (Z3)   100.0% | 100.0%    100% | 100%   100% | 100%   100|100   100|99%  100|83%
Binary numbers div by 3       100.0% | 100.0%    100% | 100%   100% | 100%   100|100   100|100  100|93%
No two consecutive 1s         100.0% | 100.0%    100% | 100%   100% | 100%   100|100   100|100  100|100
Length mod 4 == 0 (Cycle)     100.0% | 100.0%    100% | 100%   100% | 100%   100|100   100|88%  100|75%
--------------------------------------------------------------------------------------------------------
Inference Latency (avg)             1.4 µs           2.1 µs        1.8 µs     3.7 µs    605 µs   5.2 µs
Equivalence to Target Grammar       Exact            Exact         Exact      Approx    Approx   Approx
========================================================================================================
Format: [In-Distribution Test Accuracy %] | [Length-Generalization Accuracy %]
```

---

## 🏆 Which Approach Wins Under Which Conditions?

| Paradigm | Primary Strengths | Limitations | When to Use |
| :--- | :--- | :--- | :--- |
| **Ground-Truth DFA** | Zero training time, ultra-low latency (~1.4 µs), provable $100\%$ accuracy over infinite lengths. | Requires full formal specification upfront. | Compilers, static analysis, formal protocol verifiers, WAF regex rules. |
| **Active Learning (Angluin's $L^*$)** | Polynomial query complexity, provable exact minimal DFA equivalence, extremely fast convergence (<1 ms). | Requires an interactive teacher capable of answering membership and equivalence queries. | Automated black-box protocol fuzzing, reverse engineering network services, circuit verification. |
| **Passive Learning (RPNI)** | Learns minimal state machines from passive string traces $(S_+, S_-)$ without an interactive teacher. | Sensitive to label noise in training strings; requires sufficient sample diversity. | Mining workflows, intrusion detection from logs, reverse-engineering malware communication. |
| **Decision Trees & Random Forests** | High tolerance to noisy tabular data; easily inspectable feature splits. | **Brittle length generalization:** hyper-plane cuts cannot extrapolate cyclic modular arithmetic ($82–88\%$ accuracy). High inference latency (~600 µs). | General tabular classification with noisy features where inputs do not exhibit unbounded length variation. |
| **Logistic Regression** | Fast training, well-calibrated probabilities, convex loss surface. | Struggles with non-linear parity/modulo combinations; length generalization drops to $75\%$. | Baseline linear classification, sentiment polarity, simple token bag-of-words. |

---

## 📖 Scientific Report

For the in-depth theoretical analysis, mathematical derivations of the Myhill-Nerode theorem, algorithmic breakdowns of RPNI folding and $L^*$ observation tables, and complete empirical tables, see **[docs/REPORT.md](docs/REPORT.md)**.
