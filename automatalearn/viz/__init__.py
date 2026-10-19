"""Visualization package for AutomataLearn: Graphviz drawing and Plotly charts."""

from automatalearn.viz.draw import (
    dfa_to_dot,
    dfa_to_graphviz,
    nfa_to_dot,
    nfa_to_graphviz,
)
from automatalearn.viz.plots import (
    plot_in_dist_vs_gen,
    plot_learning_curves,
    plot_noise_robustness,
    plot_runtime_comparison,
    plot_complexity_vs_accuracy,
    plot_feature_importance_bar,
    plot_confusion_matrix_heatmap,
)

__all__ = [
    "dfa_to_dot",
    "dfa_to_graphviz",
    "nfa_to_dot",
    "nfa_to_graphviz",
    "plot_in_dist_vs_gen",
    "plot_learning_curves",
    "plot_noise_robustness",
    "plot_runtime_comparison",
    "plot_complexity_vs_accuracy",
    "plot_feature_importance_bar",
    "plot_confusion_matrix_heatmap",
]
