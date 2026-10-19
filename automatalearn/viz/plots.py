"""Interactive Plotly visualizations for AutomataLearn benchmarks and model profiling.

Applies a unified dark-mode template (Linear / Weights & Biases style) with transparent
backgrounds, clean gridlines, shared colorway, and publication-ready typography.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Distinct Linear/W&B color palette for all model architectures and variants
MODEL_COLOR_MAP = {
    # Automata
    "Ground-Truth DFA": "#10B981",              # Neon Emerald
    "RPNI Learned DFA": "#06B6D4",              # Cyan
    "L* Learned DFA": "#3B82F6",                # Bright Blue
    # Basic variants
    "Decision Tree (Basic)": "#F59E0B",         # Amber
    "Random Forest (Basic)": "#8B5CF6",         # Purple
    "Logistic Regression (Basic)": "#EC4899",   # Pink
    # Generic variants
    "Decision Tree (Generic)": "#F97316",       # Bright Orange
    "Random Forest (Generic)": "#6366F1",       # Indigo
    "Logistic Regression (Generic)": "#E11D48", # Crimson / Dark Rose
    # Oracle variants
    "Decision Tree (Oracle Leaky)": "#14B8A6",  # Teal
    "Random Forest (Oracle Leaky)": "#A855F7",  # Violet
    "Logistic Regression (Oracle Leaky)": "#F43F5E", # Coral Red
    # Fallback base names
    "Decision Tree": "#F59E0B",
    "Random Forest": "#8B5CF6",
    "Logistic Regression": "#EC4899",
}

COLOR_PALETTE = MODEL_COLOR_MAP

DISTINCT_FALLBACK_COLORS = [
    "#10B981", "#06B6D4", "#3B82F6", "#F59E0B", "#8B5CF6", "#EC4899",
    "#F97316", "#6366F1", "#E11D48", "#14B8A6", "#A855F7", "#F43F5E",
    "#84CC16", "#EAB308", "#22D3EE", "#D946EF",
]


def get_model_color(model_name: str, index: int = 0) -> str:
    """Return a unique, distinct hex color for any model name or variant."""
    if model_name in MODEL_COLOR_MAP:
        return MODEL_COLOR_MAP[model_name]
    m_lower = model_name.lower()
    for k, c in MODEL_COLOR_MAP.items():
        if k.lower() in m_lower or m_lower in k.lower():
            return c
    return DISTINCT_FALLBACK_COLORS[index % len(DISTINCT_FALLBACK_COLORS)]


DARK_COLORWAY = [
    "#6C8CFF", "#10B981", "#F59E0B", "#EC4899", "#8B5CF6", "#06B6D4", "#3B82F6", "#14B8A6"
]


def create_dark_template() -> go.layout.Template:
    """Create a standardized dark theme template for all Plotly charts."""
    template = go.layout.Template()
    template.layout = go.Layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, -apple-system, BlinkMacSystemFont, sans-serif", color="#F3F4F6", size=12),
        colorway=DARK_COLORWAY,
        hoverlabel=dict(
            bgcolor="#111827",
            bordercolor="#374151",
            font=dict(family="Inter, sans-serif", color="#F9FAFB", size=12),
        ),
        xaxis=dict(
            gridcolor="#1F2937",
            zerolinecolor="#374151",
            tickfont=dict(color="#9CA3AF", size=11),
            title=dict(font=dict(color="#D1D5DB", size=12)),
        ),
        yaxis=dict(
            gridcolor="#1F2937",
            zerolinecolor="#374151",
            tickfont=dict(color="#9CA3AF", size=11),
            title=dict(font=dict(color="#D1D5DB", size=12)),
        ),
        legend=dict(
            font=dict(color="#D1D5DB", size=11),
            bgcolor="rgba(17, 24, 39, 0.7)",
            bordercolor="#1F2937",
            borderwidth=1,
        ),
    )
    return template


DARK_TEMPLATE = create_dark_template()


def plot_in_dist_vs_gen(entries: List[Any], language_name: str) -> go.Figure:
    """Grouped bar chart comparing In-Distribution vs Length-Generalization accuracy."""
    models = [e.model_name for e in entries]
    in_dist = [e.in_dist_acc * 100 for e in entries]
    gen_acc = [e.gen_acc * 100 for e in entries]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=models,
        y=in_dist,
        name="In-Distribution Test Acc (%)",
        marker_color="#6C8CFF",
        text=[f"{v:.1f}%" for v in in_dist],
        textposition="outside",
        textfont=dict(color="#D1D5DB", size=11),
    ))
    fig.add_trace(go.Bar(
        x=models,
        y=gen_acc,
        name="Length-Generalization Acc (%)",
        marker_color="#EC4899",
        text=[f"{v:.1f}%" for v in gen_acc],
        textposition="outside",
        textfont=dict(color="#D1D5DB", size=11),
    ))

    fig.update_layout(
        template=DARK_TEMPLATE,
        title=dict(text=f"Generalization Gap: In-Distribution vs Length-Generalization ({language_name})", font=dict(size=14, color="#F3F4F6")),
        barmode="group",
        bargap=0.25,
        bargroupgap=0.1,
        yaxis=dict(title="Accuracy (%)", range=[0, 115]),
        xaxis=dict(title="Model Architecture"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=420,
        margin=dict(l=40, r=40, t=70, b=40),
    )
    return fig


def plot_learning_curves(df_curve: pd.DataFrame, title: str = "Learning Curve: Accuracy vs Training Sample Size") -> go.Figure:
    """Plot accuracy progression across sample sizes for in-dist and generalization."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("In-Distribution Accuracy", "Length-Generalization Accuracy"),
    )

    models = df_curve["model"].unique()
    for idx, model in enumerate(models):
        sub = df_curve[df_curve["model"] == model].sort_values("sample_size")
        color = get_model_color(model, idx)

        # In-dist
        fig.add_trace(
            go.Scatter(
                x=sub["sample_size"],
                y=sub["in_dist_acc"] * 100,
                mode="lines+markers",
                name=model,
                line=dict(color=color, width=2.5),
                marker=dict(size=6),
                legendgroup=model,
            ),
            row=1, col=1,
        )

        # Generalization
        fig.add_trace(
            go.Scatter(
                x=sub["sample_size"],
                y=sub["gen_acc"] * 100,
                mode="lines+markers",
                name=model,
                line=dict(color=color, width=2.5, dash="dot"),
                marker=dict(size=6),
                legendgroup=model,
                showlegend=False,
            ),
            row=1, col=2,
        )

    fig.update_layout(
        template=DARK_TEMPLATE,
        title_text=title,
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=-0.32, xanchor="center", x=0.5),
        margin=dict(l=40, r=40, t=60, b=80),
    )
    fig.update_yaxes(title_text="Accuracy (%)", range=[30, 105], row=1, col=1)
    fig.update_yaxes(title_text="Accuracy (%)", range=[30, 105], row=1, col=2)
    fig.update_xaxes(title_text="Training Sample Count (N)", row=1, col=1)
    fig.update_xaxes(title_text="Training Sample Count (N)", row=1, col=2)
    return fig


def plot_noise_robustness(df_noise: pd.DataFrame, title: str = "Noise Robustness: Accuracy vs Label Noise Rate") -> go.Figure:
    """Plot test accuracy vs label noise flipping percentage."""
    fig = go.Figure()
    models = df_noise["model"].unique()

    for idx, model in enumerate(models):
        sub = df_noise[df_noise["model"] == model].sort_values("noise_rate")
        color = get_model_color(model, idx)
        fig.add_trace(go.Scatter(
            x=sub["noise_rate"] * 100,
            y=sub["in_dist_acc"] * 100,
            mode="lines+markers",
            name=model,
            line=dict(color=color, width=2.5),
            marker=dict(size=7),
        ))

    fig.update_layout(
        template=DARK_TEMPLATE,
        title=dict(text=title, font=dict(size=14, color="#F3F4F6")),
        xaxis=dict(title="Label Noise Flipping Rate (%)", dtick=5),
        yaxis=dict(title="In-Distribution Accuracy (%)", range=[30, 105]),
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=70, b=40),
    )
    return fig


def plot_runtime_comparison(entries: List[Any]) -> go.Figure:
    """Side-by-side comparison of training time and per-string inference latency."""
    models = [e.model_name for e in entries]
    train_times = [e.train_time_ms for e in entries]
    infer_times = [e.infer_time_us for e in entries]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Training Duration (ms, log-scale)", "Inference Latency (µs per string)"),
    )

    fig.add_trace(
        go.Bar(
            x=models,
            y=[max(0.01, t) for t in train_times],
            marker_color="#8B5CF6",
            text=[f"{t:.1f}ms" for t in train_times],
            textposition="outside",
            textfont=dict(color="#D1D5DB", size=10),
            name="Train Time",
            showlegend=False,
        ),
        row=1, col=1,
    )
    fig.update_yaxes(type="log", title_text="Duration (ms, log)", row=1, col=1)

    fig.add_trace(
        go.Bar(
            x=models,
            y=infer_times,
            marker_color="#06B6D4",
            text=[f"{t:.1f}µs" for t in infer_times],
            textposition="outside",
            textfont=dict(color="#D1D5DB", size=10),
            name="Infer Latency",
            showlegend=False,
        ),
        row=1, col=2,
    )
    fig.update_yaxes(title_text="Latency (µs)", row=1, col=2)

    fig.update_layout(
        template=DARK_TEMPLATE,
        height=380,
        margin=dict(l=40, r=40, t=50, b=50),
    )
    return fig


def plot_complexity_vs_accuracy(entries: List[Any]) -> go.Figure:
    """Scatter plot showing Model Complexity vs Length-Generalization Accuracy."""
    data = []
    for e in entries:
        # Approximate numeric size parameter for plotting
        if "state" in e.complexity_desc.lower():
            size = e.state_count if e.state_count else 4
        elif "tree" in e.complexity_desc.lower() and "node" in e.complexity_desc.lower():
            # Extract node count
            try:
                size = int(e.complexity_desc.split("nodes")[0].split(",")[-1].strip())
            except Exception:
                size = 5
        elif "trees" in e.complexity_desc.lower():
            size = 100
        else:
            size = 62

        data.append({
            "model": e.model_name,
            "category": e.model_category,
            "gen_acc": e.gen_acc * 100,
            "in_acc": e.in_dist_acc * 100,
            "size_metric": size,
            "desc": e.complexity_desc,
        })

    df = pd.DataFrame(data)
    fig = go.Figure()

    categories = [
        ("Ground Truth", "#10B981"),
        ("Automata Learning", "#06B6D4"),
        ("Machine Learning", "#F59E0B"),
        ("Machine Learning (Oracle)", "#EC4899"),
    ]

    for cat, color in categories:
        sub = df[df["category"] == cat]
        if not sub.empty:
            custom_data = np.stack(
                (sub["model"], sub["desc"], sub["in_acc"], sub["gen_acc"]),
                axis=-1,
            )
            fig.add_trace(go.Scatter(
                x=sub["size_metric"],
                y=sub["gen_acc"],
                mode="markers",
                name=cat,
                marker=dict(size=14, color=color, line=dict(color="#FFFFFF", width=1.5)),
                customdata=custom_data,
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Complexity: %{customdata[1]}<br>"
                    "In-Dist Acc: %{customdata[2]:.1f}%<br>"
                    "Length-Gen Acc: %{customdata[3]:.1f}%"
                    "<extra></extra>"
                ),
            ))

    y_min = max(0.0, float(df["gen_acc"].min() - 5)) if not df.empty else 40.0
    fig.update_layout(
        template=DARK_TEMPLATE,
        title=dict(text="Model Size / Structural Complexity vs Length-Generalization Accuracy", font=dict(size=14, color="#F3F4F6")),
        xaxis=dict(
            title="Structural Size Parameter (Note: Mixed units — DFA states / Tree nodes / Coefficients on log scale)",
            type="log",
        ),
        yaxis=dict(title="Length-Generalization Accuracy (%)", range=[y_min, 108]),
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=70, b=40),
    )
    return fig


def plot_feature_importance_bar(importances: Dict[str, float], top_k: int = 12) -> go.Figure:
    """Horizontal bar chart showing top feature importances."""
    items = list(importances.items())[:top_k]
    if not items:
        fig = go.Figure()
        fig.update_layout(template=DARK_TEMPLATE, title="No feature importances available.")
        return fig

    # Reverse for top-down display
    names = [k for k, _ in reversed(items)]
    scores = [v for _, v in reversed(items)]

    fig = go.Figure(go.Bar(
        x=scores,
        y=names,
        orientation="h",
        marker_color="#6C8CFF",
        text=[f"{s:.3f}" for s in scores],
        textposition="outside",
        textfont=dict(color="#D1D5DB", size=10),
    ))
    fig.update_layout(
        template=DARK_TEMPLATE,
        title=dict(text=f"Top {len(items)} Feature Importances", font=dict(size=14, color="#F3F4F6")),
        xaxis=dict(title="Importance Weight"),
        yaxis=dict(title="Feature"),
        height=max(320, len(items) * 28),
        margin=dict(l=140, r=40, t=50, b=40),
    )
    return fig


def plot_confusion_matrix_heatmap(
    cm: np.ndarray,
    title: str = "Confusion Matrix",
    labels: Optional[List[str]] = None,
) -> go.Figure:
    """Interactive dark-mode heatmap representing confusion matrix."""
    labels = labels or ["Reject (0)", "Accept (1)"]
    z = cm.tolist()
    text = [[str(v) for v in row] for row in z]

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=[f"Pred: {lbl}" for lbl in labels],
        y=[f"True: {lbl}" for lbl in labels],
        text=text,
        texttemplate="%{text}",
        textfont={"size": 15, "color": "#FFFFFF"},
        colorscale=[
            [0.0, "#0F172A"],
            [0.5, "#1E3A8A"],
            [1.0, "#3B82F6"],
        ],
        showscale=False,
    ))
    fig.update_layout(
        template=DARK_TEMPLATE,
        title=dict(text=title, font=dict(size=13, color="#F3F4F6")),
        height=300,
        width=340,
        margin=dict(l=60, r=40, t=50, b=40),
    )
    return fig
