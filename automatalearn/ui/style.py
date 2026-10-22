"""Shared professional dark-theme stylesheet (Linear / Weights & Biases style) for AutomataLearn."""

from __future__ import annotations
import streamlit as st


def apply_custom_css() -> None:
    """Inject polished Linear/W&B-inspired CSS styling into the Streamlit app."""
    css = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    /* Header styling: transparent 2.5rem bar with visible sidebar toggle */
    header[data-testid="stHeader"] {
        height: 2.5rem !important;
        background: transparent !important;
        visibility: visible !important;
        display: block !important;
        z-index: 1000 !important;
    }

    /* Reduce block-container top padding to eliminate empty gap above title */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
        padding-left: 2.5rem !important;
        padding-right: 2.5rem !important;
        max-width: 98% !important;
    }

    /* Global base styling */
    html, body {
        color: #F3F4F6;
    }

    /* Scope Inter font-family to text elements only */
    p, label, h1, h2, h3, h4, li, td, th, input, textarea, button,
    [data-testid="stMarkdownContainer"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }

    code, kbd, samp, pre, [class*="stCode"] {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Restore Streamlit Material Symbols icon font */
    [data-testid="stIconMaterial"],
    span[class*="material"],
    .material-icons,
    .material-symbols-rounded,
    [data-testid="stExpandSidebarButton"] [data-testid="stIconMaterial"],
    [data-testid="stSidebarCollapsedControl"] [data-testid="stIconMaterial"],
    [data-testid="stSidebarCollapseButton"] [data-testid="stIconMaterial"],
    [data-testid="collapsedControl"] [data-testid="stIconMaterial"],
    [data-testid="stExpandSidebarButton"] span,
    [data-testid="stSidebarCollapsedControl"] span,
    [data-testid="stSidebarCollapseButton"] span,
    [data-testid="collapsedControl"] span {
        font-family: "Material Symbols Rounded" !important;
        font-feature-settings: "liga" !important;
        -webkit-font-feature-settings: "liga" !important;
        letter-spacing: normal !important;
        text-transform: none !important;
    }

    /* Sidebar collapse & expand controls: visible, clickable, light indigo, above other layers */
    [data-testid="stExpandSidebarButton"],
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="stSidebarCollapseButton"],
    [data-testid="collapsedControl"] {
        visibility: visible !important;
        display: flex !important;
        opacity: 1 !important;
        pointer-events: auto !important;
        cursor: pointer !important;
        z-index: 999999 !important;
        color: #818CF8 !important;
    }

    [data-testid="stExpandSidebarButton"] button,
    [data-testid="stSidebarCollapsedControl"] button,
    [data-testid="stSidebarCollapseButton"] button,
    [data-testid="collapsedControl"] button {
        visibility: visible !important;
        color: #818CF8 !important;
        background: transparent !important;
        border: none !important;
        cursor: pointer !important;
        pointer-events: auto !important;
    }

    [data-testid="stExpandSidebarButton"]:hover,
    [data-testid="stSidebarCollapsedControl"]:hover,
    [data-testid="stSidebarCollapseButton"]:hover,
    [data-testid="collapsedControl"]:hover,
    [data-testid="stExpandSidebarButton"] button:hover,
    [data-testid="stSidebarCollapsedControl"] button:hover,
    [data-testid="stSidebarCollapseButton"] button:hover,
    [data-testid="collapsedControl"] button:hover {
        color: #C7D2FE !important;
    }

    /* Reduced weight for section subheadings: ~1.15rem, muted slate */
    h2, h3, [data-testid="stMarkdownContainer"] h2, [data-testid="stMarkdownContainer"] h3 {
        font-size: 1.15rem !important;
        font-weight: 600 !important;
        color: #94A3B8 !important; /* Muted slate */
        letter-spacing: -0.01em !important;
        margin-top: 0.35rem !important;
        margin-bottom: 0.65rem !important;
    }

    h4, [data-testid="stMarkdownContainer"] h4 {
        font-size: 0.98rem !important;
        font-weight: 600 !important;
        color: #CBD5E1 !important;
        margin-bottom: 0.4rem !important;
    }

    /* Gradient page title */
    .dashboard-header-container {
        margin-bottom: 1.2rem;
    }

    .dashboard-title {
        font-size: 2.35rem !important;
        font-weight: 800 !important;
        letter-spacing: -0.03em !important;
        background: linear-gradient(135deg, #FFFFFF 0%, #C7D2FE 45%, #6C8CFF 100%) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        margin-bottom: 0.2rem !important;
        line-height: 1.15 !important;
    }

    .dashboard-subtitle {
        font-size: 0.96rem !important;
        font-weight: 400 !important;
        color: #94A3B8 !important;
        margin-bottom: 0.75rem !important;
    }

    /* Pill Badges Row */
    .badge-row {
        display: flex !important;
        flex-wrap: wrap !important;
        gap: 8px !important;
        margin-bottom: 1rem !important;
    }

    .pill-badge {
        display: inline-flex !important;
        align-items: center !important;
        font-size: 0.74rem !important;
        font-weight: 600 !important;
        padding: 3px 10px !important;
        border-radius: 9999px !important;
        border: 1px solid !important;
        letter-spacing: 0.02em !important;
        font-family: 'JetBrains Mono', monospace !important;
    }

    .badge-blue { background: rgba(108, 140, 255, 0.12); color: #818CF8; border-color: rgba(108, 140, 255, 0.3); }
    .badge-teal { background: rgba(6, 182, 212, 0.12); color: #22D3EE; border-color: rgba(6, 182, 212, 0.3); }
    .badge-emerald { background: rgba(16, 185, 129, 0.12); color: #34D399; border-color: rgba(16, 185, 129, 0.3); }
    .badge-purple { background: rgba(139, 92, 246, 0.12); color: #A78BFA; border-color: rgba(139, 92, 246, 0.3); }
    .badge-amber { background: rgba(245, 158, 11, 0.12); color: #FBBF24; border-color: rgba(245, 158, 11, 0.3); }
    .badge-slate { background: rgba(148, 163, 184, 0.12); color: #94A3B8; border-color: rgba(148, 163, 184, 0.3); }

    /* Pill-style Tabs */
    div[data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
        padding: 4px 0 16px 0;
        border-bottom: 1px solid #1F2937;
    }

    div[data-baseweb="tab-highlight"] {
        display: none !important;
    }

    button[data-baseweb="tab"] {
        border-radius: 9999px !important;
        padding: 6px 18px !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        background-color: #111827 !important;
        border: 1px solid #1F2937 !important;
        color: #9CA3AF !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }

    button[data-baseweb="tab"]:hover {
        background-color: #1F2937 !important;
        color: #F3F4F6 !important;
        border-color: #374151 !important;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, #6C8CFF 0%, #4F46E5 100%) !important;
        color: #FFFFFF !important;
        border-color: #6C8CFF !important;
        box-shadow: 0 4px 12px rgba(108, 140, 255, 0.35) !important;
        font-weight: 600 !important;
    }

    /* Fixed 128px KPI cards with flex column layout */
    .kpi {
        background-color: #111827 !important;
        border: 1px solid #1F2937 !important;
        border-radius: 12px !important;
        padding: 14px 16px !important;
        height: 128px !important;
        min-height: 128px !important;
        max-height: 128px !important;
        box-sizing: border-box !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: space-between !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.25), 0 2px 4px -2px rgba(0, 0, 0, 0.15) !important;
        transition: border-color 0.2s ease, transform 0.2s ease !important;
        margin-bottom: 0.5rem !important;
        user-select: none !important;
    }

    .kpi:hover {
        border-color: #374151 !important;
        transform: translateY(-1px) !important;
    }

    .kpi-label {
        font-size: 0.74rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.06em !important;
        color: #94A3B8 !important;
        line-height: 1.2 !important;
    }

    .kpi-value {
        font-size: 1.45rem !important;
        font-weight: 700 !important;
        color: #F9FAFB !important;
        font-family: 'JetBrains Mono', monospace !important;
        line-height: 1.2 !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        white-space: nowrap !important;
    }

    .kpi-pill {
        display: inline-flex !important;
        align-items: center !important;
        font-size: 0.72rem !important;
        font-weight: 500 !important;
        color: #94A3B8 !important;
        background: rgba(148, 163, 184, 0.1) !important;
        border: 1px solid rgba(148, 163, 184, 0.2) !important;
        border-radius: 9999px !important;
        padding: 2px 8px !important;
        width: fit-content !important;
        max-width: 100% !important;
        font-family: 'JetBrains Mono', monospace !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        white-space: nowrap !important;
    }

    .kpi-pill-hidden {
        visibility: hidden !important;
        height: 19px !important;
        border: none !important;
        background: transparent !important;
    }

    /* Hide heading anchor link icons */
    [data-testid="stHeaderActionElements"],
    .stHeadingAnchor {
        display: none !important;
    }

    /* Compact .kv label/value rows with fixed 120px label column */
    .kv-container {
        display: flex !important;
        flex-direction: column !important;
        gap: 6px !important;
        margin: 6px 0 10px 0 !important;
        width: 100% !important;
    }

    .kv-row {
        display: flex !important;
        align-items: flex-start !important;
        justify-content: flex-start !important;
        gap: 12px !important;
        padding: 5px 0 !important;
        border-bottom: 1px solid rgba(31, 41, 55, 0.6) !important;
        text-align: left !important;
    }

    .kv-row:last-child {
        border-bottom: none !important;
    }

    .kv-label {
        font-size: 0.72rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.06em !important;
        color: #94A3B8 !important;
        width: 120px !important;
        min-width: 120px !important;
        max-width: 120px !important;
        flex-shrink: 0 !important;
        text-align: left !important;
        padding-top: 2px !important;
    }

    .kv-text {
        font-size: 0.85rem !important;
        font-weight: 400 !important;
        color: #E2E8F0 !important;
        line-height: 1.4 !important;
        text-align: left !important;
        word-break: normal !important;
        font-family: 'Inter', -apple-system, sans-serif !important;
        flex-grow: 1 !important;
    }

    .kv-chip {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.78rem !important;
        font-weight: 500 !important;
        color: #C7D2FE !important;
        background: #1F2937 !important;
        border: 1px solid #374151 !important;
        padding: 2px 8px !important;
        border-radius: 6px !important;
        text-align: left !important;
        word-break: break-all !important;
        display: inline-block !important;
    }

    /* Card-style st.metric fallback */
    div[data-testid="stMetric"] {
        background-color: #111827 !important;
        border: 1px solid #1F2937 !important;
        border-radius: 12px !important;
        padding: 14px 18px !important;
        min-height: 108px !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: center !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.25), 0 2px 4px -2px rgba(0, 0, 0, 0.15) !important;
        transition: border-color 0.2s ease, transform 0.2s ease !important;
    }

    div[data-testid="stMetric"]:hover {
        border-color: #374151 !important;
        transform: translateY(-1px) !important;
    }

    div[data-testid="stMetricLabel"] {
        font-size: 0.76rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.06em !important;
        color: #94A3B8 !important;
    }

    div[data-testid="stMetricValue"] {
        font-size: 1.45rem !important;
        font-weight: 700 !important;
        color: #F9FAFB !important;
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Hide up/down arrows in metric delta and mute color to neutral slate */
    div[data-testid="stMetricDelta"] svg {
        display: none !important;
    }

    div[data-testid="stMetricDelta"] {
        color: #94A3B8 !important;
        font-size: 0.8rem !important;
        font-weight: 500 !important;
    }

    div[data-testid="stMetricDelta"] > div {
        color: #94A3B8 !important;
    }

    /* Gradient Primary Buttons */
    div.stButton > button[kind="primary"],
    button[kind="primary"] {
        background: linear-gradient(135deg, #6C8CFF 0%, #4F46E5 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        padding: 8px 18px !important;
        box-shadow: 0 4px 14px rgba(108, 140, 255, 0.35) !important;
        transition: all 0.2s ease !important;
    }

    div.stButton > button[kind="primary"]:hover,
    button[kind="primary"]:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(108, 140, 255, 0.5) !important;
        filter: brightness(1.08) !important;
    }

    div.stButton > button[kind="secondary"],
    button[kind="secondary"] {
        background-color: #111827 !important;
        border: 1px solid #374151 !important;
        border-radius: 8px !important;
        color: #E5E7EB !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
    }

    div.stButton > button[kind="secondary"]:hover,
    button[kind="secondary"]:hover {
        background-color: #1F2937 !important;
        border-color: #4B5563 !important;
    }

    /* Rounded DataFrames */
    div[data-testid="stDataFrame"] {
        border: 1px solid #1F2937 !important;
        border-radius: 12px !important;
        overflow: hidden !important;
        background-color: #111827 !important;
    }

    /* Container border styling */
    div[data-testid="stVerticalBlockBorderWrapper"] > div {
        border-color: #1F2937 !important;
        border-radius: 12px !important;
        background-color: #111827 !important;
        padding: 18px 22px !important;
    }

    /* Graphviz Diagram card centering, constrained to width: 100%, height: auto, max-height: 320px */
    div[data-testid="stGraphvizChart"] {
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        width: 100% !important;
        max-width: 100% !important;
        max-height: 320px !important;
        overflow: auto !important;
        padding: 4px 0 !important;
        text-align: center !important;
    }

    div[data-testid="stGraphvizChart"] svg {
        width: 100% !important;
        max-width: 100% !important;
        height: auto !important;
        max-height: 320px !important;
        display: block !important;
        margin: 0 auto !important;
    }

    /* Sidebar container styling */
    section[data-testid="stSidebar"] {
        background-color: #0B0F19 !important;
        border-right: 1px solid #1F2937 !important;
    }

    section[data-testid="stSidebar"] div[data-testid="stVerticalBlockBorderWrapper"] > div {
        background-color: #0F172A !important;
        border-color: #1E293B !important;
    }

    /* Callout & Alert & Exception Boxes */
    div[data-testid="stAlert"] {
        border-radius: 10px !important;
        border: 1px solid #1F2937 !important;
        background-color: #111827 !important;
    }

    [data-testid="stException"] {
        overflow-x: auto !important;
        max-width: 100% !important;
    }

    /* Ensure Streamlit toolbar containing the sidebar expand control remains visible */
    [data-testid="stToolbar"] {
        visibility: visible !important;
        background: transparent !important;
    }

    /* Hide only Deploy button, main menu, toolbar actions, and footer */
    [data-testid="stDeployButton"],
    [data-testid="stMainMenu"],
    #MainMenu,
    [data-testid="stToolbarActions"],
    footer {
        visibility: hidden !important;
        display: none !important;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def kpi(label: str, value: object, pill: str | None = None) -> str:
    """Render a fixed 128px height card KPI with flex column layout and identical structure."""
    pill_html = (
        f'<div class="kpi-pill">{pill}</div>'
        if pill
        else '<div class="kpi-pill kpi-pill-hidden">&nbsp;</div>'
    )
    return f"""<div class="kpi">
  <div class="kpi-label">{label}</div>
  <div class="kpi-value">{value}</div>
  {pill_html}
</div>"""


def kv_row(label: str, value: object, is_chip: bool | None = None) -> str:
    """Render a compact label-value row with fixed 120px label column and text or chip value."""
    if is_chip is None:
        is_chip = label.strip().lower() not in {"description", "summary", "notes"}
    val_class = "kv-chip" if is_chip else "kv-text"
    return f"""<div class="kv-row"><span class="kv-label">{label}</span><span class="{val_class}">{value}</span></div>"""


def kv_table(rows: list[tuple[str, object] | tuple[str, object, bool]]) -> str:
    """Render a list of KV rows inside a container with fixed 120px label columns."""
    elements = []
    for r in rows:
        if len(r) == 3:
            elements.append(kv_row(r[0], r[1], bool(r[2])))
        else:
            elements.append(kv_row(r[0], r[1]))
    inner = "\n".join(elements)
    return f"""<div class="kv-container">\n{inner}\n</div>"""
