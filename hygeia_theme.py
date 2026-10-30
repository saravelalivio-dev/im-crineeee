"""
hygeia_theme.py
===============
Visual layer: the pink design system, the CSS injected into Streamlit, and the
Plotly template every chart in the app inherits.

Palette provenance
------------------
The categorical slots below were validated with a colour-vision-deficiency
checker before being used: all six sit inside the L 0.43-0.77 lightness band,
clear the chroma floor, hold >= 3:1 contrast against the chart surface, and keep
adjacent-pair separation above the CVD floor. Rose is slot 1 so the brand hue
carries the primary series; the companion hues are deliberately *different hues*
rather than more pinks, because a pink-only categorical ramp is unreadable to a
deuteranopic reader. Every multi-series chart additionally carries a legend and
direct labels, so identity is never communicated by colour alone.

Magnitude is encoded with ROSE_SEQUENTIAL, a single-hue light-to-dark ramp.
Polarity (SHAP phi, which is signed) uses one warm and one cool pole with a
neutral grey midpoint - never a rainbow.
"""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

# ---------------------------------------------------------------------------
# TOKENS
# ---------------------------------------------------------------------------

INK = "#4A2C3A"          # primary text - warm plum-brown, readable on pink
INK_SOFT = "#7A5A69"     # secondary text
INK_MUTED = "#A98B99"    # axis labels, gridline text
SURFACE = "#FFFFFF"      # card surface
SURFACE_TINT = "#FFF5F9" # page surface
BLUSH = "#FFE4EF"        # soft fills, chips
PETAL = "#FFD0E4"        # borders
ROSE = "#D81B74"         # primary brand / accent
ROSE_DEEP = "#A4134E"    # headings, emphasis
ROSE_LIGHT = "#FF8FBF"   # hover, highlights

# Validated categorical order. Assign in this order; never cycle or recolour
# by rank - a filter that drops a series must not repaint the survivors.
CATEGORICAL = [
    "#D81B74",  # rose        - primary series, brand hue
    "#1F9E9E",  # teal        - contrast partner
    "#8E44C8",  # violet
    "#B8651A",  # amber-brown
    "#4A5BC4",  # indigo
    "#A8327A",  # plum
]

# Single-hue ramp for magnitude (light -> dark, monotone lightness).
ROSE_SEQUENTIAL = [
    "#FFE1EF", "#FFC2DC", "#F58FBB", "#E8508F", "#C2185B", "#8E0F3F",
]

# Signed values: warm pole / neutral midpoint / cool pole.
DIVERGING = ["#1F9E9E", "#8FCFCF", "#E8E4E6", "#F58FBB", "#D81B74"]

# Reserved status colours. Never reused as "series 7"; always shipped with a
# label or icon so state is not colour-alone.
STATUS = {
    "good": "#1F8A5F",
    "warning": "#B8651A",
    "serious": "#C2185B",
    "critical": "#8E0F3F",
}

GRID = "#F7DCE8"

FONT_STACK = "'Quicksand', 'Poppins', 'Segoe UI', system-ui, sans-serif"


# ---------------------------------------------------------------------------
# PLOTLY TEMPLATE
# ---------------------------------------------------------------------------

def register_template(name: str = "hygeia_pink") -> str:
    """
    Register and activate the shared Plotly template.

    Centralising this means no chart in the app sets its own fonts, grid colour
    or margins - they inherit, so the whole dashboard reads as one system.
    """
    template = go.layout.Template()
    template.layout = go.Layout(
        colorway=CATEGORICAL,
        font=dict(family=FONT_STACK, size=13, color=INK),
        title=dict(font=dict(size=17, color=ROSE_DEEP), x=0.01, xanchor="left"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=62, r=28, t=52, b=52),
        xaxis=dict(
            gridcolor=GRID, zerolinecolor=PETAL, linecolor=PETAL,
            tickfont=dict(color=INK_MUTED, size=12),
            title=dict(font=dict(color=INK_SOFT, size=13)),
        ),
        yaxis=dict(
            gridcolor=GRID, zerolinecolor=PETAL, linecolor=PETAL,
            tickfont=dict(color=INK_MUTED, size=12),
            title=dict(font=dict(color=INK_SOFT, size=13)),
        ),
        legend=dict(
            bgcolor="rgba(255,255,255,0.75)", bordercolor=PETAL, borderwidth=1,
            font=dict(size=12, color=INK), orientation="h",
            yanchor="bottom", y=1.02, xanchor="right", x=1,
        ),
        hoverlabel=dict(
            bgcolor=SURFACE, bordercolor=ROSE,
            font=dict(family=FONT_STACK, size=12, color=INK),
        ),
        colorscale=dict(sequential=ROSE_SEQUENTIAL, diverging=DIVERGING),
    )
    pio.templates[name] = template
    pio.templates.default = name
    return name


def style_3d(fig: go.Figure) -> go.Figure:
    """Apply the pink scene styling to a 3-D figure (templates miss scenes)."""
    axis = dict(
        backgroundcolor="rgba(255,245,249,0.55)",
        gridcolor=GRID, zerolinecolor=PETAL,
        showbackground=True,
        tickfont=dict(color=INK_MUTED, size=10),
        title=dict(font=dict(color=INK_SOFT, size=12)),
    )
    fig.update_layout(scene=dict(xaxis=axis, yaxis=axis, zaxis=axis))
    return fig


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Quicksand:wght@400;500;600;700&family=Poppins:wght@300;400;600&display=swap');

html, body, [class*="css"], .stMarkdown, .stMetric {{
    font-family: {FONT_STACK};
    color: {INK};
}}

/* Page surface: a soft vertical blush wash. */
.stApp {{
    background: linear-gradient(170deg, #FFF7FB 0%, #FFEDF5 42%, #FFE6F1 100%);
}}
.block-container {{ padding-top: 1.6rem; max-width: 1340px; }}

/* ---------- Hero banner ---------- */
.hygeia-hero {{
    background: linear-gradient(118deg, {ROSE} 0%, #E8508F 48%, #FF9EC4 100%);
    border-radius: 26px;
    padding: 26px 32px;
    color: #FFFFFF;
    box-shadow: 0 12px 34px rgba(216, 27, 116, 0.26);
    margin-bottom: 20px;
}}
.hygeia-hero h1 {{
    font-size: 2.35rem; font-weight: 700; margin: 0 0 4px 0;
    letter-spacing: 0.055em; color: #FFFFFF;
}}
.hygeia-hero .sub {{
    font-size: 1rem; font-weight: 500; opacity: 0.96; line-height: 1.5;
    max-width: 78ch;
}}
.hygeia-hero .byline {{
    font-size: 0.82rem; margin-top: 11px; opacity: 0.9;
    font-family: 'Poppins', sans-serif; font-weight: 300;
}}

/* ---------- Cards ---------- */
.hygeia-card {{
    background: {SURFACE};
    border: 1px solid {PETAL};
    border-radius: 20px;
    padding: 20px 24px;
    box-shadow: 0 5px 20px rgba(216, 27, 116, 0.09);
    margin-bottom: 15px;
}}
.hygeia-card h3 {{
    margin: 0 0 10px 0; font-size: 1.04rem; font-weight: 700;
    color: {ROSE_DEEP}; letter-spacing: 0.015em;
}}
.hygeia-card p {{ color: {INK_SOFT}; font-size: 0.9rem; line-height: 1.62; margin: 0.4rem 0; }}

.hygeia-verdict {{
    background: linear-gradient(135deg, {BLUSH} 0%, #FFF0F7 100%);
    border: 1.5px solid {PETAL};
    border-left: 6px solid {ROSE};
    border-radius: 18px;
    padding: 18px 22px;
    margin-bottom: 14px;
}}
.hygeia-verdict .big {{
    font-size: 2.5rem; font-weight: 700; color: {ROSE_DEEP};
    line-height: 1.05; margin: 2px 0;
}}
.hygeia-verdict .cap {{
    font-size: 0.73rem; text-transform: uppercase; letter-spacing: 0.13em;
    color: {INK_SOFT}; font-weight: 600;
}}

/* Monospace sequence display, wrapping cleanly at any width. */
.hygeia-seq {{
    font-family: 'SFMono-Regular', 'Consolas', monospace;
    font-size: 0.93rem; letter-spacing: 0.1em; word-break: break-all;
    background: {SURFACE_TINT}; border: 1px dashed {PETAL};
    border-radius: 12px; padding: 11px 14px; color: {ROSE_DEEP};
}}

/* ---------- Chips ---------- */
.chip {{
    display: inline-block; padding: 4px 13px; border-radius: 999px;
    font-size: 0.75rem; font-weight: 600; margin: 3px 5px 3px 0;
    border: 1px solid {PETAL}; background: {BLUSH}; color: {ROSE_DEEP};
}}
.chip-pass {{ background: #E6F6EE; border-color: #B6E2CC; color: #15694A; }}
.chip-fail {{ background: #FDE8EF; border-color: #F6C2D6; color: #93123F; }}
.chip-warn {{ background: #FBEEDF; border-color: #F0D4AE; color: #8A4A11; }}
.chip-lead {{ background: {ROSE}; border-color: {ROSE_DEEP}; color: #FFFFFF; }}

/* ---------- Tabs ---------- */
.stTabs [data-baseweb="tab-list"] {{ gap: 8px; background: transparent; }}
.stTabs [data-baseweb="tab"] {{
    background: rgba(255,255,255,0.72);
    border: 1px solid {PETAL};
    border-radius: 15px 15px 0 0;
    padding: 11px 20px; font-weight: 600; font-size: 0.9rem;
    color: {INK_SOFT};
}}
.stTabs [aria-selected="true"] {{
    background: {ROSE} !important; color: #FFFFFF !important;
    border-color: {ROSE_DEEP} !important;
}}

/* ---------- Metrics ---------- */
[data-testid="stMetric"] {{
    background: {SURFACE}; border: 1px solid {PETAL};
    border-radius: 17px; padding: 14px 17px;
    box-shadow: 0 3px 13px rgba(216, 27, 116, 0.08);
}}
[data-testid="stMetricLabel"] {{ color: {INK_SOFT} !important; font-weight: 600; font-size: 0.78rem; }}
[data-testid="stMetricValue"] {{ color: {ROSE_DEEP} !important; font-weight: 700; }}
[data-testid="stMetricDelta"] {{ font-size: 0.78rem; }}

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {{
    background: linear-gradient(190deg, #FFFFFF 0%, #FFF0F7 100%);
    border-right: 1px solid {PETAL};
}}
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {{ color: {ROSE_DEEP}; }}

/* ---------- Controls ---------- */
.stButton > button {{
    background: linear-gradient(118deg, {ROSE} 0%, #E8508F 100%);
    color: #FFFFFF; border: none; border-radius: 13px;
    padding: 0.56rem 1.5rem; font-weight: 700; font-size: 0.9rem;
    box-shadow: 0 5px 16px rgba(216, 27, 116, 0.3);
    transition: transform 0.14s ease, box-shadow 0.14s ease;
}}
.stButton > button:hover {{
    transform: translateY(-1.5px);
    box-shadow: 0 8px 22px rgba(216, 27, 116, 0.4);
}}
.stSlider [data-baseweb="slider"] [role="slider"] {{ background: {ROSE} !important; }}
.stExpander {{
    border: 1px solid {PETAL} !important; border-radius: 15px !important;
    background: rgba(255,255,255,0.76);
}}
.stDataFrame {{ border: 1px solid {PETAL}; border-radius: 14px; overflow: hidden; }}
.stAlert {{ border-radius: 15px; }}

/* Section rule with a small rose diamond. */
.hygeia-rule {{
    display: flex; align-items: center; gap: 11px;
    margin: 22px 0 13px 0;
}}
.hygeia-rule span {{
    font-size: 0.73rem; text-transform: uppercase; letter-spacing: 0.17em;
    color: {ROSE_DEEP}; font-weight: 700; white-space: nowrap;
}}
.hygeia-rule:after {{
    content: ''; flex: 1; height: 1.5px;
    background: linear-gradient(90deg, {PETAL}, rgba(255,208,228,0));
}}

footer, #MainMenu {{ visibility: hidden; }}
</style>
"""


def inject(st) -> None:
    """Inject the stylesheet and activate the Plotly template. Call once, first."""
    st.markdown(CSS, unsafe_allow_html=True)
    register_template()


# ---------------------------------------------------------------------------
# SMALL HTML HELPERS
# ---------------------------------------------------------------------------

def rule(st, label: str) -> None:
    """Labelled section divider."""
    st.markdown(f'<div class="hygeia-rule"><span>{label}</span></div>',
                unsafe_allow_html=True)


def card(st, title: str, body_html: str) -> None:
    """White rounded card with a rose heading."""
    st.markdown(
        f'<div class="hygeia-card"><h3>{title}</h3>{body_html}</div>',
        unsafe_allow_html=True,
    )


def chip(label: str, kind: str = "") -> str:
    """Return a pill span. ``kind`` in {'', 'pass', 'fail', 'warn', 'lead'}."""
    cls = f"chip chip-{kind}" if kind else "chip"
    return f'<span class="{cls}">{label}</span>'
