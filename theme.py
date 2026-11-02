"""Visual theme: a dark 'control room' look with neon data colours.

Change the colours in COLORS and everything (charts, cards, CSS) follows.
"""
import copy
from datetime import datetime

import gradio as gr
import plotly.graph_objects as go
import plotly.io as pio

COLORS = {
    "bg": "#0b0f17",        # page background
    "panel": "#111723",     # cards and panels
    "panel2": "#161e2d",    # raised surfaces
    "line": "#232d40",      # borders and grid
    "text": "#e6eaf2",
    "muted": "#8b97ab",
    "cyan": "#5b9cf6",      # the one accent colour (soft blue)
    "blue": "#3b6fb6",      # darker blue
    "green": "#3fb68b",     # only used for the small live dot
    "amber": "#c3cbd9",     # light grey-blue for secondary lines
    "rose": "#e5675f",      # muted red, reserved for anomalies / alerts
    "violet": "#8da2c0",    # slate
}
C = COLORS
COLORWAY = [C["cyan"], "#8da2c0", C["blue"], "#c3cbd9", "#5f6f8a", "#2f4a73"]  # blues and greys, no rainbow
SCALE = ["#16233a", "#2f5f9f", C["cyan"]]            # dark blue to blue
SCALE_RED = ["#1d1518", "#7a2f2b", C["rose"]]        # missing-data style charts
DIVERGE = [[0, C["blue"]], [0.5, "#1a2233"], [1, C["rose"]]]  # below / above expected

PLOT_TEMPLATE = "plotly_dark"
try:  # custom template: neon colours, dark grid, glowing hover labels
    _tpl = copy.deepcopy(pio.templates["plotly_dark"])
    _axis = dict(gridcolor="#1a2233", zerolinecolor=C["line"], linecolor=C["line"], tickfont=dict(color=C["muted"]))
    _tpl.layout.update(
        font=dict(color="#cbd5e1", family="Inter, system-ui, Segoe UI, sans-serif"),
        colorway=COLORWAY,
        xaxis=_axis, yaxis=_axis,
        hoverlabel=dict(bgcolor=C["panel"], bordercolor=C["cyan"], font=dict(color=C["text"])),
        title=dict(font=dict(size=16, color=C["text"])),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    pio.templates["live_dark"] = _tpl
    PLOT_TEMPLATE = "live_dark"
except Exception:  # fall back to the stock dark template
    pass

CSS = f"""
:root {{ --bg:{C['bg']}; --panel:{C['panel']}; --panel2:{C['panel2']}; --line:{C['line']};
        --text:{C['text']}; --muted:{C['muted']}; --accent:{C['cyan']}; --green:{C['green']}; --rose:{C['rose']}; }}
body, .gradio-container {{ background: var(--bg) !important; color: var(--text) !important; }}
.gradio-container {{ max-width: 1500px !important; }}
.block, .form {{ background: var(--panel) !important; border: 1px solid var(--line) !important;
                border-radius: 10px !important; box-shadow: none !important; }}
.prose, .prose p, .prose li, .md {{ color: var(--text) !important; }}
.prose h3, .prose h4 {{ color: var(--text) !important; font-weight: 600; }}
.prose code {{ background: var(--panel2) !important; color: #c3cbd9 !important; border-radius: 6px; }}
label span, .block-title, .block-label {{ color: var(--muted) !important; }}

input, textarea, select, .wrap {{ background: var(--bg) !important; color: var(--text) !important; }}
input, textarea {{ border: 1px solid var(--line) !important; border-radius: 8px !important; }}
input:focus, textarea:focus {{ border-color: var(--accent) !important; box-shadow: none !important; }}
input[type=range] {{ accent-color: var(--accent); }}

button.primary {{ background: var(--accent) !important; border: none !important; color: #0b0f17 !important;
                 font-weight: 600 !important; box-shadow: none !important; }}
button.primary:hover {{ filter: brightness(1.1); }}
button.secondary {{ background: var(--panel2) !important; border: 1px solid var(--line) !important; color: var(--text) !important; }}

button[role="tab"], .tab-nav button {{ color: var(--muted) !important; font-weight: 500 !important;
                                      background: transparent !important; border: none !important; }}
button[role="tab"][aria-selected="true"], .tab-nav button.selected {{
  color: var(--text) !important; box-shadow: inset 0 -2px 0 var(--accent) !important; }}

table {{ font-family: Consolas, 'Courier New', monospace !important; font-size: 12.5px !important; }}
thead th, th {{ background: var(--panel2) !important; color: var(--muted) !important; }}
tbody tr:nth-child(even) {{ background: rgba(255,255,255,.02) !important; }}
td {{ color: var(--text) !important; }}
::-webkit-scrollbar {{ width: 8px; height: 8px; }}
::-webkit-scrollbar-thumb {{ background: #2a3650; border-radius: 8px; }}

.hdr {{ display:flex; justify-content:space-between; align-items:center; padding:14px 18px; margin-bottom:6px;
       border:1px solid var(--line); border-radius:10px; background: var(--panel); }}
.hdr-title {{ font-size:22px; font-weight:700; color: var(--text); }}
.hdr-sub {{ color: var(--muted); font-size:13px; margin-top:2px; }}
.pill {{ display:inline-flex; align-items:center; padding:5px 12px; border-radius:999px; font-weight:600;
        font-size:11px; letter-spacing:1.2px; color: var(--muted); border:1px solid var(--line); }}

@keyframes pulse {{ 0% {{ box-shadow:0 0 0 0 rgba(63,182,139,.55); }} 70% {{ box-shadow:0 0 0 7px rgba(63,182,139,0); }}
                   100% {{ box-shadow:0 0 0 0 rgba(63,182,139,0); }} }}
.live-dot, .wait-dot {{ display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:8px; }}
.live-dot {{ background: var(--green); animation: pulse 2s infinite; }}
.wait-dot {{ background: #6b7688; }}
.strip {{ display:flex; align-items:center; color: var(--muted); font-size:12px; letter-spacing:.3px;
         margin: 2px 0 10px 2px; font-family: Consolas, 'Courier New', monospace; }}

.kpi-row {{ display:flex; gap:10px; flex-wrap:wrap; }}
.kpi {{ flex:1; min-width:150px; padding:12px 16px; border-radius:10px; border:1px solid var(--line);
       border-left:3px solid var(--accent); background: var(--panel); }}
.kpi-title {{ color: var(--muted); font-size:11px; font-weight:600; letter-spacing:1px; text-transform:uppercase; }}
.kpi-value {{ color: var(--value, var(--text)); font-size:26px; font-weight:700; margin-top:2px;
             font-family: Consolas, 'Courier New', monospace; }}
.kpi-sub {{ color: var(--muted); font-size:12px; min-height:16px; }}
"""
JS = "() => { document.body.classList.add('dark'); }"


def build_theme():
    base = gr.themes.Base(primary_hue="blue", secondary_hue="slate", neutral_hue="slate")
    try:  # the exact variable names differ a little between Gradio versions
        return base.set(
            body_background_fill=C["bg"], body_background_fill_dark=C["bg"],
            block_background_fill=C["panel"], block_background_fill_dark=C["panel"],
            block_border_color=C["line"], block_border_color_dark=C["line"],
            input_background_fill="#070e20", input_background_fill_dark="#070e20",
        )
    except Exception:
        return base


# ---------------------------------------------------------------- HTML parts
def header_html(title, subtitle):
    return (f"<div class='hdr'><div><div class='hdr-title'>{title}</div>"
            f"<div class='hdr-sub'>{subtitle}</div></div>"
            f"<div class='pill'><span class='live-dot'></span>LIVE VIEW</div></div>")


def card(title, value, sub="", color=None):
    """All cards share one accent; only a card passed the alert colour turns red."""
    alert = color == C["rose"]
    accent = C["rose"] if alert else C["cyan"]
    value_color = C["rose"] if alert else C["text"]
    return (f"<div class='kpi' style='--accent:{accent};--value:{value_color}'>"
            f"<div class='kpi-title'>{title}</div><div class='kpi-value'>{value}</div>"
            f"<div class='kpi-sub'>{sub}</div></div>")


def strip(rows=None):
    """Status line above the KPI cards: green pulsing dot when data is loaded, amber when waiting."""
    if rows is None:
        return "<div class='strip'><span class='wait-dot'></span>WAITING FOR DATA · import a file to start</div>"
    return (f"<div class='strip'><span class='live-dot'></span>LIVE VIEW · {rows:,} rows analysed · "
            f"refreshed {datetime.now():%H:%M:%S}</div>")


def kpi_row(*cards):
    return f"<div class='kpi-row'>{''.join(cards)}</div>"