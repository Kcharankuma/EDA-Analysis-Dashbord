"""E-commerce Orders Dashboard: import an Excel/CSV file and explore it (Gradio).

Run:  python app.py
No live feed: you upload the file yourself (.xlsx, .xls or .csv).
"""
import inspect
import os
import tempfile

import gradio as gr
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import advanced as adv
from anomaly import METHODS, detect
from loader import find_col, is_excel, prepare, read_any, sheet_names
from theme import COLORS as C, COLORWAY, CSS, JS, PLOT_TEMPLATE, SCALE, build_theme, card, header_html, kpi_row, strip

LAYOUT = dict(template=PLOT_TEMPLATE, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
              margin=dict(l=10, r=10, t=50, b=10), height=340)
APPEND = "Append to existing data"
REPLACE = "Replace existing data"


# ------------------------------------------------------------------ helpers
def empty_fig(msg="Upload an Excel or CSV file to begin"):
    fig = go.Figure()
    fig.update_layout(**LAYOUT, xaxis=dict(visible=False), yaxis=dict(visible=False),
                      annotations=[dict(text=msg, showarrow=False, font=dict(size=15, color="#94a3b8"))])
    return fig


def known(series):
    """Drop 'Unknown' placeholders so charts only show real values."""
    return series[~series.isin(["Unknown", "nan", "None", ""])]


def top_bar(series, title, n=10):
    s = known(series)
    if s.empty:
        return empty_fig(f"No data for: {title}")
    vc = s.value_counts().head(n).iloc[::-1].reset_index()
    vc.columns = ["label", "orders"]
    fig = px.bar(vc, x="orders", y="label", orientation="h", title=title,
                 color="orders", color_continuous_scale=SCALE)
    fig.update_layout(**LAYOUT, coloraxis_showscale=False, yaxis_title="", xaxis_title="Orders")
    return fig


def donut(series, title):
    s = known(series)
    if s.empty:
        return empty_fig(f"No data for: {title}")
    fig = px.pie(s.to_frame("label"), names="label", hole=0.55, title=title,
                 color_discrete_sequence=COLORWAY)
    fig.update_layout(**LAYOUT)
    return fig


def parse_path(file):
    return file if isinstance(file, str) else getattr(file, "name", str(file))


# ------------------------------------------------------------------ analysis
def apply_filters(df, provs, ampm, langs):
    if provs:
        df = df[df["cc_provider"].isin(provs)]
    if ampm:
        df = df[df["am_pm"].isin(ampm)]
    if langs:
        df = df[df["language"].isin(langs)]
    return df

def analyze(df, provs, ampm, langs, method_, z_thr_, cont_):
    if df is None or df.empty:
        blank = empty_fig()
        return (strip() + kpi_row(card("Orders", "0")), "", *[blank] * 10, pd.DataFrame(), pd.DataFrame())
    df = apply_filters(df, provs, ampm, langs)
    if df.empty:
        blank = empty_fig("No rows match the filters")
        return (strip() + kpi_row(card("Orders", "0")), "No rows match the filters.", *[blank] * 10,
                pd.DataFrame(), pd.DataFrame())
    df, note = detect(df, method_, z_thr_, cont_)
    bad = df[df["is_anomaly"]]

    customers = df["customer_id"].dropna().nunique()
    top_prov = known(df["cc_provider"]).mode()
    kpi = strip(len(df)) + kpi_row(
        card("Transactions", f"{len(df):,}", color=C["cyan"]),
        card("Unique customers", f"{customers:,}" if customers else "n/a", color=C["blue"]),
        card("Total revenue", f"${df['purchase_price'].sum():,.0f}", color=C["green"]),
        card("Average price", f"${df['purchase_price'].mean():,.2f}", color=C["amber"]),
        card("Top card provider", top_prov.iat[0] if len(top_prov) else "n/a", color=C["violet"]),
        card("Flagged orders", f"{len(bad):,}", f"{len(bad) / len(df):.1%} of rows", color=C["rose"]),
    )

    hist = px.histogram(df, x="purchase_price", color="is_anomaly", nbins=40,
                        title="Price distribution",
                        color_discrete_map={False: C["blue"], True: C["rose"]})
    hist.update_layout(**LAYOUT, legend_title_text="Flagged")

    if df["order_date"].notna().any():
        span = (df["order_date"].max() - df["order_date"].min()).days
        rule = "W" if span > 90 else "D"
        ts = (df.dropna(subset=["order_date"]).set_index("order_date")
              .resample(rule)["purchase_price"].agg(["sum", "count"]).reset_index())
        trend = go.Figure()
        trend.add_trace(go.Bar(x=ts["order_date"], y=ts["sum"], name="Revenue", marker_color=C["cyan"]))
        trend.add_trace(go.Scatter(x=ts["order_date"], y=ts["count"], name="Orders", yaxis="y2",
                                   mode="lines+markers", line=dict(color=C["amber"])))
        trend.update_layout(**LAYOUT, title="Revenue and orders over time",
                            yaxis2=dict(overlaying="y", side="right", showgrid=False),
                            legend=dict(orientation="h"))
    else:
        trend = empty_fig("Choose a date column to see the trend over time")

    cols = ["source_row", "reason", "purchase_price", "cc_provider", "job", "language", "anomaly_score"]
    alerts = bad.sort_values("anomaly_score", ascending=False).head(50)[cols].round(2)
    preview = df.head(25).assign(
        order_date=df.head(25)["order_date"].astype(str).replace("NaT", "")).round(2)

    return (kpi, f"ℹ️ {note}" if note else "",
            top_bar(df["cc_provider"], "Top card providers"), donut(df["am_pm"], "AM vs PM"), hist,
            top_bar(df["job"], "Top jobs"), top_bar(df["email_domain"], "Top email domains", 8),
            donut(df["browser"], "Browser share"), donut(df["os"], "Operating system share"),
            top_bar(df["language"], "Language", 8), top_bar(df["state"], "Top states"), trend,
            alerts, preview)


def export_excel(df, provs, ampm, langs, method_, z_thr_, cont_):
    if df is None or df.empty:
        gr.Info("Import a file first.")
        return None
    df, _ = detect(apply_filters(df, provs, ampm, langs), method_, z_thr_, cont_)
    summary = pd.DataFrame({
        "Metric": ["Transactions", "Total revenue", "Average price", "Median price", "Flagged orders"],
        "Value": [len(df), round(df["purchase_price"].sum(), 2), round(df["purchase_price"].mean(), 2),
                  round(df["purchase_price"].median(), 2), int(df["is_anomaly"].sum())]})
    by_prov = (df.groupby("cc_provider")["purchase_price"].agg(["count", "mean", "sum"])
               .round(2).sort_values("count", ascending=False).reset_index())
    path = os.path.join(tempfile.mkdtemp(), "orders_analysis.xlsx")
    with pd.ExcelWriter(path) as xw:
        summary.to_excel(xw, sheet_name="Summary", index=False)
        by_prov.to_excel(xw, sheet_name="By provider", index=False)
        df[df["is_anomaly"]].to_excel(xw, sheet_name="Flagged orders", index=False)
        df.to_excel(xw, sheet_name="All data", index=False)
    return path


def blank_outputs(*kinds):
    out = [empty_fig("Import a file first") if k == "f" else pd.DataFrame() if k == "t" else "Import a file first."
           for k in kinds]
    return tuple(out) if len(out) > 1 else out[0]


ADV_KINDS = ("m",) + ("f", "t", "m") + ("f", "f", "t", "m") + ("f", "t", "m") + ("f", "f", "t", "t", "m") + \
    ("f", "f", "m") + ("f", "t", "m") * 3 + ("f", "f", "m") + ("f", "f", "m") + ("f", "f", "t", "m")


def run_advanced(df, provs, ampm, langs, raw, gt, a, b, pr, pc, pagg, cn, k):
    """Run every advanced analysis on the filtered data (called automatically)."""
    if df is None or df.empty:
        return blank_outputs(*ADV_KINDS)
    d = apply_filters(df, provs, ampm, langs)
    if d.empty:
        return blank_outputs(*ADV_KINDS)
    dq = adv.data_quality(raw)
    dist = adv.price_distribution(d)
    bf = adv.benford(d)
    drv = adv.drivers(d)
    amx = adv.association_matrix(d)
    scan = adv.compose_scan(d, dq[2] if raw is not None else "", dist[3], drv[2], amx[2], bf[2])
    return (scan, *dq, *dist, *bf, *adv.group_test(d, gt), *adv.association(d, a, b), *amx, *drv,
            *adv.pivot(d, pr, pc, pagg), *adv.concentration(d, cn), *adv.time_patterns(d),
            *adv.segmentation(d, k))


# ------------------------------------------------------------------ UI
def style_kwargs():
    """Send theme/css/js to Blocks() or launch(), whichever this Gradio version expects."""
    theme = build_theme()
    wanted = {"theme": theme, "css": CSS, "js": JS}
    init_params = set(inspect.signature(gr.Blocks.__init__).parameters)
    launch_params = set(inspect.signature(gr.Blocks.launch).parameters)
    blocks_kw, launch_kw = {}, {}
    for key, value in wanted.items():
        if key in init_params:
            blocks_kw[key] = value
        elif key in launch_params:
            launch_kw[key] = value
    return blocks_kw, launch_kw


def build_app(blocks_kw):
    with gr.Blocks(title="Orders Dashboard (Excel import)", **blocks_kw) as demo:
        path_state, raw_state, df_state = gr.State(None), gr.State(None), gr.State(None)

        gr.HTML(header_html("📊 EDA Analysis Dashboard",
                            "Import an Excel or CSV file → clean data → interactive dashboard with anomaly alerts"))

        with gr.Row():
            # ------------------------------------------------ left panel
            with gr.Column(scale=1, min_width=290):
                gr.Markdown("### 1. Import data")
                file_in = gr.File(label="Excel or CSV file", file_types=[".xlsx", ".xls", ".xlsm", ".csv"])
                mode = gr.Radio([REPLACE, APPEND], value=REPLACE, label="When importing")
                dedupe = gr.Checkbox(label="Remove duplicate rows", value=True)
                sheet_dd = gr.Dropdown(label="Excel sheet", choices=[], visible=False)
                price_dd = gr.Dropdown(label="Price column (auto-detected)", choices=[])
                date_dd = gr.Dropdown(label="Date column (optional)", choices=["None"], value="None")
                status = gr.Markdown("No file loaded yet.")

                gr.Markdown("### 2. Filters")
                f_prov = gr.Dropdown(label="Card provider (empty = all)", multiselect=True, choices=[])
                f_ampm = gr.Dropdown(label="AM / PM (empty = all)", multiselect=True, choices=[])
                f_lang = gr.Dropdown(label="Language (empty = all)", multiselect=True, choices=[])

                gr.Markdown("### 3. Anomaly detection")
                method = gr.Radio(METHODS, value=METHODS[0], label="Method")
                z_thr = gr.Slider(2.0, 8.0, value=3.5, step=0.5, label="z-score threshold")
                contamination = gr.Slider(0.01, 0.10, value=0.03, step=0.01, label="Expected anomaly share")
                clear_btn = gr.Button("🗑️ Clear data")

            # ------------------------------------------------ dashboard
            with gr.Column(scale=4):
                with gr.Tabs():
                    with gr.Tab("📊 Dashboard"):
                        kpis = gr.HTML()
                        note_md = gr.Markdown()
                        with gr.Row():
                            p_prov, p_ampm, p_hist = gr.Plot(), gr.Plot(), gr.Plot()
                        with gr.Row():
                            p_jobs, p_email = gr.Plot(), gr.Plot()
                        with gr.Row():
                            p_browser, p_os, p_lang = gr.Plot(), gr.Plot(), gr.Plot()
                        with gr.Row():
                            p_state, p_trend = gr.Plot(), gr.Plot()
                        gr.Markdown("### 🚨 Flagged orders")
                        alerts_tbl = gr.DataFrame(interactive=False)
                        gr.Markdown("### 🧾 Data preview")
                        preview_tbl = gr.DataFrame(interactive=False)
                        with gr.Row():
                            export_btn = gr.Button("⬇️ Export results to Excel", variant="primary")
                            export_file = gr.File(label="orders_analysis.xlsx", interactive=False)
                    with gr.Tab("🔬 Advanced EDA"):
                        gr.Markdown("⚙️ Analyses run **automatically** when you import data, change a filter "
                                    "or change an option here. Tests use α = 0.05; p-values are corrected for "
                                    "multiple comparisons where many tests are run.")
                        adv_btn = gr.Button("🔄 Re-run all analyses")
                        with gr.Tabs():
                            with gr.Tab("⚡ Auto scan"):
                                scan_md = gr.Markdown()
                            with gr.Tab("🧹 Data quality"):
                                dq_md = gr.Markdown()
                                dq_plot = gr.Plot()
                                dq_tbl = gr.DataFrame(interactive=False)
                            with gr.Tab("📐 Price distribution"):
                                pd_md = gr.Markdown()
                                with gr.Row():
                                    pd_hist, pd_qq = gr.Plot(), gr.Plot()
                                pd_tbl = gr.DataFrame(interactive=False)
                                gr.Markdown("#### Benford's law (fraud screening)")
                                bf_md = gr.Markdown()
                                bf_plot = gr.Plot()
                                bf_tbl = gr.DataFrame(interactive=False)
                            with gr.Tab("🧪 Group tests"):
                                with gr.Row():
                                    gt_dd = gr.Dropdown(list(adv.CATS), value="Card provider",
                                                        label="Does price differ by...")
                                gt_md = gr.Markdown()
                                with gr.Row():
                                    gt_box, gt_ci = gr.Plot(), gr.Plot()
                                gt_tbl = gr.DataFrame(label="Group statistics", interactive=False)
                                gt_post = gr.DataFrame(label="Pairwise comparisons (Bonferroni)", interactive=False)
                            with gr.Tab("🔗 Associations"):
                                with gr.Row():
                                    as_a = gr.Dropdown(list(adv.CATS), value="Card provider", label="Column A")
                                    as_b = gr.Dropdown(list(adv.CATS), value="Language", label="Column B")
                                as_md = gr.Markdown()
                                with gr.Row():
                                    as_heat, as_bars = gr.Plot(), gr.Plot()
                                gr.Markdown("#### All column pairs")
                                am_md = gr.Markdown()
                                am_plot = gr.Plot()
                                am_tbl = gr.DataFrame(interactive=False)
                            with gr.Tab("🎯 Price drivers"):
                                dr_md = gr.Markdown()
                                dr_plot = gr.Plot()
                                dr_tbl = gr.DataFrame(interactive=False)
                            with gr.Tab("🧮 Pivot heatmap"):
                                with gr.Row():
                                    pv_r = gr.Dropdown(list(adv.CATS), value="Card provider", label="Rows")
                                    pv_c = gr.Dropdown(list(adv.CATS), value="AM / PM", label="Columns")
                                    pv_agg = gr.Dropdown(["mean", "median", "sum", "count"], value="mean",
                                                         label="Aggregate price")
                                pv_md = gr.Markdown()
                                pv_plot = gr.Plot()
                                pv_tbl = gr.DataFrame(interactive=False)
                            with gr.Tab("⚖️ Concentration"):
                                with gr.Row():
                                    cn_dd = gr.Dropdown(["(individual orders)"] + list(adv.CATS),
                                                        value="(individual orders)", label="Revenue concentration across")
                                cn_md = gr.Markdown()
                                with gr.Row():
                                    cn_lor, cn_par = gr.Plot(), gr.Plot()
                            with gr.Tab("🕒 Time patterns"):
                                tp_md = gr.Markdown()
                                with gr.Row():
                                    tp_heat, tp_trend = gr.Plot(), gr.Plot()
                            with gr.Tab("🧩 Segmentation"):
                                with gr.Row():
                                    sg_k = gr.Slider(2, 8, value=4, step=1, label="Number of clusters (k)")
                                sg_md = gr.Markdown()
                                with gr.Row():
                                    sg_scan, sg_scatter = gr.Plot(), gr.Plot()
                                sg_tbl = gr.DataFrame(label="Cluster profiles", interactive=False)

        filters = [f_prov, f_ampm, f_lang]
        controls = [method, z_thr, contamination]
        dash_outputs = [kpis, note_md, p_prov, p_ampm, p_hist, p_jobs, p_email, p_browser, p_os, p_lang,
                        p_state, p_trend, alerts_tbl, preview_tbl]
        load_outputs = [path_state, raw_state, df_state, sheet_dd, price_dd, date_dd, status] + filters

        # ------------------------------------------------ data loading
        def build(raw, price_col, date_col):
            """Prepare the clean table and the filter choices from the raw data."""
            df, dropped = prepare(raw, price_col, date_col)
            return df, dropped

        def filter_updates(df):
            def opts(col):
                return sorted(known(df[col]).unique().tolist())
            return {f_prov: gr.update(choices=opts("cc_provider"), value=[]),
                    f_ampm: gr.update(choices=opts("am_pm"), value=[]),
                    f_lang: gr.update(choices=opts("language"), value=[])}

        def load_file(path, sheet, mode_, raw_old, dedupe_):
            new = read_any(path, sheet)
            raw = pd.concat([raw_old, new], ignore_index=True) if (
                mode_ == APPEND and raw_old is not None) else new
            removed = 0
            if dedupe_:
                before = len(raw)
                raw = raw.drop_duplicates().reset_index(drop=True)
                removed = before - len(raw)
            return raw, len(new), removed

        def guess_columns(raw):
            price = find_col(raw.columns, "price")
            if price is None:
                numeric = [c for c in raw.columns if pd.api.types.is_numeric_dtype(raw[c])]
                price = numeric[0] if numeric else None
            date = find_col(raw.columns, "date") or "None"
            return price, date

        def status_text(path, sheet, added, removed, total, dropped, price_col):
            name = os.path.basename(path) if path else "file"
            where = f" (sheet **{sheet}**)" if sheet else ""
            parts = [f"✅ Loaded **{name}**{where}: {added:,} rows read, now **{total:,}** rows in the dashboard."]
            if removed:
                parts.append(f"{removed:,} duplicate rows removed.")
            if dropped:
                parts.append(f"{dropped:,} rows skipped (no valid price).")
            parts.append(f"Price column: `{price_col}`.")
            parts.append("Sensitive columns (card number, security code, IP, email, address) are never stored; "
                         "only safe features such as email domain and state are kept.")
            return " ".join(parts)

        def import_data(path, sheet, mode_, dedupe_, raw_old):
            """Shared loader for a new upload or a sheet switch."""
            try:
                raw, added, removed = load_file(path, sheet, mode_, raw_old, dedupe_)
            except ImportError:
                return {status: "⚠️ Reading Excel needs `openpyxl` (and `xlrd` for .xls). "
                                "Run: pip install openpyxl xlrd"}
            except Exception as exc:
                return {status: f"⚠️ Could not read the file: {exc}"}
            price_col, date_col = guess_columns(raw)
            if price_col is None:
                return {status: "⚠️ No numeric price column found. Check that the file has a price/amount column."}
            df, dropped = build(raw, price_col, date_col)
            sheets = sheet_names(path)
            out = {
                path_state: path, raw_state: raw, df_state: df,
                sheet_dd: gr.update(choices=sheets, value=sheet, visible=bool(sheets)),
                price_dd: gr.update(choices=list(raw.columns), value=price_col),
                date_dd: gr.update(choices=["None"] + list(raw.columns), value=date_col),
                status: status_text(path, sheet, added, removed, len(raw), dropped, price_col),
            }
            out.update(filter_updates(df))
            return out

        def on_upload(file, mode_, dedupe_, raw_old):
            if file is None:
                return {status: "No file loaded yet."}
            path = parse_path(file)
            sheets = sheet_names(path) if is_excel(path) else []
            return import_data(path, sheets[0] if sheets else None, mode_, dedupe_, raw_old)

        def on_sheet(path, sheet, mode_, dedupe_, raw_old):
            if not path:
                return {status: "Upload a file first."}
            return import_data(path, sheet, REPLACE, dedupe_, raw_old)  # switching sheets replaces the view

        def on_columns(raw, price_col, date_col):
            if raw is None or not price_col:
                return {status: "Upload a file first."}
            df, dropped = build(raw, price_col, date_col)
            out = {df_state: df,
                   status: f"Columns updated: price `{price_col}`, date `{date_col}`. "
                           f"{len(df):,} valid rows" + (f" ({dropped:,} skipped)." if dropped else ".")}
            out.update(filter_updates(df))
            return out

        analyze_inputs = [df_state] + filters + controls
        adv_inputs = [df_state] + filters + [raw_state, gt_dd, as_a, as_b, pv_r, pv_c, pv_agg, cn_dd, sg_k]
        adv_outputs = [scan_md, dq_plot, dq_tbl, dq_md, pd_hist, pd_qq, pd_tbl, pd_md, bf_plot, bf_tbl, bf_md,
                       gt_box, gt_ci, gt_tbl, gt_post, gt_md, as_heat, as_bars, as_md, am_plot, am_tbl, am_md,
                       dr_plot, dr_tbl, dr_md, pv_plot, pv_tbl, pv_md, cn_lor, cn_par, cn_md,
                       tp_heat, tp_trend, tp_md, sg_scan, sg_scatter, sg_tbl, sg_md]

        # ------------------------------------------------ events
        file_in.upload(on_upload, [file_in, mode, dedupe, raw_state], load_outputs).then(
            analyze, analyze_inputs, dash_outputs).then(
            run_advanced, adv_inputs, adv_outputs)
        sheet_dd.input(on_sheet, [path_state, sheet_dd, mode, dedupe, raw_state], load_outputs).then(
            analyze, analyze_inputs, dash_outputs).then(
            run_advanced, adv_inputs, adv_outputs)
        for comp in (price_dd, date_dd):
            comp.input(on_columns, [raw_state, price_dd, date_dd], [df_state, status] + filters).then(
                analyze, analyze_inputs, dash_outputs).then(
                run_advanced, adv_inputs, adv_outputs)
        for comp in filters + controls:
            comp.change(analyze, analyze_inputs, dash_outputs)

        def on_clear():
            blank = empty_fig()
            return {path_state: None, raw_state: None, df_state: None, file_in: None,
                    sheet_dd: gr.update(choices=[], value=None, visible=False),
                    price_dd: gr.update(choices=[], value=None),
                    date_dd: gr.update(choices=["None"], value="None"),
                    status: "No file loaded yet.", kpis: strip() + kpi_row(card("Orders", "0")), note_md: "",
                    p_prov: blank, p_ampm: blank, p_hist: blank, p_jobs: blank, p_email: blank,
                    p_browser: blank, p_os: blank, p_lang: blank, p_state: blank, p_trend: blank,
                    alerts_tbl: pd.DataFrame(), preview_tbl: pd.DataFrame(),
                    f_prov: gr.update(choices=[], value=[]), f_ampm: gr.update(choices=[], value=[]),
                    f_lang: gr.update(choices=[], value=[])}

        clear_btn.click(on_clear, None, [path_state, raw_state, df_state, file_in, sheet_dd, price_dd, date_dd,
                                         status, kpis, note_md, p_prov, p_ampm, p_hist, p_jobs, p_email,
                                         p_browser, p_os, p_lang, p_state, p_trend, alerts_tbl,
                                         preview_tbl] + filters).then(
            run_advanced, adv_inputs, adv_outputs)
        export_btn.click(export_excel, analyze_inputs, export_file)

        # ------------------------------------------------ advanced EDA wiring (automatic)
        base = [df_state] + filters

        def one(fn, *kinds):
            """Run a single advanced analysis on the filtered data."""
            def run(df, provs, ampm, langs, *extra):
                d = apply_filters(df, provs, ampm, langs) if df is not None and not df.empty else None
                if d is None or d.empty:
                    return blank_outputs(*kinds)
                return fn(d, *extra)
            return run

        # filters re-run every analysis; imports and clears are chained with .then() above
        for comp in filters:
            comp.change(run_advanced, adv_inputs, adv_outputs)
        adv_btn.click(run_advanced, adv_inputs, adv_outputs)

        # single-analysis controls update only their own section
        gt_dd.change(one(adv.group_test, "f", "f", "t", "t", "m"), base + [gt_dd],
                     [gt_box, gt_ci, gt_tbl, gt_post, gt_md])
        for comp in (as_a, as_b):
            comp.change(one(adv.association, "f", "f", "m"), base + [as_a, as_b], [as_heat, as_bars, as_md])
        for comp in (pv_r, pv_c, pv_agg):
            comp.change(one(adv.pivot, "f", "t", "m"), base + [pv_r, pv_c, pv_agg], [pv_plot, pv_tbl, pv_md])
        cn_dd.change(one(adv.concentration, "f", "f", "m"), base + [cn_dd], [cn_lor, cn_par, cn_md])
        sg_k.release(one(adv.segmentation, "f", "f", "t", "m"), base + [sg_k],
                     [sg_scan, sg_scatter, sg_tbl, sg_md])
    return demo


if __name__ == "__main__":
    blocks_kw, launch_kw = style_kwargs()
    build_app(blocks_kw).launch(**launch_kw)