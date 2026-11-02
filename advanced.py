"""Advanced statistical EDA for the prepared orders table.

Every function takes the prepared (filtered) dataframe from loader.prepare and
returns figures, tables and a plain-English interpretation string.
Statistical tests need scipy; clustering needs scikit-learn.
"""
import re

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from theme import COLORS as C, DIVERGE, PLOT_TEMPLATE, SCALE, SCALE_RED

try:
    from scipy import stats
except ImportError:  # reported to the user inside each function
    stats = None

PRICE = "purchase_price"
ALPHA = 0.05
LAYOUT = dict(template=PLOT_TEMPLATE, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
              margin=dict(l=10, r=10, t=50, b=10), height=380)
CATS = {"Card provider": "cc_provider", "Job": "job", "Language": "language", "AM / PM": "am_pm",
        "Email domain": "email_domain", "State": "state", "Browser": "browser", "Operating system": "os"}
SENSITIVE = re.compile(r"(card|security|cvv|ip.?addr|e-?mail|address|password|ssn|phone)", re.I)
UNKNOWN = ["Unknown", "nan", "None", ""]
NEED_SCIPY = "⚠️ This analysis needs scipy. Run: `pip install scipy`"


# =============================================================================
# helpers
# =============================================================================
def empty_fig(msg="Not enough data"):
    fig = go.Figure()
    fig.update_layout(**LAYOUT, xaxis=dict(visible=False), yaxis=dict(visible=False),
                      annotations=[dict(text=msg, showarrow=False, font=dict(size=15, color="#94a3b8"))])
    return fig


def p_text(p):
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def p_eq(p):
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def verdict(p, what="difference"):
    return (f"**statistically significant {what}** ({p_eq(p)}, α = {ALPHA})" if p < ALPHA else
            f"**no statistically significant {what}** ({p_eq(p)}, α = {ALPHA})")


def effect_label(w):
    return "negligible" if w < 0.01 else "small" if w < 0.06 else "medium" if w < 0.14 else "large"


def bh_adjust(p):
    """Benjamini-Hochberg false-discovery-rate adjusted p-values."""
    p = np.asarray(p, dtype=float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(ranked, 1.0)
    return out


def groups(df, col, top=12, min_n=5):
    """Rows of the `top` biggest groups of `col` that have at least `min_n` orders."""
    d = df[[col, PRICE]].dropna()
    d = d[~d[col].isin(UNKNOWN)]
    counts = d[col].value_counts()
    keep = counts[counts >= min_n].head(top).index
    return d[d[col].isin(keep)]


def anova_parts(d, col):
    arrays = [g[PRICE].to_numpy() for _, g in d.groupby(col)]
    k, n = len(arrays), sum(len(a) for a in arrays)
    grand = d[PRICE].mean()
    ssb = sum(len(a) * (a.mean() - grand) ** 2 for a in arrays)
    sst = ((d[PRICE] - grand) ** 2).sum()
    msw = (sst - ssb) / (n - k) if n > k else np.nan
    eta = ssb / sst if sst > 0 else 0.0
    omega = max(0.0, (ssb - (k - 1) * msw) / (sst + msw)) if (n > k and sst > 0) else 0.0
    return arrays, k, n, eta, omega


def cramers_v(ct):
    o = ct.to_numpy(dtype=float)
    n = o.sum()
    r, k = o.shape
    if min(r, k) < 2 or n < 2:
        return np.nan
    e = np.outer(o.sum(1), o.sum(0)) / n
    chi2 = ((o - e) ** 2 / e).sum()
    phi2c = max(0.0, chi2 / n - ((k - 1) * (r - 1)) / (n - 1))
    rc, kc = r - (r - 1) ** 2 / (n - 1), k - (k - 1) ** 2 / (n - 1)
    denom = min(kc - 1, rc - 1)
    return float(np.sqrt(phi2c / denom)) if denom > 0 else np.nan


def top_levels(s, n=8):
    """Keep the n most common levels, group the rest as 'Other'."""
    keep = s.value_counts().head(n).index
    return s.where(s.isin(keep), "Other")


def known_pair(df, a, b):
    d = df[[a, b]].dropna()
    return d[~d[a].isin(UNKNOWN) & ~d[b].isin(UNKNOWN)]


# =============================================================================
# 1. Data quality (works on the raw upload)
# =============================================================================
def data_quality(raw):
    if raw is None or raw.empty:
        return empty_fig("Import a file first"), pd.DataFrame(), "Import a file first."
    rows = []
    for c in raw.columns:
        s = raw[c]
        hidden = bool(SENSITIVE.search(str(c)))
        rows.append({"Column": c, "Type": str(s.dtype), "Non-null": int(s.notna().sum()),
                     "Missing %": round(s.isna().mean() * 100, 2), "Unique": int(s.nunique()),
                     "Example": "🔒 hidden" if hidden else (str(s.dropna().iloc[0])[:40] if s.notna().any() else "")})
    table = pd.DataFrame(rows).sort_values("Missing %", ascending=False)
    fig = px.bar(table.sort_values("Missing %"), x="Missing %", y="Column", orientation="h",
                 title="Missing values by column (%)", color="Missing %", color_continuous_scale=SCALE_RED)
    fig.update_layout(**LAYOUT, coloraxis_showscale=False)
    dup = int(raw.duplicated().sum())
    const = [c for c in raw.columns if raw[c].nunique(dropna=False) <= 1]
    ident = [c for c in raw.columns if raw[c].nunique() == len(raw) and len(raw) > 20]
    md = (f"**{len(raw):,} rows × {raw.shape[1]} columns**, {raw.isna().sum().sum():,} missing cells, "
          f"{dup:,} duplicate rows, {raw.memory_usage(deep=True).sum() / 1e6:.1f} MB.\n\n"
          f"- Constant columns: {', '.join(f'`{c}`' for c in const) or 'none'}\n"
          f"- Fully unique columns (identifiers, not useful for grouping): "
          f"{', '.join(f'`{c}`' for c in ident) or 'none'}")
    return fig, table, md


# =============================================================================
# 2. Price distribution, normality and transformation
# =============================================================================
def price_distribution(df):
    if stats is None:
        return empty_fig(NEED_SCIPY), empty_fig(NEED_SCIPY), pd.DataFrame(), NEED_SCIPY
    x = df[PRICE].dropna().astype(float)
    if len(x) < 8 or x.std() == 0:
        msg = "Need at least 8 orders with varying prices."
        return empty_fig(msg), empty_fig(msg), pd.DataFrame(), msg

    mu, sd = x.mean(), x.std()
    grid = np.linspace(x.min(), x.max(), 200)
    hist = px.histogram(x.to_frame(PRICE), x=PRICE, nbins=50, histnorm="probability density",
                        title="Price distribution vs normal curve")
    hist.add_trace(go.Scatter(x=grid, y=stats.norm.pdf(grid, mu, sd), mode="lines", name="Normal fit",
                              line=dict(color=C["amber"], width=3)))
    hist.update_layout(**LAYOUT)

    lx = np.log1p(x.clip(lower=0))
    qq = go.Figure()
    for name, v, color in (("Raw price", x, C["blue"]), ("log(1 + price)", lx, C["cyan"])):
        v = v.sample(min(len(v), 2000), random_state=1)
        (osm, osr), _ = stats.probplot(v, dist="norm")
        qq.add_trace(go.Scatter(x=osm, y=(osr - osr.mean()) / osr.std(), mode="markers", name=name,
                                marker=dict(size=4, color=color, opacity=0.6)))
    qq.add_trace(go.Scatter(x=[-3, 3], y=[-3, 3], mode="lines", name="Normal", line=dict(color=C["amber"], dash="dash")))
    qq.update_layout(**LAYOUT, title="Q-Q plot: raw vs log-transformed", xaxis_title="Theoretical quantiles",
                     yaxis_title="Standardised sample quantiles")

    def normality(v):
        sample = v.sample(min(len(v), 5000), random_state=1)
        return stats.shapiro(sample).pvalue, stats.normaltest(v).pvalue

    sh_raw, dk_raw = normality(x)
    sh_log, dk_log = normality(lx)
    q = x.quantile([0, .01, .05, .25, .5, .75, .95, .99, 1])
    table = pd.DataFrame({"Statistic": ["Count", "Mean", "Median", "Std", "Coefficient of variation", "Skewness",
                                        "Excess kurtosis", "Min", "1st pct", "5th pct", "25th pct", "75th pct",
                                        "95th pct", "99th pct", "Max"],
                          "Value": [len(x), mu, x.median(), sd, sd / mu if mu else np.nan, x.skew(), x.kurt(),
                                    q[0], q[.01], q[.05], q[.25], q[.75], q[.95], q[.99], q[1]]})
    table["Value"] = table["Value"].round(3)
    skew_note = ("strongly right-skewed: a few expensive orders pull the mean above the median"
                 if x.skew() > 1 else "roughly symmetric" if abs(x.skew()) < 0.5 else "moderately skewed")
    improved = abs(lx.skew()) < abs(x.skew())
    md = (f"**Price is {skew_note}** (skew {x.skew():.2f}, mean ${mu:,.2f} vs median ${x.median():,.2f}).\n\n"
          f"- Normality of raw price: Shapiro–Wilk {p_eq(sh_raw)}, D'Agostino {p_eq(dk_raw)} → "
          f"{'looks normal' if min(sh_raw, dk_raw) >= ALPHA else 'not normal'}.\n"
          f"- After log(1+price): Shapiro–Wilk {p_eq(sh_log)}, D'Agostino {p_eq(dk_log)}, skew {lx.skew():.2f} → "
          f"{'the transform helps' if improved else 'the transform does not help'}.\n"
          f"- With thousands of rows, tests flag tiny deviations; judge the Q-Q plot too. "
          f"{'Prefer non-parametric tests (Kruskal–Wallis, Mann–Whitney) for this price.' if min(sh_raw, dk_raw) < ALPHA else ''}")
    return hist, qq, table, md


# =============================================================================
# 3. Benford's law (fraud screening)
# =============================================================================
def benford(df):
    if stats is None:
        return empty_fig(NEED_SCIPY), pd.DataFrame(), NEED_SCIPY
    x = df[PRICE].dropna().astype(float)
    x = x[x > 0]
    if len(x) < 100:
        return empty_fig("Need at least 100 positive prices"), pd.DataFrame(), "Need at least 100 positive prices."
    first = (x / 10 ** np.floor(np.log10(x))).astype(int).clip(1, 9)
    digits = np.arange(1, 10)
    obs = first.value_counts(normalize=True).reindex(digits, fill_value=0).to_numpy()
    exp = np.log10(1 + 1 / digits)
    n = len(first)
    chi2 = float(n * ((obs - exp) ** 2 / exp).sum())
    p = float(stats.chi2.sf(chi2, 8))
    mad = float(np.abs(obs - exp).mean())
    level = ("close conformity" if mad < 0.006 else "acceptable conformity" if mad < 0.012
             else "marginal conformity" if mad < 0.015 else "non-conformity")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=digits, y=obs * 100, name="Observed %", marker_color=C["cyan"]))
    fig.add_trace(go.Scatter(x=digits, y=exp * 100, name="Benford expected %", mode="lines+markers",
                             line=dict(color=C["amber"], width=3)))
    fig.update_layout(**LAYOUT, title="First-digit distribution vs Benford's law",
                      xaxis_title="Leading digit", yaxis_title="%", xaxis=dict(dtick=1))
    table = pd.DataFrame({"Digit": digits, "Observed %": (obs * 100).round(2), "Expected %": (exp * 100).round(2),
                          "Difference": ((obs - exp) * 100).round(2)})
    md = (f"MAD = **{mad:.4f}** → Nigrini's rule says **{level}**. Chi-square {p_eq(p)}.\n\n"
          "⚠️ Benford's law fits numbers that spread over several orders of magnitude. Retail prices that "
          "cluster in a narrow range often do not conform even when the data is genuine, so treat this as a "
          "screening hint for manipulated amounts, never as proof.")
    return fig, table, md


# =============================================================================
# 4. Group comparison (ANOVA / Kruskal-Wallis / post hoc)
# =============================================================================
def group_test(df, label):
    blank = empty_fig()
    if stats is None:
        return blank, blank, pd.DataFrame(), pd.DataFrame(), NEED_SCIPY
    col = CATS[label]
    d = groups(df, col, top=10)
    if d[col].nunique() < 2:
        return blank, blank, pd.DataFrame(), pd.DataFrame(), "Need at least 2 groups with 5+ orders."
    arrays, k, n, eta, omega = anova_parts(d, col)
    f, p_f = stats.f_oneway(*arrays)
    h, p_k = stats.kruskal(*arrays)
    p_lev = stats.levene(*arrays).pvalue

    g = d.groupby(col)[PRICE].agg(["count", "mean", "median", "std"])
    sem = g["std"] / np.sqrt(g["count"])
    g["ci95"] = stats.t.ppf(0.975, (g["count"] - 1).clip(lower=1)) * sem
    g = g.sort_values("mean", ascending=False)
    order = g.index.tolist()

    box = px.box(d, x=col, y=PRICE, color=col, category_orders={col: order}, points=False,
                 title=f"Price by {label.lower()}")
    box.update_layout(**LAYOUT, showlegend=False, xaxis_title="")
    ci = go.Figure(go.Scatter(x=order, y=g["mean"], mode="markers", marker=dict(size=10, color=C["cyan"]),
                              error_y=dict(type="data", array=g["ci95"].fillna(0), color=C["amber"])))
    ci.add_hline(y=d[PRICE].mean(), line_dash="dash", line_color="#94a3b8", annotation_text="overall mean")
    ci.update_layout(**LAYOUT, title="Mean price with 95% confidence interval", xaxis_title="")

    table = g.rename(columns={"count": "Orders", "mean": "Mean", "median": "Median", "std": "Std",
                              "ci95": "±95% CI"}).round(2).reset_index()

    big = d[col].value_counts().head(8).index.tolist()
    pairs = [(a, b) for i, a in enumerate(big) for b in big[i + 1:]]
    rows = []
    for a, b in pairs:
        xa, xb = d.loc[d[col] == a, PRICE], d.loc[d[col] == b, PRICE]
        p_raw = stats.mannwhitneyu(xa, xb, alternative="two-sided").pvalue
        pooled = np.sqrt(((len(xa) - 1) * xa.var() + (len(xb) - 1) * xb.var()) / max(len(xa) + len(xb) - 2, 1))
        rows.append({"Group A": a, "Group B": b, "Mean diff": round(xa.mean() - xb.mean(), 2),
                     "Cohen's d": round((xa.mean() - xb.mean()) / pooled, 2) if pooled else np.nan,
                     "p (Mann–Whitney)": p_raw, "p (Bonferroni)": min(1.0, p_raw * len(pairs))})
    post = pd.DataFrame(rows)
    if not post.empty:
        post = post.sort_values("p (Bonferroni)").head(12)
        post["Significant"] = np.where(post["p (Bonferroni)"] < ALPHA, "yes", "no")
        post["p (Mann–Whitney)"] = post["p (Mann–Whitney)"].round(4)
        post["p (Bonferroni)"] = post["p (Bonferroni)"].round(4)

    both = (p_f < ALPHA) == (p_k < ALPHA)
    md = (f"**{label} vs price** ({k} groups, {n:,} orders)\n\n"
          f"- Kruskal–Wallis (robust to skew): H = {h:.2f}, {p_eq(p_k)} → {verdict(p_k)}.\n"
          f"- One-way ANOVA: F = {f:.2f}, {p_eq(p_f)}"
          f"{' (agrees)' if both else ' (**disagrees**: check skew and outliers)'}.\n"
          f"- Levene variance test {p_eq(p_lev)} → variances are "
          f"{'unequal' if p_lev < ALPHA else 'similar'} across groups.\n"
          f"- Effect size ω² = {omega:.3f} ({effect_label(omega)}): {label.lower()} explains about "
          f"{omega:.1%} of the variation in price.\n"
          f"- Post-hoc pairs use Bonferroni correction to avoid false positives from many comparisons.")
    return box, ci, table, post, md


# =============================================================================
# 5. Association between two categorical variables (chi-square)
# =============================================================================
def association(df, label_a, label_b):
    blank = empty_fig()
    if stats is None:
        return blank, blank, NEED_SCIPY
    a, b = CATS[label_a], CATS[label_b]
    if a == b:
        return blank, blank, "Pick two different columns."
    d = known_pair(df, a, b)
    d = d.assign(**{a: top_levels(d[a]), b: top_levels(d[b])})
    ct = pd.crosstab(d[a], d[b])
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return blank, blank, "Need at least 2 levels in each column."
    chi2, p, dof, exp = stats.chi2_contingency(ct)
    resid = (ct - exp) / np.sqrt(exp)
    v = cramers_v(ct)
    low = float((exp < 5).mean())
    heat = px.imshow(resid.round(2), text_auto=True, aspect="auto", color_continuous_scale=DIVERGE,
                     zmin=-4, zmax=4, title="Standardised residuals (red = more than expected, blue = fewer)")
    heat.update_layout(**LAYOUT)
    share = ct.div(ct.sum(axis=1), axis=0).mul(100).round(1).reset_index().melt(id_vars=a, value_name="%")
    bars = px.bar(share, x=a, y="%", color=b, title=f"{label_b} mix within each {label_a.lower()} (%)")
    bars.update_layout(**LAYOUT, barmode="stack")
    hot = resid.stack().abs().sort_values(ascending=False).head(1)
    cell = hot.index[0]
    md = (f"**{label_a} × {label_b}**: χ² = {chi2:.1f}, dof = {dof}, {p_eq(p)} → {verdict(p, 'association')}.\n\n"
          f"- Cramér's V = **{v:.3f}** ({'negligible' if v < 0.1 else 'weak' if v < 0.3 else 'moderate' if v < 0.5 else 'strong'}).\n"
          f"- Largest deviation: `{cell[0]}` & `{cell[1]}` (residual {resid.loc[cell]:+.2f}).\n"
          + (f"- ⚠️ {low:.0%} of cells have an expected count below 5, so the p-value is less reliable.\n"
             if low > 0.2 else ""))
    return heat, bars, md


def association_matrix(df):
    if stats is None:
        return empty_fig(NEED_SCIPY), pd.DataFrame(), NEED_SCIPY
    cols = []
    for label, c in CATS.items():
        s = df[c][~df[c].isin(UNKNOWN)]
        if 2 <= s.nunique() <= 30:
            cols.append((label, c))
    if len(cols) < 2:
        return empty_fig("Need at least 2 categorical columns"), pd.DataFrame(), "Need at least 2 categorical columns."
    labels = [l for l, _ in cols]
    mat = pd.DataFrame(np.eye(len(cols)), index=labels, columns=labels)
    rows = []
    for i, (la, a) in enumerate(cols):
        for lb, b in cols[i + 1:]:
            d = known_pair(df, a, b)
            ct = pd.crosstab(d[a], d[b])
            if ct.shape[0] < 2 or ct.shape[1] < 2:
                continue
            _, p, _, _ = stats.chi2_contingency(ct)
            v = cramers_v(ct)
            mat.loc[la, lb] = mat.loc[lb, la] = v
            rows.append({"Column A": la, "Column B": lb, "Cramér's V": round(v, 3), "p-value": p})
    table = pd.DataFrame(rows)
    if table.empty:
        return empty_fig("No valid pairs"), table, "No valid pairs."
    table["q-value (FDR)"] = bh_adjust(table["p-value"])
    table["Significant"] = np.where(table["q-value (FDR)"] < ALPHA, "yes", "no")
    table = table.sort_values("p-value")
    table["p-value"] = table["p-value"].round(4)
    table["q-value (FDR)"] = table["q-value (FDR)"].round(4)
    fig = px.imshow(mat.round(2), text_auto=True, aspect="auto", color_continuous_scale=SCALE, zmin=0, zmax=1,
                    title="Cramér's V between categorical columns")
    fig.update_layout(**{**LAYOUT, "height": 480})
    sig = table[table["Significant"] == "yes"]
    if len(sig):
        top = sig.iloc[0]
        v_top = top["Cramér's V"]
        result = (f"**{len(sig)} pair(s) significant**, strongest: `{top['Column A']}` & "
                  f"`{top['Column B']}` (V = {v_top}).")
    else:
        result = "**No pair is significantly associated**, so these columns look independent of each other."
    md = (f"Tested {len(table)} column pairs with false-discovery-rate (Benjamini–Hochberg) correction. {result} "
          "(Browser and Operating system are both derived from one field, so a strong link there is expected.)")
    return fig, table, md


# =============================================================================
# 6. Price driver analysis
# =============================================================================
def drivers(df):
    blank = empty_fig()
    if stats is None:
        return blank, pd.DataFrame(), NEED_SCIPY
    rows = []
    for label, col in CATS.items():
        d = groups(df, col, top=30)
        if d[col].nunique() < 2:
            continue
        arrays, k, n, _, omega = anova_parts(d, col)
        rows.append({"Feature": label, "Type": "categorical", "Effect size": omega,
                     "p-value": stats.kruskal(*arrays).pvalue})
    dates = df["order_date"].dropna() if "order_date" in df else pd.Series(dtype="datetime64[ns]")
    if len(dates) > 30:
        sub = df.loc[dates.index]
        for name, vals in (("Hour of day", sub["order_date"].dt.hour), ("Day of week", sub["order_date"].dt.dayofweek)):
            if vals.nunique() > 1:
                rho, p = stats.spearmanr(vals, sub[PRICE])
                rows.append({"Feature": name, "Type": "numeric (ρ²)", "Effect size": rho ** 2, "p-value": p})
    if not rows:
        return blank, pd.DataFrame(), "No usable features."
    table = pd.DataFrame(rows)
    table["q-value (FDR)"] = bh_adjust(table["p-value"])
    table["Significant"] = np.where(table["q-value (FDR)"] < ALPHA, "yes", "no")
    table = table.sort_values("Effect size", ascending=False)
    fig = px.bar(table.iloc[::-1], x="Effect size", y="Feature", orientation="h", color="Significant",
                 color_discrete_map={"yes": C["cyan"], "no": "#64748b"},
                 title="Which features explain purchase price? (effect size, grey = not significant)")
    fig.update_layout(**LAYOUT)
    table[["Effect size", "p-value", "q-value (FDR)"]] = table[["Effect size", "p-value", "q-value (FDR)"]].round(4)
    sig = table[table["Significant"] == "yes"]
    md = (f"**{len(sig)} of {len(table)} features significantly relate to price** after FDR correction"
          + (f": {', '.join(f'`{f}`' for f in sig['Feature'])}." if len(sig) else
             ". None of them explain price in a statistically meaningful way. This is a valid finding: "
             "price is not predictable from these columns, so a prediction model would perform poorly."))
    return fig, table, md


# =============================================================================
# 7. Pivot heatmap
# =============================================================================
def pivot(df, label_r, label_c, agg):
    r, c = CATS[label_r], CATS[label_c]
    if r == c:
        return empty_fig("Pick two different columns"), pd.DataFrame(), "Pick two different columns."
    d = df[[r, c, PRICE]].dropna()
    d = d[~d[r].isin(UNKNOWN) & ~d[c].isin(UNKNOWN)]
    d = d[d[r].isin(d[r].value_counts().head(10).index) & d[c].isin(d[c].value_counts().head(10).index)]
    if d.empty:
        return empty_fig(), pd.DataFrame(), "No data."
    pv = d.pivot_table(index=r, columns=c, values=PRICE, aggfunc=agg)
    n = d.pivot_table(index=r, columns=c, values=PRICE, aggfunc="count")
    shown = pv if agg == "count" else pv.where(n >= 5)
    fig = px.imshow(shown.round(2), text_auto=True, aspect="auto", color_continuous_scale=SCALE,
                    title=f"{agg.title()} price: {label_r} × {label_c}")
    fig.update_layout(**{**LAYOUT, "height": 480})
    md = ("Blank cells have fewer than 5 orders, so their value would be unreliable. "
          "A bright cell is only interesting if it also holds a decent number of orders.")
    return fig, shown.round(2).reset_index(), md


# =============================================================================
# 8. Revenue concentration (Lorenz curve, Gini, Pareto)
# =============================================================================
def concentration(df, label):
    x = df[PRICE].dropna().astype(float)
    if len(x) < 10:
        return empty_fig(), empty_fig(), "Need at least 10 orders."
    if label == "(individual orders)":
        vals = x.to_numpy()
        names = None
    else:
        s = df.groupby(CATS[label])[PRICE].sum()
        s = s[~s.index.isin(UNKNOWN)]
        vals, names = s.to_numpy(), s
    v = np.sort(vals)
    n = len(v)
    if v.sum() <= 0 or n < 2:
        return empty_fig(), empty_fig(), "Not enough variation."
    cum = np.cumsum(v) / v.sum()
    gini = float(2 * np.sum(np.arange(1, n + 1) * v) / (n * v.sum()) - (n + 1) / n)
    top_n = max(1, int(np.ceil(0.2 * n)))
    top_share = float(v[::-1][:top_n].sum() / v.sum())

    lor = go.Figure()
    lor.add_trace(go.Scatter(x=np.r_[0, np.arange(1, n + 1) / n], y=np.r_[0, cum], mode="lines", name="Lorenz curve",
                             fill="tozeroy", line=dict(color=C["cyan"])))
    lor.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect equality",
                             line=dict(color=C["amber"], dash="dash")))
    lor.update_layout(**LAYOUT, title=f"Lorenz curve (Gini = {gini:.3f})",
                      xaxis_title="Cumulative share of groups / orders", yaxis_title="Cumulative share of revenue")

    if names is None:
        dec = pd.qcut(pd.Series(vals).rank(method="first"), 10, labels=[f"D{i}" for i in range(1, 11)])
        share = pd.Series(vals).groupby(dec, observed=True).sum() / vals.sum() * 100
        par = px.bar(x=share.index.astype(str), y=share.values, title="Revenue share by price decile (D10 = priciest)",
                     labels={"x": "Decile", "y": "% of revenue"})
        par.update_traces(marker_color=C["blue"])
        par.update_layout(**LAYOUT)
    else:
        top = names.sort_values(ascending=False).head(15)
        share = top / names.sum() * 100
        par = go.Figure()
        par.add_trace(go.Bar(x=top.index.astype(str), y=share.values, name="% of revenue", marker_color=C["cyan"]))
        par.add_trace(go.Scatter(x=top.index.astype(str), y=share.cumsum().values, name="Cumulative %", yaxis="y2",
                                 mode="lines+markers", line=dict(color=C["amber"])))
        par.update_layout(**LAYOUT, title=f"Pareto chart: revenue by {label.lower()}",
                          yaxis2=dict(overlaying="y", side="right", range=[0, 105]))
    level = "very even" if gini < 0.2 else "fairly even" if gini < 0.4 else "unequal" if gini < 0.6 else "highly concentrated"
    unit = "orders" if names is None else label.lower() + " groups"
    md = (f"Gini = **{gini:.3f}** ({level}). The top 20% of {unit} generate **{top_share:.0%}** of revenue "
          f"(an 80/20 pattern would be about 80%).")
    return lor, par, md


# =============================================================================
# 9. Time patterns
# =============================================================================
def time_patterns(df):
    blank = empty_fig("Choose a date column in the sidebar to unlock time analysis")
    d = df.dropna(subset=["order_date"]) if "order_date" in df else pd.DataFrame()
    if len(d) < 20:
        return blank, blank, "No usable date column selected (needs 20+ dated orders)."
    d = d.assign(hour=d["order_date"].dt.hour, dow=d["order_date"].dt.dayofweek)
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    if d["hour"].nunique() > 1:
        heat = d.pivot_table(index="dow", columns="hour", values=PRICE, aggfunc="count", fill_value=0)
        heat.index = [days[i] for i in heat.index]
        fig1 = px.imshow(heat, aspect="auto", color_continuous_scale=SCALE, title="Orders by weekday × hour")
    else:
        by = d.groupby("dow")[PRICE].agg(["count", "sum"]).reindex(range(7), fill_value=0)
        fig1 = px.bar(x=days, y=by["sum"], title="Revenue by weekday (dates have no time of day)")
        fig1.update_traces(marker_color=C["cyan"])
    fig1.update_layout(**LAYOUT)

    daily = d.set_index("order_date")[PRICE].resample("D").agg(["sum", "count"])
    daily["rolling"] = daily["sum"].rolling(7, min_periods=1).mean()
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=daily.index, y=daily["sum"], name="Daily revenue", marker_color=C["blue"], opacity=0.6))
    fig2.add_trace(go.Scatter(x=daily.index, y=daily["rolling"], name="7-day average",
                              line=dict(color=C["amber"], width=3)))
    fig2.update_layout(**LAYOUT, title="Daily revenue with 7-day rolling average", legend=dict(orientation="h"))

    best = d.groupby("dow")[PRICE].sum().idxmax()
    md = f"Busiest weekday by revenue: **{days[best]}**. "
    if stats is not None and d["dow"].nunique() > 1:
        arrays = [g[PRICE].to_numpy() for _, g in d.groupby("dow") if len(g) >= 5]
        if len(arrays) > 1:
            p = stats.kruskal(*arrays).pvalue
            md += f"Order value by weekday: {verdict(p)}. "
    if d["hour"].nunique() > 1:
        md += f"Peak hour: **{int(d['hour'].value_counts().idxmax()):02d}:00**."
    return fig1, fig2, md


# =============================================================================
# 10. Customer / order segmentation (K-means)
# =============================================================================
def segmentation(df, k):
    blank = empty_fig()
    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
    except ImportError:
        return blank, blank, pd.DataFrame(), "⚠️ Clustering needs scikit-learn. Run: `pip install scikit-learn`"
    d = df.dropna(subset=[PRICE]).reset_index(drop=True)
    if len(d) < 60:
        return blank, blank, pd.DataFrame(), "Need at least 60 orders for clustering."
    d = d.sample(min(len(d), 20000), random_state=42).reset_index(drop=True)

    feats = pd.DataFrame({"log_price": np.log1p(d[PRICE].clip(lower=0))})
    if "order_date" in d and d["order_date"].notna().mean() > 0.9:
        feats["hour"] = d["order_date"].dt.hour.fillna(0)
        feats["weekday"] = d["order_date"].dt.dayofweek.fillna(0)
    for col in ("cc_provider", "am_pm", "language"):
        top = d[col].value_counts().head(6).index
        feats = feats.join(pd.get_dummies(d[col].where(d[col].isin(top), "Other"), prefix=col, dtype=float))
    Z = ((feats - feats.mean()) / feats.std(ddof=0).replace(0, 1)).fillna(0).to_numpy()

    sub = np.random.default_rng(1).choice(len(Z), min(len(Z), 4000), replace=False)
    ks, inertia, sil = list(range(2, 9)), [], []
    for kk in ks:
        km = KMeans(n_clusters=kk, n_init=3, random_state=42).fit(Z[sub])
        inertia.append(km.inertia_)
        sil.append(silhouette_score(Z[sub], km.labels_))
    scan = go.Figure()
    scan.add_trace(go.Scatter(x=ks, y=inertia, name="Inertia (elbow)", mode="lines+markers",
                              line=dict(color=C["blue"])))
    scan.add_trace(go.Scatter(x=ks, y=sil, name="Silhouette", yaxis="y2", mode="lines+markers",
                              line=dict(color=C["amber"])))
    scan.update_layout(**LAYOUT, title="Choosing k: elbow and silhouette", xaxis_title="Number of clusters k",
                       yaxis2=dict(overlaying="y", side="right", showgrid=False), legend=dict(orientation="h"))

    k = int(k)
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(Z)
    d["cluster"] = (km.labels_ + 1).astype(str)
    score = silhouette_score(Z[sub], km.labels_[sub])
    _, S, Vt = np.linalg.svd(Z - Z.mean(0), full_matrices=False)
    coords = (Z - Z.mean(0)) @ Vt[:2].T
    plot = pd.DataFrame({"PC1": coords[:, 0], "PC2": coords[:, 1], "Cluster": d["cluster"]}).sample(
        min(len(d), 4000), random_state=1)
    scatter = px.scatter(plot, x="PC1", y="PC2", color="Cluster", opacity=0.6,
                         title=f"Clusters projected on 2 principal components (k = {k})",
                         category_orders={"Cluster": sorted(d["cluster"].unique())})
    scatter.update_layout(**LAYOUT)

    def mode(s):
        m = s[~s.isin(UNKNOWN)].mode()
        return m.iat[0] if len(m) else ""

    prof = d.groupby("cluster").agg(Orders=(PRICE, "size"), Avg_price=(PRICE, "mean"), Median_price=(PRICE, "median"),
                                    Top_provider=("cc_provider", mode), Top_AM_PM=("am_pm", mode),
                                    Top_language=("language", mode)).round(2)
    prof["Share %"] = (prof["Orders"] / len(d) * 100).round(1)
    prof = prof.reset_index().rename(columns={"cluster": "Cluster"})
    quality = ("weak structure: the groups overlap heavily, so treat them as rough tendencies"
               if score < 0.25 else "reasonable structure" if score < 0.5 else "strong structure")
    md = (f"Silhouette score for k = {k}: **{score:.2f}** ({quality}). "
          f"Best silhouette in the scan is at k = **{ks[int(np.argmax(sil))]}** ({max(sil):.2f}). "
          "Features: log price, card provider, AM/PM, language"
          + (", hour and weekday" if "hour" in feats else "") + ".")
    return scan, scatter, prof, md


# =============================================================================
# 11. One-click statistical scan
# =============================================================================
def compose_scan(df, dq_md, dist_md, drivers_md, assoc_md, benford_md):
    """Build the auto-scan text from already computed pieces (avoids recomputing)."""
    parts = []
    if dq_md:
        parts.append("### Data quality\n" + dq_md)
    parts += ["### Price distribution\n" + dist_md,
              "### What drives price?\n" + drivers_md,
              "### Are the categorical columns related?\n" + assoc_md,
              "### Benford screening\n" + benford_md,
              "### Revenue concentration\n" + concentration(df, "(individual orders)")[2]]
    return "\n\n".join(parts)


def full_scan(df, raw=None):
    if df is None or df.empty:
        return "Import a file first."
    parts = []
    if raw is not None:
        parts.append("### Data quality\n" + data_quality(raw)[2])
    parts.append("### Price distribution\n" + price_distribution(df)[3])
    parts.append("### What drives price?\n" + drivers(df)[2])
    parts.append("### Are the categorical columns related?\n" + association_matrix(df)[2])
    parts.append("### Benford screening\n" + benford(df)[2])
    parts.append("### Revenue concentration\n" + concentration(df, "(individual orders)")[2])
    return "\n\n".join(parts)