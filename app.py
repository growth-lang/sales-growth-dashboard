# app.py — Tipsoi Sales Growth • Forecasts & Insights
# Polished layout, cohort/funnel, 4-week forecasts w/ 95% bands, anomalies, drivers,
# manual notes, CSV + PDF export (ASCII-safe to avoid FPDF Unicode crash).

import math
from io import BytesIO
from datetime import datetime
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dateutil import tz
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import streamlit as st
from fpdf import FPDF

# ---------- Page config & style ----------
st.set_page_config(page_title="Tipsoi • Sales Growth & Forecasts", layout="wide")
st.markdown("""
    <style>
    .small-note { color:#6b7280; font-size:12px; }
    .kpi-card { padding:14px; border:1px solid #e5e7eb; border-radius:12px; background:#fff; }
    .ok   { color:#059669; } .down { color:#dc2626; } .flat { color:#6b7280; }
    </style>
""", unsafe_allow_html=True)

# ---------- Helpers ----------
EXPECTED = ["date","team","mql","sql","work_orders","mrr_added_bdt","notes"]

def parse_upload(file):
    if file is None: return None
    name = file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(file)
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(file)
    st.error("Please upload a CSV or Excel file.")
    return None

def auto_map_columns(df):
    if df is None or df.empty: return None
    mapping = {}
    norm = {c.strip().lower(): c for c in df.columns}
    for want in EXPECTED:
        for k, orig in norm.items():
            if k == want or k.replace(" ", "_")==want or k.replace(" ","")==want:
                mapping[orig] = want
    ndf = df.rename(columns=mapping).copy()
    for col in EXPECTED:
        if col not in ndf.columns:
            if col in ("team","notes"):
                ndf[col] = "SME Sales" if col=="team" else ""
            else:
                ndf[col] = 0
    ndf["date"] = pd.to_datetime(ndf["date"], errors="coerce")
    ndf = ndf.dropna(subset=["date"])
    for c in ["mql","sql","work_orders","mrr_added_bdt"]:
        ndf[c] = pd.to_numeric(ndf[c], errors="coerce").fillna(0).astype(float)
    ndf["team"] = ndf["team"].replace("", "SME Sales")
    return ndf[EXPECTED].sort_values("date")

def week_floor(ts, week_start="MON"):
    wd = ts.weekday()
    if week_start.upper().startswith("SUN"):
        shift = (wd + 1) % 7
    else:  # Monday
        shift = wd
    return ts - pd.to_timedelta(shift, unit="D")

def weekly_agg(df, week_start="MON"):
    tmp = df.copy()
    tmp["week_start"] = tmp["date"].dt.floor("D").apply(lambda d: week_floor(pd.Timestamp(d), week_start)).dt.date
    g = tmp.groupby(["week_start","team"], as_index=False).agg({
        "mql":"sum","sql":"sum","work_orders":"sum","mrr_added_bdt":"sum"
    })
    g["mql_to_sql_rate"] = np.where(g.mql>0, g.sql/g.mql, np.nan)
    g["sql_to_wo_rate"]  = np.where(g.sql>0, g.work_orders/g.sql, np.nan)
    g["lead_to_wo_rate"] = np.where(g.mql>0, g.work_orders/g.mql, np.nan)
    g = g.sort_values(["team","week_start"])
    g["mrr_prev"] = g.groupby("team")["mrr_added_bdt"].shift(1)
    g["weekly_mrr_growth_pct"] = np.where(g["mrr_prev"]>0,
        ((g["mrr_added_bdt"]-g["mrr_prev"])/g["mrr_prev"])*100.0, np.nan)
    return g

def wow(prev, curr):
    if prev is None or pd.isna(prev): return None, None
    delta = curr - prev
    pct = None if prev==0 else (delta/prev*100.0)
    return delta, pct

def fit_forecast(series, periods=4):
    s = series.dropna().astype(float)
    s.index = pd.to_datetime(s.index)
    out = dict(forecast=None, conf_low=None, conf_high=None, fitted=None, backtest=None, mape=None, rmse=None)
    if len(s) < 4: return out
    try:
        m = ExponentialSmoothing(s, trend="add", seasonal=None, initialization_method="estimated").fit(optimized=True)
    except Exception:
        m = ExponentialSmoothing(s, trend=None, seasonal=None, initialization_method="estimated").fit(optimized=True)
    fitted = m.fittedvalues
    fh = m.forecast(periods)
    resid = s - fitted
    sigma = float(np.nanstd(resid.values)) if len(resid)>1 else 0.0
    conf_low = fh - 1.96*sigma
    conf_high = fh + 1.96*sigma

    # simple walk-forward backtest
    n_bt = min(8, max(0, len(s)-3))
    preds, reals, idx = [], [], []
    for i in range(n_bt, 0, -1):
        train = s.iloc[:-i]
        try:
            mm = ExponentialSmoothing(train, trend="add", seasonal=None, initialization_method="estimated").fit(optimized=True)
        except Exception:
            mm = ExponentialSmoothing(train, trend=None, seasonal=None, initialization_method="estimated").fit(optimized=True)
        preds.append(mm.forecast(1).iloc[0]); reals.append(s.iloc[-i]); idx.append(s.index[-i])
    if preds:
        bt = pd.DataFrame({"actual": reals, "pred": preds}, index=idx)
        mape = float(np.mean([abs((a-p)/a)*100 for a,p in zip(bt.actual, bt.pred) if a!=0])) if any(a!=0 for a in bt.actual) else None
        rmse = float(np.sqrt(np.mean((bt.actual - bt.pred)**2)))
    else:
        bt, mape, rmse = None, None, None

    out.update(forecast=fh, conf_low=conf_low, conf_high=conf_high, fitted=fitted, backtest=bt, mape=mape, rmse=rmse)
    return out

def line_fc(title, s, fc, ylab):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines+markers", name="Actual"))
    if fc["fitted"] is not None:
        fig.add_trace(go.Scatter(x=fc["fitted"].index, y=fc["fitted"].values, name="Fitted", line=dict(dash="dot")))
    if fc["forecast"] is not None:
        fh = fc["forecast"]
        fig.add_trace(go.Scatter(x=fh.index, y=fh.values, mode="lines+markers", name="Forecast", line=dict(dash="dash")))
        fig.add_trace(go.Scatter(
            x=list(fh.index)+list(fh.index[::-1]),
            y=list(fc["conf_high"].values)+list(fc["conf_low"].values[::-1]),
            fill="toself", name="95% CI", opacity=0.15, line=dict(color="rgba(0,0,0,0)")
        ))
    fig.update_layout(title=title, xaxis_title="Week", yaxis_title=ylab, hovermode="x unified")
    return fig

def anomalies(series, fitted, zthr=2.0):
    if series is None or fitted is None: return []
    resid = series - fitted
    mu, sd = float(np.nanmean(resid)), float(np.nanstd(resid))
    if sd==0 or np.isnan(sd): return []
    z = (resid - mu)/sd
    out = z[abs(z)>=zthr]
    return [(i.date(), float(series.loc[i]), float(fitted.loc[i]), float(z.loc[i])) for i in out.index]

def cohort_heatmaps(weekly_df, horizon=8):
    w = weekly_df.sort_values("week_start").copy()
    cohorts = sorted(w["week_start"].unique())
    sql_mat, wo_mat = [], []
    for c in cohorts:
        base = float(w.loc[w.week_start==c, "mql"].sum())
        row_sql, row_wo = [], []
        for k in range(horizon+1):
            end = (pd.Timestamp(c)+pd.Timedelta(days=7*k)).date()
            mask = (w.week_start>=c) & (w.week_start<=end)
            row_sql.append((w.loc[mask,"sql"].sum()/base*100.0) if base>0 else np.nan)
            row_wo.append((w.loc[mask,"work_orders"].sum()/base*100.0) if base>0 else np.nan)
        sql_mat.append(row_sql); wo_mat.append(row_wo)
    y = [str(c) for c in cohorts]; x = list(range(horizon+1))
    f1 = px.imshow(np.array(sql_mat), x=x, y=y, origin="lower",
                   labels=dict(x="Weeks since cohort", y="Cohort (MQL week)", color="% -> SQL"),
                   title="Cohort: % of MQL -> SQL")
    f2 = px.imshow(np.array(wo_mat), x=x, y=y, origin="lower",
                   labels=dict(x="Weeks since cohort", y="Cohort (MQL week)", color="% -> WO"),
                   title="Cohort: % of MQL -> Work Orders")
    return f1, f2

def drivers_panel(weekly_df):
    """Correlational 'drivers' using lagged features & LassoCV."""
    df = weekly_df.sort_values("week_start").copy()
    df["mql_l1"] = df["mql"].shift(1)
    df["sql_l1"] = df["sql"].shift(1)
    df["mrr_l1"] = df["mrr_added_bdt"].shift(1)
    df = df.dropna()
    if len(df) < 8: return None
    X = df[["mql","sql","mrr_added_bdt","mql_l1","sql_l1","mrr_l1"]].values
    y = df["work_orders"].values
    pipe = Pipeline([("scaler", StandardScaler()), ("lr", LassoCV(cv=3, random_state=0))]).fit(X,y)
    coefs = pipe.named_steps["lr"].coef_
    names = ["mql","sql","mrr","mql_l1","sql_l1","mrr_l1"]
    imp = pd.DataFrame({"feature":names,"importance":np.abs(coefs)})
    return imp.sort_values("importance", ascending=False)

# --- ASCII-safe PDF builder (prevents FPDF Unicode crash) ---
def pdf_from_text(txt: str) -> bytes:
    SAFE_MAP = str.maketrans({
        "–": "-", "—": "-", "−": "-",
        "→": "->", "←": "<-",
        "▲": "^", "▼": "v", "✓": "v", "✗": "x",
        "’": "'", "‘": "'", "“": '"', "”": '"',
        "\t": "  ",
    })
    clean = (txt or "").translate(SAFE_MAP)
    clean = clean.encode("latin-1", "replace").decode("latin-1")

    pdf = FPDF()
    pdf.set_margins(15, 15, 15)
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    for line in clean.split("\n"):
        if not line.strip():
            pdf.ln(4)
            continue
        pdf.multi_cell(0, 8, txt=line)
    return pdf.output(dest="S").encode("latin-1")

def seed_demo(n_weeks=16):
    rng = np.random.default_rng(7)
    start = pd.to_datetime("2025-03-03")  # Monday
    weeks = [start + pd.Timedelta(7*i, "D") for i in range(n_weeks)]
    mql = np.clip(100 + np.linspace(0,20,n_weeks) + rng.normal(0,8,n_weeks), 40, None).round()
    sql = (mql*(0.55 + rng.normal(0,0.03,n_weeks))).clip(0).round()
    wo  = (sql*(0.25 + rng.normal(0,0.02,n_weeks))).clip(0).round()
    mrr = (wo*(2500 + rng.normal(0,400,n_weeks))).clip(0).round()
    df = pd.DataFrame({"week_start":[d.date() for d in weeks], "team":"SME Sales",
                       "mql":mql,"sql":sql,"work_orders":wo,"mrr_added_bdt":mrr})
    return df

# ---------- Sidebar ----------
st.sidebar.header("Data")
up = st.sidebar.file_uploader("Upload CSV/XLSX", type=["csv","xlsx","xls"])
week_start = st.sidebar.selectbox("Week starts on", ["MON","SUN"], index=0)
currency = st.sidebar.text_input("Currency label", value="BDT")
st.sidebar.markdown("---")
st.sidebar.caption("No data yet?")
if st.sidebar.button("▶ Load demo data"):
    demo = seed_demo()
    demo["date"] = pd.to_datetime(demo["week_start"])
    demo["notes"] = ""
    demo_df = demo[["date","team","mql","sql","work_orders","mrr_added_bdt","notes"]]
    st.session_state["data"] = demo_df

# Manual quick-add
with st.sidebar.form("quick_add"):
    st.caption("Quick add a daily row")
    dt = st.date_input("Date")
    team = st.text_input("Team", "SME Sales")
    mql = st.number_input("MQL", 0, step=1)
    sql = st.number_input("SQL", 0, step=1)
    wo  = st.number_input("Work Orders", 0, step=1)
    mrr = st.number_input(f"MRR added ({currency})", 0, step=100)
    note = st.text_input("Note (optional)", "")
    add = st.form_submit_button("Add")
if add:
    row = {"date": pd.to_datetime(dt), "team": team or "SME Sales", "mql": mql, "sql": sql,
           "work_orders": wo, "mrr_added_bdt": mrr, "notes": note}
    st.session_state.setdefault("manual_rows", []).append(row)
    st.sidebar.success("Row added (in-memory).")

# ---------- Load & normalize ----------
base = None
if up is not None:
    raw = parse_upload(up)
    base = auto_map_columns(raw)
man = None
if "manual_rows" in st.session_state and st.session_state["manual_rows"]:
    man = auto_map_columns(pd.DataFrame(st.session_state["manual_rows"]))
sess = st.session_state.get("data")
if sess is not None:
    sess = auto_map_columns(sess)
dfs = [x for x in [base, man, sess] if x is not None and not x.empty]
if not dfs:
    st.info("Upload CSV/XLSX, click **Load demo data**, or add rows from the sidebar to begin.")
    st.stop()
df = pd.concat(dfs, ignore_index=True).sort_values("date")

# ---------- Header ----------
st.title("Tipsoi Sales Growth • Forecasts & Insights")
st.caption("WoW/MoM metrics, 4‑week forecasts with 95% bands, funnel & cohorts, anomaly flags, drivers, and exports.")

# ---------- Top filters ----------
cols = st.columns(3)
with cols[0]:
    teams = ["All Teams"] + sorted(df["team"].unique().tolist())
    team_choice = st.selectbox("Team", teams, index=0)
with cols[1]:
    agg_choice = st.selectbox("Compare period", ["WoW","MoM"], index=0)
with cols[2]:
    show_table = st.toggle("Show raw table", value=False)

df_team = df if team_choice=="All Teams" else df[df.team==team_choice]
if show_table:
    st.dataframe(df_team, use_container_width=True, hide_index=True)

# ---------- Weekly aggregates ----------
weekly = weekly_agg(df_team, week_start=week_start)
st.markdown("### Weekly overview")
st.dataframe(weekly, use_container_width=True, hide_index=True)

# KPI cards (last vs prev week/month)
if not weekly.empty:
    by = "week_start"
    last_key = weekly[by].max()
    if agg_choice=="WoW":
        prev_key = weekly.loc[weekly[by] < last_key, by].max() if len(weekly)>=2 else None
    else:  # MoM ~ 4 weeks earlier
        prev_key = weekly.loc[weekly[by] <= (pd.Timestamp(last_key)-pd.Timedelta(days=28)).date(), by].max()

    w_last = weekly[weekly[by]==last_key].sum(numeric_only=True)
    w_prev = weekly[weekly[by]==prev_key].sum(numeric_only=True) if prev_key is not None else None

    def kpi(label, curr, prev, money=False):
        d, p = (None, None) if w_prev is None else wow(prev, curr)
        arrow = "↔"; css = "flat"
        if p is not None:
            if p > 0: arrow, css = "▲", "ok"
            elif p < 0: arrow, css = "▼", "down"
        val = f"{curr:,.0f}{(' '+currency) if money else ''}" if money else int(curr)
        delta = "n/a" if p is None else f"{arrow} {d:+,.0f} ({p:+.1f}%)"
        st.markdown(f"""<div class="kpi-card"><div style="font-size:13px;color:#6b7280">{label}</div>
        <div style="font-size:24px;font-weight:700">{val}</div>
        <div class="{css}" style="font-size:13px">{delta}</div></div>""", unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    with c1: kpi("MQL (last period)", w_last.get("mql",0), 0 if w_prev is None else w_prev.get("mql",0))
    with c2: kpi("SQL (last period)", w_last.get("sql",0), 0 if w_prev is None else w_prev.get("sql",0))
    with c3: kpi("Work Orders (last period)", w_last.get("work_orders",0), 0 if w_prev is None else w_prev.get("work_orders",0))
    with c4: kpi(f"MRR added ({currency})", w_last.get("mrr_added_bdt",0), 0 if w_prev is None else w_prev.get("mrr_added_bdt",0), money=True)

# ---------- Tabs ----------
tab_trend, tab_forecast, tab_funnel, tab_cohort, tab_insights, tab_export = st.tabs(
    ["Trends","Forecasts","Funnel","Cohorts","Insights","Export"]
)

with tab_trend:
    st.subheader("Trends")
    pivot = weekly.groupby("week_start", as_index=True).agg(
        mql=("mql","sum"), sql=("sql","sum"), work_orders=("work_orders","sum"),
        mrr=("mrr_added_bdt","sum")
    ).sort_index()
    for metric, label in [("mql","MQL"), ("sql","SQL"), ("work_orders","Work Orders"), ("mrr","MRR added")]:
        fig = px.line(pivot.reset_index(), x="week_start", y=metric, markers=True,
                      title=f"{label} by week")
        st.plotly_chart(fig, use_container_width=True)

with tab_forecast:
    st.subheader("4‑week forecasts with 95% bands + accuracy")
    pivot = weekly.groupby("week_start", as_index=True).agg(
        mql=("mql","sum"), sql=("sql","sum"), work_orders=("work_orders","sum"),
        mrr=("mrr_added_bdt","sum")
    ).sort_index()
    for col, label in [("mql","MQL"), ("sql","SQL"), ("work_orders","Work Orders"), ("mrr","MRR added")]:
        s = pivot[col].copy(); s.index = pd.to_datetime(s.index)
        fc = fit_forecast(s, periods=4)
        fig = line_fc(f"{label} — forecast", s, fc, f"{label} ({currency})" if col=="mrr" else label)
        st.plotly_chart(fig, use_container_width=True)
        c1,c2,c3 = st.columns(3)
        with c1: st.caption("Backtest points"); st.write(0 if fc["backtest"] is None else len(fc["backtest"]))
        with c2: st.caption("MAPE (↓ better)"); st.write("n/a" if fc["mape"] is None else f"{fc['mape']:.1f}%")
        with c3: st.caption("RMSE (↓ better)"); st.write("n/a" if fc["rmse"] is None else f"{fc['rmse']:.1f}")
        an = anomalies(s, fc["fitted"], 2.0)
        if an:
            st.warning("Anomalies detected (|z| ≥ 2):")
            st.dataframe(pd.DataFrame(an, columns=["week","actual","fitted","zscore"]), use_container_width=True)
        st.markdown("---")

with tab_funnel:
    st.subheader("Funnel (last week)")
    last = weekly.groupby("week_start", as_index=True).agg(
        mql=("mql","sum"), sql=("sql","sum"), work_orders=("work_orders","sum")
    ).sort_index().tail(1)
    if not last.empty:
        vals = last.iloc[0]
        fdf = pd.DataFrame({"Stage":["MQL","SQL","Work Orders"], "Value":[vals.mql, vals.sql, vals.work_orders]})
        st.plotly_chart(px.funnel(fdf, x="Value", y="Stage"), use_container_width=True)
    rates = weekly.groupby("week_start", as_index=True).agg(
        mql_to_sql=("mql_to_sql_rate","mean"),
        sql_to_wo=("sql_to_wo_rate","mean"),
        lead_to_wo=("lead_to_wo_rate","mean")
    ).sort_index()
    rates = (rates*100).round(2)
    st.dataframe(rates.reset_index().rename(columns={
        "mql_to_sql":"MQL->SQL %", "sql_to_wo":"SQL->WO %", "lead_to_wo":"Lead->WO %"
    }), use_container_width=True, hide_index=True)

with tab_cohort:
    st.subheader("Cohort heatmaps")
    wk = weekly.groupby("week_start", as_index=False).agg(
        mql=("mql","sum"), sql=("sql","sum"), work_orders=("work_orders","sum")
    )
    if len(wk) >= 2:
        f1, f2 = cohort_heatmaps(wk, horizon=8)
        st.plotly_chart(f1, use_container_width=True)
        st.plotly_chart(f2, use_container_width=True)
    else:
        st.info("Need at least 2 weeks to render cohorts.")

with tab_insights:
    st.subheader("Bottlenecks & Drivers")
    wk = weekly.sort_values("week_start").copy()
    wk["mql_to_sql_pct"] = wk["mql_to_sql_rate"]*100
    wk["sql_to_wo_pct"]  = wk["sql_to_wo_rate"]*100
    wo_change = None
    if len(wk) >= 2:
        wk["wo_delta"] = wk["work_orders"].diff()
        wo_change = wk[["week_start","wo_delta"]].tail(1).values[0]
    bullets = []
    if wo_change is not None:
        d = int(wo_change[1])
        if d < 0: bullets.append(f"Work Orders fell by {abs(d)} vs prior week.")
        elif d > 0: bullets.append(f"Work Orders rose by {d} vs prior week.")
        else: bullets.append("Work Orders unchanged vs prior week.")
    if len(wk) >= 2:
        for col, lbl in [("mql_to_sql_pct","MQL->SQL"), ("sql_to_wo_pct","SQL->WO")]:
            prev, curr = wk[col].iloc[-2], wk[col].iloc[-1]
            if not (pd.isna(prev) or pd.isna(curr)):
                if curr < prev: bullets.append(f"{lbl} rate down {prev-curr:.1f} pts — potential bottleneck.")
                elif curr > prev: bullets.append(f"{lbl} rate up {curr-prev:.1f} pts.")
    if bullets:
        st.markdown("- " + "\n- ".join(bullets))
    imp = drivers_panel(weekly.groupby("week_start", as_index=False).agg({
        "mql":"sum","sql":"sum","work_orders":"sum","mrr_added_bdt":"sum"}))
    if imp is not None:
        fig = px.bar(imp, x="feature", y="importance", title="Drivers of Work Orders (correlational)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Need ~8+ weekly points for drivers.")
    st.markdown("### Manual reasoning (optional)")
    st.caption("Add your context/explanation; it will appear in PDF export.")
    st.session_state.setdefault("notes", {})
    if not weekly.empty:
        sel = st.selectbox("Week", sorted(weekly["week_start"].unique()), index=len(weekly["week_start"].unique())-1)
        txt = st.text_area("Note", value=st.session_state["notes"].get(str(sel),""), height=120)
        if st.button("Save note"):
            st.session_state["notes"][str(sel)] = txt
            st.success("Saved.")

with tab_export:
    st.subheader("Export")
    out_csv = weekly.to_csv(index=False).encode("utf-8")
    st.download_button("Download weekly aggregates (CSV)", out_csv, "weekly_aggregates.csv", "text/csv")
    if st.session_state.get("notes"):
        notes_df = pd.DataFrame([{"week_start":k,"note":v} for k,v in st.session_state["notes"].items()])
        st.download_button("Download manual notes (CSV)", notes_df.to_csv(index=False).encode("utf-8"),
                           "manual_notes.csv","text/csv")
    if not weekly.empty:
        last = weekly["week_start"].max()
        wk = weekly[weekly["week_start"]==last].groupby("week_start", as_index=False).agg({
            "mql":"sum","sql":"sum","work_orders":"sum","mrr_added_bdt":"sum",
            "mql_to_sql_rate":"mean","sql_to_wo_rate":"mean","lead_to_wo_rate":"mean",
            "weekly_mrr_growth_pct":"mean"
        }).iloc[0]
        lines = [
            f"Week starting: {wk['week_start']}",
            f"MQL: {int(wk['mql'])}",
            f"SQL: {int(wk['sql'])} | MQL->SQL: {wk['mql_to_sql_rate']*100:.1f}%",
            f"Work Orders: {int(wk['work_orders'])} | SQL->WO: {wk['sql_to_wo_rate']*100:.1f}%",
            f"Lead->WO: {wk['lead_to_wo_rate']*100:.1f}%",
            f"MRR added: {wk['mrr_added_bdt']:,.0f} {currency}",
        ]
        if not math.isnan(wk["weekly_mrr_growth_pct"]):
            lines.append(f"Weekly MRR Growth: {wk['weekly_mrr_growth_pct']:.2f}%")
        note = st.session_state["notes"].get(str(wk["week_start"]), "")
        if note:
            lines += ["", "Manual Note:", note]
        pdf_bytes = pdf_from_text("\n".join(lines))
        st.download_button("Download PDF summary", pdf_bytes, "sales_weekly_report.pdf", "application/pdf")

st.markdown('<div class="small-note">Tip: Use the sidebar to change week start (Mon/Sun), currency label, and to add quick daily rows. The app learns new data immediately.</div>', unsafe_allow_html=True)
