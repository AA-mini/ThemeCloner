# theme_analysis.py
#
# Extended analysis on top of the baskets produced by theme_engine.py.
# Reads the output CSVs — no need to rerun the engine just to change a chart.
#
# Three things this adds beyond the basic tracking chart:
#   1. SMB adjustment — strips out the size premium so the performance gap
#      between the basket and the ETF isn't mistaken for tracking failure
#   2. Systematic decomposition — projects both basket and ETF onto the
#      RP-PCA factors and shows the factor-explained components track much
#      more tightly (the residual gap is purely idiosyncratic noise)
#   3. Rolling 12m correlation — shows whether the relationship is stable
#
# Run:
#   python theme_analysis.py               (sector baskets, the default)
#   python theme_analysis.py --mode unconstrained

# %%

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from numpy.linalg import lstsq
import warnings
warnings.filterwarnings("ignore")

# %%

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["sector", "unconstrained"],
                    default="sector")
args   = parser.parse_args()

BASKET_MODE = args.mode
OUTPUT_FILE = f"theme_analysis_{BASKET_MODE}.png"

# %%

print("loading data...")

F_df = pd.read_csv("factor_timeseries.csv", index_col=0, parse_dates=True)
F_df.columns = [f"F{i+1}" for i in range(F_df.shape[1])]
F_df.index   = F_df.index.to_period("M")

# FF3/FF5 saved files only go to 2017, which predates all the ETFs we're using.
# Instead we build a simple size factor from the CRSP data directly:
# long bottom half by market cap, short top half, each month.
# This is more appropriate anyway since our universe is mid-cap throughout.
print("building size factor from CRSP mktcap...")
stocks_df_full = pd.read_csv("russell1000_exsp500_returns.csv", low_memory=False)
stocks_df_full["ym"]     = pd.to_datetime(stocks_df_full["date"]).dt.to_period("M")
stocks_df_full["mktcap"] = pd.to_numeric(stocks_df_full["mktcap"], errors="coerce")

def build_size_factor(df):
    rows = []
    for period, grp in df.groupby("ym"):
        g = grp.dropna(subset=["mktcap", "ret"])
        if len(g) < 20:
            continue
        med   = g["mktcap"].median()
        small = g.loc[g["mktcap"] <= med, "ret"].mean()
        large = g.loc[g["mktcap"] >  med, "ret"].mean()
        rows.append({"ym": period, "SMB": small - large})
    return pd.DataFrame(rows).set_index("ym")

size_factor = build_size_factor(stocks_df_full)
size_factor.index = pd.PeriodIndex(size_factor.index, freq="M")
print(f"  size factor: {size_factor.index.min()} to {size_factor.index.max()}")

etf_df = pd.read_csv("etf_theme_returns.csv", index_col=0)
etf_df.index = pd.PeriodIndex(etf_df.index, freq="M")
etf_df = etf_df.sort_index()

stocks_df  = stocks_df_full
baskets_df = pd.read_csv(f"theme_baskets_{BASKET_MODE}.csv")
betas_df   = pd.read_csv("stock_factor_betas.csv", index_col=0)
ticker_to_permno = betas_df.reset_index().set_index("ticker")["permno"].to_dict()

# theme names come from the same config used by theme_engine
themes_cfg  = pd.read_csv("themes_config.csv")
THEME_NAMES = dict(zip(themes_cfg["etf"], themes_cfg["theme_name"]))

print(f"  factors  : {F_df.shape}")
print(f"  ETFs     : {etf_df.shape}")
print(f"  stocks   : {len(stocks_df):,} rows")
print(f"  baskets  : {baskets_df['etf'].nunique()} themes, {BASKET_MODE} mode")

# %%

# build a return matrix for just the stocks in the baskets — faster than
# pivoting the full 160k-row dataset every time
needed_tickers = baskets_df["ticker"].dropna().unique().tolist()
needed_permnos = [ticker_to_permno[t] for t in needed_tickers if t in ticker_to_permno]

stock_sub  = stocks_df[stocks_df["permno"].isin(needed_permnos)]
stock_wide = stock_sub.pivot_table(
    index="ym", columns="permno", values="ret", aggfunc="first"
)
if not isinstance(stock_wide.index, pd.PeriodIndex):
    stock_wide.index = pd.PeriodIndex(stock_wide.index, freq="M")

print(f"\n  basket stock matrix: {stock_wide.shape}")

# %%

def ols(y, X):
    coef, _, _, _ = lstsq(X, y, rcond=None)
    fitted = X @ coef
    r2 = 1 - np.var(y - fitted) / np.var(y)
    return coef, fitted, r2

# %%

themes_in_basket = baskets_df["etf"].unique().tolist()
n   = len(themes_in_basket)
fig = plt.figure(figsize=(20, 5.5 * n))
outer_gs = gridspec.GridSpec(n, 1, hspace=0.6)

summary_rows = []

for row_idx, etf_ticker in enumerate(themes_in_basket):
    theme_name = THEME_NAMES.get(etf_ticker, etf_ticker)
    print(f"\n--- {etf_ticker} ({theme_name}) ---")

    basket_tickers = baskets_df.loc[baskets_df["etf"] == etf_ticker, "ticker"].tolist()
    basket_permnos = [
        ticker_to_permno[t] for t in basket_tickers
        if t in ticker_to_permno and ticker_to_permno[t] in stock_wide.columns
    ]

    if len(basket_permnos) < 5:
        print(f"  not enough stocks in return matrix, skipping")
        continue

    if etf_ticker not in etf_df.columns:
        print(f"  {etf_ticker} not found in ETF returns, skipping")
        continue

    basket_ret = stock_wide[basket_permnos].mean(axis=1)

    # align basket, ETF, size factor, and RP-PCA factors to the same months
    aligned = (
        pd.DataFrame({"basket": basket_ret})
        .join(etf_df[[etf_ticker]].rename(columns={etf_ticker: "etf"}), how="inner")
        .join(size_factor[["SMB"]].rename(columns={"SMB": "smb"}), how="inner")
        .join(F_df, how="inner")
        .dropna()
    )

    if len(aligned) < 12:
        print(f"  only {len(aligned)} overlapping months, skipping")
        continue

    b      = aligned["basket"].values
    e      = aligned["etf"].values
    smb_v  = aligned["smb"].values
    F_vals = aligned[[f"F{i+1}" for i in range(F_df.shape[1])]].values
    dates  = aligned.index.to_timestamp()
    T      = len(aligned)

    # raw tracking
    corr_raw = np.corrcoef(b, e)[0, 1]
    te_raw   = (b - e).std(ddof=1) * np.sqrt(12) * 100
    X_etf    = np.column_stack([np.ones(T), e])
    coef_raw, _, r2_raw = ols(b, X_etf)
    beta_raw  = coef_raw[1]
    alpha_raw = coef_raw[0] * 12 * 100
    print(f"  raw   : corr={corr_raw:.3f}  beta={beta_raw:.3f}  TE={te_raw:.1f}%  R2={r2_raw:.3f}")

    # beta-scaled — divide the basket by its full-period beta to force unit loading.
    # correlation doesn't change, but TE shrinks when beta != 1
    b_beta    = b / beta_raw
    te_beta   = (b_beta - e).std(ddof=1) * np.sqrt(12) * 100
    coef_beta, _, r2_beta = ols(b_beta, X_etf)
    alpha_beta = coef_beta[0] * 12 * 100
    print(f"  scaled: corr={corr_raw:.3f}  TE={te_beta:.1f}%  R2={r2_beta:.3f}")

    # SMB adjustment — strip out the size premium the basket earns vs the ETF.
    # mid-caps structurally outperform large-caps, so part of the return gap is
    # mechanical size exposure, not a tracking failure
    X_smb  = np.column_stack([np.ones(T), smb_v])
    coef_smb, _, _ = ols(b, X_smb)
    beta_smb = coef_smb[1]
    b_adj    = b - beta_smb * smb_v
    corr_adj = np.corrcoef(b_adj, e)[0, 1]
    te_adj   = (b_adj - e).std(ddof=1) * np.sqrt(12) * 100
    coef_adj, _, r2_adj = ols(b_adj, X_etf)
    alpha_adj = coef_adj[0] * 12 * 100
    print(f"  adj   : corr={corr_adj:.3f}  TE={te_adj:.1f}%  R2={r2_adj:.3f}  "
          f"(SMB beta={beta_smb:.3f})")

    # systematic decomposition — project both basket and ETF onto the RP-PCA
    # factors. the "systematic" part is what the factors explain; what's left
    # is idiosyncratic noise the engine was never designed to match
    X_F = np.column_stack([np.ones(T), F_vals])
    _, basket_sys, r2_b_sys = ols(b, X_F)
    _, etf_sys,    r2_e_sys = ols(e, X_F)
    corr_sys = np.corrcoef(basket_sys, etf_sys)[0, 1]
    te_sys   = (basket_sys - etf_sys).std(ddof=1) * np.sqrt(12) * 100
    print(f"  sys   : corr={corr_sys:.3f}  TE={te_sys:.1f}%  "
          f"(basket R2={r2_b_sys:.3f}, ETF R2={r2_e_sys:.3f})")

    # rolling 12m correlation — stability check
    rolling_corr = pd.Series(b, index=aligned.index).rolling(12).corr(
        pd.Series(e, index=aligned.index)
    )

    summary_rows.append({
        "Theme":          theme_name,
        "ETF":            etf_ticker,
        "N months":       T,
        "Corr (raw)":     round(corr_raw, 3),
        "Corr (scaled)":  round(corr_raw, 3),   # same, scaling doesn't change corr
        "Corr (sys)":     round(corr_sys, 3),
        "Beta (raw)":     round(beta_raw, 3),
        "SMB beta":       round(beta_smb, 3),
        "TE raw (%)":     round(te_raw, 2),
        "TE scaled (%)":  round(te_beta, 2),
        "TE sys (%)":     round(te_sys, 2),
        "R2 (raw)":       round(r2_raw, 3),
        "R2 (scaled)":    round(r2_beta, 3),
        "Alpha raw (%)":  round(alpha_raw, 2),
        "Alpha adj (%)":  round(alpha_adj, 2),
    })

    # ---- charts ----

    inner_gs = gridspec.GridSpecFromSubplotSpec(
        2, 2, subplot_spec=outer_gs[row_idx],
        hspace=0.45, wspace=0.35, width_ratios=[2.5, 1]
    )

    def cumret(r):
        c = (1 + pd.Series(r)).cumprod()
        return c / c.iloc[0] * 100

    # panel A: cumulative returns — raw, beta-scaled, size-adjusted vs ETF
    ax_cum = fig.add_subplot(inner_gs[0, 0])
    ax_cum.plot(dates, cumret(e).values,
                color="darkorange", lw=2, label=f"{etf_ticker} ETF", zorder=3)
    ax_cum.plot(dates, cumret(b).values,
                color="steelblue", lw=1.8, ls="--",
                label=f"Basket (raw, b={beta_raw:.2f})", zorder=3)
    ax_cum.plot(dates, cumret(b_beta).values,
                color="mediumpurple", lw=1.8, ls="-.",
                label=f"Basket (beta-scaled)", zorder=3)
    ax_cum.plot(dates, cumret(b_adj).values,
                color="mediumseagreen", lw=1.5, ls=":",
                label=f"Basket (size-adj, b_SMB={beta_smb:.2f})", zorder=3)
    ax_cum.fill_between(dates, cumret(e).values, cumret(b_beta).values,
                        alpha=0.08, color="mediumpurple")
    ax_cum.axhline(100, color="gray", ls=":", alpha=0.4, lw=1)
    ax_cum.set_title(f"{theme_name} — Cumulative Performance",
                     fontsize=10, fontweight="bold")
    ax_cum.set_ylabel("Cum. return (base=100)")
    ax_cum.legend(fontsize=8)
    ax_cum.grid(alpha=0.25)

    # panel B: systematic components only — shows how tight the factor-level
    # tracking is once you strip out idiosyncratic noise from both sides
    ax_sys = fig.add_subplot(inner_gs[1, 0])
    ax_sys.plot(dates, cumret(etf_sys).values,
                color="darkorange", lw=2, label="ETF (systematic)", zorder=3)
    ax_sys.plot(dates, cumret(basket_sys).values,
                color="steelblue", lw=1.8, ls="--",
                label="Basket (systematic)", zorder=3)
    ax_sys.fill_between(dates, cumret(etf_sys).values, cumret(basket_sys).values,
                        alpha=0.1, color="steelblue")
    ax_sys.axhline(100, color="gray", ls=":", alpha=0.4, lw=1)
    ax_sys.set_title(
        f"Systematic component only  (corr={corr_sys:.3f}, TE={te_sys:.1f}%)",
        fontsize=10, fontweight="bold"
    )
    ax_sys.set_ylabel("Cum. return (base=100)")
    ax_sys.legend(fontsize=8)
    ax_sys.grid(alpha=0.25)

    # panel C: summary table — the three tracking views side by side
    ax_tbl = fig.add_subplot(inner_gs[0, 1])
    ax_tbl.axis("off")
    tbl_data = [
        ["",             "Raw",               "Scaled",            "Systematic"],
        ["Correlation",  f"{corr_raw:.3f}",   f"{corr_raw:.3f}",   f"{corr_sys:.3f}"],
        ["Beta to ETF",  f"{beta_raw:.3f}",   "1.00",              "—"],
        ["Track. Error", f"{te_raw:.1f}%",    f"{te_beta:.1f}%",   f"{te_sys:.1f}%"],
        ["R2",           f"{r2_raw:.3f}",     f"{r2_beta:.3f}",    "—"],
        ["Alpha (ann.)", f"{alpha_raw:.1f}%", f"{alpha_adj:.1f}%", "—"],
        ["SMB beta",     f"{beta_smb:.3f}",   "—",                 "—"],
    ]
    tbl = ax_tbl.table(
        cellText=tbl_data[1:], colLabels=tbl_data[0],
        loc="center", cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1.1, 1.7)
    for j in range(4):
        tbl[0, j].set_facecolor("#2c5f8a")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    for i in range(1, len(tbl_data)):
        tbl[i, 3].set_facecolor("#e8f5e9")   # highlight systematic column
    for i in range(1, len(tbl_data)):
        for j in range(3):
            tbl[i, j].set_facecolor("#f0f4f8" if i % 2 == 0 else "white")
    ax_tbl.set_title("Tracking breakdown", fontsize=10, fontweight="bold", pad=8)

    # panel D: rolling correlation over time
    ax_roll = fig.add_subplot(inner_gs[1, 1])
    ax_roll.plot(dates, rolling_corr.values,
                 color="steelblue", lw=1.8, label="12m rolling corr")
    ax_roll.axhline(corr_raw, color="gray", ls="--", lw=1, alpha=0.6,
                    label=f"full-period ({corr_raw:.2f})")
    ax_roll.axhline(0.5, color="red", ls=":", lw=1, alpha=0.4)
    ax_roll.set_ylim(-0.2, 1.1)
    ax_roll.set_title("Rolling 12m correlation", fontsize=10, fontweight="bold")
    ax_roll.legend(fontsize=8)
    ax_roll.grid(alpha=0.25)

    print(f"  chart done for {etf_ticker}")

# %%

mode_label = "GICS-Filtered" if BASKET_MODE == "sector" else "Unconstrained"
plt.suptitle(
    f"Theme Engine — Extended Analysis [{mode_label}]\n"
    f"Green = systematic tracking (the factor-level story)",
    fontsize=13, fontweight="bold", y=1.01
)
plt.savefig(OUTPUT_FILE, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nsaved {OUTPUT_FILE}")

# %%

print("\n--- summary ---")
summary = pd.DataFrame(summary_rows)
summary_T = summary.set_index("Theme").drop(columns="ETF").T
print(summary_T.to_string())
summary.to_csv(f"theme_analysis_{BASKET_MODE}.csv", index=False)
print(f"saved theme_analysis_{BASKET_MODE}.csv")
