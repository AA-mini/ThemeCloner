# theme_engine.py
#
# Takes the RP-PCA factors estimated on the OpenSourceAP universe and uses them
# as a common coordinate system to extend large-cap thematic ETF baskets into
# mid-cap stocks where analyst coverage is thin.
#
# The idea: estimate each ETF's "factor fingerprint" (its exposures to the five
# RP-PCA factors), then find mid-cap stocks with the most similar profile using
# cosine similarity. Sector filtering via GICS Level 2 keeps the baskets clean.
#
# Run:
#   python theme_engine.py              (sector-constrained, the default)
#   python theme_engine.py --mode unconstrained
#
# Themes, ETF seeds, and GICS filters are all read from themes_config.csv —
# no hardcoding here. Add or change a theme by editing that file.

# %%

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from numpy.linalg import lstsq
from scipy.optimize import nnls
from sklearn.preprocessing import normalize
import warnings
warnings.filterwarnings("ignore")

# %%

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["sector", "unconstrained"],
                    default="sector",
                    help="sector = GICS-filtered; unconstrained = pure factor match")
args = parser.parse_args()

ENFORCE_SECTOR = (args.mode == "sector")
BASKET_SIZE    = 25
MIN_MONTHS     = 36   # minimum return history needed to estimate factor betas

# %%

themes_cfg = pd.read_csv("themes_config.csv")
themes_cfg["gics_l2_codes"] = themes_cfg["gics_l2_codes"].astype(str)

THEME_GICS = {}
for _, row in themes_cfg.iterrows():
    codes = [int(c.strip()) for c in row["gics_l2_codes"].split(",")]
    THEME_GICS[row["etf"]] = codes

THEMES      = dict(zip(themes_cfg["etf"], themes_cfg["theme_name"]))
THEME_NAMES = THEMES

gics_map = pd.read_csv("sic_to_gics_l2.csv")

def sic_to_gics_l2(sic):
    if pd.isna(sic):
        return None
    sic = int(sic)
    row = gics_map[(gics_map["sic_low"] <= sic) & (gics_map["sic_high"] >= sic)]
    return int(row["gics_l2_code"].iloc[0]) if len(row) > 0 else None

# %%

print("loading inputs...")

F_df = pd.read_csv("factor_timeseries.csv", index_col=0, parse_dates=True)
F_df.columns = [f"F{i+1}" for i in range(F_df.shape[1])]
F_df.index   = F_df.index.to_period("M")
factor_cols  = F_df.columns.tolist()

etf_df = pd.read_csv("etf_theme_returns.csv", index_col=0)
etf_df.index = pd.PeriodIndex(etf_df.index, freq="M")
etf_df = etf_df.sort_index()

stocks_df = pd.read_csv("russell1000_exsp500_returns.csv", low_memory=False)
stocks_df["date"] = pd.to_datetime(stocks_df["date"])
stocks_df["ym"]   = stocks_df["date"].dt.to_period("M")

print(f"  factors : {F_df.shape}")
print(f"  ETFs    : {etf_df.shape}  {etf_df.columns.tolist()}")
print(f"  stocks  : {len(stocks_df):,} rows, {stocks_df['permno'].nunique():,} stocks")

# %%

print(f"\nstep 1: ETF factor fingerprints  [{args.mode} mode]")

common_etf = F_df.index.intersection(etf_df.index)
F_etf      = F_df.loc[common_etf].values
E_etf      = etf_df.loc[common_etf]
etf_betas  = {}

for ticker, theme in THEMES.items():
    if ticker not in etf_df.columns:
        print(f"  {ticker}: not in ETF file, skipping")
        continue
    e    = E_etf[ticker].values
    mask = ~np.isnan(e)
    if mask.sum() < 12:
        print(f"  {ticker}: only {mask.sum()} months of data, skipping")
        continue
    X_   = np.column_stack([np.ones(mask.sum()), F_etf[mask]])
    coef, _, _, _ = lstsq(X_, e[mask], rcond=None)
    beta = coef[1:]
    r2   = 1 - np.var(e[mask] - X_ @ coef) / np.var(e[mask])
    etf_betas[ticker] = beta
    print(f"  {ticker} ({theme}): {mask.sum()} months, R2={r2:.3f}")

# %%

print("\nstep 2: stock factor betas")

stock_wide = stocks_df.pivot_table(
    index="ym", columns="permno", values="ret", aggfunc="first"
)
if not isinstance(stock_wide.index, pd.PeriodIndex):
    stock_wide.index = pd.PeriodIndex(stock_wide.index, freq="M")

common_stk = F_df.index.intersection(stock_wide.index)
F_stk      = F_df.loc[common_stk].values
S_wide     = stock_wide.loc[common_stk]
print(f"  {len(common_stk)} months x {S_wide.shape[1]} stocks")

stock_betas, stock_r2 = {}, {}
permnos = S_wide.columns.tolist()

for i, permno in enumerate(permnos):
    if i % 500 == 0:
        print(f"  {i}/{len(permnos)}...")
    s    = S_wide[permno].values
    mask = ~np.isnan(s)
    if mask.sum() < MIN_MONTHS:
        continue
    X_   = np.column_stack([np.ones(mask.sum()), F_stk[mask]])
    coef, _, _, _ = lstsq(X_, s[mask], rcond=None)
    stock_betas[permno] = coef[1:]
    stock_r2[permno]    = 1 - np.var(s[mask] - X_ @ coef) / np.var(s[mask])

beta_df = pd.DataFrame(stock_betas, index=factor_cols).T
beta_df.index.name = "permno"
beta_df = beta_df.join(pd.Series(stock_r2, name="r2"))

meta = (
    stocks_df.sort_values("date").groupby("permno").last()
    [["ticker", "comnam", "siccd"]]
    if "ticker" in stocks_df.columns else pd.DataFrame()
)
beta_df = beta_df.join(meta, how="left")
beta_df["gics_l2"] = pd.to_numeric(
    beta_df["siccd"], errors="coerce"
).apply(sic_to_gics_l2)

print(f"  {len(beta_df):,} stocks with enough history")

# %%

print("\nstep 3: matching stocks to themes")

results = {}
baskets = {}

for ticker, theme in THEMES.items():
    if ticker not in etf_betas:
        continue

    etf_beta = etf_betas[ticker]
    etf_norm = etf_beta / (np.linalg.norm(etf_beta) + 1e-10)

    if ENFORCE_SECTOR and ticker in THEME_GICS:
        valid_gics  = THEME_GICS[ticker]
        df_filtered = beta_df[beta_df["gics_l2"].isin(valid_gics)].copy()
        if len(df_filtered) < 10:
            print(f"  {ticker}: GICS filter too tight ({len(df_filtered)} stocks), falling back")
            df_filtered = beta_df.copy()
    else:
        df_filtered = beta_df.copy()

    bmat  = df_filtered[factor_cols].values
    bnorm = normalize(bmat, norm="l2")
    sim   = bnorm @ etf_norm
    df_filtered = df_filtered.copy()
    df_filtered["similarity"] = sim

    ranked = (
        df_filtered[["ticker", "comnam", "siccd", "gics_l2", "r2", "similarity"]]
        .sort_values("similarity", ascending=False)
    )
    ranked = ranked[ranked["r2"] > 0.05]
    top    = ranked.head(BASKET_SIZE).copy()
    top["rank"]       = range(1, len(top) + 1)
    top["weight_eq"]  = 1 / len(top)
    top["weight_sim"] = top["similarity"] / top["similarity"].sum()

    B_top = df_filtered.loc[top.index, factor_cols].values
    w_raw, _ = nnls(B_top.T, etf_beta)
    if w_raw.sum() > 1e-8:
        top["weight_nnls"] = w_raw / w_raw.sum()
    else:
        top["weight_nnls"] = 1 / len(top)

    top["theme"] = theme
    top["etf"]   = ticker
    results[ticker] = top
    baskets[ticker] = [p for p in top.index if p in S_wide.columns]

    mode_str = "GICS-filtered" if ENFORCE_SECTOR else "unconstrained"
    print(f"\n  {ticker} - {theme}  ({mode_str})")
    print(f"  {'rank':>4}  {'ticker':>7}  {'company':>30}  "
          f"{'sim':>6}  {'R2':>5}  {'w_nnls':>7}")
    print("  " + "-" * 68)
    for _, row in top.head(15).iterrows():
        nm = str(row.get("comnam", ""))[:28] if pd.notna(row.get("comnam")) else ""
        tk = str(row.get("ticker", ""))[:7]  if pd.notna(row.get("ticker")) else ""
        print(f"  {int(row['rank']):>4}  {tk:>7}  {nm:>30}  "
              f"{row['similarity']:>6.3f}  {row['r2']:>5.3f}  "
              f"{row['weight_nnls']:>7.3f}")

# %%

print("\nstep 4: tracking performance")

tracking_rows = []
n_themes = len([t for t in THEMES if t in etf_betas])
fig = plt.figure(figsize=(18, 4 * n_themes))
gs  = gridspec.GridSpec(n_themes, 2, width_ratios=[3, 1], hspace=0.5, wspace=0.3)

def norm100(s):
    c = (1 + s).cumprod()
    return c / c.iloc[0] * 100

row_idx = 0
for ticker, theme in THEMES.items():
    if ticker not in etf_betas or ticker not in baskets:
        continue

    permnos_basket = baskets[ticker]
    if len(permnos_basket) < 5:
        continue

    basket_ret = S_wide[permnos_basket].mean(axis=1)

    top_df  = results[ticker]
    nnls_w  = pd.Series(
        top_df["weight_nnls"].values,
        index=[p for p in top_df.index if p in S_wide.columns]
    )
    nnls_w  = nnls_w[nnls_w.index.isin(S_wide.columns)]
    nnls_w  = nnls_w / nnls_w.sum()
    basket_nnls = S_wide[nnls_w.index].multiply(nnls_w.values, axis=1).sum(axis=1)

    etf_ret  = etf_df[ticker].reindex(S_wide.index)
    combined = pd.DataFrame({
        "basket":      basket_ret,
        "basket_nnls": basket_nnls,
        "etf":         etf_ret,
    }).dropna()

    if len(combined) < 12:
        print(f"  {ticker}: not enough overlap, skipping")
        continue

    b_ret      = combined["basket"]
    b_nnls_ret = combined["basket_nnls"]
    e_ret      = combined["etf"]

    corr_eq = b_ret.corr(e_ret)
    te_eq   = (b_ret - e_ret).std(ddof=1) * np.sqrt(12) * 100
    X_ols   = np.column_stack([np.ones(len(e_ret)), e_ret.values])
    coef_eq, _, _, _ = lstsq(X_ols, b_ret.values, rcond=None)
    alpha_eq = coef_eq[0] * 12 * 100
    beta_eq  = coef_eq[1]
    r2_eq    = 1 - np.var(b_ret.values - X_ols @ coef_eq) / np.var(b_ret.values)
    ir_eq    = (alpha_eq / 100) / (te_eq / 100) if te_eq > 0 else np.nan

    corr_nn = b_nnls_ret.corr(e_ret)
    te_nn   = (b_nnls_ret - e_ret).std(ddof=1) * np.sqrt(12) * 100
    coef_nn, _, _, _ = lstsq(X_ols, b_nnls_ret.values, rcond=None)
    alpha_nn = coef_nn[0] * 12 * 100
    beta_nn  = coef_nn[1]
    r2_nn    = 1 - np.var(b_nnls_ret.values - X_ols @ coef_nn) / np.var(b_nnls_ret.values)
    ir_nn    = (alpha_nn / 100) / (te_nn / 100) if te_nn > 0 else np.nan

    tracking_rows.append({
        "Theme":         theme,
        "ETF":           ticker,
        "N months":      len(combined),
        "Corr (eq-wt)":  round(corr_eq, 3),
        "Corr (NNLS)":   round(corr_nn, 3),
        "Beta (eq-wt)":  round(beta_eq, 3),
        "Beta (NNLS)":   round(beta_nn, 3),
        "Alpha (eq-wt)": round(alpha_eq, 2),
        "Alpha (NNLS)":  round(alpha_nn, 2),
        "TE (eq-wt)":    round(te_eq, 2),
        "TE (NNLS)":     round(te_nn, 2),
        "R2 (eq-wt)":    round(r2_eq, 3),
        "R2 (NNLS)":     round(r2_nn, 3),
        "IR (eq-wt)":    round(ir_eq, 3),
        "IR (NNLS)":     round(ir_nn, 3),
    })

    print(f"\n  {ticker} ({theme}):")
    print(f"    {'metric':<20} {'eq-wt':>8}  {'NNLS':>8}")
    print(f"    {'correlation':<20} {corr_eq:>8.3f}  {corr_nn:>8.3f}")
    print(f"    {'beta to ETF':<20} {beta_eq:>8.3f}  {beta_nn:>8.3f}")
    print(f"    {'alpha (ann%)':<20} {alpha_eq:>8.2f}  {alpha_nn:>8.2f}")
    print(f"    {'tracking error%':<20} {te_eq:>8.2f}  {te_nn:>8.2f}")
    print(f"    {'R2':<20} {r2_eq:>8.3f}  {r2_nn:>8.3f}")
    print(f"    {'info ratio':<20} {ir_eq:>8.3f}  {ir_nn:>8.3f}")

    ax_perf = fig.add_subplot(gs[row_idx, 0])
    dates   = combined.index.to_timestamp()

    ax_perf.plot(dates, norm100(e_ret).values,
                 color="darkorange", lw=2, label=f"{ticker} ETF (seed)", zorder=3)
    ax_perf.plot(dates, norm100(b_ret).values,
                 color="steelblue", lw=1.8, ls="--",
                 label=f"Equal-weight (b={beta_eq:.2f})", zorder=3)
    ax_perf.plot(dates, norm100(b_nnls_ret).values,
                 color="mediumseagreen", lw=1.8, ls="-.",
                 label=f"NNLS-weight (b={beta_nn:.2f})", zorder=3)
    ax_perf.fill_between(dates, norm100(e_ret).values, norm100(b_nnls_ret).values,
                         alpha=0.08, color="mediumseagreen")
    ax_perf.axhline(y=100, color="gray", ls=":", alpha=0.4, lw=1)
    ax_perf.set_title(f"{theme}  |  {ticker} ETF vs Mid-Cap Baskets",
                      fontsize=11, fontweight="bold")
    ax_perf.set_ylabel("Cumulative return (base=100)")
    ax_perf.legend(fontsize=9)
    ax_perf.grid(alpha=0.3)

    ax_tbl = fig.add_subplot(gs[row_idx, 1])
    ax_tbl.axis("off")
    metrics_tbl = [
        ["Metric",       "Eq-wt",           "NNLS"],
        ["Months",       str(len(combined)), str(len(combined))],
        ["Correlation",  f"{corr_eq:.3f}",   f"{corr_nn:.3f}"],
        ["Beta to ETF",  f"{beta_eq:.3f}",   f"{beta_nn:.3f}"],
        ["Alpha (ann.)", f"{alpha_eq:.2f}%",  f"{alpha_nn:.2f}%"],
        ["Track. Err",   f"{te_eq:.2f}%",    f"{te_nn:.2f}%"],
        ["R2",           f"{r2_eq:.3f}",     f"{r2_nn:.3f}"],
        ["Info Ratio",   f"{ir_eq:.3f}",     f"{ir_nn:.3f}"],
    ]
    tbl = ax_tbl.table(
        cellText=metrics_tbl[1:], colLabels=metrics_tbl[0],
        loc="center", cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.2, 1.7)
    for j in range(3):
        tbl[0, j].set_facecolor("#2c5f8a")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    for i in range(1, len(metrics_tbl)):
        tbl[i, 2].set_facecolor("#e8f5e9")
    for i in range(1, len(metrics_tbl)):
        for j in range(2):
            tbl[i, j].set_facecolor("#f0f4f8" if i % 2 == 0 else "white")
    ax_tbl.set_title("Eq-wt vs NNLS", fontsize=10, fontweight="bold", pad=10)

    row_idx += 1

# %%

suffix = args.mode
plt.suptitle(
    f"Theme Engine: Mid-Cap ETF Replication  "
    f"[{'GICS-Filtered' if ENFORCE_SECTOR else 'Unconstrained'}]",
    fontsize=14, fontweight="bold", y=1.01
)
plt.savefig(f"theme_tracking_{suffix}.png", dpi=150, bbox_inches="tight")
plt.close()

track_df = pd.DataFrame(tracking_rows)
print("\n--- tracking summary ---")
print(track_df.to_string(index=False))
track_df.to_csv(f"theme_tracking_{suffix}.csv", index=False)

all_baskets = pd.concat(results.values(), ignore_index=True)
all_baskets.to_csv(f"theme_baskets_{suffix}.csv", index=False)
beta_df.to_csv("stock_factor_betas.csv")

print(f"""
saved:
  theme_tracking_{suffix}.png
  theme_tracking_{suffix}.csv
  theme_baskets_{suffix}.csv
  stock_factor_betas.csv
""")
