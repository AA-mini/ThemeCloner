# theme_engine.py
# Extends large-cap thematic ETF baskets to mid-cap stocks
# using RP-PCA factor exposures as a common coordinate system
#
# Key idea: estimate each ETF's "factor fingerprint" from RP-PCA factors,
# then find mid-cap stocks with the most similar factor profile.
# Performance is evaluated by how well the basket TRACKS the ETF,
# not by standalone Sharpe ratio.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from numpy.linalg import inv, lstsq
from sklearn.preprocessing import normalize
import warnings
warnings.filterwarnings("ignore")

THEMES = {
    "DTCR": "AI Infrastructure",
    "NLR":  "Energy Independence",
    "HOMZ": "US Housing & Consumer",
    "AGNG": "Bio-Economy & Longevity",
    "FITE": "Defence & Security",
}

# toggle: False = pure factor matching, True = sector-constrained
ENFORCE_SECTOR = False                   #   <-----------------------------

# SIC code ranges for each theme
THEME_SIC = {
    # electronic components, computer services, computer hardware
    "DTCR": list(range(3670, 3680)) + list(range(7370, 7380)) +
            list(range(3570, 3580)) + list(range(3670, 3700)),
    # electric/gas utilities, oil & gas extraction, petroleum refining, industrial machinery
    "NLR":  list(range(4900, 4960)) + list(range(1300, 1400)) +
            list(range(2900, 2920)) + list(range(3560, 3570)),
    # construction, real estate, home improvement retail, furniture
    "HOMZ": list(range(1500, 1800)) + list(range(6500, 6600)) +
            list(range(5200, 5300)) + list(range(2500, 2600)),
    # healthcare services, pharma, medical devices, drug wholesale
    "AGNG": list(range(8000, 8100)) + list(range(2830, 2840)) +
            list(range(3841, 3852)) + list(range(5120, 5130)),
    # guided missiles, ordnance, defence electronics, aircraft only
    # removed 7380-7390 (security services) - too broad, catches fintech
    "FITE": list(range(3760, 3770)) + list(range(3480, 3490)) +
            list(range(3812, 3813)) + list(range(3720, 3730)),
}

BASKET_SIZE = 25
MIN_MONTHS  = 36


# -----------------------------------------------------------------------
# load inputs
# -----------------------------------------------------------------------
print("loading inputs...")
F_df = pd.read_csv("factor_timeseries.csv", index_col=0, parse_dates=True)
F_df.index = F_df.index.to_period("M")
F    = F_df.values
factor_cols = F_df.columns.tolist()

labels_df     = pd.read_csv("factor_labels.csv")
factor_labels = dict(zip(labels_df["factor"], labels_df["label"]))

etf_df = pd.read_csv("etf_theme_returns.csv", index_col=0)
etf_df.index = pd.PeriodIndex(etf_df.index, freq="M")

stocks_df = pd.read_csv("russell1000_exsp500_returns.csv", low_memory=False)
stocks_df["date"] = pd.to_datetime(stocks_df["date"])
stocks_df["ym"]   = stocks_df["date"].dt.to_period("M")

print(f"  factors:  {F_df.shape}")
print(f"  ETFs:     {etf_df.shape}  {etf_df.columns.tolist()}")
print(f"  stocks:   {len(stocks_df):,} rows, {stocks_df['permno'].nunique():,} stocks")


# -----------------------------------------------------------------------
# step 1: ETF factor exposures
# -----------------------------------------------------------------------
print("\n--- step 1: ETF factor exposures ---")
common_etf  = F_df.index.intersection(etf_df.index)
F_etf       = F_df.loc[common_etf].values
E_etf       = etf_df.loc[common_etf]
etf_betas   = {}

for ticker, theme in THEMES.items():
    if ticker not in etf_df.columns:
        print(f"  {ticker}: not found, skipping")
        continue
    e    = E_etf[ticker].values
    mask = ~np.isnan(e)
    if mask.sum() < 12:
        print(f"  {ticker}: only {mask.sum()} months")
        continue
    X_   = np.column_stack([np.ones(mask.sum()), F_etf[mask]])
    coef, _, _, _ = lstsq(X_, e[mask], rcond=None)
    beta = coef[1:]
    r2   = 1 - np.var(e[mask] - X_ @ coef) / np.var(e[mask])
    etf_betas[ticker] = beta
    print(f"  {ticker} ({theme}): {mask.sum()} months  R²={r2:.3f}")


# -----------------------------------------------------------------------
# step 2: stock factor exposures
# -----------------------------------------------------------------------
print("\n--- step 2: stock factor exposures ---")
stock_wide = stocks_df.pivot_table(index="ym", columns="permno",
                                   values="ret", aggfunc="first")
if not isinstance(stock_wide.index, pd.PeriodIndex):
    stock_wide.index = pd.PeriodIndex(stock_wide.index, freq="M")

common_stk = F_df.index.intersection(stock_wide.index)
F_stk      = F_df.loc[common_stk].values
S_wide     = stock_wide.loc[common_stk]
print(f"  aligned: {len(common_stk)} months x {S_wide.shape[1]} stocks")

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
meta    = (stocks_df.sort_values("date").groupby("permno").last()
           [["ticker","comnam","siccd"]]
           if "ticker" in stocks_df.columns else pd.DataFrame())
beta_df = beta_df.join(meta, how="left")
print(f"  stocks with sufficient history: {len(beta_df):,}")


# -----------------------------------------------------------------------
# step 3: match stocks to themes
# -----------------------------------------------------------------------
print("\n--- step 3: matching stocks to themes ---")
results   = {}
baskets   = {}   # permno lists per theme

for ticker, theme in THEMES.items():
    if ticker not in etf_betas:
        continue

    etf_beta = etf_betas[ticker]
    etf_norm = etf_beta / (np.linalg.norm(etf_beta) + 1e-10)

    # sector filter
    if ENFORCE_SECTOR and ticker in THEME_SIC:
        valid_sic    = THEME_SIC[ticker]
        df_filtered  = beta_df[
            pd.to_numeric(beta_df["siccd"], errors="coerce").isin(valid_sic)
        ].copy()
        if len(df_filtered) < 10:
            df_filtered = beta_df.copy()
    else:
        df_filtered = beta_df.copy()

    bmat  = df_filtered[factor_cols].values
    bnorm = normalize(bmat, norm="l2")
    sim   = bnorm @ etf_norm
    df_filtered = df_filtered.copy()
    df_filtered["similarity"] = sim

    ranked = (df_filtered[["ticker","comnam","siccd","r2","similarity"]]
              .sort_values("similarity", ascending=False))
    ranked = ranked[ranked["r2"] > 0.05]
    top    = ranked.head(BASKET_SIZE).copy()
    top["rank"]       = range(1, len(top)+1)
    top["weight_eq"]  = 1 / BASKET_SIZE
    top["weight_sim"] = top["similarity"] / top["similarity"].sum()
    top["theme"]      = theme
    top["etf"]        = ticker
    results[ticker]   = top
    baskets[ticker]   = [p for p in top.index if p in S_wide.columns]

    print(f"\n  {ticker} - {theme}  "
          f"({'sector ON' if ENFORCE_SECTOR else 'unconstrained'})")
    print(f"  {'rank':>4}  {'ticker':>7}  {'company':>30}  "
          f"{'sim':>6}  {'R²':>5}")
    print("  " + "-"*58)
    for _, row in top.head(15).iterrows():
        nm = str(row.get("comnam",""))[:28] if pd.notna(row.get("comnam")) else ""
        tk = str(row.get("ticker",""))[:7]  if pd.notna(row.get("ticker")) else ""
        print(f"  {int(row['rank']):>4}  {tk:>7}  {nm:>30}  "
              f"{row['similarity']:>6.3f}  {row['r2']:>5.3f}")


# -----------------------------------------------------------------------
# step 4: tracking performance vs ETF seed
# metrics: correlation, tracking error, beta, cumulative return chart
# -----------------------------------------------------------------------
print("\n--- step 4: tracking performance ---")

tracking_rows = []

# one figure per theme: cumulative performance + metrics table
n_themes = len([t for t in THEMES if t in etf_betas])
fig = plt.figure(figsize=(18, 4 * n_themes))
gs  = gridspec.GridSpec(n_themes, 2, width_ratios=[3, 1],
                        hspace=0.5, wspace=0.3)

row_idx = 0
for ticker, theme in THEMES.items():
    if ticker not in etf_betas or ticker not in baskets:
        continue

    permnos_basket = baskets[ticker]
    if len(permnos_basket) < 5:
        continue

    # basket equal-weighted return series
    basket_ret = S_wide[permnos_basket].mean(axis=1)

    # ETF return series
    etf_ret = etf_df[ticker].reindex(S_wide.index)

    # align to common non-NaN period
    combined  = pd.DataFrame({"basket": basket_ret, "etf": etf_ret}).dropna()
    if len(combined) < 12:
        print(f"  {ticker}: insufficient overlapping data")
        continue

    b_ret = combined["basket"]
    e_ret = combined["etf"]

    # ── tracking metrics ────────────────────────────────────────────
    correlation  = b_ret.corr(e_ret)
    diff         = b_ret - e_ret
    tracking_err = diff.std(ddof=1) * np.sqrt(12) * 100   # annualised %

    # beta via OLS: basket = alpha + beta * ETF
    X_ols  = np.column_stack([np.ones(len(e_ret)), e_ret.values])
    coef_t, _, _, _ = lstsq(X_ols, b_ret.values, rcond=None)
    alpha_t = coef_t[0] * 12 * 100   # annualised %
    beta_t  = coef_t[1]
    resid_t = b_ret.values - X_ols @ coef_t
    r2_t    = 1 - np.var(resid_t) / np.var(b_ret.values)

    # information ratio (alpha / TE)
    ir = (alpha_t / 100) / (tracking_err / 100) if tracking_err > 0 else np.nan

    tracking_rows.append({
        "Theme":          theme,
        "ETF":            ticker,
        "N months":       len(combined),
        "Correlation":    round(correlation, 3),
        "Beta":           round(beta_t, 3),
        "Alpha (ann%)":   round(alpha_t, 2),
        "Track. Err (%)": round(tracking_err, 2),
        "R²":             round(r2_t, 3),
        "Info Ratio":     round(ir, 3),
    })

    print(f"\n  {ticker} ({theme}):")
    print(f"    months overlap : {len(combined)}")
    print(f"    correlation    : {correlation:.3f}")
    print(f"    beta to ETF    : {beta_t:.3f}")
    print(f"    alpha (ann)    : {alpha_t:.2f}%")
    print(f"    tracking error : {tracking_err:.2f}%")
    print(f"    R²             : {r2_t:.3f}")
    print(f"    info ratio     : {ir:.3f}")

    # ── plot: cumulative performance ────────────────────────────────
    ax_perf = fig.add_subplot(gs[row_idx, 0])

    cum_basket = (1 + b_ret).cumprod()
    cum_etf    = (1 + e_ret).cumprod()

    # normalise both to 100 at start
    cum_basket = cum_basket / cum_basket.iloc[0] * 100
    cum_etf    = cum_etf    / cum_etf.iloc[0]    * 100

    # x axis as timestamps for plotting
    dates = combined.index.to_timestamp()

    ax_perf.plot(dates, cum_etf.values,
                 color="darkorange", linewidth=2,
                 label=f"{ticker} ETF (seed)", zorder=3)
    ax_perf.plot(dates, cum_basket.values,
                 color="steelblue", linewidth=2,
                 linestyle="--", label="Mid-cap basket", zorder=3)

    # scatter dots showing basket vs ETF each month (beta line)
    ax_perf.fill_between(dates, cum_etf.values, cum_basket.values,
                         alpha=0.1, color="steelblue")

    # beta=1 reference line
    ax_perf.axhline(y=100, color="gray", linestyle=":", alpha=0.4,
                    linewidth=1)

    ax_perf.set_title(f"{theme}  |  {ticker} ETF vs Mid-Cap Basket",
                      fontsize=11, fontweight="bold")
    ax_perf.set_ylabel("Cumulative return (base=100)")
    ax_perf.legend(fontsize=9)
    ax_perf.grid(alpha=0.3)

    # ── metrics table in right panel ────────────────────────────────
    ax_tbl = fig.add_subplot(gs[row_idx, 1])
    ax_tbl.axis("off")

    metrics = [
        ["Months overlap",    str(len(combined))],
        ["Correlation",       f"{correlation:.3f}"],
        ["Beta to ETF",       f"{beta_t:.3f}"],
        ["Alpha (ann.)",      f"{alpha_t:.2f}%"],
        ["Tracking Error",    f"{tracking_err:.2f}%"],
        ["R²",                f"{r2_t:.3f}"],
        ["Info Ratio",        f"{ir:.3f}"],
    ]

    tbl = ax_tbl.table(
        cellText=metrics,
        colLabels=["Metric", "Value"],
        loc="center",
        cellLoc="left",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.3, 1.8)

    # colour header row
    for j in range(2):
        tbl[0, j].set_facecolor("#2c5f8a")
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    # alternate row shading
    for i in range(1, len(metrics)+1):
        for j in range(2):
            tbl[i, j].set_facecolor("#f0f4f8" if i % 2 == 0 else "white")

    ax_tbl.set_title("Tracking Metrics", fontsize=10,
                     fontweight="bold", pad=10)

    row_idx += 1

suffix = "sector" if ENFORCE_SECTOR else "unconstrained"
plt.suptitle(f"Theme Engine: Mid-Cap ETF Replication  "
             f"[{'Sector-Constrained' if ENFORCE_SECTOR else 'Unconstrained'}]",
             fontsize=14, fontweight="bold", y=1.01)
plt.savefig(f"theme_tracking_{suffix}.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"\n  saved theme_tracking_{suffix}.png")


# -----------------------------------------------------------------------
# summary tracking table
# -----------------------------------------------------------------------
print("\n--- tracking summary ---")
track_df = pd.DataFrame(tracking_rows)
print(track_df.to_string(index=False))
track_df.to_csv(f"theme_tracking_{suffix}.csv", index=False)
print(f"saved theme_tracking_{suffix}.csv")


# -----------------------------------------------------------------------
# save baskets
# -----------------------------------------------------------------------
all_baskets = pd.concat(results.values(), ignore_index=True)
all_baskets.to_csv(f"theme_baskets_{suffix}.csv", index=False)
beta_df.to_csv("stock_factor_betas.csv")
print(f"saved theme_baskets_{suffix}.csv")
print(f"saved stock_factor_betas.csv")

print(f"""
outputs:
  theme_tracking_{suffix}.png   - cumulative performance + metrics per theme
  theme_tracking_{suffix}.csv   - summary tracking table
  theme_baskets_{suffix}.csv    - ranked stock lists
  stock_factor_betas.csv        - factor exposures for all stocks
""")