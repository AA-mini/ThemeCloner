# theme_engine.py
#
# Takes the RP-PCA factors estimated on the OpenSourceAP universe and uses them
# as a common coordinate system to extend large-cap thematic ETF baskets into
# mid-cap stocks where analyst coverage is thin.
#
# Run:
#   python theme_engine.py                          (sector, no stability filter)
#   python theme_engine.py --mode unconstrained
#   python theme_engine.py --stability              (adds beta stability filter)
#   python theme_engine.py --mode unconstrained --stability

# %%

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from numpy.linalg import lstsq
from sklearn.preprocessing import normalize
import warnings
warnings.filterwarnings("ignore")

# %%

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["sector", "unconstrained"],
                    default="sector")
parser.add_argument("--stability", action="store_true",
                    help="enable beta stability filter")
parser.add_argument("--stability-threshold", type=float, default=0.5)
args = parser.parse_args()

ENFORCE_SECTOR      = (args.mode == "sector")
USE_STABILITY       = args.stability
STABILITY_THRESHOLD = args.stability_threshold
BASKET_SIZE         = 25
MIN_MONTHS          = 36

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

mode_label = f"[{args.mode}{'  +stability' if USE_STABILITY else ''}]"
print(f"  factors : {F_df.shape}")
print(f"  ETFs    : {etf_df.shape}  {etf_df.columns.tolist()}")
print(f"  stocks  : {len(stocks_df):,} rows, {stocks_df['permno'].nunique():,} stocks")
print(f"  mode    : {mode_label}")

# %%

print(f"\nstep 1: ETF factor fingerprints  {mode_label}")

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
        print(f"  {ticker}: only {mask.sum()} months, skipping")
        continue
    X_   = np.column_stack([np.ones(mask.sum()), F_etf[mask]])
    coef, _, _, _ = lstsq(X_, e[mask], rcond=None)
    beta = coef[1:]
    r2   = 1 - np.var(e[mask] - X_ @ coef) / np.var(e[mask])
    etf_betas[ticker] = beta
    print(f"  {ticker} ({theme}): {mask.sum()} months, R2={r2:.3f}")

# %%

print("\nstep 2: stock factor betas" +
      ("  +stability scoring" if USE_STABILITY else ""))

stock_wide = stocks_df.pivot_table(
    index="ym", columns="permno", values="ret", aggfunc="first"
)
if not isinstance(stock_wide.index, pd.PeriodIndex):
    stock_wide.index = pd.PeriodIndex(stock_wide.index, freq="M")

common_stk = F_df.index.intersection(stock_wide.index)
F_stk      = F_df.loc[common_stk].values
S_wide     = stock_wide.loc[common_stk]
print(f"  {len(common_stk)} months x {S_wide.shape[1]} stocks")

stock_betas    = {}
stock_r2       = {}
stock_stability = {}
permnos        = S_wide.columns.tolist()

for i, permno in enumerate(permnos):
    if i % 500 == 0:
        print(f"  {i}/{len(permnos)}...")

    s    = S_wide[permno].values
    mask = ~np.isnan(s)
    n    = mask.sum()

    if n < MIN_MONTHS:
        continue

    idx  = np.where(mask)[0]
    X_   = np.column_stack([np.ones(n), F_stk[mask]])
    coef, _, _, _ = lstsq(X_, s[mask], rcond=None)
    stock_betas[permno] = coef[1:]
    stock_r2[permno]    = 1 - np.var(s[mask] - X_ @ coef) / np.var(s[mask])

    if USE_STABILITY:
        mid  = n // 2
        idx1, idx2 = idx[:mid], idx[mid:]

        def half_beta(hi):
            if len(hi) < 12:
                return None
            Xh = np.column_stack([np.ones(len(hi)), F_stk[hi]])
            c, _, _, _ = lstsq(Xh, s[hi], rcond=None)
            return c[1:]

        b1, b2 = half_beta(idx1), half_beta(idx2)
        if b1 is not None and b2 is not None:
            n1, n2 = np.linalg.norm(b1), np.linalg.norm(b2)
            if n1 > 1e-10 and n2 > 1e-10:
                stock_stability[permno] = float(np.dot(b1, b2) / (n1 * n2))
            else:
                stock_stability[permno] = 0.0
        else:
            stock_stability[permno] = 0.0

beta_df = pd.DataFrame(stock_betas, index=factor_cols).T
beta_df.index.name = "permno"
beta_df = beta_df.join(pd.Series(stock_r2, name="r2"))

if USE_STABILITY:
    beta_df = beta_df.join(pd.Series(stock_stability, name="beta_stability"))
    n_before = len(beta_df)
    beta_df  = beta_df[beta_df["beta_stability"] >= STABILITY_THRESHOLD]
    n_after  = len(beta_df)
    print(f"  stability filter: {n_before} -> {n_after} stocks "
          f"(removed {n_before - n_after}, threshold={STABILITY_THRESHOLD})")
else:
    beta_df["beta_stability"] = np.nan

meta = (
    stocks_df.sort_values("date").groupby("permno").last()
    [["ticker", "comnam", "siccd"]]
    if "ticker" in stocks_df.columns else pd.DataFrame()
)
beta_df = beta_df.join(meta, how="left")
beta_df["gics_l2"] = pd.to_numeric(
    beta_df["siccd"], errors="coerce"
).apply(sic_to_gics_l2)

print(f"  {len(beta_df):,} stocks in universe after all filters")

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
            print(f"  {ticker}: GICS filter too tight, falling back")
            df_filtered = beta_df.copy()
    else:
        df_filtered = beta_df.copy()

    bmat  = df_filtered[factor_cols].values
    bnorm = normalize(bmat, norm="l2")
    sim   = bnorm @ etf_norm
    df_filtered = df_filtered.copy()
    df_filtered["similarity"] = sim

    ranked = (
        df_filtered[["ticker", "comnam", "siccd", "gics_l2",
                     "r2", "beta_stability", "similarity"]]
        .sort_values("similarity", ascending=False)
    )
    ranked = ranked[ranked["r2"] > 0.05]
    top    = ranked.head(BASKET_SIZE).copy()
    top["rank"]       = range(1, len(top) + 1)
    top["weight_eq"]  = 1 / len(top)
    top["weight_sim"] = top["similarity"] / top["similarity"].sum()

    # weights: equal and similarity-based
    # NNLS (factor-magnitude matching) is noted as a future improvement
    top_permnos = [p for p in top.index if p in S_wide.columns]
    top = top.copy()
    top["weight_nnls"] = top["weight_sim"]   # placeholder, same as sim for now

    top["theme"] = theme
    top["etf"]   = ticker
    results[ticker] = top
    baskets[ticker] = top_permnos   # already filtered to stocks in S_wide

    mode_str = "GICS-filtered" if ENFORCE_SECTOR else "unconstrained"
    stab_str = "  +stability" if USE_STABILITY else ""
    print(f"\n  {ticker} - {theme}  ({mode_str}{stab_str})")
    print(f"  {'rank':>4}  {'ticker':>7}  {'company':>28}  "
          f"{'sim':>6}  {'R2':>5}  {'stab':>6}")
    print("  " + "-" * 74)
    for _, row in top.head(15).iterrows():
        nm   = str(row.get("comnam", ""))[:26] if pd.notna(row.get("comnam")) else ""
        tk   = str(row.get("ticker", ""))[:7]  if pd.notna(row.get("ticker")) else ""
        stab = f"{row['beta_stability']:.3f}" if pd.notna(row.get("beta_stability")) else "  n/a"
        print(f"  {int(row['rank']):>4}  {tk:>7}  {nm:>28}  "
              f"{row['similarity']:>6.3f}  {row['r2']:>5.3f}  "
              f"{stab:>6}")

# %%

print("\nstep 4: tracking performance")

tracking_rows = []
n_themes = len([t for t in THEMES if t in etf_betas])
fig = plt.figure(figsize=(18, 4 * n_themes))
gs  = gridspec.GridSpec(n_themes, 2, width_ratios=[3, 1],
                        hspace=0.5, wspace=0.3)

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

    # equal-weighted basket
    basket_ret = S_wide[permnos_basket].mean(axis=1)

    etf_ret  = etf_df[ticker].reindex(S_wide.index)
    combined = pd.DataFrame({
        "basket": basket_ret,
        "etf":    etf_ret,
    }).dropna()

    if len(combined) < 12:
        print(f"  {ticker}: not enough overlap, skipping")
        continue

    b_ret      = combined["basket"]
    e_ret      = combined["etf"]

    corr_eq = b_ret.corr(e_ret)
    te_eq   = (b_ret - e_ret).std(ddof=1) * np.sqrt(12) * 100
    X_ols   = np.column_stack([np.ones(len(e_ret)), e_ret.values])
    coef_eq, _, _, _ = lstsq(X_ols, b_ret.values, rcond=None)
    alpha_eq = coef_eq[0] * 12 * 100
    beta_eq  = coef_eq[1]
    r2_eq    = 1 - np.var(b_ret.values - X_ols @ coef_eq) / np.var(b_ret.values)
    ir_eq    = (alpha_eq / 100) / (te_eq / 100) if te_eq > 0 else np.nan


    tracking_rows.append({
        "Theme":           theme,
        "ETF":             ticker,
        "Stability filter": USE_STABILITY,
        "N months":        len(combined),
        "Corr (eq-wt)":    round(corr_eq, 3),
        "Beta (eq-wt)":    round(beta_eq, 3),
        "Alpha (eq-wt)":   round(alpha_eq, 2),
        "TE (eq-wt)":      round(te_eq, 2),
        "R2 (eq-wt)":      round(r2_eq, 3),
        "IR (eq-wt)":      round(ir_eq, 3),
    })

    print(f"\n  {ticker} ({theme}):")
    print(f"    {'metric':<20} {'eq-wt':>8}")

    ax_perf = fig.add_subplot(gs[row_idx, 0])
    dates   = combined.index.to_timestamp()

    ax_perf.plot(dates, norm100(e_ret).values,
                 color="darkorange", lw=2,
                 label=f"{ticker} ETF (seed)", zorder=3)
    ax_perf.plot(dates, norm100(b_ret).values,
                 color="steelblue", lw=1.8, ls="--",
                 label=f"Equal-weight (b={beta_eq:.2f})", zorder=3)

    ax_perf.fill_between(dates, norm100(e_ret).values,
                         norm100(b_ret).values,
                         alpha=0.08, color="steelblue")
    ax_perf.axhline(y=100, color="gray", ls=":", alpha=0.4, lw=1)
    ax_perf.set_title(f"{theme}  |  {ticker} ETF vs Mid-Cap Baskets",
                      fontsize=11, fontweight="bold")
    ax_perf.set_ylabel("Cumulative return (base=100)")
    ax_perf.legend(fontsize=9)
    ax_perf.grid(alpha=0.3)

    ax_tbl = fig.add_subplot(gs[row_idx, 1])
    ax_tbl.axis("off")
    metrics_tbl = [
        ["Metric",       "Value"],
        ["Months",       str(len(combined))],
        ["Correlation",  f"{corr_eq:.3f}"],
        ["Beta to ETF",  f"{beta_eq:.3f}"],
        ["Alpha (ann.)", f"{alpha_eq:.2f}%"],
        ["Track. Err",   f"{te_eq:.2f}%"],
        ["R2",           f"{r2_eq:.3f}"],
        ["Info Ratio",   f"{ir_eq:.3f}"],
    ]
    tbl = ax_tbl.table(
        cellText=metrics_tbl[1:], colLabels=metrics_tbl[0],
        loc="center", cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.4, 1.7)
    for j in range(2):
        tbl[0, j].set_facecolor("#2c5f8a")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    for i in range(1, len(metrics_tbl)):
        tbl[i, 0].set_facecolor("#f0f4f8" if i % 2 == 0 else "white")
        tbl[i, 1].set_facecolor("#e8f5e9")
    ax_tbl.set_title("Tracking Metrics", fontsize=10, fontweight="bold", pad=10)

    row_idx += 1

# %%

stab_tag = "_stability" if USE_STABILITY else ""
suffix   = args.mode + stab_tag

plt.suptitle(
    f"Theme Engine: Mid-Cap ETF Replication  "
    f"[{'GICS-Filtered' if ENFORCE_SECTOR else 'Unconstrained'}"
    f"{'  +Stability Filter' if USE_STABILITY else ''}]",
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
beta_df.to_csv(f"stock_factor_betas{stab_tag}.csv")

print(f"""
saved:
  theme_tracking_{suffix}.png
  theme_tracking_{suffix}.csv
  theme_baskets_{suffix}.csv
  stock_factor_betas{stab_tag}.csv
""")
