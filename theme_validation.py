# theme_validation.py
#
# Rolling expanding window backtest to validate the stability filter.
#
# The question: do baskets built with the stability filter track their seed ETFs
# better out-of-sample than baskets built without it?
#
# Method:
#   - Pick a set of cutoff dates (quarterly, expanding window from a start date)
#   - At each cutoff, build two baskets using only data available up to that date:
#       (a) baseline: no stability filter
#       (b) filtered: beta stability filter applied
#   - Measure tracking vs the ETF over the next 12 months
#   - Average across all cutoffs and compare
#
# Also produces:
#   - Beta realisation test: cross-period beta correlation for filtered vs unfiltered stocks
#   - Turnover analysis: how many stocks change quarter-to-quarter in each approach
#
# Run:
#   python theme_validation.py
#   python theme_validation.py --start 2015-01  (change rolling window start)
#   python theme_validation.py --horizon 12     (months of OOS evaluation per window)

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
parser.add_argument("--start",    default="2014-01",
                    help="start of rolling window (YYYY-MM)")
parser.add_argument("--horizon",  type=int, default=12,
                    help="out-of-sample evaluation horizon in months")
parser.add_argument("--threshold", type=float, default=0.5,
                    help="stability filter threshold (cosine similarity)")
parser.add_argument("--step",     type=int, default=6,
                    help="months between cutoff dates (default 6 = semi-annual)")
args = parser.parse_args()

BASKET_SIZE         = 25
MIN_MONTHS          = 60
STABILITY_THRESHOLD = args.threshold
OOS_HORIZON         = args.horizon
ROLL_STEP           = args.step

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

themes_cfg  = pd.read_csv("themes_config.csv")
THEMES      = dict(zip(themes_cfg["etf"], themes_cfg["theme_name"]))
gics_map    = pd.read_csv("sic_to_gics_l2.csv")

def sic_to_gics_l2(sic):
    if pd.isna(sic):
        return None
    sic = int(sic)
    row = gics_map[(gics_map["sic_low"] <= sic) & (gics_map["sic_high"] >= sic)]
    return int(row["gics_l2_code"].iloc[0]) if len(row) > 0 else None

# build the full wide return matrix once
stock_wide = stocks_df.pivot_table(
    index="ym", columns="permno", values="ret", aggfunc="first"
)
if not isinstance(stock_wide.index, pd.PeriodIndex):
    stock_wide.index = pd.PeriodIndex(stock_wide.index, freq="M")

meta = (
    stocks_df.sort_values("date").groupby("permno").last()
    [["ticker", "comnam", "siccd"]]
    if "ticker" in stocks_df.columns else pd.DataFrame()
)

THEME_GICS = {}
for _, row in themes_cfg.iterrows():
    codes = [int(c.strip()) for c in str(row["gics_l2_codes"]).split(",")]
    THEME_GICS[row["etf"]] = codes

print(f"  factors  : {F_df.shape}")
print(f"  stocks   : {stock_wide.shape}")
print(f"  OOS horizon : {OOS_HORIZON} months  |  step : {ROLL_STEP} months")
print(f"  stability threshold : {STABILITY_THRESHOLD}")

# %%

def estimate_betas_at_cutoff(cutoff, use_stability):
    """
    Estimate full-sample and (optionally) split-sample betas for all stocks
    using only data available up to the cutoff date.
    Returns a DataFrame of betas with optional stability score.
    """
    F_sub     = F_df[F_df.index <= cutoff]
    S_sub     = stock_wide[stock_wide.index <= cutoff]
    common    = F_sub.index.intersection(S_sub.index)
    F_vals    = F_df.loc[common].values
    S_vals    = S_sub.loc[common]

    betas, r2s, stabs = {}, {}, {}

    for permno in S_vals.columns:
        s    = S_vals[permno].values
        mask = ~np.isnan(s)
        n    = mask.sum()
        if n < MIN_MONTHS:
            continue

        idx  = np.where(mask)[0]
        X_   = np.column_stack([np.ones(n), F_vals[mask]])
        coef, _, _, _ = lstsq(X_, s[mask], rcond=None)
        betas[permno] = coef[1:]
        r2s[permno]   = 1 - np.var(s[mask] - X_ @ coef) / np.var(s[mask])

        if use_stability:
            mid  = n // 2
            i1, i2 = idx[:mid], idx[mid:]

            def hb(hi):
                if len(hi) < 12:
                    return None
                Xh = np.column_stack([np.ones(len(hi)), F_vals[hi]])
                c, _, _, _ = lstsq(Xh, s[hi], rcond=None)
                return c[1:]

            b1, b2 = hb(i1), hb(i2)
            if b1 is not None and b2 is not None:
                n1, n2 = np.linalg.norm(b1), np.linalg.norm(b2)
                stabs[permno] = float(np.dot(b1, b2) / (n1 * n2)) if (n1 > 1e-10 and n2 > 1e-10) else 0.0
            else:
                stabs[permno] = 0.0

    df = pd.DataFrame(betas, index=factor_cols).T
    df.index.name = "permno"
    df = df.join(pd.Series(r2s, name="r2"))

    if use_stability:
        df = df.join(pd.Series(stabs, name="beta_stability"))
        df = df[df["beta_stability"] >= STABILITY_THRESHOLD]

    df = df.join(meta, how="left")
    df["gics_l2"] = pd.to_numeric(df["siccd"], errors="coerce").apply(sic_to_gics_l2)
    return df


def build_basket_at_cutoff(etf_ticker, beta_df_cut, etf_beta):
    """Pick top BASKET_SIZE stocks by cosine similarity."""
    etf_norm    = etf_beta / (np.linalg.norm(etf_beta) + 1e-10)
    valid_gics  = THEME_GICS.get(etf_ticker, [])
    df_filtered = beta_df_cut[beta_df_cut["gics_l2"].isin(valid_gics)].copy()
    if len(df_filtered) < 10:
        df_filtered = beta_df_cut.copy()

    bmat  = df_filtered[factor_cols].values
    bnorm = normalize(bmat, norm="l2")
    df_filtered["similarity"] = bnorm @ etf_norm
    ranked = df_filtered[df_filtered["r2"] > 0.05].sort_values(
        "similarity", ascending=False
    )
    return set(ranked.head(BASKET_SIZE).index.tolist())


def oos_tracking(basket_permnos, etf_ticker, cutoff):
    """
    Measure equal-weighted basket tracking vs ETF over the OOS_HORIZON months
    immediately following the cutoff.
    Returns (correlation, tracking_error%) or (nan, nan) if not enough data.
    """
    start = cutoff + 1
    end   = cutoff + OOS_HORIZON

    permnos_ok = [p for p in basket_permnos if p in stock_wide.columns]
    if len(permnos_ok) < 5:
        return np.nan, np.nan

    basket_ret = stock_wide.loc[start:end, permnos_ok].mean(axis=1)
    etf_ret    = etf_df[etf_ticker].reindex(basket_ret.index) if etf_ticker in etf_df.columns else None

    if etf_ret is None:
        return np.nan, np.nan

    combined = pd.DataFrame({"basket": basket_ret, "etf": etf_ret}).dropna()
    if len(combined) < 6:
        return np.nan, np.nan

    corr = combined["basket"].corr(combined["etf"])
    te   = (combined["basket"] - combined["etf"]).std(ddof=1) * np.sqrt(12) * 100
    return corr, te

# %%

# generate cutoff dates: quarterly expanding windows starting from args.start
start_period = pd.Period(args.start, freq="M")
all_periods  = F_df.index
cutoffs      = [p for p in all_periods
                if p >= start_period and
                   p <= (all_periods[-1] - OOS_HORIZON)]
cutoffs      = cutoffs[::ROLL_STEP]   # keep every ROLL_STEP months

print(f"\nrunning rolling validation: {len(cutoffs)} cutoffs "
      f"from {cutoffs[0]} to {cutoffs[-1]}")
print("(this takes a few minutes — estimating betas at each cutoff)\n")

# %%

# beta realisation test: run once on the full sample
# compare cross-period beta consistency for stocks that pass vs fail the filter
print("beta realisation test...")
full_cutoff  = all_periods[-1]
F_full       = F_df.values
S_full       = stock_wide

betas_h1, betas_h2 = {}, {}
for permno in stock_wide.columns:
    s    = S_full[permno].values
    mask = ~np.isnan(s)
    n    = mask.sum()
    if n < MIN_MONTHS * 2:
        continue
    idx   = np.where(mask)[0]
    mid   = n // 2
    i1, i2 = idx[:mid], idx[mid:]

    def hb(hi):
        Xh = np.column_stack([np.ones(len(hi)), F_full[hi]])
        c, _, _, _ = lstsq(Xh, s[hi], rcond=None)
        return c[1:]

    betas_h1[permno] = hb(i1)
    betas_h2[permno] = hb(i2)

common_p = set(betas_h1.keys()) & set(betas_h2.keys())
stab_scores = {}
for p in common_p:
    b1, b2 = betas_h1[p], betas_h2[p]
    n1, n2 = np.linalg.norm(b1), np.linalg.norm(b2)
    stab_scores[p] = float(np.dot(b1, b2) / (n1 * n2)) if (n1 > 1e-10 and n2 > 1e-10) else 0.0

scores  = np.array(list(stab_scores.values()))
passing = scores[scores >= STABILITY_THRESHOLD]
failing = scores[scores <  STABILITY_THRESHOLD]

print(f"  {len(scores)} stocks with enough history for split-sample test")
print(f"  pass filter (>={STABILITY_THRESHOLD}): {len(passing)} stocks  "
      f"mean score = {passing.mean():.3f}")
print(f"  fail filter (<{STABILITY_THRESHOLD}): {len(failing)} stocks  "
      f"mean score = {failing.mean():.3f}")
print(f"  cross-period beta consistency lift: "
      f"{passing.mean() - failing.mean():.3f} (filtered vs unfiltered)")

# %%

# main rolling loop
results_rows = []
basket_history = {ticker: {"baseline": [], "filtered": []} for ticker in THEMES}

for ci, cutoff in enumerate(cutoffs):
    if ci % 4 == 0:
        print(f"  cutoff {cutoff}  ({ci+1}/{len(cutoffs)})")

    # estimate betas at this cutoff for baseline and filtered
    beta_base = estimate_betas_at_cutoff(cutoff, use_stability=False)
    beta_filt = estimate_betas_at_cutoff(cutoff, use_stability=True)

    # ETF fingerprints at this cutoff
    F_sub    = F_df[F_df.index <= cutoff]
    E_sub    = etf_df[etf_df.index <= cutoff]
    common_e = F_sub.index.intersection(E_sub.index)
    F_e_vals = F_df.loc[common_e].values

    for ticker, theme in THEMES.items():
        if ticker not in etf_df.columns:
            continue
        e    = E_sub[ticker].values if ticker in E_sub.columns else None
        if e is None:
            continue
        mask = ~np.isnan(e)
        if mask.sum() < 12:
            continue

        X_ = np.column_stack([np.ones(mask.sum()), F_e_vals[mask]])
        coef, _, _, _ = lstsq(X_, e[mask], rcond=None)
        etf_beta = coef[1:]

        basket_base = build_basket_at_cutoff(ticker, beta_base, etf_beta)
        basket_filt = build_basket_at_cutoff(ticker, beta_filt, etf_beta)

        corr_b, te_b = oos_tracking(basket_base, ticker, cutoff)
        corr_f, te_f = oos_tracking(basket_filt, ticker, cutoff)

        results_rows.append({
            "cutoff":        str(cutoff),
            "theme":         theme,
            "etf":           ticker,
            "corr_baseline": corr_b,
            "corr_filtered": corr_f,
            "te_baseline":   te_b,
            "te_filtered":   te_f,
            "basket_size_base": len(basket_base),
            "basket_size_filt": len(basket_filt),
        })

        basket_history[ticker]["baseline"].append(basket_base)
        basket_history[ticker]["filtered"].append(basket_filt)

# %%

roll_df = pd.DataFrame(results_rows)
roll_df.to_csv("validation_rolling_results.csv", index=False)

# ── summary: average OOS tracking by theme ───────────────────────────────────
print("\n--- out-of-sample tracking: filtered vs baseline ---")
print(f"{'theme':<28} {'corr base':>10} {'corr filt':>10} "
      f"{'TE base':>9} {'TE filt':>9} {'n windows':>10}")
print("-" * 78)

summary_rows = []
for ticker, theme in THEMES.items():
    sub = roll_df[roll_df["etf"] == ticker].dropna(
        subset=["corr_baseline", "corr_filtered"]
    )
    if len(sub) == 0:
        continue
    cb = sub["corr_baseline"].mean()
    cf = sub["corr_filtered"].mean()
    tb = sub["te_baseline"].mean()
    tf = sub["te_filtered"].mean()
    print(f"  {theme:<26} {cb:>10.3f} {cf:>10.3f} "
          f"{tb:>9.1f} {tf:>9.1f} {len(sub):>10}")
    summary_rows.append({
        "Theme": theme, "ETF": ticker,
        "Corr (baseline)": round(cb, 3), "Corr (filtered)": round(cf, 3),
        "TE (baseline)":   round(tb, 2), "TE (filtered)":   round(tf, 2),
        "N windows":       len(sub)
    })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv("validation_summary.csv", index=False)

# ── turnover analysis ────────────────────────────────────────────────────────
print("\n--- quarterly turnover (avg stocks changing per period) ---")
for ticker, theme in THEMES.items():
    for label in ["baseline", "filtered"]:
        hist = basket_history[ticker][label]
        if len(hist) < 2:
            continue
        turnovers = []
        for i in range(1, len(hist)):
            prev, curr = hist[i-1], hist[i]
            if len(prev) > 0 and len(curr) > 0:
                changed = len(prev.symmetric_difference(curr))
                turnovers.append(changed / BASKET_SIZE * 100)
        if turnovers:
            print(f"  {ticker} {label:<10}: avg turnover = "
                  f"{np.mean(turnovers):.1f}%  "
                  f"(over {len(turnovers)} periods)")

# ── plot: OOS correlation over time ─────────────────────────────────────────
themes_with_data = [t for t in THEMES if t in roll_df["etf"].values]
n = len(themes_with_data)
fig, axes = plt.subplots(n, 1, figsize=(12, 3.5 * n))
if n == 1:
    axes = [axes]

for ax, ticker in zip(axes, themes_with_data):
    theme  = THEMES[ticker]
    sub    = roll_df[roll_df["etf"] == ticker].dropna(
        subset=["corr_baseline", "corr_filtered"]
    )
    if len(sub) == 0:
        continue
    x = range(len(sub))
    ax.plot(x, sub["corr_baseline"].values,
            color="steelblue", lw=1.8, ls="--", label="Baseline (no filter)")
    ax.plot(x, sub["corr_filtered"].values,
            color="mediumseagreen", lw=1.8, label="Stability filtered")
    ax.axhline(sub["corr_baseline"].mean(), color="steelblue",
               ls=":", alpha=0.5, lw=1)
    ax.axhline(sub["corr_filtered"].mean(), color="mediumseagreen",
               ls=":", alpha=0.5, lw=1)
    ax.set_title(f"{theme}  |  OOS {OOS_HORIZON}m correlation by cutoff",
                 fontsize=11, fontweight="bold")
    ax.set_ylabel("Correlation")
    ax.set_xticks(list(x)[::2])
    ax.set_xticklabels(sub["cutoff"].values[::2], rotation=45, ha="right", fontsize=8)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)

plt.suptitle(
    f"Rolling Expanding Window Validation  "
    f"[OOS horizon: {OOS_HORIZON}m  |  stability threshold: {STABILITY_THRESHOLD}]",
    fontsize=13, fontweight="bold", y=1.01
)
plt.tight_layout()
plt.savefig("validation_rolling_correlation.png", dpi=150, bbox_inches="tight")
plt.close()

print("""
saved:
  validation_rolling_results.csv
  validation_summary.csv
  validation_rolling_correlation.png
""")
