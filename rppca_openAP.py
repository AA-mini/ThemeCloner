# rppca_openAP.py
# Runs RP-PCA on the full OpenSourceAP universe (145 anomalies)
# This is the "extended replication" - same method as the paper
# but with N=145 vs paper's N=74
#
# Outputs:
#   factor_loadings.csv      - (145 x 5) loading matrix
#   factor_timeseries.csv    - (T x 5) factor returns
#   factor_labels.csv        - human readable factor descriptions
#   rppca_openAP_results.png - summary figures

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import inv
import warnings
warnings.filterwarnings("ignore")

from rppca import rppca

# -----------------------------------------------------------------------
# load data
# -----------------------------------------------------------------------
print("loading OpenAP returns...")
X_df = pd.read_csv("X_openAP.csv", index_col=0, parse_dates=True)
X    = X_df.values
T, N = X.shape
print(f"  {T} months x {N} anomalies")
print(f"  {X_df.index[0].strftime('%Y-%m')} to {X_df.index[-1].strftime('%Y-%m')}")


# -----------------------------------------------------------------------
# run RP-PCA with gamma=10 (paper's default) and K=5 factors
# also run PCA for comparison
# -----------------------------------------------------------------------
print("\nrunning RP-PCA (gamma=10, K=5)...")
res_rp  = rppca(X, gamma=10, K=5)
print("running PCA (gamma=-1, K=5)...")
res_pca = rppca(X, gamma=-1,  K=5)

Lambda_rp  = res_rp["F"]    # (T, 5) factor time series
Lambda_pca = res_pca["F"]
evals_rp   = res_rp["eigenvalues"]
fw_rp      = res_rp["factor_weights"]   # (N, 5) portfolio weights

# factor loadings (N x 5) - how much each anomaly loads on each factor
loadings_rp  = res_rp["Lambda"]    # (N, 5)
loadings_pca = res_pca["Lambda"]

F_rp  = res_rp["F"]
F_pca = res_pca["F"]


# -----------------------------------------------------------------------
# compute Sharpe ratios for each factor and the combined SDF
# -----------------------------------------------------------------------
def factor_sr(F):
    """annualised SR for each factor column"""
    return (F.mean(axis=0) / F.std(axis=0, ddof=1)) * np.sqrt(12)

def sdf_sr(F):
    """max SR of the tangency portfolio of all factors"""
    Sig = np.cov(F.T, ddof=1)
    mu  = F.mean(axis=0)
    w   = inv(Sig) @ mu
    sdf = F @ w
    return sdf.mean() / sdf.std(ddof=1) * np.sqrt(12)

sr_rp_individual  = factor_sr(F_rp)
sr_pca_individual = factor_sr(F_pca)
sr_rp_sdf         = sdf_sr(F_rp)
sr_pca_sdf        = sdf_sr(F_pca)

print(f"\n--- factor Sharpe ratios ---")
print(f"{'factor':>10}  {'RP-PCA SR':>10}  {'PCA SR':>8}")
print("-" * 35)
for k in range(5):
    print(f"  factor {k+1}  {sr_rp_individual[k]:>10.3f}  "
          f"{sr_pca_individual[k]:>8.3f}")
print(f"\n  SDF (combined K=5):")
print(f"  RP-PCA: {sr_rp_sdf:.3f}   PCA: {sr_pca_sdf:.3f}")
print(f"  ratio:  {sr_rp_sdf/sr_pca_sdf:.2f}x  (paper reports ~2x)")


# -----------------------------------------------------------------------
# identify which anomalies load most heavily on each factor
# this is how we label the factors economically
# -----------------------------------------------------------------------
print("\n--- factor interpretation (top anomalies by loading) ---")

loadings_df = pd.DataFrame(
    loadings_rp,
    index=X_df.columns,
    columns=[f"F{k+1}" for k in range(5)]
)

factor_labels = {}
for k in range(5):
    col     = f"F{k+1}"
    top_pos = loadings_df[col].nlargest(8).index.tolist()
    top_neg = loadings_df[col].nsmallest(5).index.tolist()
    print(f"\n  Factor {k+1} (SR={sr_rp_individual[k]:.3f}):")
    print(f"    positive loadings: {top_pos}")
    print(f"    negative loadings: {top_neg}")

    # auto-label based on what loads positively
    # these are heuristic labels based on what we know about anomalies
    top_str = " ".join(top_pos).lower()
    if any(x in top_str for x in ["bm", "ep", "cfp", "dp", "value"]):
        label = "Value"
    elif any(x in top_str for x in ["mom", "trend", "momentum"]):
        label = "Momentum"
    elif any(x in top_str for x in ["gp", "prof", "roe", "roa", "roaq"]):
        label = "Profitability"
    elif any(x in top_str for x in ["rev", "reversal", "strev"]):
        label = "Reversal / High-SR"
    elif any(x in top_str for x in ["size", "mktcap"]):
        label = "Size"
    elif any(x in top_str for x in ["inv", "asset", "growth"]):
        label = "Investment / Growth"
    elif k == 0:
        label = "Market"   # first factor is almost always market
    else:
        label = f"Factor {k+1}"

    factor_labels[col] = label
    print(f"    -> label: {label}")

print(f"\nfactor labels: {factor_labels}")


# -----------------------------------------------------------------------
# figures
# -----------------------------------------------------------------------
print("\ngenerating figures...")
fig = plt.figure(figsize=(15, 10))
gs  = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.35)

# top left: eigenvalue spectrum
ax1 = fig.add_subplot(gs[0, 0])
ax1.bar(range(1, 21), evals_rp[:20], color="steelblue", alpha=0.8)
ax1.set_xlabel("eigenvalue number")
ax1.set_ylabel("eigenvalue")
ax1.set_title("RP-PCA eigenvalue spectrum\n(N=145 anomalies)")
ax1.grid(alpha=0.3, axis="y")

# top middle: SR comparison RP-PCA vs PCA
ax2 = fig.add_subplot(gs[0, 1])
x   = np.arange(5)
w   = 0.35
ax2.bar(x - w/2, sr_rp_individual,  w, label="RP-PCA", color="steelblue", alpha=0.85)
ax2.bar(x + w/2, sr_pca_individual, w, label="PCA",    color="lightsteelblue", alpha=0.85)
ax2.set_xticks(x)
ax2.set_xticklabels([f"F{k+1}" for k in range(5)])
ax2.set_ylabel("annualised SR")
ax2.set_title("Individual factor SRs\nRP-PCA vs PCA")
ax2.legend()
ax2.grid(alpha=0.3, axis="y")

# top right: cumulative factor returns
ax3 = fig.add_subplot(gs[0, 2])
F_df = pd.DataFrame(F_rp, index=X_df.index,
                    columns=[factor_labels.get(f"F{k+1}", f"F{k+1}")
                             for k in range(5)])
(1 + F_df / 100).cumprod().plot(ax=ax3, linewidth=1.2)
ax3.set_title("Cumulative RP-PCA factor returns")
ax3.set_ylabel("cumulative return")
ax3.legend(fontsize=7)
ax3.grid(alpha=0.3)

# bottom: heatmap of factor loadings by anomaly (top 30 anomalies)
ax4 = fig.add_subplot(gs[1, :])
# pick top 30 anomalies by max absolute loading across all factors
max_loading = loadings_df.abs().max(axis=1).nlargest(30).index
hm_data = loadings_df.loc[max_loading].T
im = ax4.imshow(hm_data.values, aspect="auto", cmap="RdBu_r",
                vmin=-3, vmax=3)
ax4.set_xticks(range(len(max_loading)))
ax4.set_xticklabels(max_loading, rotation=45, ha="right", fontsize=7)
ax4.set_yticks(range(5))
ax4.set_yticklabels([f"F{k+1}: {factor_labels.get(f'F{k+1}','')}"
                     for k in range(5)])
ax4.set_title("Factor loadings heatmap — top 30 anomalies by max loading\n"
              "(blue = positive, red = negative)")
plt.colorbar(im, ax=ax4, shrink=0.6)

plt.suptitle(f"RP-PCA on OpenSourceAP Universe  "
             f"(N={N} anomalies, T={T} months)\n"
             f"5-factor SDF SR: RP-PCA={sr_rp_sdf:.3f}  "
             f"PCA={sr_pca_sdf:.3f}  "
             f"ratio={sr_rp_sdf/sr_pca_sdf:.2f}x",
             fontsize=12)

plt.savefig("rppca_openAP_results.png", dpi=150, bbox_inches="tight")
plt.show()
print("  saved rppca_openAP_results.png")


# -----------------------------------------------------------------------
# save outputs for theme engine
# -----------------------------------------------------------------------

# factor loadings (N x 5) - anomaly exposures to each factor
loadings_df.to_csv("factor_loadings.csv")
print("\nsaved factor_loadings.csv")

# factor time series (T x 5)
F_df.index = X_df.index
F_df.to_csv("factor_timeseries.csv")
print("saved factor_timeseries.csv")

# factor weights (N x 5) - portfolio weights to construct factors from anomalies
fw_df = pd.DataFrame(fw_rp, index=X_df.columns,
                     columns=[f"F{k+1}" for k in range(5)])
fw_df.to_csv("factor_weights.csv")
print("saved factor_weights.csv")

# factor labels
labels_df = pd.DataFrame([
    {"factor": f"F{k+1}",
     "label":  factor_labels.get(f"F{k+1}", f"Factor {k+1}"),
     "sr_rppca": sr_rp_individual[k],
     "sr_pca":   sr_pca_individual[k]}
    for k in range(5)
])
labels_df.to_csv("factor_labels.csv", index=False)
print("saved factor_labels.csv")

print(f"""
summary:
  N anomalies : {N}
  T months    : {T}
  RP-PCA SDF SR : {sr_rp_sdf:.3f}
  PCA SDF SR    : {sr_pca_sdf:.3f}
  ratio         : {sr_rp_sdf/sr_pca_sdf:.2f}x

next step: run theme_engine.py
""")
