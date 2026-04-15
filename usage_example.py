"""
usage_example.py
================
Quick-start guide showing how to call rppca() and rppcaoos() in Python.

Drop-in replacement for the MATLAB code from:
    Lettau & Pelger (2020), "Factors that Fit the Time Series and
    Cross-Section of Stock Returns", Review of Financial Studies.
"""

import numpy as np
from rppca   import rppca
from rppcaoos import rppcaoos


# ------------------------------------------------------------------ #
# 0.  Synthetic data (replace with your own returns panel)
# ------------------------------------------------------------------ #
np.random.seed(42)
T, N = 650, 74          # monthly obs × portfolios (mirrors the paper)

# Two-factor DGP:  market (strong) + value (weak)
Lambda_true = np.hstack([
    np.ones((N, 1)),                                     # market loadings
    np.random.randn(N, 1),                               # value loadings
])
F_true = np.column_stack([
    np.random.normal(0.5, 4.5, T),                       # market factor
    np.random.normal(0.3, 2.0, T),                       # value factor
])
E      = np.random.normal(0, 2.0, (T, N))
X      = F_true @ Lambda_true.T + E                     # (T, N)

print(f"Data dimensions: T={T}, N={N}")

# ------------------------------------------------------------------ #
# 1.  In-sample RP-PCA  (gamma=10 is the paper's default for N=74)
# ------------------------------------------------------------------ #
result = rppca(
    X,
    gamma=10,
    K=5,
    stdnorm=0,
    variancenormalization=0,
    orthogonalization=0,
)

Lambda_hat = result["Lambda"]          # (N, K) loadings
F_hat      = result["F"]               # (T, K) factors
evals      = result["eigenvalues"]     # eigenvalue spectrum
SDF        = result["SDF"]             # (T, K) SDF series
alpha      = result["alpha"]           # (N, K) pricing errors
fw         = result["factor_weights"]  # (N, K) portfolio weights

print("\n=== In-sample RP-PCA results ===")
print(f"Loadings shape : {Lambda_hat.shape}")
print(f"Factor shape   : {F_hat.shape}")
print(f"Top-5 eigenvalues: {evals[:5].round(4)}")

# Sharpe ratios of individual factors
SR_factors = F_hat.mean(axis=0) / F_hat.std(axis=0, ddof=1) * np.sqrt(12)
print(f"Annualised SRs of 5 factors: {SR_factors.round(3)}")

# Max SDF Sharpe ratio (K=5)
SDF_5 = SDF[:, 4]
SR_SDF = SDF_5.mean() / SDF_5.std(ddof=1) * np.sqrt(12)
print(f"Max SR (5-factor SDF, annualised): {SR_SDF:.3f}")

# RMS alpha (pricing error)
RMS_alpha = np.sqrt((alpha[:, 4] ** 2).mean())
print(f"RMS alpha (5 factors): {RMS_alpha:.4f}")

# ------------------------------------------------------------------ #
# 2.  Out-of-sample evaluation with rolling window
# ------------------------------------------------------------------ #
print("\n=== Out-of-sample RP-PCA results ===")
oos = rppcaoos(
    X_total=X,
    stdnorm=0,
    gamma=10,
    K=5,
    window=240,
)

oos_tab = oos["oos_results"]   # (3, K)  rows: SR, RMSE-alpha, Var%
is_tab  = oos["is_results"]    # (3, K)

print("\nIn-sample  [SR | RMSE-alpha | Unexplained Var %]:")
for k in range(5):
    print(f"  K={k+1}: SR={is_tab[0,k]:.3f}  RMSE-α={is_tab[1,k]:.4f}  Var%={is_tab[2,k]:.2f}%")

print("\nOut-of-sample [SR | RMSE-alpha | Unexplained Var %]:")
for k in range(5):
    print(f"  K={k+1}: SR={oos_tab[0,k]:.3f}  RMSE-α={oos_tab[1,k]:.4f}  Var%={oos_tab[2,k]:.2f}%")

avg_gc = oos["corr_loadings"].mean(axis=0)
print(f"\nMean generalised correlations (rolling vs full-sample): {avg_gc.round(3)}")

# ------------------------------------------------------------------ #
# 3.  Thematic engine hook  (placeholder)
# ------------------------------------------------------------------ #
print("\n=== Ready for thematic engine integration ===")
print("Lambda_hat columns map to latent themes:")
print("  Factor 1 → Market proxy")
print("  Factor 2 → Value / Value-interaction")
print("  Factor 3 → Momentum / Momentum-interaction")
print("  Factor 4 → Profitability")
print("  Factor 5 → High-SR / Reversal")
print("\nUse fw (factor_weights) to score any new asset universe.")
