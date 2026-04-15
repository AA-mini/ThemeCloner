"""
RP-PCA: Risk-Premium PCA estimation of factors
Python conversion of RPPCA.m by Lettau & Pelger (2020)

Reference:
    "Factors that Fit the Time Series and Cross-Section of Stock Returns"
    Review of Financial Studies (2020), Lettau & Pelger
"""

import numpy as np
from numpy.linalg import inv, eig
from scipy.linalg import qr


def rppca(X, gamma, K, stdnorm=0, variancenormalization=0, orthogonalization=0):
    """
    RP-PCA estimation of latent asset pricing factors.

    Parameters
    ----------
    X : np.ndarray, shape (T, N)
        Panel of excess returns. T = time periods, N = assets.
    gamma : float
        Risk-premium penalty weight.
        gamma=0  → standard PCA on second moment only.
        gamma>0  → overweights cross-sectional means (favors high-SR factors).
        gamma=-1 → PCA on the covariance matrix.
    K : int
        Number of factors to extract.
    stdnorm : int, default 0
        If 1, pre-standardise each asset by its time-series std before PCA.
        Use 0 (no standardisation) as default.
    variancenormalization : int, default 0
        If 0 (default): loadings have unit norm (Λ'Λ = I_K).
        If 1: loadings are scaled by sqrt(eigenvalues), so factor variances = 1.
    orthogonalization : int, default 0
        If 0 (default): return the raw RP-PCA rotation.
        If 1: apply QR so that factors are orthogonal in the time-series.

    Returns
    -------
    output : dict with keys
        'Lambda'       – (N, K) factor loadings
        'F'            – (T, K) factor time series
        'eigenvalues'  – (N,)   eigenvalues of the RP-PCA matrix (descending)
        'SDF'          – (T, K) SDF time series (for k=1..K factors each)
        'SDF_weights_assets' – list of (N,) SDF weights in asset space
        'beta'         – list of (N, k) time-series regression betas
        'alpha'        – (N, K) time-series regression intercepts
        'residuals'    – list of (T, N) residual matrices
        'factor_weights'     – (N, K) portfolio weights to construct factors from X
    """
    T, N = X.shape

    # ------------------------------------------------------------------ #
    # 1. Cross-sectional weighting matrix  WN
    # ------------------------------------------------------------------ #
    if stdnorm == 1:
        # Demean X, compute per-asset variance, take inverse of sqrt-diag
        X_dm = X - X.mean(axis=0, keepdims=True)
        asset_var = np.diag(X_dm.T @ X_dm / T)          # shape (N,)
        WN = np.diag(1.0 / np.sqrt(asset_var))
    else:
        WN = np.eye(N)

    # ------------------------------------------------------------------ #
    # 2. Time weighting matrix  WT  (overweights the mean)
    #    WT = I_T + gamma * (1/T) * 11'
    #    Applying XT' * WT * X / T  =  X'X/T  +  gamma * X_bar * X_bar'
    # ------------------------------------------------------------------ #
    ones_T = np.ones((T, T))
    WT = np.eye(T) + gamma * ones_T / T

    # ------------------------------------------------------------------ #
    # 3. Weighted covariance matrix and eigendecomposition
    # ------------------------------------------------------------------ #
    Xtilde = X @ WN                              # (T, N)
    VarWPCA = Xtilde.T @ WT @ Xtilde / T        # (N, N)

    eigenvalues_raw, eigenvectors = eig(VarWPCA)
    eigenvalues_raw = eigenvalues_raw.real
    eigenvectors    = eigenvectors.real

    # Sort descending
    order       = np.argsort(eigenvalues_raw)[::-1]
    eigenvalues = eigenvalues_raw[order]
    eigenvectors = eigenvectors[:, order]

    # ------------------------------------------------------------------ #
    # 4. Loadings: revert the cross-sectional transformation
    # ------------------------------------------------------------------ #
    Lambda = inv(WN.T) @ eigenvectors[:, :K]    # (N, K)

    # ------------------------------------------------------------------ #
    # 5. Sign normalisation: sign of the average factor return
    # ------------------------------------------------------------------ #
    F_raw     = X @ Lambda @ inv(Lambda.T @ Lambda)   # (T, K)
    sign_vec  = np.sign(F_raw.mean(axis=0))           # (K,)
    Lambda    = Lambda * sign_vec                      # broadcast over rows

    # ------------------------------------------------------------------ #
    # 6. Factor portfolio weights and factor time series
    # ------------------------------------------------------------------ #
    LtL       = Lambda.T @ Lambda                      # (K, K)
    fw        = Lambda @ inv(LtL)                      # (N, K)  raw weights
    # Normalise columns to unit norm
    col_norms = np.sqrt(np.diag(fw.T @ fw))
    fw        = fw / col_norms                         # (N, K)
    F         = X @ fw                                 # (T, K)

    # ------------------------------------------------------------------ #
    # 7. Optional normalisation / orthogonalisation
    # ------------------------------------------------------------------ #
    if variancenormalization == 1 and orthogonalization == 0:
        # Scale loadings by sqrt(eigenvalues); shrink factor weights inversely
        scale     = np.sqrt(eigenvalues[:K])
        Lambda    = Lambda * scale
        fw        = fw / scale
        F         = X @ fw

    elif variancenormalization == 1 and orthogonalization == 1:
        # QR-orthogonalise the demeaned factors
        F_dm      = (np.eye(T) - np.ones((T, T)) / T) @ F / np.sqrt(T)
        Q, R      = qr(F_dm)
        Rotation  = inv(R[:K, :K])
        fw        = fw @ Rotation
        F         = X @ fw
        sign_norm = np.diag(np.sign(F.mean(axis=0)))
        F         = F @ sign_norm
        fw        = fw @ sign_norm
        Lambda    = Lambda @ inv(Rotation) @ sign_norm

    elif variancenormalization == 0 and orthogonalization == 1:
        F_dm      = (np.eye(T) - np.ones((T, T)) / T) @ F / np.sqrt(T)
        Q, R      = qr(F_dm)
        Rdiag     = np.diag(np.diag(R[:K, :K]))
        Rotation  = inv(R[:K, :K]) @ Rdiag
        fw        = fw @ Rotation
        col_norms = np.sqrt(np.diag(fw.T @ fw))
        fw        = fw / col_norms
        F         = X @ fw
        sign_norm = np.diag(np.sign(F.mean(axis=0)))
        F         = F @ sign_norm
        fw        = fw @ sign_norm
        Lambda    = Lambda @ inv(Rotation) @ sign_norm

    # ------------------------------------------------------------------ #
    # 8. Time-series regressions and SDF construction
    # ------------------------------------------------------------------ #
    ones_col    = np.ones((T, 1))
    SDF         = np.zeros((T, K))
    SDF_weights = []
    beta_list   = []
    alpha_arr   = np.zeros((N, K))
    residual_list = []

    for k in range(1, K + 1):
        Fk = F[:, :k]                                  # (T, k)

        # Mean-variance SDF
        Sigma_k   = np.cov(Fk.T, ddof=1)              # (k, k)
        if k == 1:
            Sigma_k = np.array([[Sigma_k]])
        mu_k      = Fk.mean(axis=0)                    # (k,)
        sdf_w     = inv(Sigma_k) @ mu_k                # (k,)
        SDF[:, k-1] = Fk @ sdf_w

        # SDF weights mapped back to asset space
        LamK     = Lambda[:, :k]
        asset_w  = LamK @ inv(LamK.T @ LamK) @ sdf_w  # (N,)
        SDF_weights.append(asset_w)

        # OLS time-series regressions  X = [1, Fk] * coef + residual
        regressors = np.hstack([ones_col, Fk])         # (T, k+1)
        coef       = inv(regressors.T @ regressors) @ regressors.T @ X  # (k+1, N)
        residuals  = X - regressors @ coef             # (T, N)

        alpha_arr[:, k-1] = coef[0, :]                 # intercepts
        beta_list.append(coef[1:, :].T)                # (N, k) betas
        residual_list.append(residuals)

    return {
        "Lambda":            Lambda,
        "F":                 F,
        "eigenvalues":       eigenvalues,
        "SDF":               SDF,
        "SDF_weights_assets": SDF_weights,
        "beta":              beta_list,
        "alpha":             alpha_arr,
        "residuals":         residual_list,
        "factor_weights":    fw,
    }
