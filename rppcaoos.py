"""
RP-PCA Out-of-Sample Evaluation
Python conversion of RPPCAOOS.m by Lettau & Pelger (2020)

Reference:
    "Factors that Fit the Time Series and Cross-Section of Stock Returns"
    Review of Financial Studies (2020), Lettau & Pelger
"""

import numpy as np
from numpy.linalg import inv, eig


# ------------------------------------------------------------------ #
# Internal helper: estimate loadings from a returns window
# ------------------------------------------------------------------ #
def _estimate_loadings(X, gamma, K, stdnorm, N, T):
    """
    Core RP-PCA decomposition used inside the rolling-window loop.
    Returns Lambda (N, K) and the weighted covariance eigenvalues.
    """
    if stdnorm == 1:
        X_dm   = X - X.mean(axis=0, keepdims=True)
        asset_var = np.diag(X_dm.T @ X_dm / T / N)
        WN = np.diag(1.0 / np.sqrt(asset_var))
    else:
        WN = np.eye(N)

    ones_T   = np.ones((T, T))
    WT       = np.eye(T) + gamma * ones_T / T

    Xtilde   = X @ WN
    VarWPCA  = Xtilde.T @ WT @ Xtilde / N / T

    evals, evecs = eig(VarWPCA)
    evals  = evals.real
    evecs  = evecs.real
    order  = np.argsort(evals)[::-1]
    evals  = evals[order]
    evecs  = evecs[:, order]

    Lambda = inv(WN.T) @ evecs[:, :K]
    return Lambda, evals


def rppcaoos(X_total, stdnorm, gamma, K, window=240):
    """
    Out-of-sample evaluation of RP-PCA via a rolling estimation window.

    Parameters
    ----------
    X_total : np.ndarray, shape (T_total, N)
        Full panel of excess returns.
    stdnorm : int
        Cross-sectional standardisation flag (0 = off, 1 = on).
    gamma : float
        RP-PCA risk-premium weight.
    K : int
        Number of factors.
    window : int, default 240
        Rolling estimation window length (months).

    Returns
    -------
    output : dict with keys
        'oos_results'   – (3, K) array: [SR; RMS-alpha; Unexplained var%] OOS
        'is_results'    – (3, K) array: [SR; RMS-alpha; Unexplained var%] in-sample
        'corr_loadings' – (T_OOS, K) generalized correlations (rolling vs full)
        'Lambda_rotated'– (N, K, T_OOS) time-varying loadings rotated to full-sample space
    """
    T_total, N = X_total.shape
    T_OOS      = T_total - window
    T_start    = window          # 0-indexed: first OOS period starts here

    # ------------------------------------------------------------------ #
    # A. Full-sample (in-sample) estimation
    # ------------------------------------------------------------------ #
    Lambda_full, _ = _estimate_loadings(X_total, gamma, K, stdnorm, N, T_total)

    # Sign normalisation on full sample
    F_full_raw  = X_total @ Lambda_full @ inv(Lambda_full.T @ Lambda_full)
    sign_full   = np.sign(F_full_raw.mean(axis=0))
    Lambda_full = Lambda_full * sign_full
    Lambda_total = Lambda_full.copy()   # saved for generalized-corr computation

    # Full-sample factors
    F_full = X_total @ Lambda_full @ inv(Lambda_full.T @ Lambda_full)

    # In-sample SDF, SR, pricing errors
    ones_col = np.ones((T_total, 1))
    SR_IS            = np.zeros(K)
    RMSE_alpha_IS    = np.zeros(K)
    Varalpha_IS      = np.zeros(K)
    Varalpha_pct_IS  = np.zeros(K)
    total_var        = np.trace(np.cov(X_total.T, ddof=1))

    for k in range(1, K + 1):
        Fk       = F_full[:, :k]
        Sigma_k  = np.cov(Fk.T, ddof=1)
        if k == 1:
            Sigma_k = np.array([[Sigma_k]])
        mu_k     = Fk.mean(axis=0)
        sdf_w    = inv(Sigma_k) @ mu_k
        SDF_k    = Fk @ sdf_w
        SR_IS[k-1] = SDF_k.mean() / SDF_k.std(ddof=1)

        regressors = np.hstack([ones_col, Fk])
        coef       = inv(regressors.T @ regressors) @ regressors.T @ X_total
        residuals  = X_total - regressors @ coef
        alpha_k    = coef[0, :]

        Varalpha_IS[k-1]     = np.trace(np.cov(residuals.T, ddof=1)) / N
        Varalpha_pct_IS[k-1] = np.trace(np.cov(residuals.T, ddof=1)) / total_var * 100
        RMSE_alpha_IS[k-1]   = np.sqrt(alpha_k @ alpha_k / N)

    # ------------------------------------------------------------------ #
    # B. Rolling out-of-sample loop
    # ------------------------------------------------------------------ #
    alpha_oos     = [np.zeros((T_OOS, N)) for _ in range(K)]
    X_predict_oos = [np.zeros((T_OOS, N)) for _ in range(K)]
    max_ret_time  = np.zeros((T_OOS, K))
    Corr_final    = np.zeros((T_OOS, K))
    Lambda_rotated = np.zeros((N, K, T_OOS))

    Lambda_prev   = None

    for t in range(T_OOS):
        # Rolling window slice  [t, t+window)
        t0   = t
        t1   = t + window
        X_w  = X_total[t0:t1, :]         # (window, N)
        T_w  = window
        X_next = X_total[t1, :]          # (N,)  next-period returns

        Lambda_t, _ = _estimate_loadings(X_w, gamma, K, stdnorm, N, T_w)

        # Sign alignment
        if t == 0:
            F_w_raw  = X_w @ Lambda_t @ inv(Lambda_t.T @ Lambda_t)
            sign_t   = np.sign(F_w_raw.mean(axis=0))
        else:
            # Align signs with the previous window's loadings
            sign_t = np.sign(np.diag(Lambda_t.T @ Lambda_prev))

        Lambda_t    = Lambda_t * sign_t
        Lambda_prev = Lambda_t.copy()

        # Factors on in-window data
        F_w = X_w @ Lambda_t @ inv(Lambda_t.T @ Lambda_t)

        # -- Generalised correlation with full-sample loadings ----------
        M = inv(Lambda_t.T @ Lambda_t) @ Lambda_t.T @ Lambda_total \
            @ inv(Lambda_total.T @ Lambda_total) @ Lambda_total.T @ Lambda_t
        evals_gc = eig(M)[0].real
        Corr_final[t, :] = np.sort(np.sqrt(np.abs(evals_gc)))[::-1]

        # -- Time-varying loadings rotated into full-sample space -------
        proj = Lambda_t @ inv(Lambda_t.T @ Lambda_t) @ Lambda_t.T @ Lambda_total
        Lambda_rotated[:, :, t] = proj

        # -- OLS pricing errors and optimal portfolio -------------------
        ones_w = np.ones((T_w, 1))
        Sigma_w = np.cov(F_w.T, ddof=1)
        if K == 1:
            Sigma_w = np.array([[Sigma_w]])
        mu_w = F_w.mean(axis=0)

        for k in range(1, K + 1):
            Fk_w = F_w[:, :k]
            reg  = np.hstack([ones_w, Fk_w])
            # OLS: X_w = reg * coef' + eps
            coef_w   = X_w.T @ reg @ inv(reg.T @ reg)   # (N, k+1)
            beta_w   = coef_w[:, 1:]                     # (N, k)

            # OOS pricing error for X_next
            proj_mat = beta_w @ inv(beta_w.T @ beta_w) @ beta_w.T  # (N, N)
            alpha_oos[k-1][t, :] = X_next @ (np.eye(N) - proj_mat)
            X_predict_oos[k-1][t, :] = X_next @ proj_mat

            # OOS max-SR portfolio return
            Sigma_k = Sigma_w[:k, :k]
            mu_k    = mu_w[:k]
            if k == 1:
                Sigma_k = np.array([[Sigma_k.item()]])
            opt_w_k     = np.zeros(K)
            opt_w_k[:k] = inv(Sigma_k) @ mu_k
            max_ret_time[t, k-1] = X_next @ Lambda_t @ inv(Lambda_t.T @ Lambda_t) @ opt_w_k

    # ------------------------------------------------------------------ #
    # C. Aggregate OOS statistics
    # ------------------------------------------------------------------ #
    SR_OOS           = np.zeros(K)
    RMSE_alpha_OOS   = np.zeros(K)
    Varalpha_pct_OOS = np.zeros(K)
    X_oos_slice      = X_total[T_start:, :]
    total_var_oos    = np.trace(np.cov(X_oos_slice.T, ddof=1))

    for k in range(K):
        mean_alpha_k = alpha_oos[k].mean(axis=0)                 # (N,)
        RMSE_alpha_OOS[k]   = np.sqrt(mean_alpha_k @ mean_alpha_k / N)
        Varalpha_pct_OOS[k] = np.trace(np.cov(alpha_oos[k].T, ddof=1)) / total_var_oos * 100

    # Normalised max-return Sharpe ratios
    std_ret = max_ret_time.std(axis=0, ddof=1)
    std_ret[std_ret == 0] = 1                       # guard against zero std
    max_ret_norm = max_ret_time / std_ret
    SR_OOS = max_ret_norm.mean(axis=0)

    oos_results = np.vstack([SR_OOS, RMSE_alpha_OOS, Varalpha_pct_OOS])   # (3, K)
    is_results  = np.vstack([SR_IS,  RMSE_alpha_IS,  Varalpha_pct_IS])    # (3, K)

    return {
        "oos_results":    oos_results,
        "is_results":     is_results,
        "corr_loadings":  Corr_final,
        "Lambda_rotated": Lambda_rotated,
        # Extras for downstream analysis
        "alpha_oos":      alpha_oos,
        "max_ret_time":   max_ret_time,
        "Lambda_full":    Lambda_total,
    }
