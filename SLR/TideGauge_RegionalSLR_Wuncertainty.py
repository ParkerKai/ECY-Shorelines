"""
Created on 05/01/2026

This script Loads all tide gauge data, removes vertical land motion and reconciles
regional SLR.

@author: kaparker
    WA Dept. of Ecology
    kaip461@ecy.wa.gov
"""

__author__ = "Kai Parker"
__email__ = "kaip461@ecy.wa.gov"

# ===============================================================================
# Import Modules
# ===============================================================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt  # zero-phase, stable
import matplotlib.dates as mdates
from numpy.random import default_rng
import statsmodels.api as sm
from typing import List, Union, Optional
from scipy.stats import t

# ===============================================================================
# Functions
# ===============================================================================


def ridgeline_timeseries(
    df,
    time_col: str,
    value_col: str,
    group_col: str,
    *,
    order=None,  # list to control the order of groups (top to bottom)
    spacing=None,  # vertical gap between ridgelines (auto if None)
    scale="zscore",  # 'zscore', 'minmax', or None (raw)
    fill=True,
    palette="Spectral",
    linewidth=1.5,
    alpha=0.75,
    label_left=True,
    label_right=False,
    title="Ridgeline time series",
    # --- NEW ---
    trend=False,  # overlay trend per group
    trend_col=None,  # optional column name with precomputed trend; else OLS on value_col
    trend_style=None,  # dict: e.g., {'color': 'k', 'lw': 2, 'alpha': 0.9}
    # single scale bar for one group
    scale_bar_group=None,  # which group to place the scale bar on
    scale_bar_value=None,  # length in raw units of value_col (e.g., 0.1 m)
    scale_bar_label=None,  # e.g., "10 cm"
    scale_bar_loc="left",  # 'left' | 'right' | float in [0,1]
    scale_bar_style=None,  # dict for line style, e.g., {'color': 'k', 'lw': 2}
):
    """
    Plot a ridgeline of time series with vertical offsets, with optional trend overlay
    and a single-group y-axis scale bar.

    df: long-form DataFrame with columns [time_col, value_col, group_col]
    Returns: (fig, ax, meta) where meta={'groups': [...], 'offsets': {group: base}}
    """
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from matplotlib.dates import AutoDateLocator, ConciseDateFormatter
    import matplotlib.dates as mdates

    df = df.copy()

    # Ensure datetime
    if not np.issubdtype(df[time_col].dtype, np.datetime64):
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")

    # Determine group order
    groups = (
        list(pd.unique(df[group_col]))
        if order is None
        else [g for g in order if g in set(df[group_col])]
    )
    n = len(groups)
    if n == 0:
        raise ValueError("No groups found to plot.")

    # Choose spacing automatically based on median std dev
    if spacing is None:
        s = df.groupby(group_col)[value_col].std().median()
        spacing = 3 * s if pd.notnull(s) and s > 0 else 1.0

    # Visuals
    sns.set_theme(style="white")
    colors = sns.color_palette(palette, n)

    fig_height = max(2.5, 0.7 * n + 1.5)  # scale fig height with number of groups
    fig, ax = plt.subplots(figsize=(10, fig_height))
    ax.set_title(title, pad=12)

    # Date formatting
    locator = AutoDateLocator()
    formatter = ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    # --- Plot each group with a vertical offset ---
    offsets = []
    group_to_base = {}
    x_min = pd.to_datetime(df[time_col]).min()
    x_max = pd.to_datetime(df[time_col]).max()

    # Defaults for styles
    if trend_style is None:
        trend_style = dict(color="k", lw=2.0, alpha=0.9)
    if scale_bar_style is None:
        scale_bar_style = dict(color="k", lw=2.0, alpha=0.9)

    for i, g in enumerate(groups):
        gdf = df.loc[df[group_col] == g].sort_values(time_col)
        if gdf.empty:
            continue

        t = pd.to_datetime(gdf[time_col]).to_numpy()
        y_raw = pd.to_numeric(gdf[value_col], errors="coerce").to_numpy()

        # Scaling per group
        if scale == "zscore":
            mu = np.nanmean(y_raw)
            sd = np.nanstd(y_raw)
            sd_eff = sd if sd > 0 else 1.0
            ys = (y_raw - mu) / sd_eff
            # Factors for scaling auxiliary lines (trend, scale bar)
            scale_factor = 1.0 / sd_eff  # raw Δ -> displayed Δ

            def transform_aux(v):  # raw value -> displayed value
                return (v - mu) / sd_eff
        elif scale == "minmax":
            m, M = np.nanmin(y_raw), np.nanmax(y_raw)
            rng = M - m
            rng_eff = rng if rng > 0 else 1.0
            ys = (y_raw - m) / rng_eff
            scale_factor = 1.0 / rng_eff

            def transform_aux(v):
                return (v - m) / rng_eff
        else:
            ys = y_raw.copy()
            scale_factor = 1.0

            def transform_aux(v):
                return v

        base = i * spacing
        offsets.append(base)
        group_to_base[g] = base

        # Time plot
        ax.plot(t, ys + base, color=colors[i], lw=linewidth)
        if fill:
            ax.fill_between(
                t, base, ys + base, color=colors[i], alpha=alpha, linewidth=0
            )

        # Optional inline labels
        if label_left:
            ax.text(x_min, base, str(g), va="center", ha="left", fontsize=9)
        if label_right:
            ax.text(x_max, base, str(g), va="center", ha="right", fontsize=9)

        # --- Trend overlay (per group) ---
        if trend:
            if trend_col is not None and trend_col in gdf.columns:
                y_tr_raw = pd.to_numeric(gdf[trend_col], errors="coerce").to_numpy()
                y_tr_disp = transform_aux(y_tr_raw)
                ax.plot(t, y_tr_disp + base, **trend_style)
            else:
                # OLS on raw values vs time (days since first)
                x_num = mdates.date2num(t)
                mask = np.isfinite(x_num) & np.isfinite(y_raw)
                if mask.sum() >= 2:
                    coef = np.polyfit(x_num[mask], y_raw[mask], 1)  # m/day
                    y_fit_raw = np.poly1d(coef)(x_num)
                    y_fit_disp = transform_aux(y_fit_raw)
                    ax.plot(t, y_fit_disp + base, **trend_style)

        # --- Single scale bar for the chosen group ---
        if (
            (scale_bar_group is not None)
            and (g == scale_bar_group)
            and (scale_bar_value is not None)
        ):
            # Horizontal position
            if isinstance(scale_bar_loc, (int, float)):
                frac = float(scale_bar_loc)
                frac = 0.0 if frac < 0 else 1.0 if frac > 1.0 else frac
                x_pos = x_min + (x_max - x_min) * frac
            else:
                if str(scale_bar_loc).lower() == "right":
                    x_pos = x_max
                else:
                    x_pos = x_min

            # Display-space length for the bar
            dy_disp = scale_bar_value * scale_factor
            y0 = base
            y1 = base + dy_disp
            ax.plot([x_pos, x_pos], [y0, y1], **scale_bar_style)
            if scale_bar_label:
                ax.text(
                    x_pos,
                    (y0 + y1) / 2.0,
                    scale_bar_label,
                    ha="left" if str(scale_bar_loc).lower() == "left" else "right",
                    va="center",
                    fontsize=9,
                    bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
                )

    # Y ticks at offsets with group labels (for quick reading)
    ax.set_yticks(offsets)
    ax.set_ylabel("")
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    sns.despine(ax=ax, left=True, right=True)

    plt.tight_layout()
    meta = {"groups": groups, "offsets": group_to_base}
    return fig, ax, meta


def fit_line_with_fixed_slope(x, y, slope, time_unit="h"):
    """
    Fits a line y = slope * t + intercept, where t = (x - x0) in chosen time units.
    Slope is fixed; returns (intercept, x0).

    Parameters:
        x : array-like or pandas Series/Index (numeric or datetime64)
        y : array-like
        slope : float (units per 'time_unit')
        time_unit : str ("D", "h", "m", "s", "ns")

    Returns:
        intercept : float
        x0 : scalar (datetime64 if x is datetime-like)
    """
    if isinstance(x, (pd.Series, pd.Index)):
        x = x.to_numpy()
    if isinstance(y, (pd.Series, pd.Index)):
        y = y.to_numpy()

    # Reference x0
    x0 = x.min()

    # Convert to numeric time axis
    if np.issubdtype(x.dtype, np.datetime64):
        t = (x - x0) / np.timedelta64(1, time_unit)
        t = t.astype(float)
    else:
        t = np.asarray(x, dtype=float) - float(x0)

    y = np.asarray(y, dtype=float)

    if t.size != y.size or t.size == 0:
        raise ValueError("x and y must be non-empty arrays of the same length.")

    # Least squares intercept for fixed slope
    intercept = np.nanmean(y - slope * t)
    return intercept, x0


def moving_block_bootstrap(y: np.ndarray, X: np.ndarray, block_len: int, rng=None):
    """
    Simple moving-block bootstrap for (X, y) time series.
    Returns resampled (X*, y*) of same length.
    """
    rng = rng or default_rng()
    n = len(y)
    blocks = []
    starts = rng.integers(0, n - block_len + 1, size=int(np.ceil(n / block_len)))
    for s in starts:
        e = min(s + block_len, n)
        blocks.append((X[s:e, :], y[s:e]))
    X_star = np.vstack([b[0] for b in blocks])[:n, :]
    y_star = np.hstack([b[1] for b in blocks])[:n]
    return X_star, y_star


def nested_bootstrap_pool(
    X_list: List[np.ndarray],
    y_list: List[np.ndarray],
    B_outer: int = 500,
    B_inner: int = 1,
    block_len: int = 10,
    hac_lags: Optional[int] = None,
    rng=None,
):
    """
    Nested bootstrap:
    - Outer: resample ensemble members with replacement
    - Inner: per selected member, block-bootstrap data, refit, collect params
    Returns array of pooled parameter draws (B_outer x P).
    """
    rng = rng or default_rng()
    M = len(X_list)
    P = sm.add_constant(X_list[0]).shape[1]
    pooled_draws = np.zeros((B_outer, P))
    for b in range(B_outer):
        # sample members with replacement
        idx = rng.integers(0, M, size=M)
        params_boot = []
        for i in idx:
            X_i, y_i = X_list[i], y_list[i]
            for _ in range(B_inner):
                X_star, y_star = moving_block_bootstrap(
                    y_i, X_i, block_len=block_len, rng=rng
                )
                params_i, _, _, _ = fit_member_ols(X_star, y_star, hac_lags=hac_lags)
                params_boot.append(params_i)
        params_boot = np.vstack(params_boot)
        pooled_draws[b, :] = params_boot.mean(
            axis=0
        )  # pooled mean across resampled members
    return pooled_draws


def fit_member_ols(X: np.ndarray, y: np.ndarray, hac_lags: Optional[int] = None):
    """
    Fit OLS (optionally HAC/Newey-West) for one ensemble member.
    Returns (params, cov, sigma2, nobs).
    X should already include columns for the features (no intercept added here).
    """
    X_ = sm.add_constant(X, has_constant="add")  # add intercept
    model = sm.OLS(y, X_)
    if hac_lags is None or hac_lags == 0:
        res = model.fit()
    else:
        res = model.fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})
    params = res.params  # [const, feature1, feature2, ...]
    cov = res.cov_params()  # covariance matrix of params
    # For predictive intervals: use unbiased MSE (model.ssr/df_resid). Statsmodels stores in res.mse_resid.
    sigma2 = res.mse_resid
    nobs = int(res.nobs)
    return params, cov, sigma2, nobs


def pool_parameters(params_list: List[np.ndarray], cov_list: List[np.ndarray]):
    """
    Pool parameter estimates across ensemble members using law of total variance/Rubin's rules.
    Returns beta_bar (mean), total_cov T, W (within), B (between).
    """
    M = len(params_list)
    P = params_list[0].shape[0]
    betas = np.vstack(params_list)  # shape M x P
    beta_bar = betas.mean(axis=0)  # pooled mean

    # Between-member covariance
    centered = betas - beta_bar
    B = centered.T @ centered / (M - 1) if M > 1 else np.zeros((P, P))

    # Within-member covariance (average of covariances)
    W = sum(cov_list) / M

    # Total covariance (Rubin's-style)
    T = W + (1 + 1 / M) * B
    return beta_bar, T, W, B


def parameter_intervals(
    beta_bar: np.ndarray,
    T: np.ndarray,
    M: int,
    alpha: float = 0.05,
    W: Optional[np.ndarray] = None,
    B: Optional[np.ndarray] = None,
):
    """
    Compute t-based confidence intervals for each parameter using Rubin's df approximation (component-wise).
    For component j: nu_j = (M-1) * (1 + W_jj / ((1+1/M)*B_jj))^2, with guards.
    """
    P = beta_bar.shape[0]
    se = np.sqrt(np.diag(T))
    # Degrees of freedom per component (scalar Rubin's rule). If B_jj ~ 0, use large df fallback.
    dof = np.empty(P)
    for j in range(P):
        Bj = B[j, j] if B is not None else 0.0
        Wj = W[j, j] if W is not None else 0.0
        if Bj > 1e-12:
            nu_j = (M - 1) * (1 + Wj / ((1 + 1 / M) * Bj)) ** 2
            dof[j] = max(nu_j, 1.0)
        else:
            # If between variance is ~0, fall back to large df
            dof[j] = 10_000.0
    tcrit = t.ppf(1 - alpha / 2, dof)
    ci_lower = beta_bar - tcrit * se
    ci_upper = beta_bar + tcrit * se
    return pd.DataFrame(
        {
            "param_index": np.arange(P),
            "estimate": beta_bar,
            "std_err": se,
            "df": dof,
            f"CI{int((1 - alpha) * 100)}_lower": ci_lower,
            f"CI{int((1 - alpha) * 100)}_upper": ci_upper,
        }
    )


def predict_with_uncertainty(
    X_new: np.ndarray,
    beta_bar: np.ndarray,
    T: np.ndarray,
    sigma2_list: Union[List[float], np.ndarray],
    alpha: float = 0.05,
    add_intercept: bool = True,
    df: Optional[float] = 10_000.0,
    return_cov: bool = False,
):
    """
    Vectorized predictive intervals using pooled parameters and covariance.

    Parameters
    ----------
    X_new : np.ndarray
        Design matrix of new points without intercept, shape (N, d) or (d,).
        If 1D, it is treated as a single row.
    beta_bar : np.ndarray
        Pooled parameter estimates, shape (p,), where p = d + (1 if add_intercept else 0).
    T : np.ndarray
        Total parameter covariance (includes within and between), shape (p, p).
    sigma2_list : list or np.ndarray
        Residual variance estimates from ensemble members; we use their average.
    alpha : float, default 0.05
        Two-sided interval level (e.g., 0.05 -> 95% intervals).
    add_intercept : bool, default True
        If True, prepend a column of ones to X_new.
    df : float or None, default 10000.0
        Degrees of freedom for the t critical value. Use a large value when you
        prefer normal approximation; set None to use normal (z) explicitly.
    return_cov : bool, default False
        If True, also return the predictive covariance matrix across all N points
        (parameter uncertainty term + diagonal residual variance).

    Returns
    -------
    out : pandas.DataFrame
        Columns: 'mean', 'se_pred', 'PI{level}_lower', 'PI{level}_upper'
        One row per input point.
    cov_pred : np.ndarray, optional
        Only if return_cov=True. Predictive covariance matrix across N points:
        Cov(y_hat) = X_aug @ T @ X_aug.T  +  sigma2_bar * I_N
        Note: residual autocorrelation is NOT modeled here; see notes.

    Notes
    -----
    - Pointwise intervals are fine for most uses. For *simultaneous bands*
      across many points, use the returned covariance to build joint intervals
      (e.g., Bonferroni/Scheffé) or a bootstrap approach.
    - Residual variance is added on the diagonal (assumes uncorrelated noise in
      the forecast error at distinct points). For time series with serially
      correlated future errors, consider a model for the noise or use block
      bootstrap to capture off-diagonal residual covariance.
    """
    X_new = np.asarray(X_new)
    if X_new.ndim == 1:
        # Treat as a single row of features
        X_new = X_new.reshape(1, -1)

    N = X_new.shape[0]

    # Add intercept if requested
    if add_intercept:
        X_aug = np.column_stack([np.ones(N), X_new])
    else:
        X_aug = X_new

    # Sanity checks on shapes
    p = X_aug.shape[1]
    if beta_bar.shape[0] != p or T.shape != (p, p):
        raise ValueError(
            f"Shape mismatch: beta_bar has {beta_bar.shape[0]} params, "
            f"T is {T.shape}, X_aug has {p} columns."
        )

    # Mean predictions
    mean_pred = X_aug @ beta_bar  # shape (N,)

    # Parameter-uncertainty contribution to variance at each point:
    # var_i = x_i^T T x_i  for i = 1..N
    var_pred_params = np.einsum("ij,jk,ik->i", X_aug, T, X_aug)

    # Average residual variance across ensemble members
    sigma2_bar = float(np.mean(sigma2_list))
    var_pred_total = var_pred_params + sigma2_bar
    se_pred = np.sqrt(var_pred_total)

    # Critical value (t or z)
    if df is None or np.isinf(df):
        # Use normal approximation
        from scipy.stats import norm

        crit = norm.ppf(1 - alpha / 2)
    else:
        crit = t.ppf(1 - alpha / 2, df)

    lower = mean_pred - crit * se_pred
    upper = mean_pred + crit * se_pred

    level = int((1 - alpha) * 100)
    out = pd.DataFrame(
        {
            "mean": mean_pred,
            "se_pred": se_pred,
            f"PI{level}_lower": lower,
            f"PI{level}_upper": upper,
        }
    )

    if return_cov:
        # Full predictive covariance across N points:
        # parameter part: X_aug T X_aug^T
        cov_params = X_aug @ T @ X_aug.T
        # Add residual variance on diagonal only
        cov_total = cov_params + sigma2_bar * np.eye(N)
        return out, cov_total

    return out


def fixed_effect_mean(b, se):
    """Inverse-variance pooled mean (fixed-effect)."""
    w = 1.0 / (se**2)
    mu = np.sum(w * b) / np.sum(w)
    var_mu = 1.0 / np.sum(w)
    return mu, var_mu, w


def cochrans_Q_I2(b, se, mu, w):
    """Heterogeneity diagnostics."""
    Q = np.sum(w * (b - mu) ** 2)
    k = len(b)
    I2 = max(0.0, (Q - (k - 1)) / Q) if Q > 0 else 0.0
    return Q, I2


def dersimonian_laird_tau2(b, se, w, mu):
    """Between-station variance tau^2 (DerSimonian-Laird)."""
    k = len(b)
    Q = np.sum(w * (b - mu) ** 2)
    c = np.sum(w) - (np.sum(w**2) / np.sum(w))
    tau2 = max(0.0, (Q - (k - 1)) / c) if c > 0 else 0.0
    return tau2, Q


def random_effects_mean(b, se):
    """Random-effects pooled mean using DL tau^2."""
    mu_fe, var_mu_fe, w_fe = fixed_effect_mean(b, se)
    tau2, Q = dersimonian_laird_tau2(b, se, w_fe, mu_fe)
    w_re = 1.0 / (se**2 + tau2)
    mu_re = np.sum(w_re * b) / np.sum(w_re)
    var_mu_re = 1.0 / np.sum(w_re)
    # I2 from FE components
    _, I2 = cochrans_Q_I2(b, se, mu_fe, w_fe)
    return mu_re, var_mu_re, tau2, w_re, Q, I2


def benjamini_hochberg(p):
    """FDR-adjusted p-values (BH)."""
    p = np.asarray(p)
    m = len(p)
    order = np.argsort(p)
    ranks = np.arange(1, m + 1)
    adj_sorted = p[order] * m / ranks
    # enforce monotonicity
    adj_sorted = np.minimum.accumulate(adj_sorted[::-1])[::-1]
    adj = np.empty(m)
    adj[order] = np.clip(adj_sorted, 0.0, 1.0)
    return adj


def holm_stepdown(p):
    """Holm-Bonferroni adjusted p-values (strong FWER control)."""
    p = np.asarray(p)
    m = len(p)
    order = np.argsort(p)
    p_sorted = p[order]
    adj_sorted = np.empty(m)
    # step-down: increasing rank -> decreasing multiplier
    for j in range(m):
        adj_sorted[j] = min(1.0, (m - j) * p_sorted[j])
    # make monotone non-decreasing
    adj_sorted = np.maximum.accumulate(adj_sorted)
    adj = np.empty(m)
    adj[order] = adj_sorted
    return adj


def test_vs_mean_leave_one_out(b, se, use_random_effects=False, method_tau="DL"):
    """
    Compare each station's slope b[i] to the pooled mean of all OTHER stations.
    Returns per-station: diff_from_mean, z, p (two-sided normal approx),
    plus pooled mean and its variance (for reference).
    """

    from scipy.stats import norm

    b = np.asarray(b, dtype=float)
    se = np.asarray(se, dtype=float)
    k = len(b)

    # helper: DL tau^2
    def tau2_DL(b, se):
        w = 1.0 / (se**2)
        mu_fe = np.sum(w * b) / np.sum(w)
        Q = np.sum(w * (b - mu_fe) ** 2)
        c = np.sum(w) - (np.sum(w**2) / np.sum(w))
        tau2 = max(0.0, (Q - (k - 1)) / c) if c > 0 else 0.0
        return tau2

    diffs, zvals, pvals = [], [], []

    for i in range(k):
        # leave-one-out
        b_minus = np.delete(b, i)
        se_minus = np.delete(se, i)
        k_minus = k - 1

        if use_random_effects:
            # re-estimate tau^2 on the reduced set
            tau2_i = (
                tau2_DL(b_minus, se_minus)
                if method_tau == "DL"
                else tau2_DL(b_minus, se_minus)
            )
            w_re_minus = 1.0 / (se_minus**2 + tau2_i)
            mu_minus = np.sum(w_re_minus * b_minus) / np.sum(w_re_minus)
            var_mu_minus = 1.0 / np.sum(w_re_minus)
            # variance of difference includes tau^2
            var_diff = se[i] ** 2 + tau2_i + var_mu_minus
        else:
            # fixed-effect leave-one-out
            w_fe_minus = 1.0 / (se_minus**2)
            mu_minus = np.sum(w_fe_minus * b_minus) / np.sum(w_fe_minus)
            var_mu_minus = 1.0 / np.sum(w_fe_minus)
            var_diff = se[i] ** 2 + var_mu_minus

        diff = b[i] - mu_minus
        z = diff / np.sqrt(var_diff)
        p = 2.0 * (1.0 - norm.cdf(abs(z)))

        diffs.append(diff)
        zvals.append(z)
        pvals.append(p)

    # For reference: pooled mean & variance on full set (RE or FE)
    if use_random_effects:
        tau2 = tau2_DL(b, se)
        w_re = 1.0 / (se**2 + tau2)
        mu_hat = np.sum(w_re * b) / np.sum(w_re)
        var_mu = 1.0 / np.sum(w_re)
    else:
        w_fe = 1.0 / (se**2)
        mu_hat = np.sum(w_fe * b) / np.sum(w_fe)
        var_mu = 1.0 / np.sum(w_fe)

    return np.array(diffs), np.array(zvals), np.array(pvals), mu_hat, var_mu


def lowpass_hourly_with_gap_handling(
    df: pd.DataFrame,
    col: str = "v_obs",
    cutoff_days: float = 30.0,
    order: int = 6,
    fs_hours: float = 1.0,  # samples per hour
    small_gap_limit_hours: int = 48,    # interpolate interior gaps up to this length
    edge_fill_limit_hours: int = 48,    # temporarily fill leading/trailing NaNs up to this
    renan_buffer_hours: int = 24,       # re-NaN buffer around large gaps after filtering
    return_monthly: bool = True,
    missing_days_allowed: int = 15      # PSMSL-style: if >15 days missing, mask monthly mean
):
    """
    Create an hourly, low-pass filtered series with robust gap handling:
      - Fill small interior gaps to run the filter.
      - Temporarily fill edges and large gaps, then re-NaN those regions after filtering.
    Optionally compute monthly means from the filtered output, masking months with insufficient raw coverage.

    Parameters
    ----------
    df : DataFrame with DatetimeIndex and a column `col`.
    col : Name of the water level column.
    cutoff_days : Low-pass cutoff in days (e.g., 30).
    order : Butterworth filter order (before forward-backward).
    fs_hours : Sampling rate (samples per hour).
    small_gap_limit_hours : Max interior gap length to interpolate.
    edge_fill_limit_hours : Max edge gap length to temporarily fill without classifying as "large".
    renan_buffer_hours : Hours to buffer on each side of large gaps when re-masking as NaN post-filter.
    return_monthly : If True, also return monthly means masked by raw-data availability.
    missing_days_allowed : How many missing days are allowed per month before masking.

    Returns
    -------
    result : dict with keys:
        'df_hourly' : DataFrame with columns [col, col+'_lp'] on hourly grid.
        'monthly_lp' : (optional) DataFrame of monthly means of the filtered series (masked).
        'large_gaps' : list of (start, end) timestamps for large gaps in the hourly grid.
    """

    # --- 0) Preconditions ---
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("df must have a DatetimeIndex")

    df = df.sort_index()
    if df.index.freq is None:
        # Build a contiguous hourly index over full range
        full_idx = pd.date_range(df.index.min().floor("h"),
                                 df.index.max().ceil("h"),
                                 freq="h")
        df_hourly = df.reindex(full_idx)
    else:
        # If already regular, enforce exact hourly frequency
        df_hourly = df.asfreq("h")

    # Raw hourly series
    s_raw = pd.to_numeric(df_hourly[col], errors="coerce")

    # --- 1) Identify gap runs (NaNs) and classify "large" vs "small" ---
    def find_nan_runs(s: pd.Series):
        idx = s.index
        mask = s.isna().to_numpy()
        runs = []
        i = 0
        n = len(mask)
        while i < n:
            if mask[i]:
                j = i
                while j + 1 < n and mask[j + 1]:
                    j += 1
                    runs.append((idx[i], idx[j], j - i + 1))  # start_ts, end_ts, length_hours
                i = j + 1
            else:
                i += 1
        return runs

    nan_runs = find_nan_runs(s_raw)

    # Edge runs: those touching the first or last index
    first_idx = df_hourly.index[0]
    last_idx  = df_hourly.index[-1]

    large_gaps = []
    for start_ts, end_ts, L in nan_runs:
        is_edge = (start_ts == first_idx) or (end_ts == last_idx)
        if is_edge and L > edge_fill_limit_hours:
            large_gaps.append((start_ts, end_ts))
        elif (not is_edge) and L > small_gap_limit_hours:
            large_gaps.append((start_ts, end_ts))
        # smaller gaps will be interpolated

    # --- 2) Fill small interior gaps by time interpolation ---
    s_work = (
        s_raw
        .interpolate(method="time",
                     limit=small_gap_limit_hours,
                     limit_area="inside",
                     limit_direction="both")
    )

    # --- 3) Temporarily fill edges up to edge_fill_limit_hours ---
    # This keeps leading/trailing short gaps from breaking the filter.
    s_work = s_work.ffill(limit=edge_fill_limit_hours).bfill(limit=edge_fill_limit_hours)

    # After step 2 & 3, we may still have NaNs (very long edge gaps or interior gaps > limit).
    # To ensure the filter can run, fill any remaining NaNs *temporarily* using nearest neighbor.
    if s_work.isna().any():
        s_work = s_work.ffill().bfill()  # remove all NaNs, but we will re-NaN large gaps later

    # --- 4) Design & apply low-pass (zero-phase) ---
    cutoff_cph = 1.0 / (cutoff_days * 24.0)  # cycles per hour
    sos = butter(N=order, Wn=cutoff_cph, btype="low", output="sos", fs=fs_hours)
    # Forward–backward zero-phase filtering (effective order doubled; magnitude squared). [2](https://jp.mathworks.com/help//signal/ref/filtfilt.html?lang=en)
    v_lp = sosfiltfilt(sos, s_work.to_numpy())  # Requires finite values across the full array. [1](https://docs.scipy.org/doc//scipy-1.17.0/reference/generated/scipy.signal.sosfiltfilt.html)
    s_lp = pd.Series(v_lp, index=df_hourly.index)

    # --- 5) Re-NaN large gaps + buffer to avoid artifacts near gap edges ---
    if large_gaps:
        # Build a boolean mask to NaN the filtered output in those windows
        to_nan = pd.Series(False, index=df_hourly.index)
        # Precompute buffers as Timedelta for convenience
        buf = pd.Timedelta(hours=renan_buffer_hours)

        for (g_start, g_end) in large_gaps:
            # Extended window
            win_start = max(df_hourly.index[0], g_start - buf)
            win_end   = min(df_hourly.index[-1], g_end + buf)
            to_nan.loc[win_start:win_end] = True

        s_lp = s_lp.mask(to_nan)

    # --- 6) Assemble hourly DataFrame ---
    out = df_hourly.copy()
    out[col] = s_raw
    out[f"{col}_lp"] = s_lp

    result = {"df_hourly": out, "large_gaps": large_gaps}

    # --- 7) Optional: monthly mean of filtered series, with a RAW-data coverage mask ---
    if return_monthly:
        # Count raw observed hours per month (before any interpolation)
        monthly_raw_hours = s_raw.notna().resample("MS").sum()

        # Threshold per month based on missing_days_allowed
        dim = monthly_raw_hours.index.days_in_month
        min_hours_required = np.maximum(dim - missing_days_allowed, 0) * 24

        monthly_mean_lp = s_lp.resample("MS").mean()

        # Mask months with insufficient raw coverage
        monthly_mean_lp_masked = monthly_mean_lp.where(monthly_raw_hours >= min_hours_required)

        result["monthly_lp"] = (
            monthly_mean_lp_masked.rename(f"{col}_MonthlyMean_LP").reset_index()
        )

    return result

# ===============================================================================
# User Inputs
# ===============================================================================
dir_in = r"C:\Users\kaip461\OneDrive - Washington State Executive Branch Agencies\Documents\Data\TideGauge\Hourly"
dir_in_vlm = r'C:\Users\kaip461\OneDrive - Washington State Executive Branch Agencies\Documents\Data\VLM'
dir_out = r"C:\Users\kaip461\OneDrive - Washington State Executive Branch Agencies\Documents\Projects\RSLR\figures"
os.makedirs(dir_out, exist_ok=True)

# Monthly aggregation method
agg_method = "lowpass"  # Options: 'mean', 'lowpass'


# Record Length Trim (cut records to being within certain dates)
dt_start = pd.Timestamp("1992-01-01")   # or none

# ===============================================================================
# Tide Gauge Info
# ===============================================================================

# TG = [
#     {"station": '9440910', "name": 'TokePoint',   "vlm_mm_per_yr": 0.035921, "vlm_unc_mm_per_yr": 0.622934},
#     {"station": '9441102', "name": 'Westport',    "vlm_mm_per_yr": 1.632204, "vlm_unc_mm_per_yr": 0.841024},
#     {"station": '9442396', "name": 'LaPush',      "vlm_mm_per_yr": -0.881595, "vlm_unc_mm_per_yr": 0.829389},
#     {"station": '9443090', "name": 'NeahBay',     "vlm_mm_per_yr": 2.80886, "vlm_unc_mm_per_yr": 0.990746},
#     {"station": '9444090', "name": 'PortAngeles', "vlm_mm_per_yr": 1.075373, "vlm_unc_mm_per_yr": 0.53471},
#     {"station": '9444900', "name": 'PortTownsend',"vlm_mm_per_yr": -0.934305, "vlm_unc_mm_per_yr": 0.848552},
#     {"station": '9449880', "name": 'FridayHarbor',"vlm_mm_per_yr": 0.400442, "vlm_unc_mm_per_yr": 0.698111},
#     {"station": '9449424', "name": 'CherryPoint', "vlm_mm_per_yr": 1.254333, "vlm_unc_mm_per_yr": 0.589758},
#     {"station": '9447130', "name": 'Seattle',     "vlm_mm_per_yr": -0.648102, "vlm_unc_mm_per_yr": 0.470093},
#     {"station": '9446484', "name": 'Tacoma',      "vlm_mm_per_yr": -1.278814, "vlm_unc_mm_per_yr": 0.73966},

# ]
# TG_table = pd.DataFrame(TG)



# TG = [
    
#     #{"station": '9439099', "name": 'Wauna'},
#     {"station": '9440569', "name": 'Skamokawa'},
#     #{"station": '9440581', "name": 'CapeDisappointment'},
#     {"station": '9439040', "name": 'Astoria'},
#     {"station": '9440910', "name": 'TokePoint'},
#     {"station": '9441102', "name": 'Westport'},
#     {"station": '9442396', "name": 'LaPush'},
#     {"station": '9443090', "name": 'NeahBay'},
#     {"station": '9444090', "name": 'PortAngeles'},
#     {"station": '9444900', "name": 'PortTownsend'},
#     {"station": '9449880', "name": 'FridayHarbor'},
#     {"station": '9449424', "name": 'CherryPoint'},
#    {"station": '9447130', "name": 'Seattle'},
#    {"station": '9446484', "name": 'Tacoma'},
# ]





TG = [
    {"station": '9419750', "name": 'CrescentCity'},
    {"station": '9431647', "name": 'PortOrford'},
    {"station": '9432780', "name": 'Charleston'},
    {"station": '9435380', "name": 'SouthBeach'},
    {"station": '9437540', "name": 'Garibaldi'},
    #{"station": '9439099', "name": 'Wauna'},
    {"station": '9440569', "name": 'Skamokawa'},
    #{"station": '9440581', "name": 'CapeDisappointment'},
    {"station": '9439040', "name": 'Astoria'},
    {"station": '9440910', "name": 'TokePoint'},
    #{"station": '9441102', "name": 'Westport'},
    #{"station": '9442396', "name": 'LaPush'},
    {"station": '9443090', "name": 'NeahBay'},
    {"station": '9444090', "name": 'PortAngeles'},
    {"station": '9444900', "name": 'PortTownsend'},
    {"station": '9449880', "name": 'FridayHarbor'},
    #{"station": '9449424', "name": 'CherryPoint'},
   {"station": '9447130', "name": 'Seattle'},
   {"station": '9446484', "name": 'Tacoma'},
]



TG = pd.DataFrame(TG).set_index("station")


# Add VLM
# df_vlm = pd.read_csv(os.path.join(dir_in_vlm, "vlm_at_TGs_linear.csv"), dtype={"Station": str})
# df_vlm = df_vlm.set_index("Station")
path = os.path.join(dir_in_vlm, '2026', 'vlm_rates_tidegauges.txt')

df_vlm = pd.read_csv(
    path,
    sep=r"[ \t]+",           # split on runs of spaces or tabs (explicit)
    engine="python",         # more tolerant tokenizer
    encoding="utf-8-sig",    # strips BOM if present at file start
    header=None,
    names=["Station", "Longitude", "Latitude", "VLM_interp", "VLM_uncertainty"],
    comment="#",
    skip_blank_lines=True,
    dtype={
        "Station": "str",    # nullable int (safe if any row is missing)
        "Longitude": "float64",
        "Latitude": "float64",
        "VLM_interp": "float64",
        "VLM_uncertainty": "float64",
    },
)


print(df_vlm.head())
print(df_vlm.dtypes)
df_vlm = df_vlm.set_index("Station")


TG_table = TG.join(
    df_vlm[["VLM_interp", "VLM_uncertainty"]],
    how="left"
)

TG_table = TG_table.reset_index().rename(columns={"index": "station"})


TG_table = TG_table.rename(columns={
    "VLM_interp": "vlm_mm_per_yr",
    "VLM_uncertainty": "vlm_unc_mm_per_yr",
})


name_map = dict(zip(TG_table["station"], TG_table["name"]))


num_samples = 10000  # number of VLM samples per station for uncertainty propagation


# ===============================================================================
# Load & Process
# ===============================================================================
data_mnth = []
data_hrly = []
data_pred = []
TG_table["SLR_rate"] = np.full(
    len(TG_table), np.nan
)  # To store SLR rates for each station
TG_table["SLR_rate_upper"] = np.full(
    len(TG_table), np.nan
)  # To store upper bound of SLR rates for each station
TG_table["SLR_rate_lower"] = np.full(
    len(TG_table), np.nan
)  # To store lower bound of SLR rates for each station

# Add columns to hold slope SE and df (both in m/day and mm/yr for convenience)
for col in ["SLR_slope_m_per_day", "SLR_se_m_per_day", "SLR_df", "SLR_se_mm_per_yr"]:
    TG_table[col] = np.nan


TG_table["seasonal_amp"] = np.full(
    len(TG_table), np.nan
)  # To store seasonal amplitude for each station
for cnt, stat in enumerate(TG_table["station"]):
    print(f"Processing station {stat} ({cnt + 1}/{len(TG_table['station'])})...")

    file_in = os.path.join(dir_in, f"NOAA_TG_{stat}", "combined_resolved.parquet")
    df = pd.read_parquet(file_in)

    # Make sure t is datetime
    df["t"] = pd.to_datetime(df["t"])
    df = df.sort_values("t").set_index("t")

    
    if dt_start is not None:
        print(f'Trimming data from station {stat} to specified {dt_start}')
        df = df.loc[pd.Timestamp(dt_start):]

    
    
    # ---- Aggregation ----
    if agg_method == "mean":
        s = df["v_obs"]

        # Monthly mean and count (non-NaN)
        monthly_mean = s.resample("MS").mean()
        monthly_count = s.resample("MS").count()

        # Rule: require at least this many samples per month
        MIN_SAMPLES = 400  # A little over half a month for hourly sampled data

        # Mask insufficiently sampled months to NaN
        monthly_mean_masked = monthly_mean.mask(monthly_count < MIN_SAMPLES)

        monthly = monthly_mean_masked.rename("v_MonthlyMean").reset_index()

    elif agg_method == "lowpass":
        
        res = lowpass_hourly_with_gap_handling(
            df, col="v_obs",
            cutoff_days=30, order=6, fs_hours=1.0,
            small_gap_limit_hours=48,
            edge_fill_limit_hours=48,
            renan_buffer_hours=24,
            return_monthly=True,
            missing_days_allowed=15
        )

        df_hourly = res["df_hourly"]   # columns: v_obs (raw), v_obs_lp (low-pass) with re-NaNs over big gaps
        monthly  = res["monthly_lp"]   # masked monthly means of the low-pass series
        gaps    = res["large_gaps"]   # list of (start, end) timestamps for large gaps

        monthly = monthly.rename(columns={"index": "t", 
                                          'v_obs_MonthlyMean_LP': 'v_MonthlyMean'})
        
    else:
        raise ValueError("agg_method must be 'mean' or 'lowpass'.")

    
    
    
    # ---- Common time fields ----
    monthly["t"] = pd.to_datetime(
        monthly["t"], errors="coerce", utc=True
    ).dt.tz_localize(None)
    monthly["t_mpl"] = mdates.date2num(monthly["t"])  # float days since 0001-01-01

    monthly["station"] = stat  # Add station column

    # --- Seasonal cycle (monthly climatology) ---

    # First Temporarily detrend to calculate the monthly signal
    temp = monthly.dropna(subset=["t_mpl", "v_MonthlyMean"])
    coef = np.polyfit(temp["t_mpl"], temp["v_MonthlyMean"], 1)  # slope in m/day
    poly = np.poly1d(coef)
    trend = poly(monthly["t_mpl"])

    data_detrend = monthly.copy()
    data_detrend["v_detrend"] = monthly["v_MonthlyMean"] - trend

    # Calendar month number (1-12) and name
    data_detrend["month"] = data_detrend["t"].dt.month
    data_detrend["month_name"] = data_detrend["t"].dt.month_name()

    # Compute climatology: mean over all years for each calendar month
    climo_mean = data_detrend.groupby("month")["v_detrend"].mean()
    climo_std = data_detrend.groupby("month")["v_detrend"].std()
    climo_n = data_detrend.groupby("month")["v_detrend"].count()
    climo_se = climo_std / np.sqrt(climo_n)

    # Ensure we have all 12 months; if any missing, interpolate across months
    full_months = pd.Index(range(1, 13), name="month")
    climo_mean = climo_mean.reindex(full_months)
    climo_std = climo_std.reindex(full_months)
    climo_n = climo_n.reindex(full_months).fillna(0)
    climo_se = climo_se.reindex(full_months)

    # If some months have no data, fill mean with neighbor interpolation
    if climo_mean.isna().any():
        climo_mean = climo_mean.interpolate(limit_direction="both")
    # For uncertainty, keep NaN when count==0 (no basis), else interpolate SE gently
    if climo_se.isna().any():
        climo_se = climo_se.interpolate(limit_direction="both")

    # Repeat the seasonal cycle across the full monthly index
    monthly["seasonal_cycle"] = data_detrend["month"].map(climo_mean)
    monthly["seasonal_cycle_se"] = data_detrend["month"].map(climo_se)

    # QA: amplitude (peak-to-peak) of the seasonal cycle for this station
    TG_table.at[cnt, "seasonal_amp"] = (
        float(climo_mean.max() - climo_mean.min())
        if climo_mean.notna().any()
        else np.nan
    )

    monthly["anomaly"] = monthly["v_MonthlyMean"] - monthly["seasonal_cycle"]

    # --- Vertical land motion  ---
    # create samples of VLM by sampling from the uncertainty distribution
    # Assumed normal distribution of VLM rate with mean and std from TG_table
    vlm_rate_mm_per_yr = np.random.normal(
        loc=TG_table.loc[TG_table["station"] == stat, "vlm_mm_per_yr"].values[0],
        scale=TG_table.loc[TG_table["station"] == stat, "vlm_unc_mm_per_yr"].values[0],
        size=num_samples,
    )

    # Convert to m/day and negate so positive uplift increases corrected sea level
    vlm_rate_m_per_day = (-1.0) * (vlm_rate_mm_per_yr) / (1000.0 * 365.25)

    # Anchor VLM correction at first valid timestamp
    t0 = monthly["t"].min()
    t_days = (monthly["t"] - t0) / np.timedelta64(1, "D")
    t_days = t_days.astype(float)

    # Create a 2D array of VLM rates (num_samples X time) for uncertainty propagation
    vlm = vlm_rate_m_per_day[:, None] * t_days.to_numpy()[None, :]

    x_all = np.tile(monthly["t_mpl"].to_numpy(), (num_samples, 1))
    y_all = monthly["anomaly"].to_numpy()[None, :] - vlm  # shape (num_samples, time)

    # Fit each member with HAC (Newey-West) to get robust covariance
    hac_lags = 12  # should be chosen based on ACF or domain knowledge
    params_list = []
    cov_list = []
    sigma2_list = []
    nobs_list = []

    for x_m, y_m in zip(x_all, y_all):
        # clean nans
        mask = np.isfinite(x_m) & np.isfinite(y_m)
        x_m = x_m[mask]
        y_m = y_m[mask]

        params, cov, sigma2, nobs = fit_member_ols(x_m, y_m, hac_lags=hac_lags)
        params_list.append(params)  # [const, slope]
        cov_list.append(cov)
        sigma2_list.append(sigma2)
        nobs_list.append(nobs)

    # Pool parameters and uncertainties
    beta_bar, T_cov, W_cov, B_cov = pool_parameters(params_list, cov_list)
    print("Pooled parameter estimates:", beta_bar)
    print("Within-member covariance (W):\n", W_cov)
    print("Between-member covariance (B):\n", B_cov)
    print("Total covariance (T):\n", T_cov)

    # Parameter intervals (e.g., 95%)
    param_ci = parameter_intervals(
        beta_bar, T_cov, num_samples, alpha=0.05, W=W_cov, B=B_cov
    )
    print("\nParameter CIs:\n", param_ci)

    # Prediction at a future time point (e.g., t=220)
    x_new = np.array(monthly["t_mpl"].to_numpy())  # feature vector without intercept
    slr_pred = predict_with_uncertainty(
        x_new.reshape(-1, 1), beta_bar, T_cov, sigma2_list, alpha=0.05
    )

    # Save a couple datasets for plotting
    monthly["slr"] = slr_pred["mean"]
    monthly["slr_lower"] = slr_pred["PI95_lower"]
    monthly["slr_upper"] = slr_pred["PI95_upper"]

    monthly["vlm"] = vlm.mean(axis=0)  # mean VLM correction across samples
    monthly["vlm_lower"] = np.quantile(
        vlm, 0.025, axis=0
    )  # min VLM correction across samples
    monthly["vlm_upper"] = np.quantile(
        vlm, 0.975, axis=0
    )  # max VLM correction across samples

    TG_table.at[cnt, "SLR_rate"] = (
        beta_bar[1] * 365.25 * 1000.0
    )  # Convert slope back to mm/yr for summary table
    TG_table.at[cnt, "SLR_rate_upper"] = param_ci.at[1, "CI95_upper"] * 365.25 * 1000.0
    TG_table.at[cnt, "SLR_rate_lower"] = param_ci.at[1, "CI95_lower"] * 365.25 * 1000.0

    # Store slope (m/day), its SE (m/day), and Rubin df for the slope in TG_table
    TG_table.at[cnt, "SLR_slope_m_per_day"] = float(beta_bar[1])
    TG_table.at[cnt, "SLR_se_m_per_day"] = float(param_ci.at[1, "std_err"])
    TG_table.at[cnt, "SLR_df"] = float(param_ci.at[1, "df"])

    # Anomaly time series  (monthly mean minus the repeated seasonal cycle and vlm)
    monthly["Anomoly_offset"] = (monthly["anomaly"] - monthly["vlm"]).mean()
    monthly["AnomalySLR"] = (
        monthly["anomaly"] - monthly["vlm"] - monthly["Anomoly_offset"]
    )

    # Also lets do a prediction from 1900 to 2026 (just for plotting the SLR trend line)
    future_dates = pd.date_range(start="1900-01-01", end="2026-12-31", freq="MS")
    future_mpl = mdates.date2num(future_dates)
    future_slr_pred = predict_with_uncertainty(
        future_mpl.reshape(-1, 1), beta_bar, T_cov, sigma2_list, alpha=0.05
    )
    future_df = pd.DataFrame(
        {
            "t": future_dates,
            "t_mpl": future_mpl,
            "slr": future_slr_pred["mean"],
            "slr_lower": future_slr_pred["PI95_lower"],
            "slr_upper": future_slr_pred["PI95_upper"],
            "station": stat,
        }
    )

    # Save hourly for quick inspection
    df_out = df.reset_index()
    data_mnth.append(monthly)
    data_hrly.append(df_out)
    data_pred.append(future_df)

# Combine all stations
data_mnth = pd.concat(data_mnth, ignore_index=True)
data_hrly = pd.concat(data_hrly, ignore_index=True)
data_pred = pd.concat(data_pred, ignore_index=True)

# ===============================================================================
# Cleaning: remove obvious outliers in both raw and corrected monthly means
# ===============================================================================
# min_val, max_val = -5.0, 5.0
# for col in ["v_MonthlyMean", "v_MonthlyMean_noVLM"]:
#     if col in data_mnth.columns:
#         data_mnth[col] = data_mnth[col].mask(
#             (data_mnth[col] < min_val) | (data_mnth[col] > max_val), np.nan
#         )

# ===============================================================================
# Statistical testing of slopes
# ===============================================================================

# --- Collect station slopes & SEs (meters/day -> mm/yr) ---
rows = []
for stat in TG_table["station"]:
    row = TG_table.loc[TG_table["station"] == stat]
    beta_mpd = float(row["SLR_slope_m_per_day"].values[0])
    se_mpd = float(row["SLR_se_m_per_day"].values[0])
    df_i = float(row["SLR_df"].values[0])

    rows.append(
        {
            "station": stat,
            "name": name_map.get(stat, stat),
            "beta_m_per_day": beta_mpd,
            "se_m_per_day": se_mpd,
            "df": df_i,
            "beta_mm_per_yr": beta_mpd * 365.25 * 1000.0,
            "se_mm_per_yr": se_mpd * 365.25 * 1000.0,
        }
    )

slr_summary = pd.DataFrame(rows)

# Guard: drop any stations with missing SE
slr_summary = slr_summary.dropna(subset=["beta_mm_per_yr", "se_mm_per_yr"]).reset_index(
    drop=True
)

# --- Heterogeneity diagnostics (FE baseline) ---
mu_fe, var_mu_fe, w_fe = fixed_effect_mean(
    slr_summary["beta_mm_per_yr"].values, slr_summary["se_mm_per_yr"].values
)
Q, I2 = cochrans_Q_I2(
    slr_summary["beta_mm_per_yr"].values,
    slr_summary["se_mm_per_yr"].values,
    mu_fe,
    w_fe,
)

# --- Random-effects pooled mean (recommended if I2 > ~25%) ---
mu_re, var_mu_re, tau2, w_re, Q_re, I2_re = random_effects_mean(
    slr_summary["beta_mm_per_yr"].values, slr_summary["se_mm_per_yr"].values
)

# Choose model for the station-vs-mean tests:
use_random = I2_re > 0.25  # threshold can be tuned

diffs, zvals, pvals, mu_hat, var_mu = test_vs_mean_leave_one_out(
    slr_summary["beta_mm_per_yr"].values,
    slr_summary["se_mm_per_yr"].values,
    use_random_effects=use_random,
)

# Multiple-testing adjustment
p_bh = benjamini_hochberg(pvals)
p_holm = holm_stepdown(pvals)

station_vs_mean = slr_summary.copy()
station_vs_mean["pooled_mean_mm_per_yr"] = mu_hat
station_vs_mean["diff_from_mean_mm_per_yr"] = diffs
station_vs_mean["z_score"] = zvals
station_vs_mean["p_value"] = pvals
station_vs_mean["p_bh"] = p_bh
station_vs_mean["p_holm"] = p_holm
station_vs_mean["sig_bh_0.05"] = station_vs_mean["p_bh"] < 0.05
station_vs_mean["sig_holm_0.05"] = station_vs_mean["p_holm"] < 0.05
station_vs_mean = station_vs_mean.sort_values("p_value")

# Export results (optional)
station_vs_mean.to_csv(
    os.path.join(dir_out, "station_vs_pooled_mean_SLR.csv"), index=False
)

# Quick printout:
print(
    "\nPooled mean SLR ({}-effects): {:.3f} ± {:.3f} mm/yr (95% CI)".format(
        "random" if use_random else "fixed", mu_hat, 1.96 * np.sqrt(var_mu)
    )
)

print("Heterogeneity: Q={:.2f}, I^2={:.1%} (FE baseline)".format(Q, I2))
print("\nStations differing from pooled mean (Holm 0.05):")
for _, row in station_vs_mean[station_vs_mean["sig_holm_0.05"]].iterrows():
    print(
        f"  {row['name']} ({row['station']}): {row['diff_from_mean_mm_per_yr']:.3f} mm/yr"
    )



# Save the SLR 
station_vs_mean.to_csv(os.path.join(dir_out, "SLR_summary.csv"), index=False)

mu_hat 

# ===============================================================================
# Plot: Test VLM for a station (e.g., Neah Bay = 9443090)
# ===============================================================================
stat_plot = "9443090"

pull_hrly = data_hrly[data_hrly["station"] == stat_plot].copy()
pull_mnth = data_mnth[data_mnth["station"] == stat_plot].copy()

fig = plt.figure(figsize=(6, 10))
ax1 = plt.subplot(4, 1, 1)
plt.plot(
    pull_mnth["t"],
    pull_mnth.get("v_MonthlyMean"),
    color="black",
    label="Monthly Mean WL",
)
plt.plot(
    pull_mnth["t"], pull_mnth.get("seasonal_cycle"), color="red", label="Seasonal Cycle"
)

plt.title(
    f"{name_map.get(stat_plot, stat_plot)} TG: Monthly Mean Water Levels and Fitted VLM"
)
plt.legend()
ax1.grid(True)
ax1.set_xticklabels([])

ax2 = plt.subplot(4, 1, 2)
plt.plot(
    pull_mnth["t"],
    pull_mnth.get("anomaly"),
    color="black",
    label="Monthly Mean Anomalies (no Seasonal Cycle)",
)
plt.plot(
    pull_mnth["t"], pull_mnth.get("vlm"), color="red", label="Vertical Land Motion"
)

plt.fill_between(
    pull_mnth["t"],
    pull_mnth.get("vlm_lower"),
    pull_mnth.get("vlm_upper"),
    color="red",
    alpha=0.3,
    label="Vertical Land Motion (95% CI)",
)


plt.title("Monthly Mean anomolies and VLM")
plt.legend()
plt.tight_layout()
ax2.grid(True)
ax2.set_xticklabels([])

ax3 = plt.subplot(4, 1, 3)
plt.plot(
    pull_mnth["t"],
    pull_mnth.get("anomaly") - pull_mnth.get("vlm"),
    color="black",
    label="Absolute Monthly Mean Anomiles (no Seasonal Cycle, no VLM)",
)
plt.plot(
    pull_mnth["t"],
    pull_mnth.get("slr"),
    color="red",
    label="Sea Level Rise (linear fit)",
)

plt.fill_between(
    pull_mnth["t"],
    pull_mnth.get("slr_lower"),
    pull_mnth.get("slr_upper"),
    color="red",
    alpha=0.3,
    label="Sea Level Rise (95% CI)",
)


plt.title("Absolute Monthly Mean Anomalies and SLR")
plt.legend()
plt.tight_layout()
ax3.grid(True)
ax3.set_xticklabels([])

ax4 = plt.subplot(4, 1, 4)
plt.plot(
    pull_mnth["t"],
    pull_mnth.get("anomaly") - pull_mnth.get("vlm") - pull_mnth.get("slr"),
    color="black",
    label="Absolute Monthly Mean Anomiles (no Seasonal Cycle, no VLM)",
)
plt.title("Monthly Mean Anomolies")
plt.tight_layout()
ax4.grid(True)
plt.show()

fig.savefig(os.path.join(dir_out, "MMWL_TS_decomp.png"), dpi=300, bbox_inches="tight")


# Quick diagnostic: total applied VLM over series
if "VLM" in pull_hrly.columns and len(pull_hrly) > 1:
    delta_vlm = pull_hrly["VLM"].to_numpy()[-1] - pull_hrly["VLM"].to_numpy()[0]
    print(f"Total VLM applied over series (m): {delta_vlm:.4f}")

# ===============================================================================
# Plot: SLR for all stations
# ===============================================================================
stations = data_mnth["station"].dropna().unique()
fig = plt.figure(figsize=(10, 6))
colors = plt.cm.tab20(np.linspace(0, 1, len(stations)))

for station, color in zip(stations, colors):
    df_sub = data_mnth[data_mnth["station"] == station].dropna(
        subset=["t", "AnomalySLR"]
    )

    # Raw scatter + line
    plt.scatter(
        df_sub["t"],
        df_sub["AnomalySLR"],
        s=12,
        alpha=0.25,
        color=color,
        edgecolor="none",
        rasterized=True,
    )
    plt.plot(
        df_sub["t"], df_sub["AnomalySLR"], alpha=0.25, color=color, rasterized=True
    )

    # Fit linear trend in meters/day (x in mdates days)

    # plt.plot(
    #     df_sub["t"],
    #     df_sub["slr"] - df_sub["Anomoly_offset"],
    #     color="black",
    #     linewidth=2.5,
    # )
    # plt.plot(
    #     df_sub["t"],
    #     df_sub["slr"] - df_sub["Anomoly_offset"],
    #     color=color,
    #     linewidth=1.5,
    #     label=name_map.get(station, station),
    # )

plt.xlabel("Time")
plt.ylabel("Monthly Mean WL (m)")
plt.title("Water Level Anomolies and SLR signal by Station")
plt.legend(title="Station", ncol=2)
# Optional zoom:
plt.xlim(pd.Timestamp(2000, 1, 1), pd.Timestamp(2010, 1, 1))
plt.ylim([-0.5, 0.5])
plt.tight_layout()
plt.grid(True)
plt.show()
fig.savefig(os.path.join(dir_out, "MMWL_TS_Stacked.png"), dpi=300, bbox_inches="tight")


# ===============================================================================
# Ridgeline plot
# ===============================================================================

fig, ax, meta = ridgeline_timeseries(
    data_mnth,
    time_col="t",
    value_col="AnomalySLR",
    group_col="station",
    order=TG_table['station'],
    scale=None,
    palette="rocket",
    title="Monthly Mean Water Level Anomolies and SLR by Tide Gauge",
    trend=True,  # <-- no trend_col, so it fits OLS per group
    trend_style=dict(color="k", lw=2.0, alpha=0.7, linestyle="dashed"),
    scale_bar_group="9443090",
    scale_bar_value=0.5,
    scale_bar_label="50 cm",
    scale_bar_loc=0.15,  # 10% from the left, precise placement
)

ax.grid(True, which="both", linestyle="--", alpha=0.5)

# Safer mapping of tick labels to station names
yticks = ax.get_yticks()
ax.set_yticklabels([name_map.get(s, s) for s in TG_table['station'][: len(yticks)]])
plt.show()
fig.savefig(os.path.join(dir_out, "MMWL_ridgeline.png"), dpi=300, bbox_inches="tight")

# ===============================================================================
# Subplots of SLR for each station (Seaborn)
# ===============================================================================


fig = plt.figure(figsize=(6, 10))

for cnt, stat in enumerate(TG_table['station']):
    pull_mnth = data_mnth[data_mnth["station"] == stat].copy()
    pull_pred = data_pred[data_pred["station"] == stat].copy()

    ax = plt.subplot(int(TG_table.shape[0] / 2), 2, cnt + 1)
    ax.scatter(
        pull_mnth["t"],
        pull_mnth["anomaly"] - pull_mnth["vlm"],
        s=12,
        alpha=0.25,
        color="black",
        edgecolor="none",
        rasterized=True,
    )

    ax.plot(
        pull_pred["t"],
        pull_pred.get("slr"),
        color="red",
        label="Sea Level Rise (linear fit)",
    )

    ax.fill_between(
        pull_pred["t"],
        pull_pred.get("slr_lower"),
        pull_pred.get("slr_upper"),
        color="red",
        alpha=0.3,
        label="Sea Level Rise (95% CI)",
    )

    ax.set_xlim([pd.Timestamp(1900, 1, 1), pd.Timestamp(2026, 1, 1)])
    ax.set_ylim([-0.5, 0.5])
    ax.set_title(name_map.get(stat, stat))

    ax.text(
        0.04,
        0.95,
        f"SLR: {TG_table.loc[TG_table['station'] == stat, 'SLR_rate'].values[0]:.3g} mm/yr",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
    )

    if cnt != 8 and cnt != 9:  # Only show x-axis labels for the bottom row
        ax.set_xticklabels([])

    if cnt in [1, 3, 5, 7, 9]:
        ax.set_yticklabels([])

    ax.grid(True)

plt.tight_layout()

fig.savefig(os.path.join(dir_out, "TideGauge_SLR_TS.png"), dpi=300, bbox_inches="tight")


fig = plt.figure(figsize=(8, 12))
ax = plt.subplot(int(TG_table.shape[0] / 2), 2, cnt + 1)


for cnt, stat in enumerate(TG_table["station"]):
    pull_mnth = data_mnth[data_mnth["station"] == stat].copy()

    ax.plot(
        [cnt, cnt],
        [
            TG_table.loc[TG_table["station"] == stat, "SLR_rate_lower"].values[0],
            TG_table.loc[TG_table["station"] == stat, "SLR_rate_upper"].values[0],
        ],
        color="black",
        linewidth=2,
    )
    ax.scatter(
        cnt,
        TG_table.loc[TG_table["station"] == stat, "SLR_rate"].values[0],
        marker="o",
        color="red",
    )

ax.plot(
    [-1, len(TG_table["station"])],
    [TG_table["SLR_rate"].mean(), TG_table["SLR_rate"].mean()],
    color="blue",
    linestyle="dashed",
    label="Average SLR Rate",
)

ax.set_ylabel("SLR Rate (mm/yr)")
ax.set_title("Estimated SLR Rates with 95% CI by Station")
ax.set_xticks(range(len(TG_table["station"])))
ax.set_xticklabels([name_map.get(s, s) for s in TG_table["station"]], rotation=45, ha="right")
ax.set_xlim([-0.4, len(TG_table["station"]) - 0.4])
ax.legend()
ax.grid(True)


plt.tight_layout()

fig.savefig(
    os.path.join(dir_out, "TideGauge_SLR_Range.png"), dpi=300, bbox_inches="tight"
)


fig = plt.figure(figsize=(6, 4))
ax = plt.subplot(1, 1, 1)


for cnt, stat in enumerate(TG_table["station"]):
    pull_mnth = data_mnth[data_mnth["station"] == stat].copy()

    num_years = (pull_mnth.iloc[-1]['t'] - pull_mnth.iloc[0]['t']).days/365.25
    ax.plot(
        [num_years, num_years],
        [
            TG_table.loc[TG_table["station"] == stat, "SLR_rate_lower"].values[0],
            TG_table.loc[TG_table["station"] == stat, "SLR_rate_upper"].values[0],
        ],
        color="black",
        linewidth=2,
        zorder=1,
        alpha=0.7,
    )
    ax.scatter(
        num_years,
        TG_table.loc[TG_table["station"] == stat, "SLR_rate"].values[0],
        marker="o",
        color="red",
        zorder=2
    )

ax.set_xlabel("Record Length (years)")
ax.set_ylabel("SLR Rate (mm/yr)")
ax.set_title("Estimated SLR Rates by Record Length")
#ax.set_xlim([-0.4, len(TG_table["station"]) - 0.4])
ax.legend()
ax.grid(True)
ax.set_ylim([-2, 7])

plt.tight_layout()

fig.savefig(
    os.path.join(dir_out, "TideGauge_SLRvsRecordLength.png"), dpi=300, bbox_inches="tight"
)








