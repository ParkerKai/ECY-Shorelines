# -*- coding: utf-8 -*-
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



# ===============================================================================
# User Inputs
# ===============================================================================
dir_in = r"C:\Users\kaip461\OneDrive - Washington State Executive Branch Agencies\Documents\Data\TideGauge"
dir_out = r"C:\Users\kaip461\OneDrive - Washington State Executive Branch Agencies\Documents\Projects\RSLR\figures"
os.makedirs(dir_out, exist_ok=True)

# Monthly aggregation method
agg_method = "mean"  # Options: 'mean', 'lowpass'

stat_plot = "9440910"

# ===============================================================================
# Tide Gauge Info
# ===============================================================================
stat_list = [
    "9440910",
    "9441102",
    "9442396",
    "9443090",
    "9444090",
    "9444900",
    "9449880",
    "9449424",
    "9447130",
    "9446484",
]
stat_names = [
    "TokePoint",
    "Westport",
    "LaPush",
    "NeahBay",
    "PortAngeles",
    "PortTownsend",
    "FridayHarbor",
    "CherryPoint",
    "Seattle",
    "Tacoma",
]

vlm = [
    0.035921,
    1.632204,
    -0.881595,
    2.80886,
    1.075373,
    -0.934305,
    0.400442,
    1.254333,
    -0.648102,
    -1.278814,
]

vlm_uc = [
    0.622934,
    0.841024,
    0.829389,
    0.990746,
    0.53471,
    0.848552,
    0.698111,
    0.589758,
    0.470093,
    0.73966,
]

TG_table = pd.DataFrame(
    {
        "station": stat_list,
        "name": stat_names,
        "vlm_mm_per_yr": vlm,
        "vlm_unc_mm_per_yr": vlm_uc,
    }
)

name_map = dict(zip(TG_table["station"], TG_table["name"]))





# ===============================================================================
# Load & Process
# ===============================================================================
data_mnth = []
data_hrly = []
data_pred = []
TG_table["SLR_rate"] = np.full(
    len(stat_list), np.nan
)  # To store SLR rates for each station
TG_table["SLR_rate_upper"] = np.full(
    len(stat_list), np.nan
)  # To store upper bound of SLR rates for each station
TG_table["SLR_rate_lower"] = np.full(
    len(stat_list), np.nan
)  # To store lower bound of SLR rates for each station

# Add columns to hold slope SE and df (both in m/day and mm/yr for convenience)
for col in ["SLR_slope_m_per_day", "SLR_se_m_per_day", "SLR_df", "SLR_se_mm_per_yr","seasonal_amp"]:
    TG_table[col] = np.nan
    
    
file_in = os.path.join(dir_in, f"NOAA_TG_{stat_plot}", "combined_resolved.parquet")
df = pd.read_parquet(file_in)

# Make sure t is datetime
df["t"] = pd.to_datetime(df["t"])
df = df.sort_values("t").set_index("t")

# ---- Aggregation ----
if agg_method == "mean":
    
        
    # Assume df.index is a DateTimeIndex and df["v_obs"] is your series
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
    # Enforce hourly grid; interpolate gaps for filtering (filtfilt cannot handle NaNs)
    df_hourly = df.asfreq("H")
    v = df_hourly["v_obs_noVLM"].interpolate("time").to_numpy()

    Fs = 1.0  # samples per hour
    cutoff_days = 30
    cutoff_cph = 1.0 / (cutoff_days * 24.0)
    sos = butter(N=6, Wn=cutoff_cph, btype="low", output="sos", fs=Fs)
    v_lp = sosfiltfilt(sos, v)

    df_hourly["v_obs_lowpass_noVLM"] = v_lp

    monthly_lp = (
        df_hourly[["v_obs_lowpass_noVLM"]]
        .resample("MS")
        .nearest()
        .rename(columns={"v_obs_lowpass_noVLM": "v_MonthlyLP_noVLM"})
        .reset_index()
    )
    monthly_mean = (
        df[["v_obs"]]
        .resample("MS")
        .mean()
        .rename(columns={"v_obs": "v_MonthlyMean"})
        .reset_index()
    )
    monthly = pd.merge(monthly_mean, monthly_lp, on="t", how="outer")

else:
    raise ValueError("agg_method must be 'mean' or 'lowpass'.")

# ---- Common time fields ----
monthly["t"] = pd.to_datetime(
    monthly["t"], errors="coerce", utc=True
).dt.tz_localize(None)
monthly["t_mpl"] = mdates.date2num(monthly["t"])  # float days since 0001-01-01

monthly["station"] = stat_plot  # Add station column

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
TG_table[ "seasonal_amp"] = (
    float(climo_mean.max() - climo_mean.min())
    if climo_mean.notna().any()
    else np.nan
)

monthly["anomaly"] = monthly["v_MonthlyMean"] - monthly["seasonal_cycle"]



############### PLOT #############

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 10), sharex=True)
ax1.plot(
    monthly["t"],
    monthly.get("v_MonthlyMean"),
    color="black",
    label="Monthly Mean WL",
)
ax1.plot(
    monthly["t"], monthly.get("seasonal_cycle"), color="red", label="Seasonal Cycle"
)

ax1.set_title(
    f"{name_map.get(stat_plot, stat_plot)} TG: Monthly Mean Water Levels and Fitted VLM"
)
ax1.legend()
ax1.grid(True)
ax1.set_xticklabels([])
#ax1.set_xlim(pd.Timestamp("1980-01-01"), pd.Timestamp("1980-06-01"))


ax2.plot(
    df.index,
    df.get("v_obs"),
    color="black",
    label="Monthly Mean Anomalies (no Seasonal Cycle)",
)
ax2.grid(True)


