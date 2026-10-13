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
import matplotlib
from glob import glob
import matplotlib.pyplot as plt
from matplotlib.dates import AutoDateLocator, ConciseDateFormatter
import seaborn as sns
from scipy.signal import butter, sosfiltfilt  # zero-phase, stable
import matplotlib.dates as mdates
from pathlib import Path
from numpy.random import default_rng
import statsmodels.api as sm


from scipy.stats import t
from typing import List, Optional


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
    x_new: np.ndarray,
    beta_bar: np.ndarray,
    T: np.ndarray,
    sigma2_list: List[float],
    alpha: float = 0.05,
):
    """
    Predict at new design point x_new (features only; we add intercept).
    Returns mean prediction and CI, propagating parameter covariance (T) + average residual variance.
    """
    x_ = np.concatenate(([1.0], x_new))  # add intercept
    mean_pred = x_.dot(beta_bar)
    var_pred_params = x_.T.dot(T).dot(x_)  # parameter uncertainty
    sigma2_bar = float(np.mean(sigma2_list))  # average residual variance across members
    var_pred_total = var_pred_params + sigma2_bar
    se_pred = np.sqrt(var_pred_total)
    # conservative df: minimum of parameter dfs or large fallback
    df = 10_000.0
    tcrit = t.ppf(1 - alpha / 2, df)
    return {
        "mean": mean_pred,
        f"PI{int((1 - alpha) * 100)}_lower": mean_pred - tcrit * se_pred,
        f"PI{int((1 - alpha) * 100)}_upper": mean_pred + tcrit * se_pred,
        "se_pred": se_pred,
    }


# ===============================================================================
# User Inputs
# ===============================================================================
dir_in = r"C:\Users\kaip461\OneDrive - Washington State Executive Branch Agencies\Documents\Data\TideGauge"
dir_out = r"C:\Users\kaip461\OneDrive - Washington State Executive Branch Agencies\Documents\Projects\RSLR\figures"
os.makedirs(dir_out, exist_ok=True)

# Monthly aggregation method
agg_method = "mean"  # Options: 'mean', 'lowpass'

# ===============================================================================
# Tide Gauge Info
# ===============================================================================

TG = [
    {
        "station": "9440910",
        "name": "TokePoint",
        "vlm_mm_per_yr": 0.035921,
        "vlm_unc_mm_per_yr": 0.622934,
    },
    {
        "station": "9441102",
        "name": "Westport",
        "vlm_mm_per_yr": 1.632204,
        "vlm_unc_mm_per_yr": 0.841024,
    },
    {
        "station": "9442396",
        "name": "LaPush",
        "vlm_mm_per_yr": -0.881595,
        "vlm_unc_mm_per_yr": 0.829389,
    },
    {
        "station": "9443090",
        "name": "NeahBay",
        "vlm_mm_per_yr": 2.80886,
        "vlm_unc_mm_per_yr": 0.990746,
    },
    {
        "station": "9444090",
        "name": "PortAngeles",
        "vlm_mm_per_yr": 1.075373,
        "vlm_unc_mm_per_yr": 0.53471,
    },
    {
        "station": "9444900",
        "name": "PortTownsend",
        "vlm_mm_per_yr": -0.934305,
        "vlm_unc_mm_per_yr": 0.848552,
    },
    {
        "station": "9449880",
        "name": "FridayHarbor",
        "vlm_mm_per_yr": 0.400442,
        "vlm_unc_mm_per_yr": 0.698111,
    },
    {
        "station": "9449424",
        "name": "CherryPoint",
        "vlm_mm_per_yr": 1.254333,
        "vlm_unc_mm_per_yr": 0.589758,
    },
    {
        "station": "9447130",
        "name": "Seattle",
        "vlm_mm_per_yr": -0.648102,
        "vlm_unc_mm_per_yr": 0.470093,
    },
    {
        "station": "9446484",
        "name": "Tacoma",
        "vlm_mm_per_yr": -1.278814,
        "vlm_unc_mm_per_yr": 0.73966,
    },
]
TG_table = pd.DataFrame(TG)

name_map = dict(zip(TG_table["station"], TG_table["name"]))

# ===============================================================================
# Load & Process
# ===============================================================================
data_mnth = []
data_hrly = []
TG_table["SLR_rate"] = np.full(
    len(TG_table), np.nan
)  # To store SLR rates for each station
TG_table["RSLR_rate"] = np.full(
    len(TG_table), np.nan
)  # To store RSLR rates for each station
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

        # Keep your original naming + flat frame
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
    # Remove VLM from the observed water levels
    vlm_rate_mm_per_yr = TG_table.loc[
        TG_table["station"] == stat, "vlm_mm_per_yr"
    ].values[0]
    # Convert to m/day and negate so positive uplift increases corrected sea level
    vlm_rate_m_per_day = (-1.0) * (vlm_rate_mm_per_yr) / (1000.0 * 365.25)

    # Anchor VLM correction at first valid timestamp
    t0 = monthly["t"].min()
    t_days = (monthly["t"] - t0) / np.timedelta64(1, "D")
    t_days = t_days.astype(float)

    monthly["vlm"] = vlm_rate_m_per_day * t_days

    # --- SLR (linear fit on timeseries minus seasonal and VLM)  ---
    temp = pd.DataFrame(
        {"v_clean": monthly["anomaly"] - monthly["vlm"], "t_mpl": monthly["t_mpl"]}
    ).dropna(subset=["t_mpl", "v_clean"])
    if len(temp) >= 2:
        coef = np.polyfit(temp["t_mpl"], temp["v_clean"], 1)  # slope in m/day
        poly = np.poly1d(coef)
        monthly["slr"] = poly(monthly["t_mpl"])
        TG_table.at[cnt, "SLR_rate"] = coef[0] * (1000.0 * 365.25)  # convert to mm/yr

    else:
        monthly["slr"] = np.nan
        TG_table.at[cnt, "SLR_rate"] = np.nan

    temp = pd.DataFrame(
        {"v_clean": monthly["anomaly"], "t_mpl": monthly["t_mpl"]}
    ).dropna(subset=["t_mpl", "v_clean"])

    if len(temp) >= 2:
        coef = np.polyfit(temp["t_mpl"], temp["v_clean"], 1)  # slope in m/day
        poly = np.poly1d(coef)
        monthly["rslr"] = poly(monthly["t_mpl"])
        TG_table.at[cnt, "RSLR_rate"] = coef[0] * (1000.0 * 365.25)  # convert to mm/yr
    else:
        monthly["rslr"] = np.nan
        TG_table.at[cnt, "RSLR_rate"] = np.nan

    temp = pd.DataFrame(
        {"v_clean": monthly["v_MonthlyMean"], "t_mpl": monthly["t_mpl"]}
    ).dropna(subset=["t_mpl", "v_clean"])

    if len(temp) >= 2:
        coef = np.polyfit(temp["t_mpl"], temp["v_clean"], 1)  # slope in m/day
        poly = np.poly1d(coef)
        monthly["rslr_o"] = poly(monthly["t_mpl"])
        TG_table.at[cnt, "RSLR_Obs_rate"] = coef[0] * (
            1000.0 * 365.25
        )  # convert to mm/yr
    else:
        monthly["rslr_o"] = np.nan
        TG_table.at[cnt, "RSLR_Obs_rate"] = np.nan

    # Save a couple datasets for plotting

    # Anomaly time series  (monthly mean minus the repeated seasonal cycle and vlm)
    monthly["Anomoly_offset"] = (monthly["anomaly"] - monthly["vlm"]).mean()
    monthly["AnomalySLR"] = (
        monthly["anomaly"] - monthly["vlm"] - monthly["Anomoly_offset"]
    )

    # Save hourly for quick inspection
    df_out = df.reset_index()
    data_mnth.append(monthly)
    data_hrly.append(df_out)

# Combine all stations
data_mnth = pd.concat(data_mnth, ignore_index=True)
data_hrly = pd.concat(data_hrly, ignore_index=True)


# ===============================================================================
# Cleaning: remove obvious outliers in both raw and corrected monthly means
# ===============================================================================
min_val, max_val = -5.0, 5.0
for col in ["v_MonthlyMean", "v_MonthlyMean_noVLM"]:
    if col in data_mnth.columns:
        data_mnth[col] = data_mnth[col].mask(
            (data_mnth[col] < min_val) | (data_mnth[col] > max_val), np.nan
        )

# ===============================================================================
# Plot: Test VLM for a station (e.g., Neah Bay = 9443090)
# ===============================================================================
pull_hrly = data_hrly[data_hrly["station"] == "9443090"].copy()
pull_mnth = data_mnth[data_mnth["station"] == "9443090"].copy()

plt.figure(figsize=(6, 10))
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
    f"{name_map.get('9443090', '9443090')} TG: Monthly Mean Water Levels and Fitted VLM"
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


# Quick diagnostic: total applied VLM over series
if "VLM" in pull_hrly.columns and len(pull_hrly) > 1:
    delta_vlm = pull_hrly["VLM"].to_numpy()[-1] - pull_hrly["VLM"].to_numpy()[0]
    print(f"Total VLM applied over series (m): {delta_vlm:.4f}")

# ===============================================================================
# Plot: SLR for all stations
# ===============================================================================
stations = data_mnth["station"].dropna().unique()
plt.figure(figsize=(10, 6))
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

    plt.plot(
        df_sub["t"],
        df_sub["slr"] - df_sub["Anomoly_offset"],
        color="black",
        linewidth=2.5,
    )
    plt.plot(
        df_sub["t"],
        df_sub["slr"] - df_sub["Anomoly_offset"],
        color=color,
        linewidth=1.5,
        label=name_map.get(station, station),
    )

plt.xlabel("Time")
plt.ylabel("Monthly Mean WL (m)")
plt.title("Water Level Anomolies and SLR signal by Station")
plt.legend(title="Station", ncol=2)
# Optional zoom:
# plt.xlim(pd.Timestamp(1940,1,1), pd.Timestamp(1950,1,1))
plt.tight_layout()
plt.grid(True)
plt.show()


# ===============================================================================
# Ridgeline plot
# ===============================================================================

fig, ax, meta = ridgeline_timeseries(
    data_mnth,
    time_col="t",
    value_col="AnomalySLR",
    group_col="station",
    order=stat_list,
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
ax.set_yticklabels([name_map.get(s, s) for s in stat_list[: len(yticks)]])
plt.show()
fig.savefig(os.path.join(dir_out, "MMWL_ridgeline.png"), dpi=300, bbox_inches="tight")

# ===============================================================================
# Subplots of SLR for each station (Seaborn)
# ===============================================================================
sns.set_theme(style="whitegrid")

g = sns.lmplot(
    data=data_mnth,
    x="t_mpl",
    y="AnomalySLR",
    row="station",
    row_order=stations,
    height=2.0,
    aspect=2.0,
    scatter_kws={
        "alpha": 0.25,
        "color": "black",
        "s": 12,
        "edgecolor": "none",
        "rasterized": True,
    },
    line_kws={"color": "C0", "lw": 1.5},
    ci=95,
    truncate=False,
    facet_kws={"sharex": True, "sharey": True, "margin_titles": False},
)

# Global axis limits
tvals = data_mnth["t_mpl"].to_numpy()
mask_x = np.isfinite(tvals)
x_min, x_max = tvals[mask_x].min(), tvals[mask_x].max()

yvals = pd.to_numeric(data_mnth["AnomalySLR"], errors="coerce").to_numpy()
mask_y = np.isfinite(yvals)
y_min, y_max = yvals[mask_y].min(), yvals[mask_y].max()
if np.isfinite(y_min) and np.isfinite(y_max):
    pad = 0.02 * (y_max - y_min)
    y_min -= pad
    y_max += pad

locator = mdates.AutoDateLocator()
formatter = mdates.ConciseDateFormatter(locator)

for i, ax in enumerate(g.axes.flat):
    if i >= len(stations):
        continue
    station = stations[i]
    sub = data_mnth[data_mnth["station"] == station]

    x = sub["t_mpl"].to_numpy()
    y = pd.to_numeric(sub["AnomalySLR"], errors="coerce").to_numpy()

    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() >= 2:
        slope_m_per_day, intercept = np.polyfit(x[mask], y[mask], 1)
        slope_mm_per_year = slope_m_per_day * (1000.0 * 365.25)
        slope_text = f"SLR: {slope_mm_per_year:.3g} mm/yr"
    else:
        slope_text = "SLR: NA"

    ax.set_title(name_map.get(station, station), fontsize=10, pad=3)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.set_xlabel("")

    ax.text(
        0.02,
        0.95,
        slope_text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
    )

g.set_ylabels("MMWL (m)")
plt.tight_layout()
fig = g.figure
fig.savefig(os.path.join(dir_out, "MMWL_RSLR.png"), dpi=300, bbox_inches="tight")
