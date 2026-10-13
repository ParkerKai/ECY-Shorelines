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
from scipy import stats

from itertools import cycle
import seaborn as sns


import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from itertools import cycle


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

# Optional dependencies for trend lines
try:
    import statsmodels.api as sm
    HAS_STATSMODELS = True
except Exception:
    HAS_STATSMODELS = False

def plot_residuals_vs_mean_by_station(
    wide: pd.DataFrame,
    mean_ts: pd.Series | None = None,
    dir_out: str = ".",
    filename: str = "Residuals_vs_Mean_by_Station.png",
    col_wrap: int | None = None,
    height: float = 2.8,
    sharex: bool = True,
    sharey: bool = True,
    trend: str | None = "lowess",  # {"lowess", "linear", None}
    lowess_frac: float = 0.3,      # smoothing fraction if LOWESS used
    point_size: float = 12,
    point_alpha: float = 0.6,
    max_points_per_station: int | None = None,  # e.g., 5000 to subsample if huge
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    robust_limits: bool = True,    # use 1st–99th percentile for shared limits
    theme: str = "whitegrid",
    context: str = "talk",
):
    """
    Plot residuals (station - mean_at_time) vs the mean (x-axis), one subplot per station.

    Parameters
    ----------
    wide : DataFrame
        Columns are station IDs, index is DatetimeIndex (monthly/regular).
    mean_ts : Series, optional
        Mean across stations at each time. If None, computed from 'wide'.
    dir_out : str
        Output directory to save the figure.
    filename : str
        Output file name (PNG).
    col_wrap : int, optional
        Number of columns before wrapping to a new row. If None, autocomputed.
    height : float
        Height of each facet in inches.
    sharex, sharey : bool
        Whether to share x or y limits across facets.
    trend : {"lowess", "linear", None}
        Trend line to add within each facet. LOWESS requires statsmodels.
    lowess_frac : float
        Smoothing fraction for LOWESS (0–1).
    point_size : float
        Scatter marker size.
    point_alpha : float
        Scatter alpha.
    max_points_per_station : int, optional
        If set, randomly subsample points per station to this limit.
    xlim, ylim : tuple, optional
        Explicit axis limits (shared across facets if sharex/sharey True).
    robust_limits : bool
        If True and limits not given, use 1st–99th percentiles for x and y.
    theme, context : str
        Seaborn theme/context.

    Returns
    -------
    g : seaborn.axisgrid.FacetGrid
        The facet grid object (for further customization if needed).
    """
    assert isinstance(wide.index, pd.DatetimeIndex), "wide.index must be a DatetimeIndex"
    wide = wide.sort_index()

    # Compute mean_ts if not provided
    if mean_ts is None:
        mean_ts = wide.mean(axis=1, skipna=True)

    # Residuals per station
    residuals = wide.sub(mean_ts, axis=0)

    # Build tidy table: mean, residual, station (keep time if you want downstream filtering)
    tidy = (
        residuals
        .assign(mean=mean_ts)
        .reset_index()  # time column becomes the index column name (often 'index')
        .rename(columns={"index": "time"})
        .melt(id_vars=["time", "mean"], var_name="station", value_name="residual")
    )

    # Drop NaNs
    tidy = tidy.dropna(subset=["mean", "residual"])

    # Optional subsample per station if many points
    if max_points_per_station is not None:
        # groupby station and sample
        tidy = (
            tidy.groupby("station", group_keys=False)
            .apply(lambda df: df.sample(n=min(len(df), max_points_per_station), random_state=42))
        )

    # Seaborn theme
    sns.set_theme(style=theme, context=context)

    # Determine col_wrap automatically if not provided
    n_stations = tidy["station"].nunique()
    if col_wrap is None:
        # Aim for ~square-ish layout but keep it readable
        col_wrap = max(3, min(6, int(np.ceil(np.sqrt(n_stations)))))

    # Create FacetGrid
    g = sns.FacetGrid(
        tidy, col="station", col_wrap=col_wrap, sharex=sharex, sharey=sharey,
        height=height, margin_titles=True
    )
    g.map_dataframe(
        sns.scatterplot, x="mean", y="residual",
        s=point_size, alpha=point_alpha, edgecolor=None, linewidth=0
    )

    # Add zero reference lines and (optional) trend line per facet
    def _decorate_axis(data, color, **kwargs):
        ax = plt.gca()
        # Zero lines
        ax.axhline(0, color="0.25", lw=1.0, ls="--", alpha=0.7)
        ax.axvline(0, color="0.25", lw=1.0, ls="--", alpha=0.7)

        # Trend lines
        if trend == "lowess" and HAS_STATSMODELS:
            x = data["mean"].values
            y = data["residual"].values
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() >= 10:
                smoothed = sm.nonparametric.lowess(y[mask], x[mask], frac=lowess_frac, it=0)
                ax.plot(smoothed[:, 0], smoothed[:, 1], color=color, lw=1.6, alpha=0.9, label="LOWESS")
        elif trend == "linear":
            x = data["mean"].values
            y = data["residual"].values
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() >= 2:
                # Simple robust fit using np.polyfit; no CI drawn
                m, b = np.polyfit(x[mask], y[mask], 1)
                xs = np.linspace(np.nanmin(x[mask]), np.nanmax(x[mask]), 100)
                ax.plot(xs, m * xs + b, color=color, lw=1.6, alpha=0.9, label=f"y={m:.2f}x+{b:.2f}")

        # Make y ticks integers when range is small
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune=None))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4, prune=None))

    g.map_dataframe(_decorate_axis)

    # Set common labels and title
    g.set_axis_labels("Mean WL anomaly (m)", "Residual (m)")
    g.fig.suptitle("Residuals vs Mean (per Station)", y=1.02, fontsize=14)

    # Shared limits (robust) if requested and not explicitly provided
    if (sharex and xlim is None) or (sharey and ylim is None):
        xvals = tidy["mean"].values
        yvals = tidy["residual"].values

        if sharex and xlim is None:
            if robust_limits:
                x_low, x_high = np.nanpercentile(xvals, [1, 99])
            else:
                x_low, x_high = np.nanmin(xvals), np.nanmax(xvals)
            # pad a bit
            dx = (x_high - x_low) or 1.0
            xlim = (x_low - 0.05 * dx, x_high + 0.05 * dx)

        if sharey and ylim is None:
            if robust_limits:
                y_low, y_high = np.nanpercentile(yvals, [1, 99])
            else:
                y_low, y_high = np.nanmin(yvals), np.nanmax(yvals)
            dy = (y_high - y_low) or 1.0
            ylim = (y_low - 0.05 * dy, y_high + 0.05 * dy)

    # Apply the limits across facets
    if xlim is not None or ylim is not None:
        for ax in g.axes.flatten():
            if xlim is not None:
                ax.set_xlim(xlim)
            if ylim is not None:
                ax.set_ylim(ylim)

    # Tidy layout
    g.fig.tight_layout()

    # Save and return
    os.makedirs(dir_out, exist_ok=True)
    out_path = os.path.join(dir_out, filename)
    g.fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure to: {out_path}")

    return g


def pairplot_residuals(
    residuals: pd.DataFrame,
    dir_out: str = ".",
    filename: str = "Residuals_Pairplot.png",
    diag_kind: str | None = "kde",   # "kde", "hist", or None
    point_size: float = 12,
    point_alpha: float = 0.5,
):
    """
    Full scatter-matrix of residuals across stations.
    """
    sns.set_theme(style="whitegrid", context="talk")

    g = sns.pairplot(
        residuals,
        corner=True,                # lower triangle only
        diag_kind=diag_kind,
        plot_kws=dict(s=point_size, alpha=point_alpha, edgecolor=None, linewidth=0)
    )
    g.fig.suptitle("Residuals Across Stations (Pairwise Scatter)", y=1.02, fontsize=14)
    g.fig.tight_layout()
    out_path = os.path.join(dir_out, filename)
    g.fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_path}")


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


num_samples = 10000  # number of VLM samples per station for uncertainty propagation


# ===============================================================================
# Load & Process
# ===============================================================================

# Load regional SLR
station_vs_mean = pd.read_csv(os.path.join(dir_out, "station_vs_pooled_mean_SLR.csv"))
SLR_pooled = station_vs_mean["pooled_mean_mm_per_yr"].iloc[0]

data_mnth = []
data_hrly = []

# Add columns to hold slope SE and df (both in m/day and mm/yr for convenience)
for col in ["VLM_mm_per_year_found",
            "seasonal_amp",]:
    TG_table[col] = np.nan



for cnt, stat in enumerate(TG_table["station"]):
    print(f"Processing station {stat} ({cnt + 1}/{len(TG_table['station'])})...")

    file_in = os.path.join(dir_in, f"NOAA_TG_{stat}", "combined_resolved.parquet")
    df = pd.read_parquet(file_in)

    # Make sure t is datetime
    df["t"] = pd.to_datetime(df["t"])
    df = df.sort_values("t").set_index("t")

    # ---- Aggregation ----
    s = df["v_obs"]

    # Monthly mean and count (non-NaN)
    monthly_mean = s.resample("MS").mean()
    monthly_count = s.resample("MS").count()

    # Rule: require at least this many samples per month
    MIN_SAMPLES = 400  # A little over half a month for hourly sampled data

    # Mask insufficiently sampled months to NaN
    monthly_mean_masked = monthly_mean.mask(monthly_count < MIN_SAMPLES)

    monthly = monthly_mean_masked.rename("v_MonthlyMean").reset_index()
    
    
        
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


    # --- SLR ---
    # Convert to m/day and negate so positive uplift increases corrected sea level
    slr_rate_m_per_day = (SLR_pooled) / (1000.0 * 365.25) # SLR_pooled in mm/yr

    # Anchor VLM correction at first valid timestamp
    t0 = monthly["t"].min()
    t_days = (monthly["t"] - t0) / np.timedelta64(1, "D")
    t_days = t_days.astype(float)

    # Create a slr rate fit
    monthly["slr"] = slr_rate_m_per_day * t_days.to_numpy()    
    
    
    # --- Vertical land motion  ---
    # First Temporarily detrend to calculate the monthly signal
    temp = monthly.dropna(subset=["t_mpl", "anomaly",'slr'])
    
    coef = np.polyfit(temp["t_mpl"], temp['anomaly'] - temp['slr'], 1)  # slope in m/day
    poly = np.poly1d(coef)
    monthly["vlm"] = poly(monthly["t_mpl"])
    
    TG_table.at[cnt, "VLM_mm_per_year_found"] = float(coef[0] * 1000 * 365.25*-1)

    
    
    monthly["AnomalySLR"] = (
        monthly["anomaly"] - monthly["slr"] - monthly["vlm"])
    

    # Save hourly for quick inspection
    df_out = df.reset_index()
    data_mnth.append(monthly)
    data_hrly.append(df_out)

# Combine all stations
data_mnth = pd.concat(data_mnth, ignore_index=True)
data_hrly = pd.concat(data_hrly, ignore_index=True)


# ===============================================================================
# Plot: SLR for all stations
# ===============================================================================
stations = data_mnth["station"].dropna().unique()
colors = plt.cm.tab20(np.linspace(0, 1, len(stations)))


# Ensure datetime and sorted
data_mnth["t"] = pd.to_datetime(data_mnth["t"])

# Color mapping (stable color per station)
color_cycle = cycle(colors)
color_map = {st: c for st, c in zip(stations, color_cycle)}

# Choose decades to display (edit to your needs)
# Example: 1980s, 1990s, 2000s, 2010s
decades = [(1980, 1990), (1990, 2000), (2000, 2010), (2010, 2020)]
# Alternatively: last four decades present in data (see pattern B below)

fig, axes = plt.subplots(4, 1, figsize=(7, 10), sharex=False, sharey=True)

# --- Plot the same data on every axis ---
for ax in axes:
    for station in stations:
        df_sub = (
            data_mnth.loc[data_mnth["station"] == station]
            .dropna(subset=["t", "AnomalySLR"])
            .sort_values("t")
        )
        color = color_map.get(station, "C0")
        # line (with legend) + scatter (no legend)
        ax.plot(df_sub["t"], df_sub["AnomalySLR"], color=color, linewidth=1.2,
                rasterized=True, label=station)
        ax.scatter(df_sub["t"], df_sub["AnomalySLR"], s=12, color=color,
                   edgecolor="none", rasterized=True, label="_nolegend_")

# --- Per-axis decade windows + styling ---
for ax, (start_year, end_year) in zip(axes, decades):
    ax.set_xlim(pd.Timestamp(start_year, 1, 1), pd.Timestamp(end_year, 1, 1))
    ax.set_title(f"{start_year}–{end_year-1}", loc="left", fontsize=10)
    ax.set_ylabel("WL anomaly (m)")
    ax.set_ylim(-0.4, 0.4)
    ax.grid(True, alpha=0.3)
    # Optional: nicer ticks
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

axes[-1].set_xlabel("Time")

# One legend for all panels
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, title="Station", ncol=2,
           loc="upper right", bbox_to_anchor=(0.5, 1.02), fontsize=8)

fig.suptitle("Water Level Anomalies by Station (decade panels)", y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.98])
plt.show()
fig.savefig(os.path.join(dir_out, "MMWL_TS_decades.png"), dpi=300, bbox_inches="tight")


# ===============================================================================
# Subplots of SLR for each station (Seaborn)
# ===============================================================================

sns.set_theme(style="whitegrid", context="talk")

# Config
value_col   = "AnomalySLR"  # metric to plot
station_col = "station"
time_col    = "t"           # ensure datetime for optional filtering

# Ensure datetime and clean data
data_mnth[time_col] = pd.to_datetime(data_mnth[time_col], errors="coerce")

df = (
    data_mnth
    .dropna(subset=[station_col, value_col])
    .sort_values([station_col, time_col])
)

# Optional: limit to a time window (decade, etc.)
# df = df[(df[time_col] >= "2000-01-01") & (df[time_col] < "2010-01-01")]

# Build the figure
fig = plt.figure(figsize=(12, 6))
ax = sns.violinplot(
    data=df,
    x=station_col, y=value_col,
    bw_adjust=0.7,   # bandwidth tweak; smaller = less smoothing
    cut=0,           # don’t extend beyond the data range
    inner="quartile",# draw median & IQR inside the violin
    linewidth=1,
    palette="Set3"   # categorical palette; change if you like
)

# Optional: overlay raw points to show sampling density
sns.stripplot(
    data=df,
    x=station_col, y=value_col,
    color="k", alpha=0.35, size=2, jitter=0.15,
    dodge=False
)

ax.set_ylabel("Monthly WL anomaly (m)")
ax.set_xlabel("Station")
ax.set_title("Distribution of Monthly WL Anomalies by Station")

# Make station labels readable if you have many
ax.tick_params(axis="x", labelrotation=30, labelsize=9)

# Optional y-range
# ax.set(ylim=(-0.4, 0.4))

sns.despine(left=True, bottom=True)
plt.tight_layout()
plt.show()


fig.savefig(os.path.join(dir_out, "MMWLA_ViolinPlot.png"), dpi=300, bbox_inches="tight")

# ===============================================================================
# Subplots of SLR for each station (Seaborn)
# ===============================================================================


# --- CONFIG: set your column names here ---
time_col    = "t"
station_col = "station"
value_col   = "AnomalySLR"

# --- Prep: tidy and pivot to wide (time index, station columns) ---
df = (
    data_mnth[[time_col, station_col, value_col]]
    .dropna(subset=[time_col, station_col, value_col])
    .copy()
)
df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
df = df.dropna(subset=[time_col]).sort_values(time_col)

wide = (
    df.pivot(index=time_col, columns=station_col, values=value_col)
      .sort_index()
)

# --- Mean and 95% CI across stations for each timestamp ---
mean_ts = wide.mean(axis=1, skipna=True)
std_ts  = wide.std(axis=1, ddof=1)           # sample std
n_ts    = wide.count(axis=1)                 # number of stations contributing

# Standard error: NaN when n==0 (and std_ts is NaN when n<=1)
sem_ts  = std_ts / np.sqrt(n_ts.where(n_ts > 0, np.nan))

# Row-specific t critical values (df = n-1); NaN where n<=1
alpha = 0.05  # 95% CI
dfree = (n_ts - 1).where(n_ts > 1, np.nan)
tcrit = stats.t.ppf(1 - alpha/2, dfree)

# Margin and CI
margin = tcrit * sem_ts
lower  = mean_ts - margin
upper  = mean_ts + margin


# --- Residuals: station value minus mean at the same time ---
residuals = wide.sub(mean_ts, axis=0)        # broadcasts mean over columns

# --- Plot ---
sns.set_theme(style="whitegrid", context="talk")
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 8))

# Top axis: mean + shaded 95% CI
ax1.plot(mean_ts.index, mean_ts.values, color="C0", lw=1.8,
         label="Mean across stations")
ax1.fill_between(mean_ts.index, lower.values, upper.values,
                 color="C0", alpha=0.25, label="95% CI")
ax1.set_title("Mean Monthly WL Anomaly across Stations (±95% CI)")
ax1.set_ylabel("Monthly WL anomaly (m)")
ax1.grid(True, alpha=0.3)
ax1.legend(loc="upper left", frameon=False)
ax1.set_xlim(pd.Timestamp("1940-01-01"), pd.Timestamp("2025-01-01"))
# Bottom axis: residuals per station
# Use your existing `colors` list if available; otherwise a palette

base_colors = sns.color_palette("tab20", n_colors=max(20, residuals.shape[1]))
color_map = {st: col for st, col in zip(residuals.columns, cycle(base_colors))}

for st in residuals.columns:
    y = residuals[st]
    if y.notna().any():
        ax2.plot(y.index, y.values, color=color_map[st], lw=1.0, alpha=0.9, label=st)

ax2.axhline(0.0, color="0.2", lw=1.0, ls="--", alpha=0.6)
ax2.set_title("Residuals (Station − Mean) by Time")
ax2.set_ylabel("Residual (m)")
ax2.set_xlabel("Time")
ax2.grid(True, alpha=0.3)

# Legend handling (large number of stations → place outside)
handles, labels = ax2.get_legend_handles_labels()
if len(labels) <= 20:
    ax2.legend(title="Station", ncol=2, fontsize=8, frameon=False)
else:
    fig.legend(handles, labels, title="Station", ncol=4,
               loc="upper center", bbox_to_anchor=(0.5, 1.02), fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.98])


ax2.set_xlim(pd.Timestamp("2000-01-01"), pd.Timestamp("2010-01-01"))
ax2.set_ylim(-0.1, 0.1)
fig.tight_layout()
plt.show()
fig.savefig(os.path.join(dir_out, "Mean_Monthly_WL_Residuals.png"), dpi=300, bbox_inches="tight")



# ===============================================================================
# Residual plot (seaborn)
# ===============================================================================

mean_ts = mean_ts.rename_axis("time")
wide = wide.rename_axis("time")


# Assuming you already have `wide` (monthly WL anomalies) and optional `mean_ts`
g = plot_residuals_vs_mean_by_station(
    wide=wide,
    mean_ts=mean_ts,              # or omit to compute internally
    dir_out=dir_out,
    filename="Residuals_vs_Mean_by_Station.png",
    col_wrap=4,                   # adjust columns per row (auto if None)
    trend="lowess",               # "lowess", "linear", or None
    lowess_frac=0.25,             # smoother amount
    point_size=10,
    point_alpha=0.6,
    max_points_per_station=None,  # e.g., 5000 if very dense
    sharex=True,
    sharey=True,
    robust_limits=True
)


pairplot_residuals(residuals, dir_out=dir_out)

#########################################################


# import pandas as pd
# import numpy as np
# from statsmodels.tsa.stattools import ccf
# from statsmodels.graphics.tsaplots import plot_ccf
# import matplotlib.pyplot as plt



# # 1) Long -> wide
# def to_wide(df_long):
#     """
#     Convert long-format ('station', 't', 'v_MonthlyMean') to wide timeseries DataFrame.
#     Rows = timestamps (t), columns = stations, values = v_MonthlyMean.
#     Handles duplicates at a given station,t by averaging.
#     """
#     # Ensure time is datetime and sorted
#     df = df_long.copy()
#     df["t"] = pd.to_datetime(df["t"])
#     df.sort_values("t", inplace=True)

#     # If duplicates exist per station,t, aggregate (mean); adjust if you prefer 'last' or 'sum'
#     df_agg = (
#         df.groupby(["t", "station"], as_index=False)["v_MonthlyMean"]
#           .mean()
#     )
#     df_wide = df_agg.pivot(index="t", columns="station", values="v_MonthlyMean")
#     df_wide = df_wide.sort_index()
#     return df_wide


# # 2) CCF for both positive and negative lags built from statsmodels.ccf
# def ccf_both_lags(series_x, series_y, max_lag=40, adjusted=True, fft=True):
#     """
#     Compute cross-correlation for lags -max_lag..+max_lag using statsmodels ccf.

#     Consistent definition:
#       Corr at lag k is Corr(X_{t+k}, Y_t) for k >= 0 (statsmodels' forward side).
#       Negative lags are obtained by reversing both series and then reversing the output.

#     Returns: lags (np.array), corr (np.array)
#     """
#     # Pairwise align & drop NaNs so both series have identical non-missing timestamps
#     aligned = pd.concat([series_x, series_y], axis=1).dropna()
#     x = aligned.iloc[:, 0].to_numpy()
#     y = aligned.iloc[:, 1].to_numpy()

#     # Forward lags: 0..max_lag
#     # Note: ccf returns only non-negative lags; set nlags=max_lag+1 to include lag 0..max_lag
#     forward = ccf(x, y, adjusted=adjusted, fft=fft, nlags=max_lag + 1)

#     # Backward lags: -max_lag..-1 via reverse trick (then reverse output)
#     # See statsmodels behavior and community guidance.  [1](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.ccf.html)[4](https://stackoverflow.com/questions/63491991/how-to-use-the-ccf-method-in-the-statsmodels-library)
#     backward = ccf(x[::-1], y[::-1], adjusted=adjusted, fft=fft, nlags=max_lag + 1)[::-1]

#     # Combine: drop duplicated lag 0 from one side
#     corr = np.concatenate([backward[:-1], forward])
#     lags = np.arange(-max_lag, max_lag + 1)
#     return lags, corr


# # 3) Pairwise CCF across all stations using the same definition above
# def pairwise_ccf_all(df_long, max_lag=40, adjusted=True, fft=True):
#     """
#     Compute lagged correlations (-max_lag..+max_lag) for all station pairs.
#     Returns:
#         results[station_i][station_j] = {
#             'lags': np.array,
#             'correlations': np.array,
#             'best_lag': int,
#             'best_corr': float
#         }
#     """
#     df_wide = to_wide(df_long)
#     stations = df_wide.columns
#     results = {}

#     for s1 in stations:
#         results[s1] = {}
#         for s2 in stations:
#             if s1 == s2:
#                 continue
#             lags, corr = ccf_both_lags(
#                 df_wide[s1], df_wide[s2],
#                 max_lag=max_lag, adjusted=adjusted, fft=fft
#             )
#             best_idx = np.argmax(np.abs(corr))
#             results[s1][s2] = {
#                 "lags": lags,
#                 "correlations": corr,
#                 "best_lag": int(lags[best_idx]),
#                 "best_corr": float(corr[best_idx]),
#             }
#     return results


# # 4) Plot with statsmodels.plot_ccf in a way that matches pairwise_ccf_all
# def plot_station_ccf(df_long, station_x, station_y, max_lag=40,
#                      adjusted=True, fft=True, overlay_computed=True):
#     """
#     Plot CCF between two stations using statsmodels' plot_ccf with negative lags.
#     To ensure consistency, use the same 'adjusted' and 'fft' options and the same alignment.
#     Optionally overlay the numerically computed points from ccf_both_lags.
#     """
#     x = df_wide[station_x]
#     y = df_wide[station_y]
#     aligned = pd.concat([x, y], axis=1).dropna()
#     x = aligned.iloc[:, 0]
#     y = aligned.iloc[:, 1]

#     fig, ax = plt.subplots(figsize=(10, 6))

#     # Important: ask plot_ccf to show negative lags, and keep the parameters consistent
#     # plot_ccf computes Corr(x_t, y_{t-k}) for displayed lags k; with negative_lags=True it
#     # includes the negative side on the axis.  [3](https://www.statsmodels.org/stable/generated/statsmodels.graphics.tsaplots.plot_ccf.html)
#     plot_ccf(x, y, ax=ax,
#              lags=max_lag,           # integer lag window
#              negative_lags=True,     # include -lags..+lags on the axis
#              adjusted=adjusted,
#              fft=fft)

#     ax.set_title(f"CCF: {station_x} vs {station_y} (±{max_lag} lags)")
#     ax.set_xlabel("Lag (k)")
#     ax.set_ylabel("Correlation")

#     # Optional: overlay the explicitly computed symmetric CCF to visually confirm they match
#     if overlay_computed:
#         lags, corr = ccf_both_lags(x, y, max_lag=max_lag, adjusted=adjusted, fft=fft)
#         ax.plot(lags, corr, 'o', color='tab:orange', alpha=0.7, label='Computed (ccf_both_lags)')
#         ax.legend()

#     plt.tight_layout()
#     plt.show()



# # Example sanity check between compute and plot:
# t= residuals.index
# residuals = residuals.reset_index() 
# residuals['t'] = t
# df_wide = residuals
# x, y = df_wide["9443090"], df_wide["9443090"]
# lags, corr = ccf_both_lags(x, y, max_lag=40)

# # Confirm location of peak lag
# best = lags[np.argmax(np.abs(corr))]
# print("Best lag:", best, "corr:", corr[lags == best][0])

# # Plot and visually confirm orange markers sit on top of the blue stems
# plot_station_ccf(df_wide, "9440910", "9443090", max_lag=40, overlay_computed=True)


