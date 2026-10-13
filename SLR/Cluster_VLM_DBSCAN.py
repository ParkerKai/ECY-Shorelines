"""Cluster VLM point data with DBSCAN using space, value and uncertainty as features.

Input CSV columns (produced by an earlier VLM clustering/aggregation step and
carried through unchanged by this script):

``lon``, ``lat``
    Point location in decimal degrees, WGS84 (EPSG:4326).
``value``
    Vertical land motion rate at the point (positive = uplift), mm/yr.
``unc``
    Reported 1-sigma uncertainty on that point's ``value``, mm/yr.
``cluster_labels``
    Cluster id from the *upstream* clustering run. This is not the labelling
    produced here; see ``dbscan_label`` below.
``mean_val``
    Mean ``value`` of all points in the upstream cluster, mm/yr. Repeated on
    every row of that cluster.
``MC_std``
    Monte-Carlo uncertainty of the cluster mean, i.e. the spread obtained by
    resampling each point within its own ``unc``. Measurement-error term.
``spatial_std``
    Standard deviation of the ``value``s of the points within the cluster,
    i.e. how variable the VLM signal is across the cluster footprint. Spatial
    variability (representativeness) term.
``combined_std``
    Total 1-sigma uncertainty of the cluster mean: the two independent terms
    added in quadrature, ``sqrt(MC_std**2 + spatial_std**2)``. This is the
    value to use as the uncertainty on ``mean_val``.
``number_of_values_in_cluster``
    Point count in the upstream cluster.

Columns added by this script:

``geometry``
    Shapely ``Point`` built from ``lon``/``lat``; CRS EPSG:4326 until
    reprojected.
``dbscan_label``
    Cluster id from the DBSCAN run in :func:`cluster`. Integers ``0..n-1`` for
    clustered points, ``-1`` for noise (points with fewer than ``min_samples``
    neighbours within ``eps`` in the scaled feature space). Labels are
    arbitrary identifiers with no ordering meaning.

Note that the ``*_std`` columns describe the *upstream* clusters, so they are
only consistent with ``cluster_labels``, not with ``dbscan_label``. Recompute
them per DBSCAN cluster if you need equivalent statistics for this run.
"""

import os

import folium
import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xyzservices
from matplotlib.colors import ListedColormap, to_rgba_array
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN
from dotenv import load_dotenv

load_dotenv()

CSV_PATH = (
    r"C:\Users\kaip461\OneDrive - Washington State Executive Branch Agencies"
    r"\Documents\Data\VLM\cluster_vals_full_0.5-1-10.csv"
)

HTML_OUT = "vlm_clusters.html"
CSV_OUT = "vlm_clusters.csv"

_STADIA_ATTR = (
    '&copy; <a href="https://stadiamaps.com/">Stadia Maps</a> '
    '&copy; <a href="https://stamen.com/">Stamen Design</a> '
    '&copy; <a href="https://openmaptiles.org/">OpenMapTiles</a> '
    '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
)

# Tilesets folium has no built-in entry for, so the URL and attribution must be
# supplied explicitly. Stamen moved to Stadia Maps in 2023 and now needs a key.
TILESETS = {
    "stamen watercolor": (
        "https://tiles.stadiamaps.com/tiles/stamen_watercolor/{z}/{x}/{y}.jpg",
        _STADIA_ATTR,
        True,
    ),
    "stamen toner": (
        "https://tiles.stadiamaps.com/tiles/stamen_toner/{z}/{x}/{y}.png",
        _STADIA_ATTR,
        True,
    ),
    "stamen terrain": (
        "https://tiles.stadiamaps.com/tiles/stamen_terrain/{z}/{x}/{y}.png",
        _STADIA_ATTR,
        True,
    ),
    "esri imagery": (
        (
            "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery"
            "/MapServer/tile/{z}/{y}/{x}"
        ),
        (
            "Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, "
            "GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community"
        ),
        False,
    ),
}


def resolve_tiles(tiles):
    """Map a tileset name to a ``(url_or_name, attribution)`` pair for folium.

    Names folium already knows are passed straight through with no attribution
    (folium supplies its own). Entries in :data:`TILESETS` are expanded to an
    explicit URL template, and Stadia-hosted Stamen styles get the API key from
    the ``STADIA_API_KEY`` environment variable appended.

    CARTO now meters its basemaps, so if ``CARTO_API_KEY`` is set the CartoDB
    styles are built with it; without it they still work but folium warns.
    """
    key = tiles.strip().lower()

    if key.startswith("cartodb"):
        carto_key = os.environ.get("CARTO_API_KEY")
        if not carto_key:
            return tiles, None
        provider = xyzservices.providers.query_name(tiles)
        return (
            provider.build_url(apikey=carto_key, fill_subdomain=False),
            provider.html_attribution,
        )

    if key not in TILESETS:
        return tiles, None

    url, attr, needs_key = TILESETS[key]
    if needs_key:
        api_key = os.environ.get("STADIA_API_KEY")
        if not api_key:
            raise RuntimeError(
                f"'{tiles}' is hosted by Stadia Maps and requires an API key. "
                "Get a free one at https://client.stadiamaps.com/ and set it in "
                "the environment as STADIA_API_KEY (do not hardcode it)."
            )
        url = f"{url}?api_key={api_key}"
    return url, attr


# tab20 as hex, recycled across clusters.
PALETTE = [
    "#1f77b4",
    "#aec7e8",
    "#ff7f0e",
    "#ffbb78",
    "#2ca02c",
    "#98df8a",
    "#d62728",
    "#ff9896",
    "#9467bd",
    "#c5b0d5",
    "#8c564b",
    "#c49c94",
    "#e377c2",
    "#f7b6d2",
    "#7f7f7f",
    "#c7c7c7",
    "#bcbd22",
    "#dbdb8d",
    "#17becf",
    "#9edae5",
]

# Projected CRS in metres so geographic separation is a true distance.
PROJ_CRS = "EPSG:32610"  # UTM zone 10N (US West Coast)

# Per-feature tolerances: points are "close" if they differ by less than these
# amounts. Each feature is divided by its scale so eps = 1.0 is isotropic.
SPATIAL_SCALE = 40000.0  # metres
VALUE_SCALE = 0.25  # mm/yr (same units as `value`)
UNC_SCALE = 0.25  # mm/yr (same units as `unc`)

MIN_SAMPLES = 20


def load_points(csv_path=CSV_PATH):
    """Read the VLM csv into a GeoDataFrame of lon/lat points.

    Parameters
    ----------
    csv_path : str
        Path to the VLM csv. Must contain ``lon``, ``lat``, ``value`` and
        ``unc``.

    Returns
    -------
    geopandas.GeoDataFrame
        The four selected input columns plus a ``geometry`` column of points in
        EPSG:4326.
    """
    df = pd.read_csv(csv_path, usecols=["lon", "lat", "value", "unc"])
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["lon"], df["lat"]),
        crs="EPSG:4326",
    )
    return gdf


def build_features(
    gdf,
    spatial_scale=SPATIAL_SCALE,
    value_scale=VALUE_SCALE,
    unc_scale=UNC_SCALE,
    proj_crs=PROJ_CRS,
):
    """Return the scaled ``[x, y, value, unc]`` feature matrix used by DBSCAN.

    Coordinates are reprojected to a metric CRS so that spatial separation is a
    true distance, then every feature is divided by its own tolerance so all
    four are dimensionless.

    Caution: the resulting metric is the Euclidean *sum in quadrature* of the
    four scaled differences, not an "AND" of per-feature tolerances. A pair of
    points can exceed one tolerance and still fall inside ``eps`` if the other
    differences are small. The scales therefore set the relative *weight* of
    each feature, not a hard limit on any of them. With the current constants
    the two spatial features supply ~96% of the squared distance to a typical
    point's 10 nearest neighbours, so clustering is driven almost entirely by
    geography; reduce ``spatial_scale`` relative to the others (or increase
    ``value_scale``/``unc_scale``) to give ``value`` and ``unc`` more say.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Output of :func:`load_points`.
    spatial_scale : float
        Distance tolerance in metres.
    value_scale, unc_scale : float
        Tolerances on ``value`` and ``unc``, in mm/yr.
    proj_crs : str
        Metric CRS used for the x/y features.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(len(gdf), 4)``, rows in the same order as ``gdf``.
    """
    projected = gdf.to_crs(proj_crs)
    features = np.column_stack(
        [
            projected.geometry.x.to_numpy() / spatial_scale,
            projected.geometry.y.to_numpy() / spatial_scale,
            gdf["value"].to_numpy() / value_scale,
            gdf["unc"].to_numpy() / unc_scale,
        ]
    )
    return features


def cluster(gdf, eps=1.0, min_samples=MIN_SAMPLES, **feature_kwargs):
    """Run DBSCAN on the scaled features and label every point.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Output of :func:`load_points`.
    eps : float
        Neighbourhood radius in the *scaled* feature space (see
        :func:`build_features`); 1.0 corresponds to the configured tolerances.
    min_samples : int
        Minimum neighbours (including the point itself) for a point to be a
        core point. Points not reachable from any core point become noise.
    **feature_kwargs
        Passed through to :func:`build_features` (``spatial_scale``,
        ``value_scale``, ``unc_scale``, ``proj_crs``).

    Returns
    -------
    geopandas.GeoDataFrame
        A copy of ``gdf`` with an added integer ``dbscan_label`` column:
        ``0..n-1`` for clustered points and ``-1`` for noise.
    """
    features = build_features(gdf, **feature_kwargs)
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(features)
    gdf = gdf.copy()
    gdf["dbscan_label"] = labels
    return gdf


def plot_clusters(gdf, label_col="dbscan_label", proj_crs=PROJ_CRS, ax=None):
    """Map the clustered points: noise in grey, each cluster a distinct colour.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Output of :func:`cluster`.
    label_col : str
        Column holding the cluster ids; ``-1`` is treated as noise.
    proj_crs : str
        Metric CRS the points are drawn in, so the map has equal aspect.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new figure is created when omitted.

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the map.
    """
    projected = gdf.to_crs(proj_crs)
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 10))

    noise = projected[projected[label_col] == -1]
    clustered = projected[projected[label_col] != -1]

    if not noise.empty:
        noise.plot(ax=ax, color="0.7", markersize=4, label=f"noise (n={len(noise)})")

    if not clustered.empty:
        labels = np.sort(clustered[label_col].unique())
        # tab20 has only 20 entries, so colours are deliberately recycled when
        # there are more clusters than that; adjacent clusters may repeat.
        colors = plt.cm.tab20(np.arange(len(labels)) % 20)
        # Map labels onto contiguous indices so the colormap covers every cluster.
        codes = np.searchsorted(labels, clustered[label_col].to_numpy())
        clustered.plot(
            ax=ax,
            column=codes,
            cmap=ListedColormap(colors),
            markersize=8,
            legend=False,
        )

    ax.set_aspect("equal")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.grid()
    ax.set_title(f"DBSCAN clusters (n={clustered[label_col].nunique()})")
    if not noise.empty:
        ax.legend(loc="best", fontsize="small")
    return ax


def plot_cluster_boxplot(
    gdf,
    column,
    label_col="dbscan_label",
    include_noise=False,
    ax=None,
):
    """Boxplot of ``column`` for each DBSCAN cluster, ordered by label.

    Shows the spread of a per-point quantity *within* each cluster, i.e. a
    visual equivalent of ``spatial_std`` (for ``value``) or of the spread of
    the reported measurement errors (for ``unc``).

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Output of :func:`cluster`.
    column : str
        Per-point column to summarise, e.g. ``"value"`` or ``"unc"``.
    label_col : str
        Column holding the cluster ids.
    include_noise : bool
        When True, the noise group (``-1``) is plotted as its own box.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new figure is created when omitted.

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the boxplot. Outliers are hidden, so whiskers span
        1.5x the interquartile range at most.
    """
    data = gdf if include_noise else gdf[gdf[label_col] != -1]
    labels = np.sort(data[label_col].unique())
    groups = [data.loc[data[label_col] == lab, column].to_numpy() for lab in labels]

    if ax is None:
        width = max(8.0, 0.35 * len(labels))
        _, ax = plt.subplots(figsize=(width, 5))

    ax.boxplot(groups, tick_labels=[str(lab) for lab in labels], showfliers=False)
    ax.set_xlabel("Cluster label")
    ax.set_ylabel(column)
    ax.set_title(
        f"Within-cluster distribution of '{column}' (n={len(labels)} clusters)"
    )
    ax.tick_params(axis="x", labelrotation=90, labelsize="small")
    ax.grid(axis="y", alpha=0.3)
    return ax


def cluster_spread_stats(gdf, columns=("value", "unc"), label_col="dbscan_label"):
    """Per-cluster internal spread of each column: range, std and IQR.

    One row per cluster (noise excluded). ``*_range`` is ``max - min``, the
    worst-case disagreement between two points in that cluster; ``*_std`` is
    the sample standard deviation (ddof=1, matching ``spatial_std``).

    Returns
    -------
    pandas.DataFrame
        Indexed by cluster label, with an ``n`` column plus
        ``<col>_range``, ``<col>_std`` and ``<col>_iqr`` for each column.
    """
    grouped = gdf[gdf[label_col] != -1].groupby(label_col)
    out = pd.DataFrame({"n": grouped.size()})
    for col in columns:
        s = grouped[col]
        out[f"{col}_range"] = s.max() - s.min()
        out[f"{col}_std"] = s.std(ddof=1)
        out[f"{col}_iqr"] = s.quantile(0.75) - s.quantile(0.25)
    return out


def plot_cluster_spread(gdf, columns=("value", "unc"), label_col="dbscan_label"):
    """Summarise how internally consistent the clusters are, one panel per column.

    Each panel boxplots the distribution *across clusters* of three within-cluster
    spread metrics, so a tight box means the clusters are internally consistent.
    Fliers are shown because the worst clusters are the point of the figure, and
    the mean and max of each metric are annotated.

    Returns
    -------
    (matplotlib.figure.Figure, pandas.DataFrame)
        The figure and the per-cluster stats from :func:`cluster_spread_stats`.
    """
    stats = cluster_spread_stats(gdf, columns=columns, label_col=label_col)
    metrics = ["range", "std", "iqr"]

    fig, axes = plt.subplots(1, len(columns), figsize=(6 * len(columns), 5.5))
    axes = np.atleast_1d(axes)

    for ax, col in zip(axes, columns):
        # std is NaN for single-point clusters; drop per-metric rather than per-row.
        data = [stats[f"{col}_{m}"].dropna().to_numpy() for m in metrics]
        ax.boxplot(data, tick_labels=[m.upper() if m == "iqr" else m for m in metrics])
        ax.set_ylabel(f"within-cluster {col} spread (mm/yr)")
        ax.set_title(f"'{col}' consistency across {len(stats)} clusters")
        ax.grid(axis="y", alpha=0.3)

        lines = [
            f"{m:>5}: mean {d.mean():.3f}   max {d.max():.3f}"
            for m, d in zip(metrics, data)
        ]
        ax.text(
            0.02,
            0.98,
            "\n".join(lines),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize="small",
            family="monospace",
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "0.7"},
        )

    fig.tight_layout()
    return fig, stats


def map_clusters_interactive(
    gdf,
    label_col="dbscan_label",
    out_html=HTML_OUT,
    radius=2.5,
    show_noise=True,
    tiles="CartoDB positron",
):
    """Write a zoomable slippy-map of the clusters to an HTML file.

    Uses folium/Leaflet rather than matplotlib, so panning and zooming happen in
    the browser and a real coastline basemap sits underneath -- which is the
    point when checking how clusters behave in complex coastal geometry. Noise
    is drawn as small grey dots on a separate toggleable layer.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Output of :func:`cluster`, in any CRS (reprojected to WGS84 internally).
    out_html : str
        Path of the HTML file to write.
    radius : float
        Marker radius in screen pixels; markers stay this size as you zoom.
    show_noise : bool
        Include the noise (``-1``) layer.
    tiles : str
        A folium built-in name (``"OpenStreetMap"``, ``"CartoDB positron"``,
        ``"CartoDB dark_matter"``) or a key of :data:`TILESETS` such as
        ``"stamen watercolor"`` or ``"esri imagery"``. Stamen styles are served
        by Stadia Maps and need ``STADIA_API_KEY`` set in the environment.

    Returns
    -------
    folium.Map
    """
    wgs = gdf.to_crs("EPSG:4326")
    lat = wgs.geometry.y.to_numpy()
    lon = wgs.geometry.x.to_numpy()
    labels = wgs[label_col].to_numpy()
    value = wgs["value"].to_numpy()
    unc = wgs["unc"].to_numpy()

    tile_url, tile_attr = resolve_tiles(tiles)

    # preferCanvas keeps tens of thousands of markers responsive; the default
    # SVG renderer creates one DOM node per point and stalls the browser.
    fmap = folium.Map(
        location=[float(lat.mean()), float(lon.mean())],
        zoom_start=7,
        tiles=tile_url,
        attr=tile_attr,
        prefer_canvas=True,
    )

    clustered_layer = folium.FeatureGroup(name="clusters", show=True)
    noise_layer = folium.FeatureGroup(name="noise", show=False)

    uniq = np.unique(labels[labels != -1])
    color_of = {lab: PALETTE[i % len(PALETTE)] for i, lab in enumerate(uniq)}

    def feature(i):
        lab = int(labels[i])
        return {
            "type": "Feature",
            # 5 dp is ~1 m; full precision just bloats the HTML.
            "geometry": {
                "type": "Point",
                "coordinates": [round(float(lon[i]), 5), round(float(lat[i]), 5)],
            },
            "properties": {
                "cluster": lab,
                "value": round(float(value[i]), 3),
                "unc": round(float(unc[i]), 3),
                "c": "#999999" if lab == -1 else color_of[lab],
            },
        }

    def add_layer(idx, layer, opacity):
        if len(idx) == 0:
            return
        folium.GeoJson(
            {"type": "FeatureCollection", "features": [feature(i) for i in idx]},
            marker=folium.CircleMarker(radius=radius, weight=0, fill=True),
            style_function=lambda f, o=opacity: {
                "fillColor": f["properties"]["c"],
                "fillOpacity": o,
                "weight": 0,
            },
            tooltip=folium.GeoJsonTooltip(fields=["cluster", "value", "unc"]),
        ).add_to(layer)

    add_layer(np.flatnonzero(labels != -1), clustered_layer, 0.85)
    if show_noise:
        add_layer(np.flatnonzero(labels == -1), noise_layer, 0.35)

    clustered_layer.add_to(fmap)
    if show_noise:
        noise_layer.add_to(fmap)
    folium.LayerControl(collapsed=False).add_to(fmap)

    fmap.fit_bounds([[lat.min(), lon.min()], [lat.max(), lon.max()]])
    fmap.save(out_html)
    return fmap


def plot_clusters_interactive(
    gdf,
    label_col="dbscan_label",
    proj_crs=PROJ_CRS,
    show_noise=True,
    pick_radius_px=8,
    markersize=10,
    ax=None,
):
    """Zoomable cluster map with hover tooltips, for ipykernel notebooks.

    Run ``%matplotlib widget`` (ipympl) or ``%matplotlib qt`` in the notebook
    first; the default ``inline`` backend renders a static PNG and cannot pan,
    zoom or respond to the mouse.

    Interactions
    ------------
    hover
        Tooltip showing the nearest point's cluster id, ``value`` and ``unc``.
    click a point
        Isolate that cluster; every other point fades to 5% alpha.
    click empty space, or the same cluster again
        Clear the isolation.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Output of :func:`cluster`.
    pick_radius_px : float
        Hit-test radius in screen pixels. Converted to data units on every
        event, so it stays constant as you zoom into dense coastal areas.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if "inline" in matplotlib.get_backend().lower():
        raise RuntimeError(
            "The inline backend is static. Run `%matplotlib widget` (needs "
            "ipympl) or `%matplotlib qt`, then call this function again."
        )

    projected = gdf.to_crs(proj_crs)
    if not show_noise:
        projected = projected[projected[label_col] != -1]

    x = projected.geometry.x.to_numpy()
    y = projected.geometry.y.to_numpy()
    labels = projected[label_col].to_numpy()
    value = projected["value"].to_numpy()
    unc = projected["unc"].to_numpy()

    uniq = np.unique(labels[labels != -1])
    color_of = {lab: PALETTE[i % len(PALETTE)] for i, lab in enumerate(uniq)}
    base = to_rgba_array(["#bbbbbb" if lab == -1 else color_of[lab] for lab in labels])
    base[labels == -1, 3] = 0.35

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 10))
    fig = ax.figure

    scat = ax.scatter(x, y, s=markersize, c=base, linewidths=0)
    ax.set_aspect("equal")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_title(f"{len(uniq)} clusters - hover to inspect, click to isolate")
    ax.grid(alpha=0.3)

    tree = cKDTree(np.column_stack([x, y]))
    annot = ax.annotate(
        "",
        xy=(0, 0),
        xytext=(12, 12),
        textcoords="offset points",
        bbox={"boxstyle": "round", "fc": "lightyellow", "ec": "0.5"},
        arrowprops={"arrowstyle": "->"},
        fontsize="small",
        family="monospace",
    )
    annot.set_visible(False)
    selected = {"label": None}

    def nearest(event):
        inv = ax.transData.inverted()
        (x0, _), (x1, _) = inv.transform([(0, 0), (pick_radius_px, 0)])
        dist, idx = tree.query(
            [event.xdata, event.ydata], distance_upper_bound=abs(x1 - x0)
        )
        return None if not np.isfinite(dist) else int(idx)

    def on_move(event):
        if event.inaxes is not ax or event.xdata is None:
            return
        i = nearest(event)
        if i is None:
            if annot.get_visible():
                annot.set_visible(False)
                fig.canvas.draw_idle()
            return
        annot.xy = (x[i], y[i])
        annot.set_text(
            f"cluster {labels[i]}\nvalue   {value[i]:+.3f}\nunc     {unc[i]:.3f}"
        )
        annot.set_visible(True)
        fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes is not ax or event.xdata is None:
            return
        i = nearest(event)
        lab = None if i is None else int(labels[i])
        if lab is None or lab == selected["label"]:
            selected["label"] = None
            scat.set_facecolors(base)
        else:
            selected["label"] = lab
            dimmed = base.copy()
            dimmed[:, 3] = 0.05
            dimmed[labels == lab] = base[labels == lab]
            scat.set_facecolors(dimmed)
        fig.canvas.draw_idle()

    # Keep the connection ids alive on the figure so the callbacks are not
    # garbage collected while the widget is still on screen.
    fig._vlm_cids = [
        fig.canvas.mpl_connect("motion_notify_event", on_move),
        fig.canvas.mpl_connect("button_press_event", on_click),
    ]
    return ax


if __name__ == "__main__":
    gdf = load_points()
    gdf = cluster(gdf, eps=0.5)
    gdf.drop(columns="geometry").to_csv(CSV_OUT, index=False)
    print(f"Clustered data written to {CSV_OUT}")

    n_clusters = gdf.loc[gdf["dbscan_label"] != -1, "dbscan_label"].nunique()
    n_noise = int((gdf["dbscan_label"] == -1).sum())
    print(f"{len(gdf)} points -> {n_clusters} clusters, {n_noise} noise points")
    print(gdf.groupby("dbscan_label")["value"].agg(["count", "mean", "std"]))

    map_clusters_interactive(gdf)
    print(f"\nInteractive map written to {HTML_OUT}")

    _, spread = plot_cluster_spread(gdf)
    print("\nWithin-cluster spread summary (across clusters):")
    print(spread.drop(columns="n").agg(["mean", "median", "max"]).T)

    plot_clusters(gdf)
    plt.tight_layout()

    plot_cluster_boxplot(gdf, "value")
    plt.tight_layout()

    plot_cluster_boxplot(gdf, "unc")
    plt.tight_layout()

    plt.show()
