import numpy as np
import pandas as pd

import geopandas as gpd

from shapely.geometry import Point
from shapely.geometry import LineString

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm

import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ==========================================================
# SCALE BAR (Cartopy native)
# ==========================================================

def add_scale_bar(ax, length_km=50, location=(0.08, 0.08)):
    """
    Add a simple scale bar to a Cartopy map.

    Parameters
    ----------
    ax : cartopy axis
    length_km : float
        Scale bar length in km
    location : tuple
        Position in axes coordinates
    """

    # Current map extent
    x0, x1, y0, y1 = ax.get_extent(
        crs=ccrs.PlateCarree()
    )

    # Latitude at bottom of map
    lat = y0 + 0.05*(y1-y0)

    # Convert km to degrees longitude
    # approximation valid for regional maps
    deg_lon = length_km / (
        111.32 * np.cos(np.deg2rad(lat))
    )


    x_start = x0 + location[0]*(x1-x0)
    x_end = x_start + deg_lon

    y = y0 + location[1]*(y1-y0)


    # Draw bar
    ax.plot(
        [x_start, x_end],
        [y, y],
        transform=ccrs.PlateCarree(),
        color="black",
        linewidth=3,
        solid_capstyle="butt",
        zorder=10
    )


    # End ticks
    ax.plot(
        [x_start, x_start],
        [y-0.01*(y1-y0), y+0.01*(y1-y0)],
        transform=ccrs.PlateCarree(),
        color="black",
        linewidth=2,
        zorder=10
    )


    ax.plot(
        [x_end, x_end],
        [y-0.01*(y1-y0), y+0.01*(y1-y0)],
        transform=ccrs.PlateCarree(),
        color="black",
        linewidth=2,
        zorder=10
    )


    # Label
    ax.text(
        (x_start+x_end)/2,
        y+0.025*(y1-y0),
        f"{length_km} km",
        transform=ccrs.PlateCarree(),
        ha="center",
        va="bottom",
        fontsize=9,
        zorder=10
    )


# ============================================================
# USER INPUTS
# ============================================================

# Produced previously
DELTA_M0_FILE = "Delta_m0_percent_by_transect.csv"

# CSV containing all transect points
TRANSECT_POINTS_FILE = "../swash_landes/transects_points_with_depth.csv"

# Run-up differences
RUNUP_FILE = "../swash_landes/swash_cases/runup_comparison.csv"

# Outputs
OUTPUT_TRANSECTS = "transects_runup_q95.geojson"
OUTPUT_SPECTRAL = "spectral_points_delta.geojson"

# ------------------------------------------------------------

TRANSECT_ID = "transect_id"

TRANSECT_ID_run ="Transect_ID"

TRANSECT_LON = "longitude"
TRANSECT_LAT = "latitude"

RUNUP_DIFF = "Difference"

# ============================================================
# LOAD FILES
# ============================================================

print("Loading Δm0 dataframe...")

delta = pd.read_csv(DELTA_M0_FILE)

print("Loading transects...")

transects = pd.read_csv(TRANSECT_POINTS_FILE)

print("Loading run-up...")

runup = pd.read_csv(RUNUP_FILE)
runup = runup.rename(columns={"Transect_ID":"transect_id"})
# ============================================================
# COMPUTE Q95 OF RUNUP DIFFERENCE
# ============================================================

print("Computing Q95...")

runup_q95 = (
    runup
    .groupby(TRANSECT_ID)[RUNUP_DIFF]
    .quantile(0.95)
    .reset_index()
)

runup_q95.rename(
    columns={
        RUNUP_DIFF: "Runup_Q95"
    },
    inplace=True
)

print(runup_q95.head())

# ============================================================
# BUILD TRANSECT LINES
# ============================================================

print("Building LineStrings...")

lines = []

for tid, group in transects.groupby(TRANSECT_ID):

    group = group.copy()

    group = group.sort_index()

    coords = list(
        zip(
            group[TRANSECT_LON],
            group[TRANSECT_LAT]
        )
    )

    if len(coords) < 2:
        continue

    line = LineString(coords)

    lines.append(
        {
            TRANSECT_ID: tid,
            "geometry": line
        }
    )

transects_gdf = gpd.GeoDataFrame(
    lines,
    crs="EPSG:4326"
)

print(
    f"{len(transects_gdf)} transects created."
)

# ============================================================
# JOIN Q95
# ============================================================

transects_gdf = transects_gdf.merge(
    runup_q95,
    on=TRANSECT_ID,
    how="left"
)

# ============================================================
# BUILD SPECTRAL POINTS
# ============================================================
points_gdf = transects_gdf.copy()

# Replace each LineString geometry with its first point
points_gdf["geometry"] = points_gdf.geometry.apply(
    lambda line: Point(line.coords[-1])
)

# Ensure it is still a GeoDataFrame
points_gdf = gpd.GeoDataFrame(
    points_gdf,
    geometry="geometry",
    crs=transects_gdf.crs
)
#geometry = gpd.points_from_xy(
#    delta["Spectrum_Lon"],
#    delta["Spectrum_Lat"]
#)

spectral_gdf = gpd.GeoDataFrame(
    delta,
    geometry=points_gdf.geometry,#geometry,
    crs="EPSG:4326"
)

# ============================================================
# CHECK
# ============================================================

print()

print(spectral_gdf.head())

print()

print(transects_gdf.head())

# ============================================================
# SAVE
# ============================================================

spectral_gdf.to_file(
    OUTPUT_SPECTRAL,
    driver="GeoJSON"
)

transects_gdf.to_file(
    OUTPUT_TRANSECTS,
    driver="GeoJSON"
)

print()

print("Saved:")

print(" ", OUTPUT_SPECTRAL)

print(" ", OUTPUT_TRANSECTS)



# ==========================================================
# USER INPUTS
# ==========================================================

SPECTRAL_FILE = "spectral_points_delta.geojson"
TRANSECTS_FILE = "transects_runup_q95.geojson"

OUTPUT_FIG = "Delta_m0_Runup_maps.png"


# ==========================================================
# LOAD DATA
# ==========================================================

print("Loading data...")

spectral = gpd.read_file(SPECTRAL_FILE)

transects = gpd.read_file(TRANSECTS_FILE)


# ==========================================================
# COLOR LIMITS
# ==========================================================

# Delta m0 %

vmax_energy = np.nanpercentile(
    np.abs(
        spectral["Delta_m0_percent"]
    ),
    98
)

norm_energy = colors.TwoSlopeNorm(
    vmin=-vmax_energy,
    vcenter=0,
    vmax=vmax_energy
)


# Runup

vmax_runup = np.nanpercentile(
    np.abs(
        transects["Runup_Q95"]
    ),
    98
)

norm_runup = colors.TwoSlopeNorm(
    vmin=-vmax_runup,
    vcenter=0,
    vmax=vmax_runup
)


cmap = plt.cm.RdBu_r


# ==========================================================
# MAP EXTENT
# ==========================================================

bounds = spectral.total_bounds

margin = 0.3

extent = [
    bounds[0]-margin,
    bounds[2]+margin,
    bounds[1]-margin,
    bounds[3]+margin
]


# ==========================================================
# FIGURE
# ==========================================================

fig = plt.figure(
    figsize=(14,7)
)


projection = ccrs.PlateCarree()


ax1 = fig.add_subplot(
    1,
    2,
    1,
    projection=projection
)

ax2 = fig.add_subplot(
    1,
    2,
    2,
    projection=projection
)



axes = [ax1, ax2]


for ax in axes:

    ax.set_extent(
        extent,
        crs=projection
    )

    ax.add_feature(
        cfeature.LAND,
        facecolor="lightgray",
        zorder=0
    )

    ax.add_feature(
        cfeature.OCEAN,
        facecolor="white",
        zorder=0
    )

    ax.add_feature(
        cfeature.COASTLINE,
        linewidth=0.8,
        zorder=3
    )

    ax.gridlines(
        draw_labels=True,
        linewidth=0.3,
        alpha=0.5
    )


# ==========================================================
# PANEL A
# ==========================================================

print("Plotting Δm0...")

sc = ax1.scatter(
    spectral.geometry.x,
    spectral.geometry.y,

    c=spectral["Delta_m0_percent"],

    cmap=cmap,

    norm=norm_energy,

    s=15,

    edgecolor="none",

    transform=projection,

    zorder=5
)


ax1.set_title(
    "(a) Relative wave energy change",
    fontsize=13
)


# ==========================================================
# PANEL B
# ==========================================================

print("Plotting runup...")

transects.plot(
    ax=ax2,

    column="Runup_Q95",

    cmap=cmap,

    norm=norm_runup,

    linewidth=2.5,

    transform=projection,

    zorder=5,
    vmin=-1.5,
    vmax=1.5
)


ax2.set_title(
    "(b) 95th percentile wave run-up difference",
    fontsize=13
)

                                                                                        
