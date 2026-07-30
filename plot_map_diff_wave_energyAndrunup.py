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

from matplotlib_scalebar.scalebar import ScaleBar

# ============================================================
# USER INPUTS
# ============================================================

# Produced previously
DELTA_M0_FILE = "Delta_m0_percent_by_transect.parquet"

# CSV containing all transect points
TRANSECT_POINTS_FILE = "transect_points.csv"

# Run-up differences
RUNUP_FILE = "Runup_difference.csv"

# Outputs
OUTPUT_TRANSECTS = "transects_runup_q95.geojson"
OUTPUT_SPECTRAL = "spectral_points_delta.geojson"

# ------------------------------------------------------------

TRANSECT_ID = "Transect_ID"

TRANSECT_LON = "Longitude"
TRANSECT_LAT = "Latitude"

RUNUP_DIFF = "Difference"

# ============================================================
# LOAD FILES
# ============================================================

print("Loading Δm0 dataframe...")

delta = pd.read_parquet(DELTA_M0_FILE)

print("Loading transects...")

transects = pd.read_csv(TRANSECT_POINTS_FILE)

print("Loading run-up...")

runup = pd.read_csv(RUNUP_FILE)

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

geometry = gpd.points_from_xy(
    delta["Spectrum_Lon"],
    delta["Spectrum_Lat"]
)

spectral_gdf = gpd.GeoDataFrame(
    delta,
    geometry=geometry,
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

    zorder=5
)


ax2.set_title(
    "(b) 95th percentile wave run-up difference",
    fontsize=13
)


# ==========================================================
# COLORBARS
# ==========================================================


cbar1 = fig.colorbar(
    cm.ScalarMappable(
        norm=norm_energy,
        cmap=cmap
    ),

    ax=ax1,

    orientation="horizontal",

    pad=0.05,

    fraction=0.05
)


cbar1.set_label(
    r"$\Delta m_0$ (%)"
)



cbar2 = fig.colorbar(
    cm.ScalarMappable(
        norm=norm_runup,
        cmap=cmap
    ),

    ax=ax2,

    orientation="horizontal",

    pad=0.05,

    fraction=0.05
)


cbar2.set_label(
    r"$\Delta R_{95}$ (m)"
)


# ==========================================================
# NORTH ARROWS
# ==========================================================

def add_north_arrow(ax):

    ax.annotate(
        "N",

        xy=(0.93,0.85),

        xycoords="axes fraction",

        fontsize=14,

        ha="center",

        va="center",

        fontweight="bold"
    )

    ax.arrow(
        0.93,
        0.75,

        0,
        0.08,

        transform=ax.transAxes,

        width=0.005,

        head_width=0.03,

        head_length=0.03
    )


for ax in axes:
    add_north_arrow(ax)



# ==========================================================
# SCALE BAR
# ==========================================================

for ax in axes:

    scalebar = ScaleBar(
        1,
        units="deg",
        dimension="si-length",
        location="lower right"
    )

    ax.add_artist(
        scalebar
    )


# ==========================================================
# SAVE
# ==========================================================

plt.tight_layout()


plt.savefig(
    OUTPUT_FIG,
    dpi=600,
    bbox_inches="tight"
)


plt.show()


print()
print("Saved:")
print(OUTPUT_FIG)
