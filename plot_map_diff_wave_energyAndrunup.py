import numpy as np
import pandas as pd
import geopandas as gpd

from shapely.geometry import Point

import cartopy.io.shapereader as shpreader


# ==========================================================
# USER INPUTS
# ==========================================================

SPECTRAL_FILE = "spectral_points_delta.geojson"

TRANSECTS_FILE = "transects_runup_q95.geojson"

TRANSECT_POINTS_FILE = "transect_points.csv"


OUTPUT_OFFSHORE = "offshore_delta_m0_points.geojson"

OUTPUT_TRANSECTS = "final_transects_runup.geojson"


TRANSECT_ID = "Transect_ID"

LON = "Longitude"
LAT = "Latitude"


# ==========================================================
# LOAD DATA
# ==========================================================

print("Loading files...")


spectral = gpd.read_file(
    SPECTRAL_FILE
)


transects = gpd.read_file(
    TRANSECTS_FILE
)


transect_points = pd.read_csv(
    TRANSECT_POINTS_FILE
)



# ==========================================================
# CREATE TRANSECT POINT GEODATAFRAME
# ==========================================================

points_gdf = gpd.GeoDataFrame(
    transect_points,

    geometry=gpd.points_from_xy(
        transect_points[LON],
        transect_points[LAT]
    ),

    crs="EPSG:4326"
)



# ==========================================================
# LOAD COASTLINE
# ==========================================================

print("Loading coastline...")


land_shp = shpreader.natural_earth(
    resolution="10m",
    category="physical",
    name="land"
)


land = gpd.read_file(
    land_shp
)


land = land.to_crs(
    "EPSG:3857"
)



# coastline boundary

coastline = (
    land
    .boundary
)



# ==========================================================
# PROJECT DATA TO METRIC CRS
# ==========================================================

print("Projecting...")


points_m = points_gdf.to_crs(
    "EPSG:3857"
)

spectral_m = spectral.to_crs(
    "EPSG:3857"
)



# ==========================================================
# FIND OFFSHORE POINT PER TRANSECT
# ==========================================================

print("Finding offshore points...")


offshore_points = []


for tid, group in points_m.groupby(
        TRANSECT_ID
):

    distances = []

    for geom in group.geometry:

        d = (
            coastline
            .distance(geom)
            .min()
        )

        distances.append(d)


    group = group.copy()

    group["coast_distance"] = distances


    offshore = group.loc[
        group["coast_distance"]
        .idxmax()
    ]


    offshore_points.append(
        offshore
    )



offshore_points = gpd.GeoDataFrame(
    offshore_points,

    crs="EPSG:3857"
)



print(
    f"{len(offshore_points)} offshore points found"
)



# ==========================================================
# MATCH OFFSHORE POINT WITH SPECTRAL POINT
# ==========================================================

print("Matching spectral points...")


matched = []


for _, row in offshore_points.iterrows():


    tid = row[TRANSECT_ID]


    candidates = spectral_m[
        spectral_m[TRANSECT_ID]
        ==
        tid
    ]


    if len(candidates) == 0:

        continue



    distances = (
        candidates.geometry
        .distance(
            row.geometry
        )
    )


    idx = distances.idxmin()


    selected = candidates.loc[idx]


    matched.append(
        selected
    )



offshore_spectral = gpd.GeoDataFrame(
    matched,

    crs="EPSG:3857"
)



print(
    f"{len(offshore_spectral)} offshore spectral points matched"
)



# ==========================================================
# SAVE OFFSHORE POINTS
# ==========================================================

offshore_spectral.to_crs(
    "EPSG:4326"
).to_file(
    OUTPUT_OFFSHORE,
    driver="GeoJSON"
)



# ==========================================================
# SAVE TRANSECTS
# ==========================================================

transects.to_crs(
    "EPSG:4326"
).to_file(
    OUTPUT_TRANSECTS,
    driver="GeoJSON"
)



print()
print("Saved:")
print(
    OUTPUT_OFFSHORE
)

print(
    OUTPUT_TRANSECTS
)
import numpy as np
import geopandas as gpd

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm

import cartopy.crs as ccrs
import cartopy.feature as cfeature

from matplotlib.patches import Rectangle


# ==========================================================
# INPUTS
# ==========================================================

OFFSHORE_FILE = "offshore_delta_m0_points.geojson"

TRANSECTS_FILE = "final_transects_runup.geojson"


OUTPUT_FIG = "Delta_m0_vs_Runup_map.png"


# ==========================================================
# LOAD DATA
# ==========================================================

print("Loading data...")


offshore = gpd.read_file(
    OFFSHORE_FILE
)


transects = gpd.read_file(
    TRANSECTS_FILE
)



# ==========================================================
# PROJECTION FOR PLOTTING
# ==========================================================

# Web Mercator is convenient for Cartopy
crs_plot = ccrs.PlateCarree()


offshore_plot = offshore.to_crs(
    "EPSG:4326"
)

transects_plot = transects.to_crs(
    "EPSG:4326"
)



# ==========================================================
# COLOR NORMALIZATION
# ==========================================================


energy_limit = np.nanpercentile(
    np.abs(
        offshore_plot["Delta_m0_percent"]
    ),
    98
)


runup_limit = np.nanpercentile(
    np.abs(
        transects_plot["Runup_Q95"]
    ),
    98
)


norm_energy = colors.TwoSlopeNorm(
    vmin=-energy_limit,
    vcenter=0,
    vmax=energy_limit
)


norm_runup = colors.TwoSlopeNorm(
    vmin=-runup_limit,
    vcenter=0,
    vmax=runup_limit
)


cmap = plt.cm.RdBu_r



# ==========================================================
# MAP EXTENT
# ==========================================================

all_geom = (
    offshore_plot
    .geometry
    .union_all()
)


xmin, ymin, xmax, ymax = (
    all_geom.bounds
)


margin_x = 0.35
margin_y = 0.25


extent = [
    xmin-margin_x,
    xmax+margin_x,
    ymin-margin_y,
    ymax+margin_y
]



# ==========================================================
# FIGURE
# ==========================================================

fig = plt.figure(
    figsize=(14,7)
)


ax1 = fig.add_subplot(
    1,
    2,
    1,
    projection=crs_plot
)


ax2 = fig.add_subplot(
    1,
    2,
    2,
    projection=crs_plot
)


axes = [
    ax1,
    ax2
]


# ==========================================================
# COMMON MAP SETTINGS
# ==========================================================


for ax in axes:

    ax.set_extent(
        extent,
        crs=crs_plot
    )


    ax.add_feature(
        cfeature.LAND,
        facecolor="0.85",
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


    gl = ax.gridlines(
        draw_labels=True,
        linewidth=0.3,
        alpha=0.5
    )

    gl.top_labels = False
    gl.right_labels = False



# ==========================================================
# PANEL A : DELTA M0
# ==========================================================


sc = ax1.scatter(

    offshore_plot.geometry.x,

    offshore_plot.geometry.y,

    c=offshore_plot["Delta_m0_percent"],

    cmap=cmap,

    norm=norm_energy,

    s=35,

    edgecolor="black",

    linewidth=0.3,

    transform=crs_plot,

    zorder=5
)


ax1.set_title(
    "(a) Offshore relative wave-energy change",
    fontsize=13
)



# ==========================================================
# PANEL B : RUNUP
# ==========================================================


transects_plot.plot(

    ax=ax2,

    column="Runup_Q95",

    cmap=cmap,

    norm=norm_runup,

    linewidth=2.0,

    transform=crs_plot,

    zorder=5
)


ax2.set_title(
    "(b) Coastal extreme run-up response",
    fontsize=13
)



# ==========================================================
# CITY LOCATIONS
# ==========================================================


cities = {

    "Arcachon":
        (-1.17,44.66),

    "Mimizan":
        (-1.16,44.20),

    "Capbreton":
        (-1.43,43.64),

    "Biarritz":
        (-1.56,43.48)

}



for name,(lon,lat) in cities.items():

    for ax in axes:

        ax.plot(
            lon,
            lat,
            marker="o",
            markersize=4,
            color="black",
            transform=crs_plot,
            zorder=8
        )


        ax.text(

            lon+0.05,

            lat+0.03,

            name,

            fontsize=9,

            transform=crs_plot,

            zorder=8
        )



# ==========================================================
# BAY OF BISCAY LABEL
# ==========================================================


for ax in axes:

    ax.text(

        -2.8,

        44.2,

        "Bay of Biscay",

        fontsize=13,

        fontstyle="italic",

        color="0.25",

        transform=crs_plot,

        rotation=0

    )



# ==========================================================
# NORTH ARROW
# ==========================================================


def north_arrow(ax):

    ax.annotate(

        "N",

        xy=(0.92,0.85),

        xycoords="axes fraction",

        fontsize=14,

        fontweight="bold",

        ha="center"

    )


    ax.arrow(

        0.92,

        0.73,

        0,

        0.08,

        transform=ax.transAxes,

        width=0.004,

        head_width=0.025,

        head_length=0.025,

        color="black"

    )


for ax in axes:

    north_arrow(ax)



# ==========================================================
# SCALE BAR
# ==========================================================


def add_scale_bar(ax, length_km=50):

    xmin,xmax,ymin,ymax = ax.get_extent(
        crs_plot
    )


    lat = ymin + 0.08*(ymax-ymin)


    deg = length_km / (
        111.32*np.cos(
            np.deg2rad(lat)
        )
    )


    x0 = xmin + 0.08*(xmax-xmin)

    x1 = x0 + deg


    ax.plot(

        [x0,x1],

        [lat,lat],

        transform=crs_plot,

        color="black",

        linewidth=3

    )


    ax.text(

        (x0+x1)/2,

        lat+0.03,

        f"{length_km} km",

        transform=crs_plot,

        ha="center",

        fontsize=9

    )



for ax in axes:

    add_scale_bar(ax)



# ==========================================================
# COLORBARS
# ==========================================================


cb1 = fig.colorbar(

    cm.ScalarMappable(
        norm=norm_energy,
        cmap=cmap
    ),

    ax=ax1,

    orientation="horizontal",

    pad=0.05

)


cb1.set_label(
    r"$\Delta m_0$ (%)"
)



cb2 = fig.colorbar(

    cm.ScalarMappable(
        norm=norm_runup,
        cmap=cmap
    ),

    ax=ax2,

    orientation="horizontal",

    pad=0.05

)


cb2.set_label(
    r"$Q_{95}(\Delta R)$ (m)"
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
