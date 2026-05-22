import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from shapely.geometry import LineString
import numpy as np
import rasterio
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

# ============================================================
# READ BATHYMETRY ASC
# ============================================================
asc_file = "/scratch/work/langercostaw/swash_landes/MNT_FACADE_ATLANTIQUE_HOMONIM_NM/DONNEES/MNT_ATL100m_HOMONIM_WGS84_NM_ZNEG.asc"

with rasterio.open(asc_file) as src:

    bathy = src.read(1)
    bounds = src.bounds

# ============================================================
# KEEP ONLY OCEAN (negative depths)
# ============================================================
bathy = np.where(bathy < 0, bathy, np.nan)

# ============================================================
# LOAD MFWAM OUTPUT POINTS
# ============================================================
def read_aut_file(file):

    blocks = []

    with open(file) as f:
        lines = f.readlines()

    i = 0

    while i < len(lines):

        parts = lines[i].split()

        # Detect coordinate lines
        if len(parts) == 7:

            lat = float(parts[0])
            lon = float(parts[1])

            blocks.append({
                "lat": lat,
                "lon": lon,
            })

        i += 1

    return pd.DataFrame(blocks)

mfwam_pts = read_aut_file(
    "/scratch/work/langercostaw/swash_landes/swash_cases/OUT/OUT20250102120000"
)

print("\nMFWAM points:")
print(mfwam_pts.head())

# ============================================================
# SANITY CHECK
# ============================================================
print("\nLongitude range:")
print(mfwam_pts.lon.min(), mfwam_pts.lon.max())

print("\nLatitude range:")
print(mfwam_pts.lat.min(), mfwam_pts.lat.max())

# ============================================================
# CONVERT TO GEODATAFRAME
# IMPORTANT:
# x = longitude
# y = latitude
# ============================================================
mfwam_gdf = gpd.GeoDataFrame(
    mfwam_pts,
    geometry=gpd.points_from_xy(
        mfwam_pts.lon,
        mfwam_pts.lat
    ),
    crs="EPSG:4326"
)

# ============================================================
# LOAD TRANSECTS CSV
# ============================================================
df = pd.read_csv(
    "/scratch/work/langercostaw/swash_landes/transects_processed/transects_processed.csv"
)

gdf = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(df.lon, df.lat),
    crs="EPSG:4326"
)

# ============================================================
# BUILD TRANSECTS
# ============================================================
lines = []

for tid, group in gdf.groupby("transect_id"):

    group = group.reset_index(drop=True)

    start = group.geometry.iloc[0]
    end = group.geometry.iloc[-1]

    lines.append({
        "transect_id": tid,
        "geometry": LineString([start, end])
    })

gdf_lines = gpd.GeoDataFrame(lines, crs="EPSG:4326")

# ============================================================
# CARTOPY SETUP
# ============================================================
proj = ccrs.PlateCarree()

fig = plt.figure(figsize=(10, 10))
ax = plt.axes(projection=proj)

# ============================================================
# MAP EXTENT
# ============================================================
ax.set_extent(
    [-2.0, -0.8, 43.2, 44.8],
    crs=proj
)

# ============================================================
# PLOT BATHYMETRY
# ============================================================
img = ax.imshow(
    bathy,
    extent=[
        bounds.left,
        bounds.right,
        bounds.bottom,
        bounds.top
    ],
    origin='upper',
    transform=proj,
    cmap='viridis',
    vmin=-200,
    vmax=0,
    alpha=0.9,
    zorder=1
)

# ============================================================
# COLORBAR
# ============================================================
cbar = plt.colorbar(
    img,
    ax=ax,
    orientation='vertical',
    shrink=0.7,
    pad=0.02
)

cbar.set_label(
    "Bathymetry (m)",
    fontsize=16
)

cbar.ax.tick_params(labelsize=12)

# ============================================================
# LAND / COASTLINES
# ============================================================
ax.add_feature(
    cfeature.LAND.with_scale('10m'),
    facecolor='lightgray',
    zorder=2
)

# IMPORTANT:
# REMOVE OCEAN FEATURE
# It can hide bathymetry + points
#
# ax.add_feature(cfeature.OCEAN...)

ax.add_feature(
    cfeature.COASTLINE.with_scale('10m'),
    linewidth=1,
    zorder=3
)

ax.add_feature(
    cfeature.BORDERS.with_scale('10m'),
    linestyle=':',
    zorder=3
)

# ============================================================
# PLOT TRANSECTS
# ============================================================
first = True

for _, row in gdf_lines.iterrows():

    geom = row.geometry
    x, y = geom.xy

    ax.plot(
        x,
        y,
        transform=proj,
        color="red",
        linewidth=3,
        zorder=5,
        label='Transects' if first else ""
    )

    first = False

# ============================================================
# LABEL TRANSECTS
# ============================================================
def offset_point(line, dlon, dlat):

    mid = line.interpolate(0.5, normalized=True)

    return mid.x + dlon, mid.y - dlat

for _, row in gdf_lines.iterrows():

    line = row.geometry

    x_lab, y_lab = offset_point(
        line,
        dlon=0.06,
        dlat=0.01
    )

    ax.text(
        x_lab,
        y_lab,
        f"T{row.transect_id}",
        fontsize=12,
        color="k",
        weight="bold",
        ha="center",
        va="center",
        transform=proj,
        zorder=10
    )

# ============================================================
# PLOT MFWAM POINTS
# ============================================================
ax.scatter(
    mfwam_gdf.geometry.x,
    mfwam_gdf.geometry.y,
    transform=proj,
    s=40,
    marker='o',
    facecolor='yellow',
    edgecolor='black',
    linewidth=0.8,
    zorder=20,
    label='MFWAM spectral points'
)

# ============================================================
# SCALE BAR
# ============================================================
def add_scalebar(ax):

    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()

    x_start = x0 + 0.78 * (x1 - x0)
    y_start = y0 + 0.03 * (y1 - y0)

    dx = 0.2

    ax.plot(
        [x_start, x_start + dx],
        [y_start, y_start],
        color="black",
        linewidth=3,
        transform=proj,
        zorder=20
    )

    ax.text(
        x_start + dx / 2,
        y_start + 0.03,
        "≈20 km",
        transform=proj,
        ha="center",
        fontsize=12,
        bbox=dict(
            facecolor="white",
            alpha=0.7,
            edgecolor="none"
        ),
        zorder=20
    )

add_scalebar(ax)

# ============================================================
# NORTH ARROW
# ============================================================
def add_north_arrow(ax):

    ax.annotate(
        "N",
        xy=(0.1, 0.92),
        xytext=(0.1, 0.85),
        xycoords="axes fraction",
        ha="center",
        fontsize=14,
        fontweight="bold",
        arrowprops=dict(
            facecolor="black",
            width=5,
            headwidth=15
        )
    )

add_north_arrow(ax)

# ============================================================
# OCEAN LABEL
# ============================================================
ax.text(
    -1.75,
    43.9,
    "Bay of Biscay",
    transform=proj,
    fontsize=16,
    color="lightblue",
    weight="bold",
    ha="center",
    alpha=0.9,
    zorder=4,
    rotation=45
)

# ============================================================
# CITIES
# ============================================================
cities = [
    ("Biarritz", -1.558, 43.48),
    ("Capbreton", -1.43, 43.64),
    ("Arcachon", -1.17, 44.66),
    ("Mimizan", -1.2311, 44.2)
]

for name, lon, lat in cities:

    ax.plot(
        lon,
        lat,
        marker="s",
        color="black",
        markersize=5,
        transform=proj,
        zorder=15
    )

    ax.text(
        lon + 0.04,
        lat - 0.01,
        name,
        transform=proj,
        fontsize=12,
        color='k',
        weight='bold',
        bbox=dict(
            facecolor='white',
            alpha=0.7,
            edgecolor='none'
        ),
        zorder=15
    )

# ============================================================
# GRIDLINES
# ============================================================
gl = ax.gridlines(
    crs=proj,
    draw_labels=True,
    linewidth=0.5,
    color='gray',
    alpha=0.5,
    linestyle='--',
    zorder=2
)

gl.top_labels = False
gl.right_labels = False

gl.xlabel_style = {'size': 12}
gl.ylabel_style = {'size': 12}

# ============================================================
# LEGEND
# ============================================================
ax.legend(
    fontsize=12,
    loc='lower left'
)

# ============================================================
# SHOW
# ============================================================
plt.tight_layout()

plt.show()

# fig.savefig('SWASH_profiles_map.jpg', dpi=500)
