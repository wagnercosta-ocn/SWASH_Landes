
import geopandas as gpd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from shapely.geometry import LineString
import numpy as np
import cartopy.io.img_tiles as cimgt
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import rasterio

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

#=========================
# LOAD MFWAm Output wave spectral points
#=========================
# =========================
# read OUT/AUT file
# =========================
def read_aut_file(file):

    blocks = []

    with open(file) as f:
        lines = f.readlines()

    i = 0

    while i < len(lines):

        parts = lines[i].split()

        if len(parts) == 7:

            lat,lon = float(parts[0]), float(parts[1])

            i += 1

            blocks.append({
                "lat": lat,
                "lon": lon,
            })

        else:
            i += 1
    blocks=pd.DataFrame(blocks)
    return blocks

mfwam_pts=read_aut_file("/scratch/work/langercostaw/swash_landes/swash_cases/OUT/OUT20250102120000")
print("MFWAM points:")
print(mfwam_pts)


# ============================================================
# 1. LOAD CSV
# ============================================================
df = pd.read_csv("/scratch/work/langercostaw/swash_landes/transects_processed/transects_processed.csv")

gdf = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(df.lon, df.lat),
    crs="EPSG:4326"
)

# ============================================================
# 2. BUILD TRANSECTS (first + last point only)
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
# 3. CARTOPY SETUP
# ============================================================
proj = ccrs.PlateCarree()

fig = plt.figure(figsize=(8, 8))
ax = plt.axes(projection=proj)

# Map extent (Biarritz → Arcachon approx)
ax.set_extent([-2.0, -0.8, 43.2, 44.8], crs=proj)

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
    transform=ccrs.PlateCarree(),
    cmap='viridis',
    vmin=-200,
    vmax=0,
    alpha=0.9,
    zorder=1
)

cbar = plt.colorbar(
    img,
    ax=ax,
    orientation='vertical',
    shrink=0.7,
    pad=0.02
)

cbar.set_label(
    "Bathymetry (m)",
    fontsize=18
)

cbar.ax.tick_params(labelsize=14)


# ============================================================
# 4. HIGH-RES BACKGROUND (Cartopy tiles)
# ============================================================
#tiler=cimgt.OSM()
'''
tiler = cimgt.StadiaMapsTiles('terrain-background')
ax = plt.axes(projection=tiler.crs)
ax.set_extent([-2,-0.8,43.2,44.8],crs=ccrs.PlateCarree())
ax.add_image(tiler,10)
'''
ax.add_feature(cfeature.LAND.with_scale('10m'),facecolor='lightgray',zorder=2)
ax.add_feature(cfeature.OCEAN.with_scale('10m'),facecolor='white')

# OPTIONAL better coastline detail
ax.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=1,zorder=2)
ax.add_feature(cfeature.BORDERS.with_scale('10m'), linestyle=':',zorder=2)

# ============================================================
# 5. PLOT TRANSECTS
# ============================================================
for _, row in gdf_lines.iterrows():
    geom = row.geometry

    x, y = geom.xy

    ax.plot(
        x, y,
        transform=proj,
        color="red",
        linewidth=4,
        zorder=5,
    )

ax.plot(
  x, y,
  transform=proj,
  color="red",
  linewidth=4,
  zorder=5,
  label='transects')

# ============================================================
# 6. LABEL TRANSECTS (midpoint)
# ============================================================
def offset_point(line, dlon,dlat): # meters if using EPSG:3857
    mid = line.interpolate(0.5, normalized=True)

    return mid.x + dlon, mid.y - dlat


for _, row in gdf_lines.iterrows():
    line = row.geometry
    x_lab, y_lab = offset_point(line, dlon=0.06,dlat=0.01)

    ax.text(
        x_lab,
        y_lab,
        f"T{row.transect_id}",
        fontsize=14,
        color="k",
        weight="bold",
        ha="center",
        va="center",
        #bbox=dict(facecolor="black", alpha=0.5, edgecolor="none"),
        transform=ccrs.PlateCarree(),
        zorder=10
    )

#
# plot MFWAM points
#
mfwam_pts = gpd.GeoDataFrame(
    mfwam_pts,
    geometry=gpd.points_from_xy(mfwam_pts.lon, mfwam_pts.lat),
    crs="EPSG:4326"
)

#mfp=ax.scatter(mfwam_pts.geometry.x,mfwam_pts.geometry.y,color="red",edgecolor='k',transform=ccrs.PlateCarree(),zorder=11,label='MFWAM-IBI wave spectra')
for _, row in mfwam_pts.iterrows():
    geom = row.geometry

    x, y = geom.xy

    ax.plot(
        x, y,
        transform=proj,
        marker='o',
        color="red",
        linewidth=1,
        zorder=5,
    )


# ============================================================
# 7. SCALE BAR (approx in degrees → km)
# ============================================================
def add_scalebar(ax):
    # 20 km approximation in degrees (~0.18° lat)
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()

    x_start = x0 + 0.8 * (x1 - x0)
    y_start = y0 + 0.02 * (y1 - y0)

    dx = 0.2 # ~10 km near France coast

    ax.plot(
        [x_start, x_start + dx],
        [y_start, y_start],
        color="black",
        linewidth=3,
        transform=proj
    )

    ax.text(
        x_start + dx / 2,
        y_start + 0.03,
        "≈20 km",
        transform=proj,
        ha="center",
        fontsize=12,
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none")
    )

add_scalebar(ax)

# ============================================================
# 8. NORTH ARROW
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
        arrowprops=dict(facecolor="black", width=5, headwidth=15)
    )

add_north_arrow(ax)

# ADD OCEAN TEXT

ax.text(
    -1.75, 43.9,
    "Bay of Biscay",
    transform=ccrs.PlateCarree(),
    fontsize=16,
    color="lightblue",
    weight="bold",
    ha="center",
    alpha=0.9,
    zorder=2,
    rotation=45
)

# ADD CITIES
cities= [
   ("Biarritz",-1.558, 43.48),
   ("Capbreton", -1.43,43.64),
   ("Arcachon",-1.17,44.66),
   ("Mimizan",-1.2311,44.2)
]

for name,lon,lat in cities:
   ax.plot(
      lon,
      lat,
      marker="s",
      color="black",
      transform=ccrs.PlateCarree(),
      zorder=10
   )

   ax.text(
      lon+0.04,
      lat-0.01,
      name,
      transform=ccrs.PlateCarree(),
      fontsize = 14,
      color='k',
      weight='bold',
      bbox=dict(facecolor='w',alpha=0.5,edgecolor='none'),
      zorder=10
   )
# ============================================================
# 9. FINAL STYLE
# ============================================================
#ax.set_title("Transects – Biarritz to Arcachon", fontsize=14)
gl = ax.gridlines(
     crs=ccrs.PlateCarree(),
     draw_labels=True,
     linewidth=0.5,
     color='gray',
     alpha=0.5,
     linestyle='--',zorder=2)
gl.top_labels = False
gl.right_labels=False
gl.xlabel_style={'size':14}
gl.ylabel_style ={'size':14}
ax.legend(fontsize=14,loc='lower left' )
plt.show()
#fig.savefig('SWASH_profiles_map.jpg',dpi=500)
~
