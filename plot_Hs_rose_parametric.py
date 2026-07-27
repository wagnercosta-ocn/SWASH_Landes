import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# ==========================================================
# USER SETTINGS
# ==========================================================

csv_file = "wave_data.csv"

lat_col = "lat"
lon_col = "lon"
dir_col = "Dir"
hs_col = "Hs"

n_sectors = 16

output_dir = "Hs_roses"
os.makedirs(output_dir, exist_ok=True)

# ==========================================================
# READ DATA
# ==========================================================

df = pd.read_csv(csv_file)

df = df[[lat_col, lon_col, dir_col, hs_col]].dropna()

# Keep directions between 0 and 360
df[dir_col] = df[dir_col] % 360

# ==========================================================
# GLOBAL COLOR SCALE
# (same for every figure)
# ==========================================================

global_norm = Normalize(
    vmin=df[hs_col].min(),
    vmax=df[hs_col].max()
)

cmap = plt.cm.viridis

# ==========================================================
# DIRECTION BINS
# ==========================================================

edges = np.linspace(0, 360, n_sectors + 1)
centers = np.deg2rad((edges[:-1] + edges[1:]) / 2)

width = np.deg2rad(360 / n_sectors)

# ==========================================================
# LOOP OVER POINTS
# ==========================================================

for (lat, lon), group in df.groupby([lat_col, lon_col]):

    direction = group[dir_col].values
    hs = group[hs_col].values

    probability = np.zeros(n_sectors)
    mean_hs = np.full(n_sectors, np.nan)

    for i in range(n_sectors):

        mask = (
            (direction >= edges[i]) &
            (direction < edges[i + 1])
        )

        probability[i] = 100 * mask.sum() / len(direction)

        if mask.any():
            mean_hs[i] = hs[mask].mean()

    colors = cmap(global_norm(np.nan_to_num(mean_hs,
                                            nan=df[hs_col].min())))

    # ======================================================
    # FIGURE
    # ======================================================

    fig = plt.figure(figsize=(7.5,7.5))
    ax = plt.subplot(111, projection='polar')

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    ax.bar(
        centers,
        probability,
        width=width,
        color=colors,
        edgecolor='k',
        linewidth=0.5
    )

    ax.set_xticks(np.deg2rad(np.arange(0,360,45)))
    ax.set_xticklabels(
        ["N","NE","E","SE","S","SW","W","NW"],
        fontsize=12
    )

    ax.set_rlabel_position(225)
    ax.tick_params(labelsize=11)

    ax.grid(True, alpha=0.4)

    ax.set_title(
        f"Lat = {lat:.4f}°, Lon = {lon:.4f}°",
        fontsize=14,
        pad=20
    )

    sm = ScalarMappable(norm=global_norm, cmap=cmap)
    sm.set_array([])

    cbar = plt.colorbar(
        sm,
        ax=ax,
        pad=0.10,
        shrink=0.85
    )

    cbar.set_label(
        "Mean $H_s$ (m)",
        fontsize=12
    )

    filename = f"Hs_rose_lat_{lat:.4f}_lon_{lon:.4f}"

    plt.savefig(
        os.path.join(output_dir, filename + ".png"),
        dpi=600,
        bbox_inches="tight"
    )

    plt.savefig(
        os.path.join(output_dir, filename + ".pdf"),
        bbox_inches="tight"
    )

    plt.close(fig)

print(f"\nFinished! {df.groupby([lat_col, lon_col]).ngroups} figures created.")
