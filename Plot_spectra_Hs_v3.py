import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# INPUT
# ============================================================

# Choose one
df = pd.read_parquet("outputs/spectra_stats.parquet")
# df = pd.read_csv("outputs/spectra_stats.csv")

# Parameter used for radial axis
period_column = "Tm01"       # or "Tp"

# Output folder
outdir = Path("outputs/rose_plots")
outdir.mkdir(parents=True, exist_ok=True)

# ============================================================
# BIN DEFINITIONS
# ============================================================

direction_edges = np.arange(0, 361, 15)

period_edges = np.arange(2, 22.5, 0.5)

# ============================================================
# LOOP OVER POINTS
# ============================================================

for (lon, lat), point in df.groupby(["lon", "lat"]):

    direction = point["mean_direction"].values
    period = point[period_column].values
    hs = point["Hs"].values

    # Remove invalid values
    mask = (
        np.isfinite(direction)
        & np.isfinite(period)
        & np.isfinite(hs)
    )

    direction = direction[mask]
    period = period[mask]
    hs = hs[mask]

    if len(hs) == 0:
        continue

    # --------------------------------------------------------
    # Mean Hs in each (direction,period) bin
    # --------------------------------------------------------

    sum_hs, _, _ = np.histogram2d(
        direction,
        period,
        bins=[direction_edges, period_edges],
        weights=hs
    )

    count, _, _ = np.histogram2d(
        direction,
        period,
        bins=[direction_edges, period_edges]
    )

    hs_mean = sum_hs / np.maximum(count, 1)

    hs_mean[count == 0] = np.nan

    # --------------------------------------------------------
    # Polar mesh
    # --------------------------------------------------------

    theta = np.deg2rad(direction_edges)

    Theta, Radius = np.meshgrid(
        theta,
        period_edges
    )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    fig = plt.figure(figsize=(8,8))

    ax = plt.subplot(
        111,
        projection="polar"
    )

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    pcm = ax.pcolormesh(
        Theta,
        Radius,
        hs_mean.T,
        shading="auto",
        cmap="viridis"
    )

    # --------------------------------------------------------
    # Axes
    # --------------------------------------------------------

    ax.set_ylim(
        period_edges.min(),
        period_edges.max()
    )

    ax.set_yticks([4,6,8,10,12,15,18,20])

    ax.set_yticklabels(
        [f"{x}s" for x in [4,6,8,10,12,15,18,20]]
    )

    ax.set_thetagrids(
        np.arange(0,360,45),
        labels=["N","NE","E","SE","S","SW","W","NW"]
    )

    cbar = plt.colorbar(
        pcm,
        pad=0.10,
        shrink=0.70
    )

    cbar.set_label(
        "Mean Hs (m)",
        fontsize=13
    )

    plt.title(
        f"Lon={lon:.3f}   Lat={lat:.3f}",
        fontsize=14
    )

    plt.tight_layout()

    plt.savefig(
        outdir / f"rose_{lon:.3f}_{lat:.3f}.png",
        dpi=500
    )

    plt.close()

print("Finished!")
