import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# INPUT
# ============================================================

df = pd.read_pickle("outputs/spectra_database.pkl")

outdir = Path("outputs/rose_plots")
outdir.mkdir(exist_ok=True)

# ============================================================
# GRID DEFINITIONS
# ============================================================

growth = 1.1
f0 = 0.035
nfreq = 30
ndir = 24

frequencies = f0 * growth**np.arange(nfreq)

periods = 1.0 / frequencies

# Same logic as your original script
periods = periods[::-1]

directions_deg = 7.5 + np.arange(ndir) * 15
directions = np.deg2rad(directions_deg)

# ------------------------------------------------------------
# Frequency bin widths
# ------------------------------------------------------------

dfreq = np.empty(nfreq)

dfreq[1:-1] = 0.5 * (frequencies[2:] - frequencies[:-2])
dfreq[0] = frequencies[1] - frequencies[0]
dfreq[-1] = frequencies[-1] - frequencies[-2]

dtheta = np.deg2rad(15)

# ============================================================
# LOOP OVER POINTS
# ============================================================

for (lon, lat), point in df.groupby(["lon", "lat"]):

    spectra = point["spectrum"].values

    # --------------------------------------------------------
    # Accumulator
    # --------------------------------------------------------

    hs_contribution = np.zeros((nfreq, ndir))

    # --------------------------------------------------------
    # Loop over all time steps
    # --------------------------------------------------------

    for spec in spectra:

        # reshape back to (30,24)
        spec = np.asarray(spec).reshape(nfreq, ndir)

        # Hs contribution of every spectral cell
        hs_cell = 4.0 * np.sqrt(
            np.maximum(spec, 0.0) *
            dfreq[:, None] *
            dtheta
        )

        hs_contribution += hs_cell

    # --------------------------------------------------------
    # Mean through time
    # --------------------------------------------------------

    hs_contribution /= len(spectra)

    # Same radial orientation as original figure
    hs_contribution = hs_contribution[::-1, :]

    # --------------------------------------------------------
    # Mesh
    # --------------------------------------------------------

    Theta, Radius = np.meshgrid(
        directions,
        periods
    )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    fig = plt.figure(figsize=(8,8))

    ax = plt.subplot(111, projection="polar")

    # Same orientation as original code
    ax.set_theta_zero_location("N")

    pcm = ax.pcolormesh(
        Theta,
        Radius,
        hs_contribution,
        cmap="viridis",
        shading="auto"
    )

    # --------------------------------------------------------
    # Period ticks
    # --------------------------------------------------------

    ticks = [5,8,10,12,15,20]

    ax.set_ylim(periods.min(), periods.max())

    ax.set_yticks(ticks)

    ax.set_yticklabels(
        [f"{t}s" for t in ticks],
        fontsize=12
    )

    ax.set_thetagrids(
        np.arange(0,360,45),
        labels=["N","NE","E","SE","S","SW","W","NW"],
        fontsize=13
    )

    cbar = plt.colorbar(
        pcm,
        pad=0.10,
        shrink=0.70
    )

    cbar.set_label(
        "Mean Hs contribution (m)",
        fontsize=13
    )

    plt.title(
        f"{lon:.3f}, {lat:.3f}",
        fontsize=14
    )

    plt.tight_layout()

    plt.savefig(
        outdir /
        f"rose_{lon:.3f}_{lat:.3f}.png",
        dpi=300
    )

    plt.close()

print("Finished.")
