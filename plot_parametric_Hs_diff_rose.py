import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.cm import ScalarMappable


# ==========================================================
# USER SETTINGS
# ==========================================================

input_file = "outputs/Hs_direction_database.pkl"

output_dir = "outputs/dHs_roses"

os.makedirs(output_dir, exist_ok=True)


# Available:
# "mean"
# "median"
# "p90"
# "p95"
# "p99"
# "max"

STATISTIC = "p95"


# Number of directional sectors

n_sectors = 16


# ==========================================================
# READ DATABASE
# ==========================================================

df = pd.read_pickle(input_file)

print(f"Loaded {len(df)} spectra")



# ==========================================================
# SELECT STATISTIC
# ==========================================================

def calculate_statistic(values, statistic):

    if len(values) == 0:
        return np.nan

    if statistic == "mean":
        return np.mean(values)

    elif statistic == "median":
        return np.median(values)

    elif statistic == "p90":
        return np.percentile(values,90)

    elif statistic == "p95":
        return np.percentile(values,95)

    elif statistic == "p99":
        return np.percentile(values,99)

    elif statistic == "max":
        return np.max(values)

    else:
        raise ValueError(
            "Unknown statistic"
        )



# ==========================================================
# DIRECTION BINS
# ==========================================================

edges = np.linspace(
    0,
    360,
    n_sectors+1
)

centers_deg = (
    edges[:-1] +
    edges[1:]
) / 2


centers = np.deg2rad(
    centers_deg
)


width = np.deg2rad(
    360/n_sectors
)



# ==========================================================
# FIRST PASS
# GLOBAL COLOR SCALE
# ==========================================================

all_values = []


for (lat, lon), group in df.groupby(
        ["lat","lon"]):

    for i in range(n_sectors):

        mask = (
            (group["direction"] >= edges[i])
            &
            (group["direction"] < edges[i+1])
        )

        values = group.loc[
            mask,
            "dHs"
        ].values

        stat = calculate_statistic(
            values,
            STATISTIC
        )

        if not np.isnan(stat):
            all_values.append(stat)


all_values = np.asarray(all_values)


max_abs = np.nanmax(
    np.abs(all_values)
)


print(
    f"Color scale: "
    f"{-max_abs:.3f} to {max_abs:.3f} m"
)



norm = TwoSlopeNorm(
    vmin=-max_abs,
    vcenter=0,
    vmax=max_abs
)


cmap = plt.cm.RdBu_r



# ==========================================================
# LOOP OVER STATIONS
# ==========================================================

for (lat, lon), group in df.groupby(
        ["lat","lon"]):


    probability = np.zeros(
        n_sectors
    )


    values_sector = np.full(
        n_sectors,
        np.nan
    )


    # ----------------------------------------------
    # Directional statistics
    # ----------------------------------------------

    for i in range(n_sectors):

        mask = (
            (group["direction"] >= edges[i])
            &
            (group["direction"] < edges[i+1])
        )


        probability[i] = (
            100 *
            mask.sum() /
            len(group)
        )


        values = group.loc[
            mask,
            "dHs"
        ].values


        values_sector[i] = calculate_statistic(
            values,
            STATISTIC
        )


    # ----------------------------------------------
    # Colors
    # ----------------------------------------------

    colors = cmap(
        norm(
            np.nan_to_num(
                values_sector,
                nan=0
            )
        )
    )


    # ----------------------------------------------
    # Figure
    # ----------------------------------------------

    fig = plt.figure(
        figsize=(7.5,7.5)
    )


    ax = plt.subplot(
        111,
        projection="polar"
    )


    # Oceanographic convention

    ax.set_theta_zero_location(
        "N"
    )

    ax.set_theta_direction(
        -1
    )


    ax.bar(

        centers,

        probability,

        width=width,

        color=colors,

        edgecolor="k",

        linewidth=0.5

    )


    # Direction labels

    ax.set_xticks(
        np.deg2rad(
            np.arange(0,360,45)
        )
    )


    ax.set_xticklabels(
        [
            "N",
            "NE",
            "E",
            "SE",
            "S",
            "SW",
            "W",
            "NW"
        ],
        fontsize=12
    )


    ax.set_rlabel_position(
        225
    )


    ax.tick_params(
        labelsize=11
    )


    ax.grid(
        True,
        alpha=0.4
    )


    # ----------------------------------------------
    # Title
    # ----------------------------------------------

    ax.set_title(

        f"ΔHs {STATISTIC}\n"
        f"lat={lat:.4f}°, lon={lon:.4f}°",

        fontsize=14,

        pad=20

    )


    # ----------------------------------------------
    # Colorbar
    # ----------------------------------------------

    sm = ScalarMappable(
        norm=norm,
        cmap=cmap
    )

    sm.set_array([])


    cbar = plt.colorbar(
        sm,
        ax=ax,
        pad=0.10,
        shrink=0.85
    )


    cbar.set_label(
        f"{STATISTIC} "
        r"$\Delta H_s$ (AUT - OUT) [m]",
        fontsize=12
    )


    # ----------------------------------------------
    # Save
    # ----------------------------------------------

    filename = (
        f"dHs_{STATISTIC}_"
        f"lat_{lat:.4f}_"
        f"lon_{lon:.4f}"
    )


    plt.savefig(
        os.path.join(
            output_dir,
            filename+".png"
        ),
        dpi=600,
        bbox_inches="tight"
    )


    plt.savefig(
        os.path.join(
            output_dir,
            filename+".pdf"
        ),
        bbox_inches="tight"
    )


    plt.close(fig)


    print(
        f"Saved {filename}"
    )


print("\nFinished.")
