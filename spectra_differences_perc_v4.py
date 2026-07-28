import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ==========================================================
# FOLDERS
# ==========================================================

out_folder = Path(
    "/scratch/work/langercostaw/swash_landes/swash_cases/OUT"
)

aut_folder = Path(
    "/scratch/work/langercostaw/swash_landes/swash_cases/AUT"
)

save_folder = Path("outputs_landes")
save_folder.mkdir(exist_ok=True)


# ==========================================================
# FIND FILES
# ==========================================================

out_files = sorted(out_folder.glob("mean_spectrum*.npy"))

print(f"{len(out_files)} spectra found")


# ==========================================================
# GRID DEFINITIONS
# ==========================================================

n_dir = 24

directions = np.linspace(
    0,
    2*np.pi,
    n_dir,
    endpoint=False
)


f0 = 0.035
growth = 1.1
n_freq = 30

frequencies = (
    f0 *
    growth**np.arange(n_freq)
)

periods = 1 / frequencies


# ==========================================================
# EXACT LOG FREQUENCY BIN WIDTHS
# ==========================================================

freq_edges = np.zeros(n_freq + 1)

freq_edges[1:-1] = np.sqrt(
    frequencies[:-1] *
    frequencies[1:]
)

freq_edges[0] = (
    frequencies[0] /
    np.sqrt(growth)
)

freq_edges[-1] = (
    frequencies[-1] *
    np.sqrt(growth)
)


df = np.diff(freq_edges)

dtheta = 2*np.pi / n_dir



# ==========================================================
# FIRST PASS
# Find global color scale
# ==========================================================

print("\nComputing global color scale...")

all_values = []


for file1 in out_files:

    file2 = aut_folder / file1.name.replace(
        "noassi",
        "swot_cfo"
    )

    if not file2.exists():
        continue


    spec1 = np.load(file1)
    spec2 = np.load(file2)


    deltaS = spec2 - spec1


    m0_control = np.sum(
        spec1 *
        df[:,None] *
        dtheta
    )


    m0_assim = np.sum(
        spec2 *
        df[:,None] *
        dtheta
    )


    delta_m0 = m0_assim - m0_control


    if abs(delta_m0) < 1e-12:
        continue


    contribution = (
        100 *
        deltaS *
        df[:,None] *
        dtheta /
        delta_m0
    )


    all_values.extend(
        contribution.flatten()
    )


all_values = np.array(all_values)

global_vmax = np.percentile(
    np.abs(all_values),
    99
)

global_vmax = np.ceil(global_vmax)


print(
    f"Global color scale: "
    f"-{global_vmax:.0f} to +{global_vmax:.0f} %"
)



# ==========================================================
# SECOND PASS
# CREATE FIGURES
# ==========================================================

print("\nCreating figures...")


for counter, file1 in enumerate(out_files, start=1):


    # ------------------------------------------------------
    # Find assimilation file
    # ------------------------------------------------------

    file2 = aut_folder / file1.name.replace(
        "noassi",
        "swot_cfo"
    )


    if not file2.exists():

        print(
            f"Missing: {file2.name}"
        )

        continue



    # ------------------------------------------------------
    # Coordinates
    # ------------------------------------------------------

    parts = file1.stem.split("_")

    lon = float(parts[2])
    lat = float(parts[3])


    print(
        f"[{counter}/{len(out_files)}] "
        f"lon={lon:.4f}, lat={lat:.4f}"
    )


    # ------------------------------------------------------
    # Load spectra
    # ------------------------------------------------------

    spec1 = np.load(file1)
    spec2 = np.load(file2)



    # ------------------------------------------------------
    # Energies
    # ------------------------------------------------------

    m0_control = np.sum(
        spec1 *
        df[:,None] *
        dtheta
    )


    m0_assim = np.sum(
        spec2 *
        df[:,None] *
        dtheta
    )


    delta_m0 = (
        m0_assim -
        m0_control
    )


    delta_m0_percent = (
        100 *
        delta_m0 /
        m0_control
    )


    Hs_control = (
        4 *
        np.sqrt(m0_control)
    )


    Hs_assim = (
        4 *
        np.sqrt(m0_assim)
    )



    # ------------------------------------------------------
    # Contribution metric
    # ------------------------------------------------------

    contribution = (
        100 *
        (spec2-spec1) *
        df[:,None] *
        dtheta /
        delta_m0
    )


    contribution = contribution[::-1,:]

    periods_plot = periods[::-1]



    Theta, R = np.meshgrid(
        directions,
        periods_plot
    )


    # ------------------------------------------------------
    # Plot
    # ------------------------------------------------------

    fig = plt.figure(
        figsize=(7.5,7.5)
    )


    ax = plt.subplot(
        111,
        projection="polar"
    )


    ax.set_theta_zero_location("N")
    #ax.set_theta_direction(-1)



    pcm = ax.pcolormesh(
        Theta,
        R,
        contribution,
        cmap="seismic",
        shading="auto",
        vmin=-global_vmax,
        vmax=global_vmax
    )



    # Period axis

    period_ticks = [
        5,
        8,
        10,
        12,
        15,
        20
    ]


    ax.set_ylim(
        periods_plot.min(),
        periods_plot.max()
    )


    ax.set_yticks(
        period_ticks
    )


    ax.set_yticklabels(
        [
            f"{p} s"
            for p in period_ticks
        ],
        fontsize=12
    )



    # Directions

    ax.set_thetagrids(
        np.arange(0,360,45),
        labels=[
            "N",
            "NE",
            "E",
            "SE",
            "S",
            "SW",
            "W",
            "NW"
        ],
        fontsize=13
    )


    ax.grid(alpha=0.35)



    # ------------------------------------------------------
    # Annotation
    # ------------------------------------------------------

    text = (
        f"$H_s$: "
        f"{Hs_control:.2f}"
        f" → "
        f"{Hs_assim:.2f} m\n"
        f"$\\Delta m_0$: "
        f"{delta_m0_percent:+.2f}%"
    )


    ax.text(
        0.02,
        0.98,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        bbox=dict(
            facecolor="white",
            edgecolor="black",
            alpha=0.85
        )
    )



    # Title

    #ax.set_title(
    #    f"SWOT+CFOSAT - No assimilation\n"
    #    f"lon={lon:.4f}, lat={lat:.4f}",
    #    fontsize=14,
    #    pad=20
    #)



    # Colorbar

    cbar = plt.colorbar(
        pcm,
        pad=0.10,
        shrink=0.72
    )


    cbar.set_label(
        "Contribution to total Δm₀ (%)",
        fontsize=13
    )



    plt.tight_layout()



    # ------------------------------------------------------
    # Save
    # ------------------------------------------------------

    filename = (
        save_folder /
        f"Contribution_Delta_m0_"
        f"lon_{lon:.4f}_"
        f"lat_{lat:.4f}"
    )


    plt.savefig(
        filename.with_suffix(".png"),
        dpi=600,
        bbox_inches="tight"
    )


    plt.savefig(
        filename.with_suffix(".pdf"),
        bbox_inches="tight"
    )


    plt.close(fig)



print("\nFinished!")
