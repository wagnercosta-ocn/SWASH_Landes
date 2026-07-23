import numpy as np
import matplotlib.pyplot as plt
import os

# -------------------------
# FILES
# -------------------------

file1 = "/scratch/work/langercostaw/swash_landes/swash_cases/OUT/mean_spectrum_-1.504125049628507_43.58564849638961_noassi.npy"
file2 = "/scratch/work/langercostaw/swash_landes/swash_cases/AUT/mean_spectrum_-1.504125049628507_43.58564849638961_swot_cfo.npy"

lon = file1.split("_")[3]
lat = file1.split("_")[4]


spec1 = np.load(file1)
spec2 = np.load(file2)

name_spec1 = "noassi"
name_spec2 = "SWOT_CFO"


if spec1.shape != spec2.shape:
    raise ValueError("Spectra must have the same shape")


# -------------------------
# GRID DEFINITIONS
# -------------------------

n_dir = 24
directions = np.linspace(0, 2*np.pi, n_dir, endpoint=False)

# Geometric frequency grid
f0 = 0.035
growth = 1.1
n_freq = 30

frequencies = f0 * growth**np.arange(n_freq)


# -------------------------
# FREQUENCY BIN WIDTHS
# (non-uniform frequency spacing)
# -------------------------

df = np.zeros_like(frequencies)

df[1:-1] = 0.5 * (frequencies[2:] - frequencies[:-2])
df[0] = frequencies[1] - frequencies[0]
df[-1] = frequencies[-1] - frequencies[-2]


# Direction bin width (radians)
dtheta = 2*np.pi / n_dir


# -------------------------
# DIFFERENCE SPECTRUM
# -------------------------

diff_spectrum = spec2 - spec1


# -------------------------
# CONVERT TO Hs CONTRIBUTION
# -------------------------

# Each spectral cell contribution to variance
dm0 = diff_spectrum * df[:, None] * dtheta

# Convert variance contribution to equivalent Hs contribution
# Keep sign of the difference
Hs_contribution = np.sign(dm0) * 4*np.sqrt(np.abs(dm0))


# -------------------------
# PERIOD GRID
# -------------------------

periods = 1 / frequencies


# Reverse frequency ordering:
# low frequency (long period) outside
Hs_contribution = Hs_contribution[::-1, :]
periods = periods[::-1]


# -------------------------
# MESHGRID
# -------------------------

Theta, R = np.meshgrid(directions, periods)


# -------------------------
# PLOT
# -------------------------

fig = plt.figure(figsize=(7, 7))
ax = plt.subplot(111, projection="polar")


ax.set_theta_zero_location("N")


vmax = np.max(np.abs(Hs_contribution))
vmin = -vmax


pcm = ax.pcolormesh(
    Theta,
    R,
    Hs_contribution,
    cmap="seismic",
    shading="auto",
    vmin=vmin,
    vmax=vmax
)


# -------------------------
# RADIAL AXIS
# -------------------------

period_ticks = [5, 8, 10, 12, 15, 20]

ax.set_yticks(period_ticks)
ax.set_yticklabels(
    [f"{p}s" for p in period_ticks],
    fontsize=12
)

ax.set_ylim(
    periods.min(),
    periods.max()
)


# -------------------------
# DIRECTIONS
# -------------------------

ax.set_thetagrids(
    np.arange(0, 360, 45),
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
    fontsize=14
)


# -------------------------
# COLORBAR
# -------------------------

cbar = plt.colorbar(
    pcm,
    pad=0.1,
    shrink=0.65
)

cbar.set_label(
    r"$\Delta H_s$ contribution (m)",
    fontsize=14
)


plt.tight_layout()


# -------------------------
# SAVE
# -------------------------

os.makedirs(
    "outputs_landes",
    exist_ok=True
)


outfile = (
    "outputs_landes/"
    "Hs_contribution_"
    + lon +
    "_" +
    lat +
    "_" +
    name_spec1 +
    "-" +
    name_spec2 +
    "_Jan_May_2025.png"
)


plt.savefig(
    outfile,
    dpi=500,
    bbox_inches="tight"
)


plt.show()
