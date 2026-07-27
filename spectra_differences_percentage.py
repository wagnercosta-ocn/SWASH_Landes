import numpy as np
import matplotlib.pyplot as plt
import os

# ============================================================
# FILES
# ============================================================

file1 = "/scratch/work/langercostaw/swash_landes/swash_cases/OUT/mean_spectrum_-1.504125049628507_43.58564849638961_noassi.npy"
file2 = "/scratch/work/langercostaw/swash_landes/swash_cases/AUT/mean_spectrum_-1.504125049628507_43.58564849638961_swot_cfo.npy"

name_spec1 = "noassi"
name_spec2 = "SWOT_CFO"

lon = file1.split("_")[3]
lat = file1.split("_")[4]

# ============================================================
# READ SPECTRA
# ============================================================

spec1 = np.load(file1)
spec2 = np.load(file2)

if spec1.shape != spec2.shape:
    raise ValueError("Spectra must have identical dimensions.")

# ============================================================
# SPECTRAL GRID
# ============================================================

n_dir = 24
directions = np.linspace(0, 2*np.pi, n_dir, endpoint=False)

f0 = 0.035
growth = 1.1
n_freq = 30

frequencies = f0 * growth**np.arange(n_freq)
periods = 1.0 / frequencies

# ============================================================
# SYMMETRIC PERCENTAGE DIFFERENCE
# ============================================================

denominator = spec1 + spec2

# Ignore cells with almost no energy
threshold = 1e-6

percent_diff = np.full_like(spec1, np.nan)

mask = denominator > threshold

percent_diff[mask] = (
    200.0 * (spec2[mask] - spec1[mask]) /
    denominator[mask]
)

# Reverse frequency axis so long periods are outside
percent_diff = percent_diff[::-1, :]
periods = periods[::-1]

# ============================================================
# POLAR GRID
# ============================================================

Theta, R = np.meshgrid(directions, periods)

# ============================================================
# ROBUST COLOR SCALE
# ============================================================

valid = np.abs(percent_diff[np.isfinite(percent_diff)])

if len(valid) == 0:
    vmax = 1
else:
    vmax = np.nanpercentile(valid, 99)

# Optional clipping to avoid a few extreme pixels
vmax = min(vmax, 100)

vmin = -vmax

# ============================================================
# FIGURE
# ============================================================

plt.rcParams.update({
    "font.size": 12,
    "axes.linewidth": 1.2,
})

fig = plt.figure(figsize=(7,7))

ax = plt.subplot(111, projection="polar")

ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)

pcm = ax.pcolormesh(
    Theta,
    R,
    percent_diff,
    cmap="RdBu_r",
    shading="auto",
    vmin=vmin,
    vmax=vmax
)

# ============================================================
# RADIAL AXIS
# ============================================================

period_ticks = [5, 8, 10, 12, 15, 20]

ax.set_ylim(periods.min(), periods.max())

ax.set_yticks(period_ticks)
ax.set_yticklabels(
    [f"{p} s" for p in period_ticks],
    fontsize=12
)

# ============================================================
# DIRECTION LABELS
# ============================================================

ax.set_thetagrids(
    np.arange(0,360,45),
    labels=["N","NE","E","SE","S","SW","W","NW"],
    fontsize=13
)

# ============================================================
# GRID STYLE
# ============================================================

ax.grid(alpha=0.35)

# ============================================================
# COLORBAR
# ============================================================

cbar = plt.colorbar(
    pcm,
    pad=0.10,
    shrink=0.70
)

cbar.set_label(
    "Symmetric spectral energy difference (%)",
    fontsize=13
)

# ============================================================
# TITLE
# ============================================================

ax.set_title(
    f"({lon}, {lat})",
    fontsize=14,
    pad=20
)

plt.tight_layout()

# ============================================================
# SAVE
# ============================================================

os.makedirs("outputs_landes", exist_ok=True)

outfile = (
    f"outputs_landes/"
    f"symmetric_percent_difference_{lon}_{lat}_"
    f"{name_spec1}_vs_{name_spec2}"
)

plt.savefig(outfile + ".png", dpi=500, bbox_inches="tight")
plt.savefig(outfile + ".pdf", bbox_inches="tight")

plt.show()
