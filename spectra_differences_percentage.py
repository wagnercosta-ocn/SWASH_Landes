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
# FREQUENCY/DIRECTION GRID
# ============================================================

n_dir = 24
directions = np.linspace(0, 2*np.pi, n_dir, endpoint=False)

f0 = 0.035
growth = 1.1
n_freq = 30

frequencies = f0 * growth**np.arange(n_freq)
periods = 1 / frequencies

# ============================================================
# TOTAL SPECTRAL ENERGY OF CONTROL
# ============================================================

# Since frequency spacing is logarithmic, compute Δf
df = np.empty_like(frequencies)
df[:-1] = np.diff(frequencies)
df[-1] = df[-2]

# Direction increment (radians)
dtheta = 2*np.pi / n_dir

# Total integrated spectral energy (m²)
total_energy = np.sum(spec1 * df[:, None] * dtheta)

print(f"Total control spectral energy = {total_energy:.4f} m²")

# ============================================================
# PERCENT CONTRIBUTION TO TOTAL ENERGY CHANGE
# ============================================================

percent_change = (
    100.0 *
    (spec2 - spec1) /
    total_energy
)

# Reverse for plotting
percent_change = percent_change[::-1, :]
periods = periods[::-1]

Theta, R = np.meshgrid(directions, periods)

# ============================================================
# COLOR SCALE
# ============================================================

vmax = np.nanpercentile(np.abs(percent_change), 99)

# Optional cap to make different stations comparable
vmax = min(vmax, 5)

vmin = -vmax

# ============================================================
# FIGURE
# ============================================================

plt.rcParams.update({
    "font.size": 12,
    "axes.linewidth": 1.2
})

fig = plt.figure(figsize=(7,7))

ax = plt.subplot(111, projection="polar")

ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)

pcm = ax.pcolormesh(
    Theta,
    R,
    percent_change,
    cmap="RdBu_r",
    shading="auto",
    vmin=vmin,
    vmax=vmax
)

# ============================================================
# PERIOD AXIS
# ============================================================

period_ticks = [5,8,10,12,15,20]

ax.set_ylim(periods.min(), periods.max())

ax.set_yticks(period_ticks)
ax.set_yticklabels([f"{p} s" for p in period_ticks])

# ============================================================
# DIRECTION LABELS
# ============================================================

ax.set_thetagrids(
    np.arange(0,360,45),
    labels=["N","NE","E","SE","S","SW","W","NW"]
)

ax.grid(alpha=0.35)

# ============================================================
# COLORBAR
# ============================================================

cbar = plt.colorbar(
    pcm,
    pad=0.10,
    shrink=0.72
)

cbar.set_label(
    "Contribution to total spectral energy change (%)",
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
    "outputs_landes/"
    f"relative_energy_change_{lon}_{lat}_{name_spec1}_vs_{name_spec2}"
)

plt.savefig(outfile + ".png", dpi=500, bbox_inches="tight")
plt.savefig(outfile + ".pdf", bbox_inches="tight")

plt.show()
