import numpy as np
import matplotlib.pyplot as plt
import os

# ==========================================================
# INPUT FILES
# ==========================================================

file1 = "/scratch/work/langercostaw/swash_landes/swash_cases/OUT/mean_spectrum_-1.504125049628507_43.58564849638961_noassi.npy"
file2 = "/scratch/work/langercostaw/swash_landes/swash_cases/AUT/mean_spectrum_-1.504125049628507_43.58564849638961_swot_cfo.npy"

name_spec1 = "No assimilation"
name_spec2 = "SWOT + CFOSAT"

lon = file1.split("_")[3]
lat = file1.split("_")[4]

# ==========================================================
# LOAD SPECTRA
# ==========================================================

spec1 = np.load(file1)
spec2 = np.load(file2)

if spec1.shape != spec2.shape:
    raise ValueError("Spectra must have identical dimensions.")

# ==========================================================
# SPECTRAL GRID
# ==========================================================

n_dir = 24
directions = np.linspace(0, 2*np.pi, n_dir, endpoint=False)

f0 = 0.035
growth = 1.1
n_freq = 30

frequencies = f0 * growth**np.arange(n_freq)
periods = 1/frequencies

# ==========================================================
# EXACT LOGARITHMIC BIN WIDTHS
# ==========================================================

ratio = growth

# Frequency bin edges
edges = np.zeros(n_freq + 1)

edges[1:-1] = np.sqrt(frequencies[:-1] * frequencies[1:])

edges[0] = frequencies[0] / np.sqrt(ratio)
edges[-1] = frequencies[-1] * np.sqrt(ratio)

df = np.diff(edges)

# Direction increment
dtheta = 2*np.pi / n_dir

# ==========================================================
# COMPUTE m0 (TOTAL WAVE ENERGY)
# ==========================================================

m0 = np.sum(spec1 * df[:, None] * dtheta)

Hs = 4*np.sqrt(m0)

print(f"Integrated spectral energy m0 = {m0:.4f} m²")
print(f"Hs derived from spectrum = {Hs:.2f} m")

# ==========================================================
# RELATIVE CONTRIBUTION TO TOTAL ENERGY
# ==========================================================

percent_change = 100 * (spec2 - spec1) / m0

# Reverse frequency axis so longest periods are outside
percent_change = percent_change[::-1, :]
periods = periods[::-1]

Theta, R = np.meshgrid(directions, periods)

# ==========================================================
# COLOR SCALE
# ==========================================================

valid = np.abs(percent_change[np.isfinite(percent_change)])

vmax = np.percentile(valid,99)

# Optional clipping for easier comparison
vmax = np.ceil(vmax*10)/10

vmin = -vmax

# ==========================================================
# FIGURE STYLE
# ==========================================================

plt.rcParams.update({
    "font.size":13,
    "axes.linewidth":1.2,
    "xtick.direction":"out",
    "ytick.direction":"out"
})

fig = plt.figure(figsize=(7.2,7.2))

ax = plt.subplot(111,projection="polar")

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

# ==========================================================
# RADIAL AXIS
# ==========================================================

period_ticks = [5,8,10,12,15,20]

ax.set_ylim(periods.min(),periods.max())

ax.set_yticks(period_ticks)
ax.set_yticklabels(
    [f"{p} s" for p in period_ticks],
    fontsize=12
)

# ==========================================================
# DIRECTION LABELS
# ==========================================================

ax.set_thetagrids(
    np.arange(0,360,45),
    labels=["N","NE","E","SE","S","SW","W","NW"],
    fontsize=13
)

ax.grid(alpha=0.35)

# ==========================================================
# COLORBAR
# ==========================================================

cbar = plt.colorbar(
    pcm,
    pad=0.10,
    shrink=0.72
)

cbar.set_label(
    "Contribution to total spectral energy (%)",
    fontsize=13
)

# ==========================================================
# TITLE
# ==========================================================

ax.set_title(
    f"{name_spec2} − {name_spec1}\n({lon}, {lat})",
    fontsize=14,
    pad=20
)

plt.tight_layout()

# ==========================================================
# SAVE
# ==========================================================

os.makedirs("outputs_landes",exist_ok=True)

outfile = (
    "outputs_landes/"
    f"spectral_energy_contribution_percent_{lon}_{lat}"
)

plt.savefig(outfile+".png",dpi=600,bbox_inches="tight")
plt.savefig(outfile+".pdf",bbox_inches="tight")

plt.show()
