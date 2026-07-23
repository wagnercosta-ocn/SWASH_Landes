import numpy as np
import matplotlib.pyplot as plt
import os

# -------------------------
# FILE
# -------------------------

file1 = "/scratch/work/langercostaw/swash_landes/swash_cases/OUT/mean_spectrum_-1.504125049628507_43.58564849638961_noassi.npy"

lon = file1.split("_")[3]
lat = file1.split("_")[4]

spec = np.load(file1)

name_spec = "noassi"

# -------------------------
# GRID
# -------------------------

n_dir = 24
directions = np.linspace(0, 2*np.pi, n_dir, endpoint=False)

f0 = 0.035
growth = 1.1
n_freq = 30

frequencies = f0 * growth**np.arange(n_freq)

periods = 1/frequencies

# -------------------------
# VARIABLE FREQUENCY BINS
# -------------------------

df = np.zeros_like(frequencies)

df[1:-1] = 0.5*(frequencies[2:] - frequencies[:-2])
df[0] = frequencies[1] - frequencies[0]
df[-1] = frequencies[-1] - frequencies[-2]

# Direction spacing
dtheta = 2*np.pi/n_dir

# -------------------------
# ENERGY CONTRIBUTION
# -------------------------

# contribution of every spectral bin to m0
dm0 = spec * df[:,None] * dtheta

# total Hs
Hs = 4*np.sqrt(np.sum(dm0))

print(f"Hs = {Hs:.2f} m")

# -------------------------
# PREPARE FOR PLOT
# -------------------------

dm0 = dm0[::-1,:]
periods = periods[::-1]

Theta, R = np.meshgrid(directions, periods)

# -------------------------
# PLOT
# -------------------------

fig = plt.figure(figsize=(7,7))
ax = plt.subplot(111, projection="polar")

ax.set_theta_zero_location("N")

pcm = ax.pcolormesh(
    Theta,
    R,
    dm0,
    shading="auto",
    cmap="viridis"
)

# -------------------------
# RADIAL AXIS
# -------------------------

period_ticks = [5,8,10,12,15,20]

ax.set_yticks(period_ticks)
ax.set_yticklabels(
    [f"{p}s" for p in period_ticks],
    fontsize=12
)

ax.set_ylim(periods.min(), periods.max())

# -------------------------
# DIRECTIONS
# -------------------------

ax.set_thetagrids(
    np.arange(0,360,45),
    labels=["N","NE","E","SE","S","SW","W","NW"],
    fontsize=14
)

# -------------------------
# COLORBAR
# -------------------------

cbar = plt.colorbar(
    pcm,
    pad=0.10,
    shrink=0.65
)

cbar.set_label(
    r"Contribution to $m_0$ ($m^2$)",
    fontsize=14
)

# Display Hs in the title
ax.set_title(
    f"{name_spec}\n$H_s$ = {Hs:.2f} m",
    fontsize=15,
    pad=20
)

plt.tight_layout()

# -------------------------
# SAVE
# -------------------------

os.makedirs("outputs_landes", exist_ok=True)

plt.savefig(
    f"outputs_landes/Hs_spectrum_{lon}_{lat}_{name_spec}.png",
    dpi=500,
    bbox_inches="tight"
)

plt.show()
