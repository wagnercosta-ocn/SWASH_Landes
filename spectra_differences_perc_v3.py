import numpy as np
import matplotlib.pyplot as plt
import os

# ==========================================================
# INPUT FILES
# ==========================================================

file1 = "/scratch/work/langercostaw/swash_landes/swash_cases/OUT/mean_spectrum_-1.504125049628507_43.58564849638961_noassi.npy"
file2 = "/scratch/work/langercostaw/swash_landes/swash_cases/AUT/mean_spectrum_-1.504125049628507_43.58564849638961_swot_cfo.npy"

name1 = "No assimilation"
name2 = "SWOT + CFOSAT"

lon = file1.split("_")[3]
lat = file1.split("_")[4]

# ==========================================================
# LOAD
# ==========================================================

spec1 = np.load(file1)
spec2 = np.load(file2)

if spec1.shape != spec2.shape:
    raise ValueError("Spectra have different dimensions.")

# ==========================================================
# SPECTRAL GRID
# ==========================================================

n_dir = 24
theta = np.linspace(0, 2*np.pi, n_dir, endpoint=False)

f0 = 0.035
growth = 1.1
n_freq = 30

freq = f0 * growth**np.arange(n_freq)
period = 1/freq

# ==========================================================
# EXACT FREQUENCY BIN WIDTHS
# ==========================================================

edges = np.zeros(n_freq + 1)

edges[1:-1] = np.sqrt(freq[:-1] * freq[1:])

edges[0] = freq[0] / np.sqrt(growth)
edges[-1] = freq[-1] * np.sqrt(growth)

df = np.diff(edges)

dtheta = 2*np.pi/n_dir

# ==========================================================
# INTEGRATED ENERGIES
# ==========================================================

m0_control = np.sum(spec1 * df[:, None] * dtheta)
m0_assim = np.sum(spec2 * df[:, None] * dtheta)

delta_m0 = m0_assim - m0_control

print(f"m0 control : {m0_control:.5f} m²")
print(f"m0 assim   : {m0_assim:.5f} m²")
print(f"Δm0        : {delta_m0:.5f} m²")

if np.abs(delta_m0) < 1e-12:
    raise ValueError("Δm0 is approximately zero.")

# ==========================================================
# CONTRIBUTION OF EACH BIN TO Δm0
# ==========================================================

deltaS = spec2 - spec1

contribution = (
    100
    * deltaS
    * df[:, None]
    * dtheta
    / delta_m0
)

print(f"Check: contributions sum to {np.nansum(contribution):.2f}%")

# ==========================================================
# PREPARE FOR POLAR PLOT
# ==========================================================

contribution = contribution[::-1, :]
period = period[::-1]

Theta, R = np.meshgrid(theta, period)

# ==========================================================
# COLOR SCALE
# ==========================================================

valid = contribution[np.isfinite(contribution)]

v = np.percentile(np.abs(valid),99)

v = np.ceil(v)

vmin = -v
vmax = v

# ==========================================================
# FIGURE
# ==========================================================

plt.rcParams.update({
    "font.size":13,
    "axes.linewidth":1.2
})

fig = plt.figure(figsize=(7.5,7.5))

ax = plt.subplot(111, projection="polar")

ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)

pcm = ax.pcolormesh(
    Theta,
    R,
    contribution,
    cmap="RdBu_r",
    shading="auto",
    vmin=vmin,
    vmax=vmax
)

# ==========================================================
# PERIOD AXIS
# ==========================================================

ticks = [5,8,10,12,15,20]

ax.set_ylim(period.min(),period.max())

ax.set_yticks(ticks)
ax.set_yticklabels(
    [f"{t} s" for t in ticks],
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
    "Contribution to total Δ$m_0$ (%)",
    fontsize=13
)

# ==========================================================
# TITLE
# ==========================================================

Hs_control = 4*np.sqrt(m0_control)
Hs_assim = 4*np.sqrt(m0_assim)

ax.set_title(
    f"{name2} − {name1}\n"
    f"Hs: {Hs_control:.2f} → {Hs_assim:.2f} m",
    fontsize=14,
    pad=18
)

plt.tight_layout()

# ==========================================================
# SAVE
# ==========================================================

os.makedirs("outputs_landes", exist_ok=True)

outfile = (
    f"outputs_landes/"
    f"Contribution_Delta_m0_{lon}_{lat}"
)

plt.savefig(outfile + ".png", dpi=600, bbox_inches="tight")
plt.savefig(outfile + ".pdf", bbox_inches="tight")

plt.show()
