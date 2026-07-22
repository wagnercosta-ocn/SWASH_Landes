import numpy as np
import matplotlib.pyplot as plt
import os
# -------------------------
# FILES
# -------------------------

#file1="spectra_exp1/mean_spectrum_2.878788_52.7297287_5.npy"
#file2="spectra_exp6/mean_spectrum_2.878788_52.7297287_5.npy"

#file1 = "spectra_exp1/mean_spectrum_-9.69697_50.5675659_2.npy"
#file2 = "spectra_exp6/mean_spectrum_-9.69697_50.5675659_2.npy"
#file1 = "spectra_exp1/mean_spectrum_-2.5151515_46.3513527_1.npy"
#file2 = "spectra_exp2/mean_spectrum_-2.5151515_46.3513527_1.npy"

#file1="spectra_exp1/mean_spectrum_-5.151515_52.7702713_4.npy"
#file2="spectra_exp2/mean_spectrum_-5.151515_52.7702713_4.npy"

#file1="spectra_exp1/mean_spectrum_-5.3030305_46.2027016_0.npy"
#file2="spectra_exp6/mean_spectrum_-5.3030305_46.2027016_0.npy"

#file1="spectra_exp1/mean_spectrum_-7.6060605_46.0270271_3.npy"
#file2="spectra_exp6/mean_spectrum_-7.6060605_46.0270271_3.npy"

#file1="/scratch/work/langercostaw/swash_landes/swash_cases/OUT/mean_spectrum_-1.3609655347233844_44.029755529722415_noassi.npy"

#file2="/scratch/work/langercostaw/swash_landes/swash_cases/AUT/mean_spectrum_-1.3609655347233844_44.029755529722415_swot_cfo.npy"

#file1="/scratch/work/langercostaw/swash_landes/swash_cases/OUT/mean_spectrum_-1.274972636787726_44.45260322858395_noassi.npy"

#file2="/scratch/work/langercostaw/swash_landes/swash_cases/AUT/mean_spectrum_-1.274972636787726_44.45260322858395_swot_cfo.npy"

file1 = "/scratch/work/langercostaw/swash_landes/swash_cases/OUT/mean_spectrum_-1.504125049628507_43.58564849638961_noassi.npy"
file2 = "/scratch/work/langercostaw/swash_landes/swash_cases/AUT/mean_spectrum_-1.504125049628507_43.58564849638961_swot_cfo.npy"

lon=file1.split("_")[3]
lat=file1.split("_")[4]


spec1 = np.load(file1)
spec2 = np.load(file2)
name_spec1 = 'noassi'
name_spec2 = 'SWOT_CFO'


if spec1.shape != spec2.shape:
    raise ValueError("Spectra must have the same shape")

# -------------------------
# GRID DEFINITIONS (FIXED)
# -------------------------

# Directions: CLEAN cyclic grid (IMPORTANT FIX)
n_dir = 24
directions = np.linspace(0, 2*np.pi, n_dir, endpoint=False)

# Frequencies (geometric)
f0 = 0.035
growth = 1.1
n_freq = 30
frequencies = f0 * growth**np.arange(n_freq)

# Getting Hs


# Convert to PERIOD (this is your radial coordinate)
periods = 1 / frequencies

# -------------------------
# DIFFERENCE SPECTRUM
# -------------------------
diff_spectrum = spec2 - spec1

# IMPORTANT:
# rows = frequency (must match periods order)
# ensure consistency: low freq -> high period outward

diff_spectrum = diff_spectrum[::-1, :] # align with increasing period

periods = periods[::-1] # match spectrum order

# -------------------------
# MESHGRID (CORRECT USAGE)
# -------------------------
Theta, R = np.meshgrid(directions, periods)

# -------------------------
# PLOT
# -------------------------
fig = plt.figure(figsize=(6, 6))
ax = plt.subplot(111, projection="polar")

ax.set_theta_zero_location("N")

vmax = np.max(np.abs(diff_spectrum))
vmin = -vmax

pcm = ax.pcolormesh(
    Theta,
    R,
    diff_spectrum,
    cmap="seismic",
    shading="auto",
    vmin=vmin,
    vmax=vmax
)

# -------------------------
# RADIAL AXIS (PERIODS)
# -------------------------
period_ticks = [5, 8, 10, 12, 15, 20]
ax.set_yticks(period_ticks)
ax.set_yticklabels([f"{p}s" for p in period_ticks], fontsize=12)

# Ensure correct radial direction (short → center, long → edge)
ax.set_ylim(periods.min(), periods.max())

# -------------------------
# DIRECTION LABELS (FIXED)
# -------------------------
ax.set_thetagrids(
    np.arange(0, 360, 45),
    labels=["N", "NE", "E", "SE", "S", "SW", "W", "NW"],
    fontsize=14
)

# -------------------------
# COLORBAR
# -------------------------
unit = r"$\mathrm{m}^2\!\cdot\!\mathrm{Hz}^{-1}$"
cbar = plt.colorbar(pcm, pad=0.1, shrink=0.6)
cbar.set_label(f"difference ({unit})", fontsize=14)

plt.tight_layout()

# -------------------------
# SAVE FIGURE
# -------------------------
os.makedirs("outputs_landes", exist_ok=True)
plt.savefig("outputs_landes/diff_spectrum_"+lon+"_"+lat+"_"+name_spec1+"-"+name_spec2+"_Jan_May_2025.png", dpi=500)

plt.show()
