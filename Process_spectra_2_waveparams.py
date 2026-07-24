import numpy as np
from pathlib import Path
import pandas as pd
from datetime import datetime

# ============================================================
# CONSTANTS
# ============================================================

GROWTH_FACTOR = 1.1


# ============================================================
# FREQUENCY GRID
# ============================================================

def build_frequency_grid(f0, nfreq, growth=GROWTH_FACTOR):
    """
    Reconstruct WW3/SWASH geometric frequency grid.

    Parameters
    ----------
    f0 : float
        First frequency (Hz)

    nfreq : int
        Number of frequencies

    growth : float
        Frequency growth factor (default=1.1)

    Returns
    -------
    frequencies : ndarray (nfreq,)
    """

    return f0 * growth ** np.arange(nfreq)


# ============================================================
# DIRECTION GRID
# ============================================================

def build_direction_grid(first_direction, ndir):
    """
    Reconstruct equally-spaced directional grid.

    Parameters
    ----------
    first_direction : float
        First direction (degrees)

    ndir : int

    Returns
    -------
    directions : ndarray (ndir,)
    """

    dtheta_deg = 360.0 / ndir

    return first_direction + np.arange(ndir) * dtheta_deg


# ============================================================
# VARIABLE FREQUENCY BIN WIDTHS
# ============================================================

def frequency_bin_widths(freq):

    df = np.empty_like(freq)

    df[1:-1] = 0.5 * (freq[2:] - freq[:-2])

    df[0] = freq[1] - freq[0]

    df[-1] = freq[-1] - freq[-2]

    return df


# ============================================================
# COMPUTE WAVE PARAMETERS
# ============================================================

def compute_wave_parameters(
        spectrum,
        frequencies,
        directions_deg):
    """
    Parameters
    ----------
    spectrum : ndarray (nfreq, ndir)

    frequencies : ndarray (nfreq,)

    directions_deg : ndarray (ndir,)

    Returns
    -------
    dict containing

        Hs
        Tp
        Tm01
        Tm02
        mean_direction
        peak_direction
    """

    ndir = len(directions_deg)

    dtheta = np.deg2rad(360.0 / ndir)

    df = frequency_bin_widths(frequencies)

    # ---------------------------------------------------------
    # Total spectral moments
    # ---------------------------------------------------------

    S_f = np.sum(spectrum, axis=1)

    m0 = np.sum(S_f * df) * dtheta

    m1 = np.sum(S_f * frequencies * df) * dtheta

    m2 = np.sum(S_f * frequencies**2 * df) * dtheta

    # ---------------------------------------------------------
    # Significant wave height
    # ---------------------------------------------------------

    Hs = 4.0 * np.sqrt(max(m0, 0.0))

    # ---------------------------------------------------------
    # Peak period
    # ---------------------------------------------------------

    peak_index = np.argmax(S_f)

    Tp = 1.0 / frequencies[peak_index]

    # ---------------------------------------------------------
    # Mean periods
    # ---------------------------------------------------------

    if m1 > 0:
        Tm01 = m0 / m1
    else:
        Tm01 = np.nan

    if m2 > 0:
        Tm02 = np.sqrt(m0 / m2)
    else:
        Tm02 = np.nan

    # ---------------------------------------------------------
    # Directional energy distribution
    # ---------------------------------------------------------

    S_theta = np.sum(
        spectrum * df[:, None],
        axis=0
    )

    # ---------------------------------------------------------
    # Peak direction
    # ---------------------------------------------------------

    peak_direction = directions_deg[np.argmax(S_theta)] % 360

    # ---------------------------------------------------------
    # Mean direction
    # ---------------------------------------------------------

    theta = np.deg2rad(directions_deg)

    x = np.sum(S_theta * np.cos(theta))

    y = np.sum(S_theta * np.sin(theta))

    mean_direction = np.rad2deg(
        np.arctan2(y, x)
    ) % 360

    return {
        "Hs": Hs,
        "Tp": Tp,
        "Tm01": Tm01,
        "Tm02": Tm02,
        "mean_direction": mean_direction,
        "peak_direction": peak_direction,
    }
# ============================================================
# PART 2
# READ ALL AUT FILES AND COMPUTE STATISTICS
# ============================================================

from tqdm import tqdm
import glob

# ------------------------------------------------------------
# DIRECTORY CONTAINING AUT FILES
# ------------------------------------------------------------

input_dir = "/path/to/AUT/files"

aut_files = sorted(
    glob.glob(f"{input_dir}/AUT*")
)

print(f"Found {len(aut_files)} AUT files")

# ------------------------------------------------------------
# STORAGE
# ------------------------------------------------------------

records = []

# ------------------------------------------------------------
# LOOP OVER FILES
# ------------------------------------------------------------

for aut_file in tqdm(aut_files, desc="Reading AUT files"):

    with open(aut_file, "r") as f:

        lines = f.readlines()

    i = 0
    nlines = len(lines)

    while i < nlines:

        line = lines[i].strip()

        # Skip empty lines
        if len(line) == 0:
            i += 1
            continue

        tokens = line.split()

        # ----------------------------------------------------
        # HEADER DETECTION
        # Expected:
        # lon lat datetime ndir nfreq firstdir firstfreq
        # ----------------------------------------------------

        if len(tokens) != 7:
            i += 1
            continue

        try:

            lon = float(tokens[0])
            lat = float(tokens[1])

            datetime_str = tokens[2]

            ndir = int(float(tokens[3]))
            nfreq = int(float(tokens[4]))

            first_direction = float(tokens[5])
            first_frequency = float(tokens[6])

        except Exception:

            i += 1
            continue

        # ----------------------------------------------------
        # READ SPECTRUM
        # ----------------------------------------------------

        spectrum = np.zeros((nfreq, ndir))

        success = True

        for k in range(nfreq):

            if i + 1 + k >= nlines:
                success = False
                break

            row = lines[i + 1 + k].split()

            if len(row) < ndir:
                success = False
                break

            spectrum[k, :] = np.asarray(
                row[:ndir],
                dtype=float
            )

        if not success:
            print(
                f"Warning: incomplete spectrum in "
                f"{aut_file}"
            )
            break

        # ----------------------------------------------------
        # RECONSTRUCT GRIDS
        # ----------------------------------------------------

        frequencies = build_frequency_grid(
            first_frequency,
            nfreq
        )

        directions = build_direction_grid(
            first_direction,
            ndir
        )

        # ----------------------------------------------------
        # COMPUTE PARAMETERS
        # ----------------------------------------------------

        stats = compute_wave_parameters(
            spectrum,
            frequencies,
            directions
        )

        # ----------------------------------------------------
        # STORE
        # ----------------------------------------------------

        records.append({
            "lon": lon,
            "lat": lat,
            "datetime": datetime_str,
            "Hs": stats["Hs"],
            "Tp": stats["Tp"],
            "Tm01": stats["Tm01"],
            "Tm02": stats["Tm02"],
            "mean_direction": stats["mean_direction"],
            "peak_direction": stats["peak_direction"]
        })

        # ----------------------------------------------------
        # MOVE TO NEXT HEADER
        # ----------------------------------------------------

        i += nfreq + 1

# ============================================================
# CREATE DATAFRAME
# ============================================================

df = pd.DataFrame(records)

# Optional: convert datetime column

df["datetime"] = pd.to_datetime(
    df["datetime"],
    format="%Y%m%d%H%M",
    errors="coerce"
)

print()
print(df.head())
print()
print(f"Total spectra processed: {len(df):,}")
print(f"Unique points: {df[['lon','lat']].drop_duplicates().shape[0]:,}")
from pathlib import Path

# Output directory
output_dir = Path("outputs")
output_dir.mkdir(parents=True, exist_ok=True)

# Save CSV
df.to_csv(
    output_dir / "spectra_stats.csv",
    index=False
)

# Save Parquet
df.to_parquet(
    output_dir / "spectra_stats.parquet",
    index=False
)

print(f"CSV saved to: {output_dir / 'spectra_stats.csv'}")
print(f"Parquet saved to: {output_dir / 'spectra_stats.parquet'}")
