import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# ============================================================
# INPUT
# ============================================================

database = "outputs/spectra_database.pkl"
# database = "outputs/spectra_database.parquet"

# ============================================================
# LOAD DATABASE
# ============================================================

if database.endswith(".pkl"):
    df = pd.read_pickle(database)
else:
    df = pd.read_parquet(database)

print(f"{len(df)} spectra loaded.")

# ============================================================
# FUNCTIONS
# ============================================================

def frequency_bin_widths(freq):
    df = np.empty_like(freq)

    df[1:-1] = 0.5 * (freq[2:] - freq[:-2])
    df[0] = freq[1] - freq[0]
    df[-1] = freq[-1] - freq[-2]

    return df


def compute_wave_parameters(spectrum, frequencies, directions):

    df = frequency_bin_widths(frequencies)

    dtheta = np.deg2rad(
        directions[1] - directions[0]
    )

    # -------------------------------------------------------
    # Frequency spectrum
    # -------------------------------------------------------

    Sf = np.sum(spectrum, axis=1)

    m0 = np.sum(Sf * df) * dtheta
    m1 = np.sum(Sf * frequencies * df) * dtheta
    m2 = np.sum(Sf * frequencies**2 * df) * dtheta

    Hs = 4 * np.sqrt(max(m0, 0))

    peak = np.argmax(Sf)
    Tp = 1 / frequencies[peak]

    Tm01 = np.nan
    Tm02 = np.nan

    if m1 > 0:
        Tm01 = m0 / m1

    if m2 > 0:
        Tm02 = np.sqrt(m0 / m2)

    # -------------------------------------------------------
    # Directional distribution
    # -------------------------------------------------------

    Sdir = np.sum(
        spectrum * df[:, None],
        axis=0
    )

    peak_direction = directions[np.argmax(Sdir)]

    theta = np.deg2rad(directions)

    x = np.sum(Sdir * np.cos(theta))
    y = np.sum(Sdir * np.sin(theta))

    mean_direction = np.rad2deg(
        np.arctan2(y, x)
    ) % 360

    return (
        Hs,
        Tp,
        Tm01,
        Tm02,
        mean_direction,
        peak_direction,
    )

# ============================================================
# LOOP
# ============================================================

records = []

for row in tqdm(df.itertuples(index=False), total=len(df)):

    Hs, Tp, Tm01, Tm02, mean_dir, peak_dir = compute_wave_parameters(
        row.spectrum,
        row.frequencies,
        row.directions,
    )

    records.append({

        "lon": row.lon,
        "lat": row.lat,
        "datetime": row.datetime,

        "Hs": Hs,
        "Tp": Tp,
        "Tm01": Tm01,
        "Tm02": Tm02,

        "mean_direction": mean_dir,
        "peak_direction": peak_dir

    })

# ============================================================
# SAVE
# ============================================================

stats = pd.DataFrame(records)

outdir = Path("outputs")
outdir.mkdir(exist_ok=True)

stats.to_csv(
    outdir / "spectra_stats.csv",
    index=False
)

stats.to_parquet(
    outdir / "spectra_stats.parquet",
    index=False
)

print(stats.head())

print("\nFinished.")
