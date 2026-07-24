import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# ============================================================
# INPUT
# ============================================================

input_dir = Path("/path/to/AUT/files")

aut_files = sorted(input_dir.glob("AUT*"))

print(f"{len(aut_files)} AUT files found")

# ============================================================
# CONSTANTS
# ============================================================

growth = 1.1

records = []

# ============================================================
# LOOP OVER FILES
# ============================================================

for file in tqdm(aut_files):

    with open(file, "r") as f:

        while True:

            header = f.readline()

            if header == "":
                break

            if not header.strip():
                continue

            parts = header.split()

            # --------------------------------------------------
            # HEADER
            # --------------------------------------------------

            lon = float(parts[0])
            lat = float(parts[1])

            date = datetime.strptime(parts[2], "%Y%m%d%H%M")

            ndir = int(float(parts[3]))
            nfreq = int(float(parts[4]))

            first_direction = float(parts[5])
            first_frequency = float(parts[6])

            # --------------------------------------------------
            # REBUILD GRIDS
            # --------------------------------------------------

            frequencies = first_frequency * growth**np.arange(nfreq)

            directions = first_direction + np.arange(ndir) * (360 / ndir)

            periods = 1.0 / frequencies

            # --------------------------------------------------
            # READ SPECTRUM
            # --------------------------------------------------

            spectrum = np.zeros((nfreq, ndir), dtype=np.float32)

            for i in range(nfreq):
                spectrum[i] = np.fromstring(
                    f.readline(),
                    sep=" ",
                    dtype=np.float32
                )

            # --------------------------------------------------
            # SAVE EVERYTHING
            # --------------------------------------------------

            records.append(
                {
                    "lon": lon,
                    "lat": lat,
                    "datetime": date,
                    "frequencies": frequencies.astype(np.float32),
                    "periods": periods.astype(np.float32),
                    "directions": directions.astype(np.float32),
                    "spectrum": spectrum
                }
            )

# ============================================================
# DATAFRAME
# ============================================================

df = pd.DataFrame(records)

print(df.head())

print(f"\nNumber of spectra: {len(df)}")

# ============================================================
# SAVE
# ============================================================

output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

df.to_parquet(
    output_dir / "spectra_database.parquet",
    index=False
)

df.to_pickle(
    output_dir / "spectra_database.pkl"
)

print("\nDatabase saved.")
