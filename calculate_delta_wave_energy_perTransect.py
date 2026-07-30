import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from tqdm import tqdm

# ==========================================================
# USER INPUTS
# ==========================================================

AUT_FILE = "AUT_spectra.pkl"
OUT_FILE = "OUT_spectra.pkl"

TRANSECTS_FILE = "transect_points.csv"

OUTPUT_FILE = "Delta_m0_percent_by_transect.parquet"
OUTPUT_CSV = "Delta_m0_percent_by_transect.csv"

TRANSECT_ID_COL = "Transect_ID"
LON_COL = "Longitude"
LAT_COL = "Latitude"

# ==========================================================
# LOAD DATA
# ==========================================================

print("Loading AUT...")
aut = pd.read_pickle(AUT_FILE)

print("Loading OUT...")
out = pd.read_pickle(OUT_FILE)

print("Loading transects...")
transects = pd.read_csv(TRANSECTS_FILE)

# ==========================================================
# BUILD KD TREE
# ==========================================================

tree = cKDTree(
    np.column_stack(
        (
            transects[LON_COL].values,
            transects[LAT_COL].values,
        )
    )
)

# ==========================================================
# GROUP BY SPECTRAL LOCATION
# ==========================================================

aut_groups = aut.groupby(["lon", "lat"])
out_groups = out.groupby(["lon", "lat"])

common_points = sorted(
    set(aut_groups.groups.keys()).intersection(
        out_groups.groups.keys()
    )
)

print(f"{len(common_points)} common spectral points")

results = []

# ==========================================================
# LOOP OVER POINTS
# ==========================================================

for lon, lat in tqdm(common_points):

    aut_point = aut_groups.get_group((lon, lat))
    out_point = out_groups.get_group((lon, lat))

    # ------------------------------------------------------
    # Mean spectrum
    # ------------------------------------------------------

    mean_aut = np.mean(
        np.stack(aut_point["spectrum"].values),
        axis=0,
    )

    mean_out = np.mean(
        np.stack(out_point["spectrum"].values),
        axis=0,
    )

    # ------------------------------------------------------
    # Frequency vector
    # ------------------------------------------------------

    freqs = np.asarray(
        aut_point.iloc[0]["frequencies"]
    )

    df = np.diff(freqs)

    # last interval
    df = np.append(df, df[-1])

    # ------------------------------------------------------
    # Direction spacing
    # ------------------------------------------------------

    ndir = mean_aut.shape[1]

    dtheta = 360.0 / ndir
    dtheta = np.deg2rad(dtheta)

    # ------------------------------------------------------
    # Compute m0
    # ------------------------------------------------------

    m0_aut = np.sum(mean_aut * df[:, None]) * dtheta
    m0_out = np.sum(mean_out * df[:, None]) * dtheta

    # ------------------------------------------------------
    # Relative change
    # ------------------------------------------------------

    if m0_out > 0:

        delta_percent = (
            (m0_aut - m0_out)
            / m0_out
            * 100.0
        )

    else:

        delta_percent = np.nan

    # ------------------------------------------------------
    # Find nearest transect point
    # ------------------------------------------------------

    dist, idx = tree.query([lon, lat])

    nearest = transects.iloc[idx]

    results.append(
        {
            "Transect_ID": nearest[TRANSECT_ID_COL],
            "Spectrum_Lon": lon,
            "Spectrum_Lat": lat,
            "Transect_Lon": nearest[LON_COL],
            "Transect_Lat": nearest[LAT_COL],
            "Distance_deg": dist,
            "Delta_m0_percent": delta_percent,
            "m0_AUT": m0_aut,
            "m0_OUT": m0_out,
        }
    )

# ==========================================================
# SAVE
# ==========================================================

results = pd.DataFrame(results)

results.to_parquet(
    OUTPUT_FILE,
    index=False,
)

results.to_csv(
    OUTPUT_CSV,
    index=False,
)

print()
print(results.head())

print()
print(f"Saved {len(results)} points")
