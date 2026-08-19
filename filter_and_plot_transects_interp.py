import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import os

# -----------------------------
# SETTINGS
# -----------------------------
INPUT_CSV = "transects_points_with_depth.csv"
OUTPUT_FOLDER = "transects_points_with_depth_filtered"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(f"{OUTPUT_FOLDER}/profiles_before", exist_ok=True)
os.makedirs(f"{OUTPUT_FOLDER}/profiles_after", exist_ok=True)


# -----------------------------
# Fast distance computation (no geopy)
# -----------------------------
def compute_distance(lat, lon):
    R = 6371000 # Earth radius in meters

    lat = np.radians(lat)
    lon = np.radians(lon)

    dlat = np.diff(lat)
    dlon = np.diff(lon)

    a = dlat**2 + (np.cos(lat[:-1]) * dlon)**2
    d = R * np.sqrt(a)

    dist = np.insert(np.cumsum(d), 0, 0)
    return dist


# -----------------------------
# Plot profile
# -----------------------------
def plot_profile(distance, elevation, title, filepath):
    plt.figure(figsize=(8, 4))
    plt.plot(distance, elevation)
    plt.axhline(0, linestyle="--")
    plt.axhline(10, linestyle="--")
    plt.xlabel("Distance (m)")
    plt.ylabel("Elevation (m)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close()


# -----------------------------
# Process one transect
# -----------------------------
def process_transect(df_tr):
    df_tr = df_tr.copy()

    # --------------------------------------------------
    # Sort transect points in their original transect order
    # --------------------------------------------------
    df_tr = df_tr.reset_index(drop=True)

    # Compute cumulative distance
    df_tr["distance"] = compute_distance(
        df_tr["latitude"].values,
        df_tr["longitude"].values
    )

    # Make sure distance is increasing
    df_tr = df_tr.sort_values("distance").reset_index(drop=True)

    distance = df_tr["distance"].values
    elevation = df_tr["Depth"].values
    lat = df_tr["latitude"].values
    lon = df_tr["longitude"].values

    # --------------------------------------------------
    # BEFORE PLOT
    # --------------------------------------------------
    plot_profile(
        distance,
        elevation,
        f"Before - Transect {df_tr['transect_id'].iloc[0]}",
        f"{OUTPUT_FOLDER}/profiles_before/tr_{df_tr['transect_id'].iloc[0]}.png"
    )

    # --------------------------------------------------
    # Determine direction: we want SEA -> LAND
    #
    # Assume:
    #   sea = negative elevation
    #   land = positive elevation
    #
    # If the first point is land and the last point is sea,
    # reverse the transect.
    # --------------------------------------------------
    if elevation[0] > elevation[-1]:
        distance = distance[::-1]
        elevation = elevation[::-1]
        lat = lat[::-1]
        lon = lon[::-1]

        # Rebuild distance so it starts at 0
        distance = np.concatenate([
            [0],
            np.cumsum(
                np.sqrt(
                    np.diff(lat)**2 + np.diff(lon)**2
                )
            )
        ])

        # NOTE:
        # The above is only an ordering correction.
        # We will replace distance below with the proper
        # geographic distance calculation.
        distance = compute_distance(lat, lon)

    # --------------------------------------------------
    # Find the FIRST LAND POINT
    # elevation >= 0
    # --------------------------------------------------
    land_idx = np.where(elevation >= 0)[0]

    if len(land_idx) == 0:
        return None

    first_land = land_idx[0]

    # Need some offshore points before reaching land
    if first_land < 2:
        return None

    # --------------------------------------------------
    # Find +10 m point
    # --------------------------------------------------
    end_idx = None

    for i in range(first_land, len(elevation)):
        if elevation[i] >= 10:
            end_idx = i
            break

    if end_idx is None:
        return None

    # --------------------------------------------------
    # KEEP EVERYTHING:
    #
    # SEA -> BEACH -> LAND (+10 m)
    # --------------------------------------------------
    trimmed_dist = distance[:end_idx + 1]
    trimmed_elev = elevation[:end_idx + 1]
    trimmed_lat = lat[:end_idx + 1]
    trimmed_lon = lon[:end_idx + 1]

    # --------------------------------------------------
    # Length check
    # --------------------------------------------------
    if (trimmed_dist[-1] - trimmed_dist[0]) < 500:
        return None

    # --------------------------------------------------
    # Interpolate every 1 metre
    # --------------------------------------------------
    new_dist = np.arange(
        trimmed_dist[0],
        trimmed_dist[-1] + 1,
        1
    )

    interp_elev = interp1d(
        trimmed_dist,
        trimmed_elev,
        kind="linear"
    )

    new_elev = interp_elev(new_dist)

    # --------------------------------------------------
    # Interpolate latitude / longitude
    # --------------------------------------------------
    interp_lat = interp1d(
        trimmed_dist,
        trimmed_lat,
        kind="linear"
    )

    interp_lon = interp1d(
        trimmed_dist,
        trimmed_lon,
        kind="linear"
    )

    new_lat = interp_lat(new_dist)
    new_lon = interp_lon(new_dist)

    # --------------------------------------------------
    # AFTER PLOT
    # --------------------------------------------------
    plot_profile(
        new_dist,
        new_elev,
        f"After - Transect {df_tr['transect_id'].iloc[0]}",
        f"{OUTPUT_FOLDER}/profiles_after/tr_{df_tr['transect_id'].iloc[0]}.png"
    )

    # --------------------------------------------------
    # Return
    # --------------------------------------------------
    return pd.DataFrame({
        "transect_id": df_tr["transect_id"].iloc[0],
        "distance": new_dist,
        "latitude": new_lat,
        "longitude": new_lon,
        "elevation": new_elev
    })
# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    process_all()

