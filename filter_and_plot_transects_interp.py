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

    # Compute distance
    df_tr["distance"] = compute_distance(df_tr["latitude"].values,
                                         df_tr["longitude"].values)

    df_tr = df_tr.sort_values("distance")

    distance = df_tr["distance"].values
    elevation = df_tr["Depth"].values

    # ---- Plot BEFORE
    plot_profile(distance, elevation,
                 f"Before - Transect {df_tr['transect_id'].iloc[0]}",
                 f"{OUTPUT_FOLDER}/profiles_before/tr_{df_tr['transect_id'].iloc[0]}.png")

    # --- Ensure offshore start (optional auto-fix)
    if elevation[0] > elevation[-1]:
        distance = distance[::-1]
        elevation = elevation[::-1]
    # --- Find first land point
    land_idx = np.where(elevation >= 0)[0]
    if len(land_idx) == 0:
        return None

    first_land = land_idx[0]
    if first_land < 2:
        return None

    # --- Find +10 m
    end_idx = None
    for i in range(first_land, len(elevation)):
        if elevation[i] >= 10:
            end_idx = i
            break

    if end_idx is None:
        return None

    # --- Trim
    trimmed_dist = distance[:end_idx + 1]
    trimmed_elev = elevation[:end_idx + 1]

    # --- Length check
    if (trimmed_dist[-1] - trimmed_dist[0]) < 500:
        return None

    # --- Interpolation (1 m)
    new_dist = np.arange(trimmed_dist[0], trimmed_dist[-1], 1)

    interp_func = interp1d(trimmed_dist, trimmed_elev, kind='linear')
    new_elev = interp_func(new_dist)

    # ---- Plot AFTER
    plot_profile(new_dist, new_elev,
                 f"After - Transect {df_tr['transect_id'].iloc[0]}",
                 f"{OUTPUT_FOLDER}/profiles_after/tr_{df_tr['transect_id'].iloc[0]}.png")

    # --- Interpolate lat/lon too
    interp_lat = interp1d(trimmed_dist, df_tr["latitude"].values[:end_idx + 1])
    interp_lon = interp1d(trimmed_dist, df_tr["longitude"].values[:end_idx + 1])

    new_lat = interp_lat(new_dist)
    new_lon = interp_lon(new_dist)

    return pd.DataFrame({
        "transect_id": df_tr["transect_id"].iloc[0],
        "distance": new_dist,
        "latitude": new_lat,
        "longitude": new_lon,
        "elevation": new_elev
    })


# -----------------------------
# Main
# -----------------------------
def process_all():
    df = pd.read_csv(INPUT_CSV)

    all_before = df.copy()
    all_after = []

    for tid, group in df.groupby("transect_id"):
        result = process_transect(group)
        if result is not None:
            all_after.append(result)

    if len(all_after) == 0:
        print("No valid transects.")
        return

    df_after = pd.concat(all_after, ignore_index=True)

    # -----------------------------
    # MAP PLOT (before vs after)
    # -----------------------------
    plt.figure(figsize=(10, 8))

    # BEFORE
    sc1 = plt.scatter(all_before["longitude"], all_before["latitude"],
                      c=all_before["Depth"], s=5, alpha=0.4)

    # AFTER
    sc2 = plt.scatter(df_after["longitude"], df_after["latitude"],
                      c=df_after["elevation"], s=8)

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Transects Before (faded) and After (bold)")

    cbar = plt.colorbar(sc2)
    cbar.set_label("Elevation (m)")

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_FOLDER}/map_transects.png", dpi=200)
    plt.close()

    # Save data
    df_after.to_csv(f"{OUTPUT_FOLDER}/processed_transects.csv", index=False)

    print("Processing complete.")


# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    process_all()

