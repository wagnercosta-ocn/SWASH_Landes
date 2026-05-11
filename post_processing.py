import os
import glob
import numpy as np
import pandas as pd
from scipy.io import loadmat

# =========================================================
# USER INPUT
# =========================================================
base_dir = "/scratch/work/langercostaw/swash_landes/swash_cases/swash_runs"

output_csv = "swash_runup_results.csv"

# =========================================================
# FIND ALL MATLAB OUTPUT FILES
# =========================================================
mat_files = glob.glob(
    os.path.join(base_dir, "*", "case_*", "output_*.mat")
)

print(f"Found {len(mat_files)} matlab files")

# =========================================================
# STORAGE
# =========================================================
results = []

# =========================================================
# MAIN LOOP
# =========================================================
for mat_file in mat_files:

    try:

        print(f"\nProcessing: {mat_file}")

        # -------------------------------------------------
        # Extract transect ID and date from filename
        # -------------------------------------------------
        filename = os.path.basename(mat_file)

        # Example:
        # output_12_202501250000.mat

        parts = filename.replace(".mat", "").split("_")

        transect_id = parts[1]
        date = parts[2]

        # -------------------------------------------------
        # Load MATLAB file
        # -------------------------------------------------
        mat = loadmat(mat_file)

        # -------------------------------------------------
        # Find WaterLev variable automatically
        # -------------------------------------------------
        waterlev_var = None

        for key in mat.keys():

            key_lower = key.lower()

            if key_lower.startswith("watlev") or key_lower.startswith("waterlev"):

                waterlev_var = key
                break

        if waterlev_var is None:

            print("⚠️ No WaterLev variable found")
            continue

        print(f"Using variable: {waterlev_var}")

        # -------------------------------------------------
        # Find Botlev variable automatically
        # -------------------------------------------------
        botlev_var = None

        for key in mat.keys():

            key_lower = key.lower()

            if key_lower.startswith("botlev"):

                botlev_var = key
                break

        if botlev_var is None:

            print("⚠️ No Botlev variable found")
            continue

        print(f"Using variable: {botlev_var}")

        # -------------------------------------------------
        # Read variables
        # -------------------------------------------------
        waterlev = np.squeeze(mat[waterlev_var])
        botlev = np.squeeze(mat[botlev_var])

        # -------------------------------------------------
        # Ensure same length
        # -------------------------------------------------
        n = min(len(waterlev), len(botlev))

        waterlev = waterlev[:n]
        botlev = botlev[:n]

        # -------------------------------------------------
        # Compute difference
        # -------------------------------------------------
        diff = waterlev - botlev

        # -------------------------------------------------
        # Find crossing points
        # -------------------------------------------------
        crossings = np.where(np.diff(np.sign(diff)) != 0)[0]

        if len(crossings) == 0:

            print("⚠️ No crossing found")

            runup = np.nan

        else:

            # Use last crossing = highest runup position
            idx = crossings[-1]

            # Elevation at crossing
            runup = botlev[idx]

            print(f"Run-up elevation: {runup:.3f}")

        # -------------------------------------------------
        # Save result
        # -------------------------------------------------
        results.append({
            "Transect_ID": transect_id,
            "Date": date,
            "Runup": runup
        })

    except Exception as e:

        print(f"❌ Error processing {mat_file}")
        print(e)

# =========================================================
# SAVE CSV
# =========================================================
df = pd.DataFrame(results)

df = df.sort_values(
    by=["Transect_ID", "Date"]
)

df.to_csv(output_csv, index=False)

print("\n====================================")
print(f"Results saved to: {output_csv}")
print("====================================")
