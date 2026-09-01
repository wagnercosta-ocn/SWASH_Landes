import os
import glob
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

from shapely.geometry import Point
from scipy.stats import pearsonr


# ============================================================
# USER SETTINGS
# ============================================================

# SWOT-HR GeoJSON files
SWOT_FOLDER = "/scratch/work/langercostaw/HR_SWOT"

# Transect CSV
TRANSECT_FILE = (
    "/scratch/work/langercostaw/"
    "swash_landes/transects_points_with_depth_filtered_v2/"
    "processed_transects_10.csv"
)

# SWASH results
SWASH_FOLDER = (
    "/scratch/work/langercostaw/"
    "swash_landes/swash_cases/results_per_transect"
)

# Case pattern
CASE_PATTERN = "case_2025012603*_{id}.csv"

# Buffer radius in metres
BUFFER_DISTANCE = 50

# CRS
WGS84_CRS = "EPSG:4326"

# Lambert-93 is a suitable metric CRS for mainland France.
# Distances/buffers are therefore expressed in metres.
METRIC_CRS = "EPSG:2154"

# Variables used in the comparison
SWASH_VARIABLE = "watlev_noassim"
SWOT_VARIABLE = "surface"


# ============================================================
# 1. LOAD SWOT-HR DATA
# ============================================================

print("=" * 70)
print("LOADING SWOT-HR DATA")
print("=" * 70)

geojson_files = glob.glob(
    os.path.join(SWOT_FOLDER, "*.geojson")
)

print(f"Number of SWOT-HR files found: {len(geojson_files)}")

if len(geojson_files) == 0:
    raise FileNotFoundError(
        f"No GeoJSON files found in {SWOT_FOLDER}"
    )


gdf_list = []

for file in geojson_files:

    try:
        tmp = gpd.read_file(file)

        print(
            f"Loaded {os.path.basename(file)}: "
            f"{len(tmp)} features"
        )

        gdf_list.append(tmp)

    except Exception as e:
        print(
            f"WARNING: Could not read {file}: {e}"
        )


if len(gdf_list) == 0:
    raise RuntimeError(
        "No SWOT-HR GeoJSON files could be loaded."
    )


combined_gdf = gpd.GeoDataFrame(
    pd.concat(gdf_list, ignore_index=True),
    crs=gdf_list[0].crs
)


print("\nSWOT-HR files concatenated!")
print(f"Total SWOT features: {len(combined_gdf)}")
print(f"SWOT CRS: {combined_gdf.crs}")

print("\nSWOT geometry types:")
print(combined_gdf.geometry.geom_type.value_counts())


# ============================================================
# 2. CHECK SWOT CRS
# ============================================================

if combined_gdf.crs is None:

    raise ValueError(
        "\nThe SWOT GeoJSON files do not contain a CRS.\n"
        "The correct CRS must be assigned before continuing.\n"
        "Do NOT simply assume EPSG:4326 unless you know that "
        "the coordinates are longitude/latitude."
    )


# Remove empty geometries
combined_gdf = combined_gdf.loc[
    combined_gdf.geometry.notna()
].copy()


# ============================================================
# 3. CHECK SWOT VARIABLE
# ============================================================

if SWOT_VARIABLE not in combined_gdf.columns:

    print("\nAvailable SWOT columns:")
    print(combined_gdf.columns.tolist())

    raise KeyError(
        f"\nColumn '{SWOT_VARIABLE}' was not found "
        "in the SWOT-HR data."
    )


# ============================================================
# 4. CONVERT SWOT DATA TO METRIC CRS
# ============================================================

print("\nConverting SWOT data to metric CRS...")

combined_gdf_metric = combined_gdf.to_crs(
    METRIC_CRS
)

print(
    "SWOT metric bounds:",
    combined_gdf_metric.total_bounds
)


# ============================================================
# 5. LOAD TRANSECT DATA
# ============================================================

print("\n" + "=" * 70)
print("LOADING TRANSECT DATA")
print("=" * 70)

df2 = pd.read_csv(TRANSECT_FILE)

print(f"Number of rows in transect file: {len(df2)}")

required_columns = [
    "transect_id",
    "longitude",
    "latitude"
]

for col in required_columns:

    if col not in df2.columns:

        raise KeyError(
            f"Required column '{col}' not found "
            f"in {TRANSECT_FILE}"
        )


# Remove rows without coordinates
df2 = df2.dropna(
    subset=["longitude", "latitude"]
).copy()


id_tr = df2["transect_id"].unique()

print(
    f"Number of transects found: {len(id_tr)}"
)


# ============================================================
# 6. PROCESS ALL TRANSECTS
# ============================================================

print("\n" + "=" * 70)
print("STARTING SPATIAL MATCHING")
print("=" * 70)

results_list = []

total_matches = 0
total_transect_points = 0


for id in id_tr:

    print("\n" + "-" * 70)
    print(f"TRANSECT: {id}")
    print("-" * 70)

    # --------------------------------------------------------
    # Extract this transect
    # --------------------------------------------------------

    transect_df = df2.loc[
        df2["transect_id"] == id
    ].copy()

    print(
        f"Number of transect points: "
        f"{len(transect_df)}"
    )

    total_transect_points += len(transect_df)


    # --------------------------------------------------------
    # Find SWASH result files
    # --------------------------------------------------------

    csv_files = glob.glob(
        os.path.join(
            SWASH_FOLDER,
            CASE_PATTERN.format(id=id)
        )
    )

    print(
        f"SWASH files found: {len(csv_files)}"
    )

    if len(csv_files) == 0:

        print(
            f"WARNING: No SWASH files found "
            f"for transect {id}"
        )

        continue


    # --------------------------------------------------------
    # Coordinates
    # --------------------------------------------------------

    lon = transect_df[
        "longitude"
    ].to_numpy()

    lat = transect_df[
        "latitude"
    ].to_numpy()


    # ========================================================
    # PROCESS EACH SWASH FILE
    # ========================================================

    for f in csv_files:

        print(
            f"\nProcessing: {os.path.basename(f)}"
        )

        df = pd.read_csv(f)


        # ----------------------------------------------------
        # Check SWASH variable
        # ----------------------------------------------------

        if SWASH_VARIABLE not in df.columns:

            print(
                f"WARNING: '{SWASH_VARIABLE}' not found "
                f"in {os.path.basename(f)}"
            )

            print(
                "Available columns:"
            )
            print(df.columns.tolist())

            continue


        # ----------------------------------------------------
        # Check number of SWASH points
        # ----------------------------------------------------

        if len(df) != len(lon):

            print(
                "\nWARNING: Number of SWASH rows does not "
                "match number of transect points!"
            )

            print(
                f"SWASH rows: {len(df)}"
            )

            print(
                f"Transect points: {len(lon)}"
            )

            continue


        # ----------------------------------------------------
        # Add coordinates
        # ----------------------------------------------------

        df = df.copy()

        df["longitude"] = lon
        df["latitude"] = lat


        # ----------------------------------------------------
        # Create unique transect-point ID
        # ----------------------------------------------------

        df["transect_point_index"] = np.arange(
            len(df)
        )


        # ----------------------------------------------------
        # Create GeoDataFrame
        # ----------------------------------------------------

        transect_gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(
                df["longitude"],
                df["latitude"]
            ),
            crs=WGS84_CRS
        )


        # ----------------------------------------------------
        # Convert transect points to metric CRS
        # ----------------------------------------------------

        transect_metric = transect_gdf.to_crs(
            METRIC_CRS
        )


        # ----------------------------------------------------
        # Create 50-m buffer
        #
        # IMPORTANT:
        # The buffered polygons become the active geometry.
        # ----------------------------------------------------

        buffer_gdf = transect_metric.copy()

        buffer_gdf["geometry"] = (
            buffer_gdf.geometry.buffer(
                BUFFER_DISTANCE
            )
        )


        # ----------------------------------------------------
        # Keep only columns necessary for join
        # ----------------------------------------------------

        buffer_gdf = buffer_gdf[
            [
                "transect_point_index",
                "geometry"
            ]
        ].copy()


        # ----------------------------------------------------
        # Spatial join
        #
        # SWOT points are joined to the 50-m buffers.
        #
        # 'intersects' is used instead of 'within' so that
        # points exactly on the buffer boundary are included.
        # ----------------------------------------------------

        joined = gpd.sjoin(
            combined_gdf_metric,
            buffer_gdf,
            how="inner",
            predicate="intersects"
        )


        n_matches = len(joined)

        total_matches += n_matches

        print(
            f"SWOT points matched: {n_matches}"
        )


        # ----------------------------------------------------
        # No matches
        # ----------------------------------------------------

        if joined.empty:

            print(
                "NO SWOT POINTS FOUND WITHIN "
                f"{BUFFER_DISTANCE} m."
            )

            # Diagnostic information
            print(
                "Transect bounds:",
                transect_metric.total_bounds
            )

            print(
                "SWOT bounds:",
                combined_gdf_metric.total_bounds
            )

            continue


        # ====================================================
        # 7. AVERAGE SWOT SSH IN EACH BUFFER
        # ====================================================

        # Remove NaN SWOT SSH values
        joined_valid = joined.dropna(
            subset=[SWOT_VARIABLE]
        ).copy()


        if joined_valid.empty:

            print(
                "Matched SWOT points exist, but all "
                f"'{SWOT_VARIABLE}' values are NaN."
            )

            continue


        # ----------------------------------------------------
        # Mean SWOT SSH per transect point
        # ----------------------------------------------------

        result = (
            joined_valid
            .groupby("transect_point_index")
            .agg(
                **{
                    SWOT_VARIABLE: (
                        SWOT_VARIABLE,
                        "mean"
                    ),
                    "n_swot": (
                        SWOT_VARIABLE,
                        "count"
                    )
                }
            )
            .reset_index()
        )


        # ====================================================
        # 8. ADD SWASH VALUES
        # ====================================================

        swash_columns = [
            col for col in df.columns
            if col != "geometry"
        ]

        swash_point_data = df[
            swash_columns
        ].copy()


        result = result.merge(
            swash_point_data,
            on="transect_point_index",
            how="left"
        )


        # ----------------------------------------------------
        # Add transect ID
        # ----------------------------------------------------

        result["transect_id"] = id


        # ----------------------------------------------------
        # Add original transect geometry
        # ----------------------------------------------------

        point_geometry = (
            transect_gdf
            .set_index("transect_point_index")
            .geometry
        )

        result["geometry"] = (
            result["transect_point_index"]
            .map(point_geometry)
        )


        result_gdf = gpd.GeoDataFrame(
            result,
            geometry="geometry",
            crs=WGS84_CRS
        )


        # ----------------------------------------------------
        # Store result
        # ----------------------------------------------------

        results_list.append(
            result_gdf
        )


        print(
            f"Transect points with SWOT matches: "
            f"{len(result_gdf)}"
        )


# ============================================================
# 9. COMBINE ALL RESULTS
# ============================================================

print("\n" + "=" * 70)
print("COMBINING RESULTS")
print("=" * 70)


if len(results_list) == 0:

    raise RuntimeError(
        "\nNo valid SWOT/SWASH spatial matches were found.\n\n"
        "Check the CRS, longitude/latitude coordinates, "
        "SWOT geometry type, and the 50-m buffer."
    )


GDF = gpd.GeoDataFrame(
    pd.concat(
        results_list,
        ignore_index=True
    ),
    crs=WGS84_CRS
)


print(
    f"Total matched transect points: {len(GDF)}"
)

print(
    f"Total SWOT points used: "
    f"{GDF['n_swot'].sum()}"
)


# ============================================================
# 10. CHECK RESULTS
# ============================================================

print("\nFirst results:")

columns_to_show = [
    "transect_id",
    "transect_point_index",
    "n_swot",
    SWOT_VARIABLE,
    SWASH_VARIABLE
]

columns_to_show = [
    c for c in columns_to_show
    if c in GDF.columns
]

print(
    GDF[columns_to_show].head(20)
)


# ============================================================
# 11. OPTIONAL: REQUIRE A MINIMUM NUMBER OF SWOT POINTS
# ============================================================

# A buffer containing only one SWOT point may be less robust.
#
# Change this value if desired.

MIN_SWOT_POINTS = 1

GDF = GDF.loc[
    GDF["n_swot"] >= MIN_SWOT_POINTS
].copy()


print(
    f"\nResults after requiring at least "
    f"{MIN_SWOT_POINTS} SWOT point(s) per buffer:"
)

print(
    f"{len(GDF)} transect points retained."
)


# ============================================================
# 12. CREATE COPY FOR STATISTICAL ANALYSIS
# ============================================================

GDF2 = GDF.copy()


# ------------------------------------------------------------
# Keep every 5th point, as in the original script
# ------------------------------------------------------------

GDF2 = GDF2.iloc[::5].copy()


# ------------------------------------------------------------
# Check required variables
# ------------------------------------------------------------

if SWASH_VARIABLE not in GDF2.columns:

    raise KeyError(
        f"SWASH variable '{SWASH_VARIABLE}' "
        "not present in results."
    )


if SWOT_VARIABLE not in GDF2.columns:

    raise KeyError(
        f"SWOT variable '{SWOT_VARIABLE}' "
        "not present in results."
    )


# ============================================================
# 13. FILTER DATA
# ============================================================

# Keep only finite values
GDF2 = GDF2.replace(
    [np.inf, -np.inf],
    np.nan
)

GDF2 = GDF2.dropna(
    subset=[
        SWASH_VARIABLE,
        SWOT_VARIABLE
    ]
).copy()


# Position filter
if "position" in GDF2.columns:

    GDF2 = GDF2.loc[
        GDF2["position"] >= 1000
    ].copy()


# SWOT SSH filter
GDF2 = GDF2.loc[
    GDF2[SWOT_VARIABLE] < 16
].copy()


# Difference filter
GDF2 = GDF2.loc[
    np.abs(
        GDF2[SWOT_VARIABLE]
        -
        GDF2[SWASH_VARIABLE]
    ) < 3
].copy()


print("\n" + "=" * 70)
print("FINAL STATISTICAL DATASET")
print("=" * 70)

print(
    f"Number of points used: {len(GDF2)}"
)


# ============================================================
# 14. STOP IF NOT ENOUGH DATA
# ============================================================

if len(GDF2) < 2:

    raise RuntimeError(
        "\nNot enough valid points for statistical analysis."
    )


# ============================================================
# 15. EXTRACT VARIABLES
# ============================================================

x = GDF2[SWOT_VARIABLE].to_numpy()
y = GDF2[SWASH_VARIABLE].to_numpy()


# ============================================================
# 16. CALCULATE STATISTICS
# ============================================================

corr, p_value = pearsonr(x, y)

difference = y - x

bias = np.mean(difference)

mse = np.mean(
    difference ** 2
)

rmse = np.sqrt(mse)

mae = np.mean(
    np.abs(difference)
)

std = np.std(
    difference
)


# ============================================================
# 17. PRINT STATISTICS
# ============================================================

print("\nResults Scatter")
print("---------------------------")
print(f"N     = {len(GDF2)}")
print(f"Corr  = {corr:.4f}")
print(f"p     = {p_value:.4e}")
print(f"Bias  = {bias:.4f} m")
print(f"RMSE  = {rmse:.4f} m")
print(f"MAE   = {mae:.4f} m")
print(f"STD   = {std:.4f} m")


# ============================================================
# 18. SCATTER PLOT
# ============================================================

fig, ax = plt.subplots(
    figsize=(8, 7)
)


ax.plot(
    x,
    y,
    ".k",
    markersize=5
)


# Determine sensible range for 1:1 line
line_min = min(
    np.nanmin(x),
    np.nanmin(y)
)

line_max = max(
    np.nanmax(x),
    np.nanmax(y)
)

line_range = np.linspace(
    line_min,
    line_max,
    200
)


ax.plot(
    line_range,
    line_range,
    "--k",
    linewidth=1
)


ax.grid(
    True,
    alpha=0.3
)


ax.set_xlabel(
    "SWOT-HR SSH (m)"
)

ax.set_ylabel(
    f"SWASH {SWASH_VARIABLE} (m)"
)

ax.set_title(
    "SWOT-HR vs SWASH"
)


# ------------------------------------------------------------
# Statistics on plot
# ------------------------------------------------------------

if p_value < 0.05:

    ax.text(
        0.05,
        0.95,
        f"corr = {corr:.2f}\n"
        f"bias = {bias:.2f} m\n"
        f"RMSE = {rmse:.2f} m\n"
        f"MAE = {mae:.2f} m\n"
        f"STD = {std:.2f} m\n"
        f"N = {len(GDF2)}",
        transform=ax.transAxes,
        verticalalignment="top"
    )

else:

    ax.text(
        0.05,
        0.95,
        f"corr = {corr:.2f}\n"
        f"p = {p_value:.3f}\n"
        f"N = {len(GDF2)}",
        transform=ax.transAxes,
        verticalalignment="top"
    )


fig.tight_layout()


scatter_filename = (
    f"scatter_swash_swotHR_{SWASH_VARIABLE}.png"
)

fig.savefig(
    scatter_filename,
    dpi=300,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# 19. QQ PLOT
# ============================================================

quantiles = np.arange(
    0.01,
    0.99,
    0.01
)


q1 = GDF2[
    SWOT_VARIABLE
].quantile(
    quantiles
).to_numpy()


q2 = GDF2[
    SWASH_VARIABLE
].quantile(
    quantiles
).to_numpy()


# QQ statistics
qq_difference = q2 - q1

qq_bias = np.mean(
    qq_difference
)

qq_mse = np.mean(
    qq_difference ** 2
)

qq_rmse = np.sqrt(
    qq_mse
)

qq_mae = np.mean(
    np.abs(qq_difference)
)

qq_std = np.std(
    qq_difference
)


# ============================================================
# 20. QQ PLOT
# ============================================================

fig2, ax2 = plt.subplots(
    figsize=(8, 7)
)


ax2.plot(
    q1,
    q2,
    ".b",
    markersize=5
)


qq_min = min(
    np.nanmin(q1),
    np.nanmin(q2)
)

qq_max = max(
    np.nanmax(q1),
    np.nanmax(q2)
)

qq_line = np.linspace(
    qq_min,
    qq_max,
    200
)


ax2.plot(
    qq_line,
    qq_line,
    "--k",
    linewidth=1
)


ax2.grid(
    True,
    alpha=0.3
)


ax2.set_xlabel(
    "SWOT-HR SSH (m)"
)

ax2.set_ylabel(
    f"SWASH {SWASH_VARIABLE} (m)"
)

ax2.set_title(
    "QQ-plot (1%-99%)"
)


ax2.text(
    0.05,
    0.95,
    f"bias = {qq_bias:.2f} m\n"
    f"RMSE = {qq_rmse:.2f} m\n"
    f"MAE = {qq_mae:.2f} m\n"
    f"STD = {qq_std:.2f} m",
    transform=ax2.transAxes,
    verticalalignment="top"
)


fig2.tight_layout()


qq_filename = (
    f"qqplot_swash_swotHR_{SWASH_VARIABLE}.png"
)

fig2.savefig(
    qq_filename,
    dpi=300,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# 21. PRINT QQ STATISTICS
# ============================================================

print("\nResults QQ plot")
print("---------------------------")
print(f"Bias = {qq_bias:.4f} m")
print(f"RMSE = {qq_rmse:.4f} m")
print(f"MAE  = {qq_mae:.4f} m")
print(f"STD  = {qq_std:.4f} m")


# ============================================================
# 22. SAVE MATCHED DATA
# ============================================================

output_file = (
    f"matched_swash_swotHR_{SWASH_VARIABLE}.geojson"
)

GDF.to_file(
    output_file,
    driver="GeoJSON"
)


print("\n" + "=" * 70)
print("PROCESSING COMPLETE")
print("=" * 70)

print(
    f"Matched data saved to: {output_file}"
)

print(
    f"Scatter plot saved to: {scatter_filename}"
)

print(
    f"QQ plot saved to: {qq_filename}"
)
