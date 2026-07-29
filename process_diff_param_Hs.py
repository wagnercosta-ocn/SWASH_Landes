import numpy as np
import pandas as pd

# ==========================================================
# USER SETTINGS
# ==========================================================

aut_file = "outputs/spectra_AUT.pkl"
out_file = "outputs/spectra_OUT.pkl"

output_file = "outputs/Hs_direction_database.pkl"

# Statistic to compute later
STATISTIC = "p95"

# ==========================================================
# READ DATABASES
# ==========================================================

print("Reading databases...")

aut = pd.read_pickle(aut_file)
out = pd.read_pickle(out_file)

print(f"AUT spectra : {len(aut)}")
print(f"OUT spectra : {len(out)}")

# ==========================================================
# MERGE
# ==========================================================

print("\nMatching spectra...")

df = aut.merge(
    out,
    on=["lat", "lon", "datetime"],
    suffixes=("_AUT", "_OUT")
)

print(f"Matched spectra : {len(df)}")

# ==========================================================
# Hs COMPUTATION
# ==========================================================

def compute_hs(spectrum, frequencies, directions):
    """
    Computes Hs from a directional spectrum.
    """

    frequencies = np.asarray(frequencies)
    directions = np.asarray(directions)

    # frequency bin widths

    dfreq = np.empty_like(frequencies)

    dfreq[1:-1] = (
        frequencies[2:] -
        frequencies[:-2]
    ) / 2

    dfreq[0] = frequencies[1] - frequencies[0]
    dfreq[-1] = frequencies[-1] - frequencies[-2]

    # direction spacing

    dtheta = np.deg2rad(
        directions[1] - directions[0]
    )

    # spectral moment m0

    m0 = np.sum(
        spectrum *
        dfreq[:, None]
    ) * dtheta

    return 4 * np.sqrt(m0)


# ==========================================================
# PEAK DIRECTION
# ==========================================================

def peak_direction(spec_aut,
                   spec_out,
                   directions):
    """
    Peak direction computed from
    the average spectrum.
    """

    spec = 0.5 * (
        spec_aut +
        spec_out
    )

    energy = spec.sum(axis=0)

    return directions[
        np.argmax(energy)
    ]


# ==========================================================
# LOOP
# ==========================================================

print("\nComputing Hs...")

records = []

for _, row in df.iterrows():

    freq = row["frequencies_AUT"]
    dirs = row["directions_AUT"]

    spec_aut = row["spectrum_AUT"]
    spec_out = row["spectrum_OUT"]

    hs_aut = compute_hs(
        spec_aut,
        freq,
        dirs
    )

    hs_out = compute_hs(
        spec_out,
        freq,
        dirs
    )

    direction = peak_direction(
        spec_aut,
        spec_out,
        dirs
    )

    records.append(
        {
            "lat": row["lat"],
            "lon": row["lon"],
            "datetime": row["datetime"],
            "direction": direction,
            "Hs_AUT": hs_aut,
            "Hs_OUT": hs_out,
            "dHs": hs_aut - hs_out
        }
    )

analysis = pd.DataFrame(records)

print()

print(analysis.head())

print()

print(
    f"Stations : "
    f"{analysis.groupby(['lat','lon']).ngroups}"
)

print(
    f"Spectra : {len(analysis)}"
)

print(
    f"Mean ΔHs : {analysis['dHs'].mean():.3f} m"
)

print(
    f"Std ΔHs : {analysis['dHs'].std():.3f} m"
)

# ==========================================================
# SAVE
# ==========================================================

analysis.to_pickle(output_file)

analysis.to_csv(
    output_file.replace(".pkl",".csv"),
    index=False
)

print()

print(f"Database saved to\n{output_file}")
