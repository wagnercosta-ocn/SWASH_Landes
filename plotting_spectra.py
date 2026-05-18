import numpy as np
import matplotlib.pyplot as plt
import glob
import os

# -------------------------
# USER INPUT
# -------------------------
data_folder = "/scratch/work/langercostaw/swash_landes/swash_cases/OUT"
target_points = [(44.029755529722415,-1.3609655347233844),
(44.23120758322239 ,-1.3211017669921352) ,
(44.45260322858395 ,-1.274972636787726) ,
(44.39783016610618 ,-1.28590554211243)  ,
(43.58564849638961 ,-1.504125049628507) ,
(43.72406991287593 ,-1.4585462977535448) ,
(43.78514533105822 ,-1.440447939322108) ,
(43.861694463021 ,-1.4127540163492067),
(43.90896808200986 ,-1.3939617280801304) ,
(44.1550895298274 ,-1.3337178446686613)]
'''
# ﻦﻗﺎﻃ you want (lat, lon)
target_points = [(52.7297287 , 1.969697 ),
(52.7297287 , 2.2727273 ),
(52.7297287 , 2.5757575 ),
(52.7297287 , 2.878788 ),
(52.7432442 , 3.1818182 ),
(52.7432442 , 3.4848485 ),
(52.7432442 , 3.7878788 ),
(52.7432442 , 4.090909 ),
(52.7432442 , 4.3939395 ),
(52.716217 , -6 ),
(52.7297287 , -5.787879 ),
(52.7432442 , -5.5757575 ),
(52.7567558 , -5.3939395 ),
(52.7702713 , -5.151515 ),
(52.783783 , -4.939394 ),
(51.2972984 , -9.939394 ),
(51.1081085 , -9.878788 ),
(50.9324341 , -9.818182 ),
(50.7432442 , -9.757576 ),
(50.5675659 , -9.69697 ),
(50.3783798 , -9.636364 ),
(50.1891899 , -9.606061 ),
(50.0135117 , -9.545455 ),
(49.8378372 , -9.484848 ),
(49.6486473 , -9.454545 ),
(49.4729729 , -9.393939 ),
(49.283783 , -9.333333 ),
(49.094593 , -9.30303 ),
(48.905407 , -9.242424 ),
(48.716217 , -9.181818 ),
(48.5540543 , -9.151515 ),
(48.3918915 , -9.121212 ),
(48.216217 , -9.060606 ),
(48.0540543 , -9.030303 ),
(47.8648643 , -8.969697 ),
(47.6486473 , -8.909091 ),
(47.4729729 , -8.878788 ),
(47.283783 , -8.818182 ),
(47.1081085 , -8.787879 ),
(46.9459457 , -8.757576 ),
(46.7567558 , -8.69697 ),
(46.5810814 , -8.666667 ),
(46.3918915 , -8.606061 ),
(46.216217 , -8.575758 ),
(46.0405388 , -8.545455 ),
(46 , -8.454545 ),
(46 , -8.242424 ),
(46 , -8.030303 ),
(46.0135117 , -7.818182 ),
(46.0270271 , -7.6060605 ),
(46.0540543 , -7.3636365 ),
(46.0675659 , -7.151515 ),
(46.094593 , -6.878788 ),
(46.1081085 , -6.6363635),
(46.1216202 , -6.4242425 ),
(46.1351357 , -6.212121 ),
(46.1486473 , -6 ),
(46.1756744 , -5.787879 ),
(46.1891899 , -5.5454545 ),
(46.2027016 , -5.3030305 ),
(46.216217 , -5.090909 ),
(46.216217 ,-4.848485 ),
(46.2432442 , -4.6363635 ),
(46.2567558 , -4.4545455 ),
(46.2702713 , -4.2727275 ),
(46.2702713 , -4.060606 ),
(46.283783 , -3.8484848 ),
(46.2972984 , -3.6363637 ),
(46.3108101 , -3.3939395 ),
(46.3243256 , -3.1818182 ),
(46.3243256 , -2.969697 ),
(46.3378372 , -2.7575758 ),
(46.3513527 , -2.5151515 ),
(46.3513527 , -2.2727273 )]
'''

# -------------------------
# READ AUT FILE (MULTI-BLOCK)
# -------------------------
def read_aut_file(file_path):
    spectra_list = []

    with open(file_path, "r") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):

        parts = lines[i].split()

        # Skip invalid lines
        if len(parts) < 5:
            i += 1
            continue

        try:
            lat = float(parts[1])
            lon = float(parts[0])
            date = parts[2]
            n_dir = int(float(parts[3]))
            n_freq = int(float(parts[4]))
        except:
            i += 1
            continue

        # ﺎﺘﺟﺎﻫﺎﺗ ﻭ ﺕﺭﺩﺩﺎﺗ (adjust if your file contains them explicitly)
        directions = np.linspace(7.5, 360, n_dir, endpoint=False)
        frequencies = np.linspace(0.05, 0.5, n_freq)

        # -------------------------
        # READ SPECTRUM MATRIX
        # -------------------------
        spectrum = []
        for j in range(n_freq):
            if i + 1 + j >= len(lines):
                break

            row = lines[i + 1 + j].split()

            if len(row) != n_dir:
                break

            spectrum.append([float(x) for x in row])

        # Only keep valid blocks
        if len(spectrum) == n_freq:
            spectra_list.append({
                "lat": lat,
                "lon": lon,
                "date": date,
                "directions": directions,
                "frequencies": frequencies,
                "spectrum": np.array(spectrum),
            })

            i += n_freq + 1
        else:
            i += 1

    return spectra_list


# -------------------------
# LOAD ALL FILES
# -------------------------
files = sorted(glob.glob(os.path.join(data_folder, "OUT*")))

data = []
for f in files:
    data.extend(read_aut_file(f))

print(f"Total spectra loaded: {len(data)}")

# -------------------------
# FIND CLOSEST POINT
# -------------------------
def find_closest(target_lat, target_lon, dataset):
    dataset = [d for d in dataset if d is not None]

    distances = [
        np.sqrt((d["lat"] - target_lat)**2 + (d["lon"] - target_lon)**2)
        for d in dataset
    ]

    return np.argmin(distances)


# -------------------------
# LOOP OVER REQUESTED POINTS
# -------------------------
for (tlon, tlat) in target_points:

    idx = find_closest(tlat, tlon, data)
    ref = data[idx]

    print(f"Target ({tlat},{tlon}) -> using ({ref['lat']},{ref['lon']})")

    # -------------------------
    # SELECT SAME LOCATION (ALL TIMES)
    # -------------------------
    selected_data = [
        d for d in data
        if abs(d["lat"] - ref["lat"]) < 1e-3
        and abs(d["lon"] - ref["lon"]) < 1e-3
    ]

    print(f"Time steps found: {len(selected_data)}")

    # Stack spectra over time
    spectra_all = np.array([d["spectrum"] for d in selected_data])
    mean_spectrum = spectra_all.mean(axis=0)
    thr=0.03*mean_spectrum.max()
    masked_spectrum = np.ma.masked_less(mean_spectrum,thr)
    directions = ref["directions"]
    frequencies = ref["frequencies"]
    periods = 1 / frequencies

    # -------------------------
    # POLAR GRID
    # -------------------------
    theta_edges = np.linspace(0, 2*np.pi, len(directions) + 1)
    period_edges = np.linspace(periods.min(), periods.max(), len(periods) + 1)

    Theta, R = np.meshgrid(theta_edges, period_edges)

    # -------------------------
    # PLOT
    # -------------------------
    fig = plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, projection="polar")

    # Oceanographic convention
    ax.set_theta_zero_location("N")
    #ax.set_theta_direction(-1)

    pcm = ax.pcolormesh(
        Theta,
        R,
        masked_spectrum,#mean_spectrum,
        cmap="viridis",
        shading="auto"
    )

    # Period circles
    period_ticks = [5, 8, 10, 12, 15, 20]
    ax.set_yticks(period_ticks)
    ax.set_yticklabels([f"{p}s" for p in period_ticks])

    # Direction labels
    ax.set_thetagrids(
        angles=np.arange(0, 360, 45),
        labels=["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    )

    # Colorbar
    cbar = plt.colorbar(pcm, pad=0.1)
    cbar.set_label("Mean Spectral Density")

    # Title
    plt.title(
        f"Mean Directional Spectrum\n"
        f"Requested: ({tlat},{tlon}) | Used: ({ref['lat']},{ref['lon']})\n"
        f"{len(selected_data)} time steps"
    )

    plt.tight_layout()
    output_dir="/scratch/work/langercostaw/swash_landes/swash_cases/OUT/"
    os.makedirs(output_dir,exist_ok=True)
    fig_filename = os.path.join(output_dir,f"spectrum_{ref['lat']:.3f}_{ref['lon']:.3f}_noassi.png")
    plt.savefig(fig_filename, dpi=300)
    print(f"Figure saved to: {fig_filename}")
    #plt.show()

    np.save(os.path.join(output_dir,"mean_spectrum_"+str(tlon)+"_"+str(tlat)+"_noassi.npy"),mean_spectrum)
