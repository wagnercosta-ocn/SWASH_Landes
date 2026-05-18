import os
import glob
import numpy as np
from scipy.io import loadmat
from matplotlib import pyplot as plt


# ============================================================
# TEXT FILE CONTAINING THE LIST OF CASE FOLDERS
#
# Expected folder structure:
# swash_runs/transect_ID/case_date
#
# Example:
# swash_runs/T001/20250125
# swash_runs/T002/20250126
# ============================================================
txt_file = 'case_list.txt'


# ============================================================
# ROOT DIRECTORY
# ============================================================
root_dir = '/scratch/work/langercostaw/swash_landes/swash_cases/swash_runs'


# ============================================================
# OUTPUT DIRECTORY FOR ALL FIGURES
#
# All figures will be saved inside:
# swash_runs/figures
# ============================================================
figures_dir = os.path.join(root_dir, 'figures')

os.makedirs(figures_dir, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================
# MATLAB file pattern
mat_pattern = '*.mat'

# Threshold used to define wave runup (1 cm)
threshold = 0.01  # meters

# Plot limits
#xlim = [1750, 2200]
#ylim = [-5, 5]


# ============================================================
# READ LIST OF CASE FOLDERS
# ============================================================
with open(txt_file, 'r') as f:
    relative_folders = [line.strip() for line in f if line.strip()]
folders = [os.path.join(root_dir, rel_folder) for rel_folder in relative_folders]

# ============================================================
# MAIN LOOP
# ============================================================
for folder in folders:

    print(f'\nProcessing folder: {folder}')

    # --------------------------------------------------------
    # Search for MATLAB files
    # --------------------------------------------------------
    mat_files = glob.glob(os.path.join(folder, mat_pattern))

    if len(mat_files) == 0:
        print('  No .mat file found.')
        continue

    # Use first MATLAB file found
    mat_file = mat_files[0]

    print(f'  MAT file: {os.path.basename(mat_file)}')

    try:

        # ----------------------------------------------------
        # LOAD MATLAB DATA
        # ----------------------------------------------------
        data = loadmat(mat_file)

        # ----------------------------------------------------
        # AUTOMATICALLY FIND Watlev VARIABLE
        # ----------------------------------------------------
        watlev_key = None

        for key in data.keys():
            if 'Watlev' in key:
                watlev_key = key
                break

        if watlev_key is None:
            print('  Watlev variable not found.')
            continue

        # ----------------------------------------------------
        # READ VARIABLES
        # ----------------------------------------------------
        xp = data['Xp'][0]

        watlev = data[watlev_key][0]

        # Invert Botlev as in original example
        botlev = data['Botlev'][0] * -1

        # Hrunup is optional
        #if 'Hrunup' in data:
        #    hrunup = data['Hrunup'][0]
        #else:
        #    hrunup = None

        # ----------------------------------------------------
        # COMPUTE WAVE RUNUP
        #
        # Wave runup is defined as the last location where:
        #
        # |Watlev - Botlev| < 0.01 m
        # ----------------------------------------------------
        diff = np.abs(watlev - botlev)

        indices = np.where(diff < threshold)[0]

        if len(indices) > 0:

            runup_idx = indices[0]

            runup_x = xp[runup_idx]

            runup_z = watlev[runup_idx]

            runup_text = (
                f'Wave Runup\n'
                f'x = {runup_x:.2f} m\n'
                f'z = {runup_z:.2f} m'
            )

        else:

            runup_idx = None

            runup_text = 'Wave Runup not found'

        # ----------------------------------------------------
        # CREATE FIGURE
        # ----------------------------------------------------
        plt.figure(figsize=[10, 5])
        ax=plt.gca()
        plt.plot(
            xp,
            watlev,
            'b',
            label='water level'
        )

        plt.plot(
            xp,
            botlev,
            'k',
            label='beach profile'
        )

        #if hrunup is not None:

        #    plt.plot(
        #        xp,
        #        hrunup,
        #        'g',
        #        label='Hrunup'
        #    )

        # ----------------------------------------------------
        # MARK WAVE RUNUP LOCATION
        # ----------------------------------------------------
        if runup_idx is not None:

            plt.plot(
                runup_x,
                runup_z,
                'ro',
                markersize=8
            )

            plt.text(
                0.75,
                0.2,
                runup_text,
                fontsize=10,
                color='red',
                bbox=dict(
                    facecolor='white',
                    alpha=0.8
                ),
                transform=ax.transAxes
            )

        # ----------------------------------------------------
        # FIGURE SETTINGS
        # ----------------------------------------------------
        #plt.xlim(xlim)

        #plt.ylim(ylim)

        plt.grid(True)

        plt.xlabel(
            'extension (m)',
            fontsize=16
        )

        plt.ylabel(
            'elevation (m)',
            fontsize=16
        )

        plt.title(
            os.path.basename(folder),
            fontsize=14
        )

        # plt.legend(fontsize=12)

        # ----------------------------------------------------
        # BUILD OUTPUT FIGURE NAME
        #
        # Example:
        # swash_runs/figures/T001_20250125.png
        # ----------------------------------------------------
        relative_path= os.path.relpath(folder,root_dir)

        folder_parts = relative_path.split(os.sep)

        transect_id = folder_parts[0]

        case_date = folder_parts[1]

        output_name = f'{transect_id}_{case_date}.png'

        output_path = os.path.join(
            figures_dir,
            output_name
        )

        # ----------------------------------------------------
        # SAVE FIGURE
        # ----------------------------------------------------
        plt.savefig(
            output_path,
            dpi=500,
            bbox_inches='tight'
        )

        plt.close()

        print(f'  Figure saved: {output_path}')

    except Exception as e:

        print(f'  Error: {e}')
~
~
