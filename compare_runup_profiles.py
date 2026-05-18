import os
import glob
import numpy as np
from scipy.io import loadmat
from matplotlib import pyplot as plt


# ============================================================
# TEXT FILE CONTAINING THE LIST OF CASE FOLDERS
# ============================================================
txt_file = 'case_list.txt'


# ============================================================
# ROOT DIRECTORIES
# ============================================================
# Simulation WITH assimilation
root_dir_assim = '/scratch/work/langercostaw/swash_landes/swash_cases/swash_runs_assim'

# Simulation WITHOUT assimilation
root_dir_noassim = '/scratch/work/langercostaw/swash_landes/swash_cases/swash_runs_noassim'


# ============================================================
# OUTPUT DIRECTORY
# ============================================================
figures_dir = os.path.join(root_dir_assim, 'figures')

os.makedirs(figures_dir, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================
mat_pattern = '*.mat'

threshold = 0.01  # meters


# ============================================================
# READ LIST OF CASE FOLDERS
# ============================================================
with open(txt_file, 'r') as f:
    relative_folders = [line.strip() for line in f if line.strip()]


# ============================================================
# MAIN LOOP
# ============================================================
for rel_folder in relative_folders:

    # --------------------------------------------------------
    # Build paths for both simulations
    # --------------------------------------------------------
    folder_assim = os.path.join(root_dir_assim, rel_folder)

    folder_noassim = os.path.join(root_dir_noassim, rel_folder)

    print(f'\nProcessing case: {rel_folder}')

    # --------------------------------------------------------
    # Search for MATLAB files
    # --------------------------------------------------------
    mat_files_assim = glob.glob(
        os.path.join(folder_assim, mat_pattern)
    )

    mat_files_noassim = glob.glob(
        os.path.join(folder_noassim, mat_pattern)
    )

    if len(mat_files_assim) == 0:
        print('  No assimilation .mat file found.')
        continue

    if len(mat_files_noassim) == 0:
        print('  No non-assimilation .mat file found.')
        continue

    mat_file_assim = mat_files_assim[0]

    mat_file_noassim = mat_files_noassim[0]

    try:

        # ====================================================
        # LOAD MATLAB FILES
        # ====================================================
        data_assim = loadmat(mat_file_assim)

        data_noassim = loadmat(mat_file_noassim)

        # ====================================================
        # FIND Watlev VARIABLE
        # ====================================================
        watlev_key_assim = None
        watlev_key_noassim = None

        for key in data_assim.keys():
            if 'Watlev' in key:
                watlev_key_assim = key
                break

        for key in data_noassim.keys():
            if 'Watlev' in key:
                watlev_key_noassim = key
                break

        if watlev_key_assim is None:
            print('  Assimilation Watlev variable not found.')
            continue

        if watlev_key_noassim is None:
            print('  No-assimilation Watlev variable not found.')
            continue

        # ====================================================
        # READ VARIABLES
        # ====================================================
        xp = data_assim['Xp'][0]

        watlev_assim = data_assim[watlev_key_assim][0]

        watlev_noassim = data_noassim[watlev_key_noassim][0]

        # Beach profile
        botlev = data_assim['Botlev'][0] * -1

        # ====================================================
        # COMPUTE RUNUP FOR ASSIMILATION CASE
        # ====================================================
        diff_assim = np.abs(watlev_assim - botlev)

        indices_assim = np.where(diff_assim < threshold)[0]

        if len(indices_assim) > 0:

            runup_idx_assim = indices_assim[0]

            runup_x_assim = xp[runup_idx_assim]

            runup_z_assim = watlev_assim[runup_idx_assim]

        else:

            runup_idx_assim = None

        # ====================================================
        # COMPUTE RUNUP FOR NON-ASSIMILATION CASE
        # ====================================================
        diff_noassim = np.abs(watlev_noassim - botlev)

        indices_noassim = np.where(diff_noassim < threshold)[0]

        if len(indices_noassim) > 0:

            runup_idx_noassim = indices_noassim[0]

            runup_x_noassim = xp[runup_idx_noassim]

            runup_z_noassim = watlev_noassim[runup_idx_noassim]

        else:

            runup_idx_noassim = None

        # ====================================================
        # CREATE FIGURE
        # ====================================================
        plt.figure(figsize=[10, 5])

        ax = plt.gca()

        # ----------------------------------------------------
        # Water level WITH assimilation
        # ----------------------------------------------------
        plt.plot(
            xp,
            watlev_assim,
            'b',
            linewidth=2,
            label='water level (assimilation)'
        )

        # ----------------------------------------------------
        # Water level WITHOUT assimilation
        # ----------------------------------------------------
        plt.plot(
            xp,
            watlev_noassim,
            'r--',
            linewidth=2,
            label='water level (no assimilation)'
        )

        # ----------------------------------------------------
        # Beach profile
        # ----------------------------------------------------
        plt.plot(
            xp,
            botlev,
            'k',
            linewidth=2,
            label='beach profile'
        )

        # ====================================================
        # MARK RUNUP LOCATIONS
        # ====================================================
        if runup_idx_assim is not None:

            plt.plot(
                runup_x_assim,
                runup_z_assim,
                'bo',
                markersize=8
            )

        if runup_idx_noassim is not None:

            plt.plot(
                runup_x_noassim,
                runup_z_noassim,
                'ro',
                markersize=8
            )

        # ====================================================
        # TEXT BOX
        # ====================================================
        text_string = ''

        if runup_idx_assim is not None:

            text_string += (
                f'Assimilation Runup\n'
                f'x = {runup_x_assim:.2f} m\n'
                f'z = {runup_z_assim:.2f} m\n\n'
            )

        if runup_idx_noassim is not None:

            text_string += (
                f'No Assimilation Runup\n'
                f'x = {runup_x_noassim:.2f} m\n'
                f'z = {runup_z_noassim:.2f} m'
            )

        plt.text(
            0.72,
            0.20,
            text_string,
            fontsize=10,
            bbox=dict(
                facecolor='white',
                alpha=0.8
            ),
            transform=ax.transAxes
        )

        # ====================================================
        # FIGURE SETTINGS
        # ====================================================
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
            rel_folder,
            fontsize=14
        )

        plt.legend(fontsize=12)

        # ====================================================
        # OUTPUT NAME
        # ====================================================
        folder_parts = rel_folder.split(os.sep)

        transect_id = folder_parts[0]

        case_date = folder_parts[1]

        output_name = f'{transect_id}_{case_date}.png'

        output_path = os.path.join(
            figures_dir,
            output_name
        )

        # ====================================================
        # SAVE FIGURE
        # ====================================================
        plt.savefig(
            output_path,
            dpi=500,
            bbox_inches='tight'
        )

        plt.close()

        print(f'  Figure saved: {output_path}')

    except Exception as e:

        print(f'  Error: {e}')
