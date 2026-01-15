import sys
import os
import numpy as np
import scipy.io
import pandas as pd
import matplotlib.pyplot as plt
import mne

# Ensure we can import from src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model_NMM_plastic import nmm_plastic
from parameters_whole_brain import get_whole_brain_params
from parameters_EZ import get_EZ_params
from parameters_PZ import get_PZ_params

def save_to_edf(data, channel_names, fs, filename):
    """
    Saves the LFP data to an EDF file using MNE
    data: (n_channels, n_samples) numpy array
    channel_names: list of strings
    fs: sampling frequency (Hz)
    """

    n_channels = data.shape[0]
    n_samples = data.shape[1]
    print(f"Creating MNE Raw object ({n_channels} channels, {n_samples} samples)...")

    data_V = data * 1e-3

    # Create MNE Info
    # Ensure channel names are strings
    ch_names_str = [str(x) for x in channel_names]

    # We define channel types as 'seeg'
    info = mne.create_info(ch_names=ch_names_str, sfreq=fs, ch_types='seeg')

    # Create RawArray
    raw = mne.io.RawArray(data_V, info)

    # Export to EDF
    print(f"Exporting to {filename} using MNE...")
    try:
        mne.export.export_raw(filename, raw, fmt='edf', overwrite=True)
        print("EDF file saved successfully.")
    except Exception as e:
        print(f"Error exporting EDF with MNE: {e}")

def run_whole_brain_simulation():
    print("--- Initializing Whole-Brain Simulation (NE2LOC1) ---")

    # 1. Load Whole Brain Parameters & Region Names
    # ---------------------------------------------
    print("Loading parameters...")
    wb_params = get_whole_brain_params()
    region_names = wb_params['region_names']
    n_regions = len(region_names)
    print(f"Detected {n_regions} regions.")

    # 2. Initialize Model
    # -------------------
    model = nmm_plastic()
    model.NbNMMs = n_regions

    # Initialize vectors (parameters for each node)
    model.init_vector_param()
    model.init_vector()

    # 3. Apply Whole Brain Parameters
    # -------------------------------
    count = 0
    for key, val in wb_params.items():
        if hasattr(model, key) and isinstance(val, np.ndarray):
            try:
                # Direct array copy for speed and Numba compatibility
                getattr(model, key)[:] = val
                count += 1
            except Exception:
                pass
    print(f"Applied {count} parameter arrays.")

    # 4. Identify EZ and PZ Indices
    # -----------------------------
    ez_params = get_EZ_params()
    pz_params = get_PZ_params()

    ez_name = ez_params['Name'] # String
    pz_names = pz_params['Name'] # List of strings

    try:
        ez_idx = region_names.index(ez_name)
    except ValueError:
        print(f"Error: EZ {ez_name} not found.")
        return

    pz_indices = []
    for name in pz_names:
        try:
            pz_indices.append(region_names.index(name))
        except ValueError:
            print(f"Warning: PZ {name} not found.")

    print(f"EZ Index: {ez_idx} ({ez_name})")
    print(f"PZ Indices: {pz_indices}")

    # 5. Overwrite Parameters for EZ and PZ
    # -------------------------------------
    # Helper to apply specific scalar params to specific indices
    def apply_local_params(indices, params):
        idx_list = indices if isinstance(indices, list) else [indices]
        for key, val in params.items():
            if hasattr(model, key) and isinstance(val, (int, float)):
                attr = getattr(model, key)
                if isinstance(attr, np.ndarray):
                    for i in idx_list:
                        attr[i] = val

    apply_local_params(ez_idx, ez_params)
    apply_local_params(pz_indices, pz_params)
    print("EZ/PZ specific parameters applied.")

    # 6. Load Connectivity Matrix
    # ---------------------------
    cm_path = os.path.join(os.path.dirname(__file__), '../data/CMs/CM_whole_brain_NE2LOC1.mat')
    print(f"Loading Connectivity: {cm_path}")

    if os.path.exists(cm_path):
        mat = scipy.io.loadmat(cm_path)
        if 'CM_Matrix' in mat:
            cm = mat['CM_Matrix']
            # Safety check on shape
            rows, cols = cm.shape
            r_lim = min(rows, n_regions)
            c_lim = min(cols, n_regions)
            model.CM_P_P[:r_lim, :c_lim] = cm[:r_lim, :c_lim]
            print("Connectivity matrix loaded.")
        else:
            print("Error: 'CM_Matrix' not found in MAT file.")
    else:
        print("Error: Matrix file not found.")
        return

    # 7. Configure Plasticity
    # --------------------------------------------------
    print("Configuring Plasticity Indices...")

    plastic_links = []
    for pz_i in pz_indices:
        # [Post, Pre] -> [PZ, EZ]
        plastic_links.append([pz_i, ez_idx])

    model.n_plastic_links = len(plastic_links)
    model.plasticity_indices = np.array(plastic_links, dtype=np.int32)

    # Initialize Plasticity
    # Values from parameters_EZ metadata
    init_rho = ez_params.get('init_rho_EZ_to_PZ', 0.75)
    init_ampa_state = ez_params.get('init_AMPAr_EZ_to_PZ', 7.5)

    print(f"Initializing Plasticity States (Rho={init_rho}, AMPA_State={init_ampa_state})")

    for pz_i in pz_indices:
        # p[1] = Rho (Synaptic Efficacy)
        model.p[1, pz_i, ez_idx] = init_rho

        # p[2] = AMPA State (Conductance/Receptor Availability)
        model.p[2, pz_i, ez_idx] = init_ampa_state

    # 8. Configure E-Normals and Stimulation
    # --------------------------------------
    print("Configuring Stimulation with E-Normals...")

    # Load Averaged E-Normals
    enormal_path = os.path.join(os.path.dirname(__file__), '../data/roi_averaged_enormals_BIP_NE2LOC1.txt')
    # Temporary array to hold E-normals locally
    e_normals_local = np.ones(n_regions)

    if os.path.exists(enormal_path):
        try:
            df_enormals = pd.read_csv(enormal_path, sep='\t')

            # Map E-normals to model indices based on ROI name
            enormal_dict = dict(zip(df_enormals['Region'], df_enormals['Average_Enormal']))

            match_count = 0
            for i, name in enumerate(region_names):
                if name in enormal_dict:
                    val = enormal_dict[name]
                    e_normals_local[i] = val
                    match_count += 1
                else:
                    e_normals_local[i] = 0.0 # default implies no field effect

            print(f"Mapped E-normals to {match_count}/{n_regions} regions.")

        except Exception as e:
            print(f"Error reading E-normal file: {e}")
    else:
        print(f"Warning: E-normal file not found at {enormal_path}. Using default 1.0.")

    # Stimulate all regions, weighted by their E-normal
    STIM_MAGNITUDE = 1 #scaling factor useful to compare different tDCS montages.
    for i in range(n_regions):
        # Calculate weighted stimulation value
        stim_value = STIM_MAGNITUDE * e_normals_local[i]

        # Apply to model.Stim array
        model.Stim[i] = stim_value

        # Set lambda_E values for each subpopulation:
        # PYR (k_P) = 1.0
        # SST (k_I2) = 0.6
        # PV (k_I1) = 0.1
        model.k_P[i] = 1.0
        model.k_I2[i] = 0.6
        model.k_I1[i] = 0.1

        # others are 0
        model.k_I3[i] = 0.0
        model.k_I4[i] = 0.0

    print("Stimulation gains set and Stim array populated based on E-normals.")

    # 9. Run Simulation for duration seconds
    # -----------------
    duration = 30.0 # seconds
    steps = int(duration / model.dt) # to keep track of progress.
    print(f"Starting Simulation: {duration}s ({steps} steps)")

    # Store full results for EDF export
    # Using float32 to reduce memory footprint
    all_lfp_results = np.zeros((n_regions, steps), dtype=np.float32)

    # Buffers for plotting specific nodes (EZ and first PZ)
    res_ez_lfp = np.zeros(steps)
    res_pz_lfp = np.zeros(steps)
    res_calcium = np.zeros(steps)
    res_ampa = np.zeros(steps)

    # Use first PZ for tracking
    track_pz = pz_indices[0]

    for t in range(steps):
        model.Eul_Maruyama()

        # Capture full LFP state
        all_lfp_results[:, t] = model.LFPoutput

        # Track specific indices for plot
        res_ez_lfp[t] = model.LFPoutput[ez_idx]
        res_pz_lfp[t] = model.LFPoutput[track_pz]

        # Track plasticity variables for the EZ->PZ link
        # p[0] is Calcium, p[1] is AMPA State
        res_calcium[t] = model.p[0, track_pz, ez_idx]
        res_ampa[t] = model.p[1, track_pz, ez_idx]

        if t % 1024 == 0:
            print(f"Progress: {t/steps*100:.1f}%", end='\r')

    print("\nSimulation Complete.")

    # 10. Save to EDF
    # ---------------
    edf_filename = 'whole_brain_LFP.edf'
    fs = 1.0 / model.dt
    save_to_edf(all_lfp_results, region_names, fs, edf_filename)

    # 11. Plot Results (Subset)
    # -------------------------
    print("Generating Plots...")
    ds = 50 # Downsample factor for lighter plots
    time_ax = np.linspace(0, duration, steps)[::ds]

    fig, axs = plt.subplots(4, 1, figsize=(12, 14), sharex=True)

    # Plot 1: EZ LFP
    axs[0].plot(time_ax, res_ez_lfp[::ds], color='#1f77b4', lw=1)
    axs[0].set_title(f'EZ LFP ({ez_name}) - E-Norm: {e_normals_local[ez_idx]:.3f}')
    axs[0].set_ylabel('AU')
    axs[0].grid(True, alpha=0.3)

    # Plot 2: PZ LFP
    axs[1].plot(time_ax, res_pz_lfp[::ds], color='#ff7f0e', lw=1)
    axs[1].set_title(f'PZ LFP ({region_names[track_pz]}) - E-Norm: {e_normals_local[track_pz]:.3f}')
    axs[1].set_ylabel('AU')
    axs[1].grid(True, alpha=0.3)

    # Plot 3: Calcium
    axs[2].plot(time_ax, res_calcium[::ds], color='#2ca02c', lw=1.5)
    axs[2].set_title('Calcium Concentration (EZ->PZ)')
    axs[2].set_ylabel('[Ca]')
    axs[2].grid(True, alpha=0.3)

    # Plot 4: AMPA State
    axs[3].plot(time_ax, res_ampa[::ds], color='#d62728', lw=1.5)
    axs[3].set_title('AMPA State (p[2])')
    axs[3].set_ylabel('Conductance')
    axs[3].set_xlabel('Time (s)')
    axs[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('whole_brain_simulation_results.png')
    print("Saved plot to 'whole_brain_simulation_results.png'")

if __name__ == "__main__":
    run_whole_brain_simulation()
