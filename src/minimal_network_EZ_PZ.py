import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Ensure we can import from src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Use Legacy Model Class
from model_NMM_plastic_nomask import nmm_plastic
from parameters_EZ import get_EZ_params
from parameters_PZ import get_PZ_params

def run_minimal_network():
    print("--- Initializing Minimal EZ-PZ Network (2 Regions) [Legacy Model] ---")

    # 1. Setup Model for 2 Regions
    # ----------------------------
    n_regions = 2
    model = nmm_plastic()
    model.NbNMMs = n_regions

    # Initialize memory vectors (legacy calls)
    model.init_vector_param()
    model.init_vector()

    # 2. Load Parameters
    # ------------------
    ez_params = get_EZ_params()
    pz_params = get_PZ_params()

    # Define indices
    IDX_EZ = 0
    IDX_PZ = 1

    print(f"Assigning EZ parameters to Region {IDX_EZ}")
    print(f"Assigning PZ parameters to Region {IDX_PZ}")

    # Helper to safe-assign parameters
    def apply_params(target_idx, param_dict):
        for key, val in param_dict.items():
            # Check if model has this attribute and it is an array (parameter)
            if hasattr(model, key):
                attr = getattr(model, key)
                # Ensure it's a numpy array of the right size (n_regions)
                if isinstance(attr, np.ndarray) and attr.shape == (n_regions,):
                    if isinstance(val, (int, float)):
                        attr[target_idx] = val
                    elif isinstance(val, list) and len(val) > 0:
                        pass

                        # Apply EZ params to index 0
    apply_params(IDX_EZ, ez_params)

    # Apply PZ params to index 1
    apply_params(IDX_PZ, pz_params)

    # 3. Connectivity Matrix (2x2)
    # ----------------------------
    # CM[target, source]
    # We want EZ -> PZ = 1.0
    # Source = EZ (0), Target = PZ (1)

    model.CM_P_P[:] = 0.0 # Reset
    model.CM_P_P[IDX_PZ, IDX_EZ] = 1.0

    print("Connectivity Matrix set manually:")
    print(model.CM_P_P)

    # 4. Setup Plasticity (EZ -> PZ only)
    # -----------------------------------
    # We rely on the mask logic or simply
    # initialize only the relevant link and let the others evolve (or stay 0 if inputs are 0).

    print("Configuring Plasticity (EZ -> PZ)...")

    # Initialize Specific Plastic State Variables from EZ params
    init_rho = ez_params.get('init_rho_EZ_to_PZ', 0.75)
    init_ampar = ez_params.get('init_AMPAr_EZ_to_PZ', 7.5)

    print(f"Initializing Plasticity: Rho={init_rho}, AMPAr={init_ampar}")

    # p[1] is Synaptic Efficacy (Rho)
    model.p[1, IDX_PZ, IDX_EZ] = init_rho

    # p[2] is AMPA State (init_AMPAr) - Treated as 'position' in 2nd order ODE
    model.p[2, IDX_PZ, IDX_EZ] = init_ampar

    # 5. Run Simulation Loop
    # ----------------------
    duration = 30.0 # seconds
    steps = int(duration / model.dt)
    print(f"Running simulation for {duration}s ({steps} steps)...")

    results_ez = np.zeros(steps)
    results_pz = np.zeros(steps)

    for t in range(steps):
        model.Eul_Maruyama()

        results_ez[t] = model.LFPoutput[IDX_EZ]
        results_pz[t] = model.LFPoutput[IDX_PZ]

        if t % 1024 == 0:
            print(f"Step {t}/{steps}", end='\r')

    print("\nSimulation Finished.")

    # 6. Plot
    # -------
    time_axis = np.linspace(0, duration, steps)

    plt.figure(figsize=(10, 6))
    plt.plot(time_axis, results_ez, label='EZ (Region 0)')
    plt.plot(time_axis, results_pz, label='PZ (Region 1)')

    plt.title('Minimal Network PoC [Legacy Model]: EZ driving PZ')
    plt.xlabel('Time (s)')
    plt.ylabel('LFP (mV)')
    plt.legend()
    plt.grid(True)

    outfile = 'minimal_network_result.png'
    plt.savefig(outfile)
    print(f"Plot saved to '{outfile}'")


if __name__ == "__main__":
    run_minimal_network()
