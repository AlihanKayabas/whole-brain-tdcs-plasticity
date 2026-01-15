def get_PZ_params():
    """
    Returns the parameter dictionary for the Propagation Zone (PZ).
    """
    p = {}

    # --- Synaptic Gains (mV) ---
    p['A'] = 3.8
    p['B'] = 50.0
    p['BB'] = 40.0
    p['G'] = 30.0

    # --- Time Constants (1/sec) ---
    p['a1'] = 100.0
    p['a2'] = 100.0
    p['b1'] = 50.0
    p['b2'] = 50.0
    p['bb1'] = 50.0
    p['bb2'] = 50.0
    p['g1'] = 400.0
    p['g2'] = 400.0

    # --- Connectivity Constants ---
    p['CPP'] = 0.0
    p['CP1P'] = 108.0
    p['CI1P'] = 50.0
    p['CI2aP'] = 10.0
    p['CI2bP'] = 11.0
    p['CI4P'] = 0.0
    p['CPP1'] = 135.0
    p['CPI1'] = 40.0
    p['CI1I1'] = 0.0
    p['CI1bI1'] = 0.0
    p['CI2I1'] = 15.0
    p['CI4I1'] = 0.0
    p['CPI1b'] = 0.0
    p['CI1I1b'] = 0.0
    p['CPI2'] = 40.0
    p['CI3I2'] = 0.0
    p['CI4I2'] = 0.0
    p['CPI3'] = 0.0
    p['CI2I3'] = 0.0
    p['CI4I3'] = 0.0
    p['CI2I4'] = 0.0
    p['CI4I4'] = 0.0
    p['CI4bI4'] = 0.0
    p['CI4I4b'] = 0.0

    # --- Sigmoid Parameters ---
    p['Pv0'] = 6.0
    p['I1v0'] = 6.0
    p['I2v0'] = 6.0
    p['I3v0'] = 6.0
    p['I4v0'] = 6.0
    p['Pe0'] = 5.0
    p['I1e0'] = 5.0
    p['I2e0'] = 5.0
    p['I3e0'] = 5.0
    p['I4e0'] = 5.0
    p['Pr0'] = 0.56
    p['I1r0'] = 0.56
    p['I2r0'] = 0.56
    p['I3r0'] = 0.56
    p['I4r0'] = 0.56
    p['Pv0_UD'] = 6.0
    p['Pe0_UD'] = 5.0
    p['Pr0_UD'] = 0.56

    # --- Input / Stimulation ---
    p['Pm'] = 100.0
    p['I1m'] = 0.0
    p['I2m'] = 0.0
    p['I3m'] = 0.0
    p['I4m'] = 0.0
    p['Ps'] = 1.8
    p['I1s'] = 0.0
    p['I2s'] = 0.0
    p['I3s'] = 0.0
    p['I4s'] = 0.0
    p['Pcoef'] = 1.0
    p['I1coef'] = 0.0
    p['I2coef'] = 0.0
    p['I3coef'] = 0.0
    p['I4coef'] = 0.0
    p['k_P'] = -0.0046352280172104405
    p['k_Pp'] = -0.0
    p['k_I1'] = -0.0
    p['k_I2'] = -0.002781136810326264
    p['k_I3'] = -0.0
    p['k_I4'] = -0.0
    p['stim_sigmoid_rate'] = 0.25

    # --- Plasticity & Biophysics ---
    p['tau_d'] = 0.6
    p['tau_f'] = 0.05
    p['Use'] = 0.4
    p['Use_max'] = 0.8
    p['tau_Use'] = 100.0
    p['A_ampa'] = 5.0
    p['aa_ampa'] = 200.0
    p['A_nmda'] = 4.0
    p['sigm_NMDApost_Ca_factor'] = 200.0
    p['tauCa'] = 0.055

    # Omega parameters (Graupner-Brunel)
    p['omega_gamma_p'] = 5.0
    p['omega_beta_p'] = 80.0
    p['omega_alpha_p'] = 0.55
    p['omega_gamma_d'] = 1.0
    p['omega_beta_d'] = 80.0
    p['omega_alpha_d'] = 0.095
    p['tauP'] = 50.0

    # NMDA Ext / Other Kinetics
    p['A_nmda_ext'] = 1.0
    p['aa_nmda_ext'] = 25.0
    p['aa_nmda'] = 50.0
    p['tau_K'] = 10.0
    p['sigm_NMDApost_amplitude'] = 1.0
    p['sigm_NMDApost_slope'] = 1.0
    p['sigm_NMDApost_threshold'] = 5.0
    p['g_kdrive'] = 1.0

    # Conductance dynamics
    p['campad'] = 2.0
    p['campap'] = 10.0
    p['tau_campa'] = 100.0
    p['cnmda'] = 2.0
    p['decay_rate'] = 0.05
    p['V0_init'] = 6.0

    # Metadata

    #PZ list, for NE2LOC1 scenario:
    p['Name'] = [
        'l-C-PARO',
        'l-C-IP-7',
        'l-PARO-1',
        'l-C-PREC-1',
        'l-C-PREC-5',
        'l-C-PREC-7',
        'l-C-RMF-2',
        'l-C-SF-10',
        'l-C-SP-2',
        'l-C-INS-1'
    ]

    return p
