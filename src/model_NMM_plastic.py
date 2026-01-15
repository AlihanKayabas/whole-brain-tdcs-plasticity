import numpy as np
from numba.experimental import jitclass
from numba import boolean, int32, float64, uint8

spec = [('A' ,float64[:]),
        ('B' ,float64[:]),
        ('BB' ,float64[:]),
        ('G' ,float64[:]),
        ('D' ,float64[:]),
        ('R' ,float64[:]),
        ('a1' ,float64[:]),
        ('a2' ,float64[:]),
        ('b1',float64[:]),
        ('b2',float64[:]),
        ('bb1',float64[:]),
        ('bb2',float64[:]),
        ('g1' ,float64[:]),
        ('g2' ,float64[:]),
        ('d1' ,float64[:]),
        ('d2' ,float64[:]),
        ('r1' ,float64[:]),
        ('r2' ,float64[:]),
        ('CPP' ,float64[:]),
        ('CP1P' ,float64[:]),
        ('CI1P' ,float64[:]),
        ('CI2aP' ,float64[:]),
        ('CI2bP' ,float64[:]),
        ('CI4P' ,float64[:]),
        ('CPP1' ,float64[:]),
        ('CPI1' ,float64[:]),
        ('CI1I1' ,float64[:]),
        ('CI1bI1' ,float64[:]),
        ('CI2I1' ,float64[:]),
        ('CI4I1' ,float64[:]),
        ('CPI1b' ,float64[:]),
        ('CI1I1b' ,float64[:]),
        ('CPI2' ,float64[:]),
        ('CI3I2' ,float64[:]),
        ('CI4I2' ,float64[:]),
        ('CPI3' ,float64[:]),
        ('CI2I3' ,float64[:]),
        ('CI4I3' ,float64[:]),
        ('CI2I4' ,float64[:]),
        ('CI4I4' ,float64[:]),
        ('CI4bI4' ,float64[:]),
        ('CI4I4b' ,float64[:]),
        ('Pv0' ,float64[:]),
        ('I1v0' ,float64[:]),
        ('I2v0' ,float64[:]),
        ('I3v0' ,float64[:]),
        ('I4v0' ,float64[:]),
        ('Pe0' ,float64[:]),
        ('I1e0' ,float64[:]),
        ('I2e0' ,float64[:]),
        ('I3e0' ,float64[:]),
        ('I4e0' ,float64[:]),
        ('Pr0' ,float64[:]),
        ('I1r0' ,float64[:]),
        ('I2r0' ,float64[:]),
        ('I3r0' ,float64[:]),
        ('I4r0' ,float64[:]),
        ('Pm' ,float64[:]),
        ('I1m' ,float64[:]),
        ('I2m' ,float64[:]),
        ('I3m' ,float64[:]),
        ('I4m' ,float64[:]),
        ('Ps' ,float64[:]),
        ('I1s' ,float64[:]),
        ('I2s' ,float64[:]),
        ('I3s' ,float64[:]),
        ('I4s' ,float64[:]),
        ('Pcoef' ,float64[:]),
        ('I1coef' ,float64[:]),
        ('I2coef' ,float64[:]),
        ('I3coef' ,float64[:]),
        ('I4coef' ,float64[:]),
        ('dt',float64),
        ('NbODEs',int32),
        ('NbODEsPlast',int32),
        ('NbNMMs',int32),
        ('dydx' ,float64[:,:]),
        ('dydx1' ,float64[:,:]),
        ('dydx2' ,float64[:,:]),
        ('dydx3' ,float64[:,:]),
        ('y'    ,float64[:,:]),
        ('yt'  ,float64[:,:]),

        # Plasticity
        ('dydP', float64[:, :, :]),
        ('p', float64[:, :, :]),
        ('diagonal_mask', float64[:, :]),

        # Optimization Fields
        ('plasticity_indices', int32[:, :]),
        ('n_plastic_links', int32),

        ('tau_d', float64[:]),
        ('tau_f', float64[:]),
        ('Use', float64[:]),
        ('Use_max', float64[:]),
        ('tau_Use', float64[:]),
        ('A_ampa', float64[:]),
        ('aa_ampa', float64[:]),
        ('A_nmda', float64[:]),
        ('sigm_NMDApost_Ca_factor', float64[:]),
        ('tauCa', float64[:]),
        ('omega_gamma_p', float64[:]),
        ('omega_beta_p', float64[:]),
        ('omega_alpha_p', float64[:]),
        ('omega_gamma_d', float64[:]),
        ('omega_beta_d', float64[:]),
        ('omega_alpha_d', float64[:]),
        ('tauP', float64[:]),
        ('A_nmda_ext', float64[:]),
        ('aa_nmda_ext', float64[:]),
        ('aa_nmda', float64[:]),
        ('tau_K', float64[:]),
        ('sigm_NMDApost_amplitude', float64[:]),
        ('sigm_NMDApost_slope', float64[:]),
        ('sigm_NMDApost_threshold', float64[:]),
        ('g_kdrive', float64[:]),
        ('campad', float64[:]),
        ('campap', float64[:]),
        ('tau_campa', float64[:]),
        ('cnmda', float64[:]),

        ('bruitP' ,float64[:]),
        ('bruitI1' ,float64[:]),
        ('bruitI2' ,float64[:]),
        ('bruitI3' ,float64[:]),
        ('bruitI4' ,float64[:]),

        ('EEGoutput' ,float64[:]),
        ('LFPoutput' ,float64[:]),
        ('OutSigmoidEXC' ,float64[:]),
        ('OutSigmoidI1' ,float64[:]),
        ('OutSigmoidI1b' ,float64[:]),
        ('OutSigmoidI2' ,float64[:]),
        ('OutSigmoidEXC1' ,float64[:]),
        ('OutSigmoidI3' ,float64[:]),
        ('OutSigmoidI4' ,float64[:]),
        ('OutSigmoidI4b' ,float64[:]),

        ('PPSEXC' ,float64[:]),
        ('PPSI1' ,float64[:]),
        ('PPSI1b' ,float64[:]),
        ('PPSI2apical' ,float64[:]),
        ('PPSI2basal' ,float64[:]),
        ('PPSEXC1' ,float64[:]),
        ('PPSI3' ,float64[:]),
        ('PPSI4' ,float64[:]),
        ('PPSI4b' ,float64[:]),
        ('PPSExtI_P_P', float64[:]),

        ('ExtI_P_P'  ,float64[:]),
        ('ExtI_P_I1' ,float64[:]),
        ('ExtI_P_I2' ,float64[:]),
        ('ExtI_P_I3' ,float64[:]),
        ('ExtI_P_I4' ,float64[:]),
        ('ExtI_I2_I2' ,float64[:]),

        ('k_P',float64[:]),
        ('k_Pp',float64[:]),
        ('k_I1',float64[:]),
        ('k_I2',float64[:]),
        ('k_I3',float64[:]),
        ('k_I4',float64[:]),
        ('Stim',float64[:]),
        ('Pre_Post',int32),
        ('CM_P_P' ,float64[:,:]),
        ('CM_P_I1' ,float64[:,:]),
        ('CM_P_I2' ,float64[:,:]),
        ('CM_P_I3' ,float64[:,:]),
        ('CM_P_I4' ,float64[:,:]),
        ('CM_I2_I2' ,float64[:,:]),
        ('DelayMat' ,float64[:,:]),
        ('Delay_in_index_Mat' ,int32[:,:]),
        ('EmptyMat' ,boolean[:]),
        ('Nb_NMM_m1' ,int32),
        ('history_pulseP' ,float64[:,:]),
        ('history_pulseI2' ,float64[:,:]),
        ('max_delay_index' ,int32),
        ('pulseP_delay_Mat' ,float64[:,:]),
        ('pulseI2_delay_Mat' ,float64[:,:]),

        ('sigma',float64),
        ('sources',float64[:,:]),
        ('position',float64[:]),
        ('distance',float64[:]),

        ('Thresh', float64[:]),
        ('Pv0_UD', float64[:]),
        ('Pe0_UD', float64[:]),
        ('Pr0_UD', float64[:]),
        ('Pv0_used', float64[:]),
        ('Pe0_used', float64[:]),
        ('Pr0_used', float64[:]),
        ('V0_init', float64[:]),
        ('p2', float64[:, :]),
        ('dydP_acute', float64[:, :]),
        ('stim_sigmoid_rate', float64[:])
        ]

@jitclass(spec)
class nmm_plastic:
    def __init__(self,):
        self.dt = 1./1024.
        self.NbODEs = 16
        self.NbODEsPlast = 7

        self.NbNMMs = 1
        self.Nb_NMM_m1=1
        self.init_vector()
        self.init_vector_param()
        self.Pre_Post = True

    def init_vector(self):
        self.dydx = np.zeros((self.NbODEs,self.NbNMMs))
        self.dydx1 = np.zeros((self.NbODEs,self.NbNMMs))
        self.dydx2 = np.zeros((self.NbODEs,self.NbNMMs))
        self.dydx3 = np.zeros((self.NbODEs,self.NbNMMs))
        self.y    =np.zeros((self.NbODEs,self.NbNMMs))
        self.yt    =np.zeros((self.NbODEs,self.NbNMMs))
        self.ExtI_P_P      = np.zeros((self.NbNMMs))
        self.ExtI_P_I1     = np.zeros((self.NbNMMs))
        self.ExtI_P_I2     = np.zeros((self.NbNMMs))
        self.ExtI_P_I3     = np.zeros((self.NbNMMs))
        self.ExtI_P_I4     = np.zeros((self.NbNMMs))
        self.ExtI_I2_I2     = np.zeros((self.NbNMMs))
        self.Stim = np.zeros((self.NbNMMs))
        self.bruitP = np.zeros((self.NbNMMs))
        self.bruitI1 = np.zeros((self.NbNMMs))
        self.bruitI2 = np.zeros((self.NbNMMs))
        self.bruitI3 = np.zeros((self.NbNMMs))
        self.bruitI4 = np.zeros((self.NbNMMs))

        self.EEGoutput = np.zeros((self.NbNMMs))
        self.LFPoutput = np.zeros((self.NbNMMs))

        self.OutSigmoidEXC = np.zeros((self.NbNMMs))
        self.OutSigmoidI1 = np.zeros((self.NbNMMs))
        self.OutSigmoidI1b = np.zeros((self.NbNMMs))
        self.OutSigmoidI2 = np.zeros((self.NbNMMs))
        self.OutSigmoidEXC1 = np.zeros((self.NbNMMs))
        self.OutSigmoidI3 = np.zeros((self.NbNMMs))
        self.OutSigmoidI4 = np.zeros((self.NbNMMs))
        self.OutSigmoidI4b = np.zeros((self.NbNMMs))
        self.PPSEXC = np.zeros((self.NbNMMs))
        self.PPSI1 = np.zeros((self.NbNMMs))
        self.PPSI1b = np.zeros((self.NbNMMs))
        self.PPSI3 = np.zeros((self.NbNMMs))
        self.PPSI4 = np.zeros((self.NbNMMs))
        self.PPSI2apical = np.zeros((self.NbNMMs))
        self.PPSI2basal = np.zeros((self.NbNMMs))
        self.PPSExtI_P_P = np.zeros((self.NbNMMs))

        self.PPSEXC1 = np.zeros((self.NbNMMs))
        self.PPSI3 = np.zeros((self.NbNMMs))
        self.PPSI4 = np.zeros((self.NbNMMs))
        self.PPSI4b = np.zeros((self.NbNMMs))

        self.CM_P_P = np.zeros((self.NbNMMs, self.NbNMMs))
        self.CM_P_I1 = np.zeros((self.NbNMMs, self.NbNMMs))
        self.CM_P_I2 = np.zeros((self.NbNMMs, self.NbNMMs))
        self.CM_P_I3 = np.zeros((self.NbNMMs, self.NbNMMs))
        self.CM_P_I4 = np.zeros((self.NbNMMs, self.NbNMMs))
        self.CM_I2_I2 = np.zeros((self.NbNMMs, self.NbNMMs))
        self.DelayMat = np.zeros((self.NbNMMs, self.NbNMMs))
        self.Delay_in_index_Mat = np.zeros((self.NbNMMs, self.NbNMMs), dtype=np.int32)
        self.EmptyMat = np.full((6), True)
        self.history_pulseP = np.zeros((self.NbNMMs, 1))
        self.history_pulseI2 = np.zeros((self.NbNMMs, 1))
        self.max_delay_index = 0
        self.pulseP_delay_Mat = np.zeros((self.NbNMMs, self.NbNMMs))
        self.pulseI2_delay_Mat = np.zeros((self.NbNMMs, self.NbNMMs))

        # Plasticity State Matrices
        self.dydP = np.zeros((self.NbODEsPlast,self.NbNMMs,self.NbNMMs))
        self.p    = np.zeros((self.NbODEsPlast,self.NbNMMs,self.NbNMMs))
        self.diagonal_mask = np.ones((self.NbNMMs, self.NbNMMs)) - np.eye(self.NbNMMs)
        self.dydP_acute = np.zeros((1, self.NbNMMs))
        self.p2 = np.ones((1, self.NbNMMs)) * 6

        # Optimization: Plasticity Indices
        self.plasticity_indices = np.zeros((1, 2), dtype=np.int32)
        self.n_plastic_links = 0

        self.sources = np.array([[0., 2.],[0., 0.5],[0., 0.2]], dtype=float64)
        self.position = np.array([10., 2], dtype=np.float64)
        self.sigma = 4e-4
        self.compute_distance()

    def init_vector_param(self):
        # Initialization of all parameters values
        self.A = np.ones((self.NbNMMs)) * 5.
        self.B = np.ones((self.NbNMMs)) * 50.
        self.BB = np.ones((self.NbNMMs)) * 25.
        self.G = np.ones((self.NbNMMs)) * 40.
        self.D = np.ones((self.NbNMMs)) * 0.
        self.R = np.ones((self.NbNMMs)) * 0.
        self.a1 = np.ones((self.NbNMMs)) * 100.
        self.a2 = np.ones((self.NbNMMs)) * 100.
        self.b1 = np.ones((self.NbNMMs)) * 30.
        self.b2 = np.ones((self.NbNMMs)) * 30.
        self.bb1 = np.ones((self.NbNMMs)) * 30.
        self.bb2 = np.ones((self.NbNMMs)) * 30.
        self.g1 = np.ones((self.NbNMMs)) * 340.
        self.g2 = np.ones((self.NbNMMs)) * 340.
        self.d1 = np.ones((self.NbNMMs)) * 0.
        self.d2 = np.ones((self.NbNMMs)) * 0.
        self.r1 = np.ones((self.NbNMMs)) * 0.
        self.r2 = np.ones((self.NbNMMs)) * 0.

        self.CPP = np.ones((self.NbNMMs)) * 0.
        self.CP1P = np.ones((self.NbNMMs)) * 108.
        self.CI1P = np.ones((self.NbNMMs)) * 50.
        self.CI2aP = np.ones((self.NbNMMs)) * 15.
        self.CI2bP = np.ones((self.NbNMMs)) * 10.
        self.CI4P = np.ones((self.NbNMMs)) * 0.
        self.CPP1 = np.ones((self.NbNMMs)) * 135.
        self.CPI1 = np.ones((self.NbNMMs)) * 50.
        self.CI1I1 = np.ones((self.NbNMMs)) * 0.
        self.CI1bI1 = np.ones((self.NbNMMs)) * 0.
        self.CI2I1 = np.ones((self.NbNMMs)) * 20.
        self.CI4I1 = np.ones((self.NbNMMs)) * 0.
        self.CPI1b = np.ones((self.NbNMMs)) * 0.
        self.CI1I1b = np.ones((self.NbNMMs)) * 0.
        self.CPI2 = np.ones((self.NbNMMs)) * 40.
        self.CI3I2 = np.ones((self.NbNMMs)) * 0.
        self.CI4I2 = np.ones((self.NbNMMs)) * 0.
        self.CPI3 = np.ones((self.NbNMMs)) * 0.
        self.CI2I3 = np.ones((self.NbNMMs)) * 0.
        self.CI4I3 = np.ones((self.NbNMMs)) * 0.
        self.CI2I4 = np.ones((self.NbNMMs)) * 0.
        self.CI4I4 = np.ones((self.NbNMMs)) * 0.
        self.CI4bI4 = np.ones((self.NbNMMs)) * 0.
        self.CI4I4b = np.ones((self.NbNMMs)) * 0.

        self.Pv0 = np.ones((self.NbNMMs)) * 6.
        self.I1v0 = np.ones((self.NbNMMs)) * 6.
        self.I2v0 = np.ones((self.NbNMMs)) * 6.
        self.I3v0 = np.ones((self.NbNMMs)) * 6.
        self.I4v0 = np.ones((self.NbNMMs)) * 6.
        self.Pe0 = np.ones((self.NbNMMs)) * 5.
        self.I1e0 = np.ones((self.NbNMMs)) * 5.
        self.I2e0 = np.ones((self.NbNMMs)) * 5.
        self.I3e0 = np.ones((self.NbNMMs)) * 5.
        self.I4e0 = np.ones((self.NbNMMs)) * 5.
        self.Pr0 = np.ones((self.NbNMMs)) * 0.56
        self.I1r0 = np.ones((self.NbNMMs)) * 0.56
        self.I2r0 = np.ones((self.NbNMMs)) * 0.56
        self.I3r0 = np.ones((self.NbNMMs)) * 0.56
        self.I4r0 = np.ones((self.NbNMMs)) * 0.56
        self.Pm = np.ones((self.NbNMMs)) * 100
        self.I1m = np.ones((self.NbNMMs)) * 0.
        self.I2m = np.ones((self.NbNMMs)) * 0.
        self.I3m = np.ones((self.NbNMMs)) * 0.
        self.I4m = np.ones((self.NbNMMs)) * 0.
        self.Ps = np.ones((self.NbNMMs)) * 3.0
        self.I1s = np.ones((self.NbNMMs)) * 0.
        self.I2s = np.ones((self.NbNMMs)) * 0.
        self.I3s = np.ones((self.NbNMMs)) * 0.
        self.I4s = np.ones((self.NbNMMs)) * 0.
        self.Pcoef = np.ones((self.NbNMMs)) * 1.
        self.I1coef = np.ones((self.NbNMMs)) * 0.
        self.I2coef = np.ones((self.NbNMMs)) * 0.
        self.I3coef = np.ones((self.NbNMMs)) * 0.
        self.I4coef = np.ones((self.NbNMMs)) * 0.

        self.k_P = np.ones((self.NbNMMs)) * 1.
        self.k_Pp = np.ones((self.NbNMMs)) * 1.
        self.k_I1 = np.ones((self.NbNMMs)) * 1.
        self.k_I2 = np.ones((self.NbNMMs)) * 1.
        self.k_I3 = np.ones((self.NbNMMs)) * 1.
        self.k_I4 = np.ones((self.NbNMMs)) * 1.

        self.Thresh = np.ones((self.NbNMMs)) * 6.
        self.Pv0_UD = np.ones((self.NbNMMs)) * 6.
        self.Pe0_UD = np.ones((self.NbNMMs)) * 5.
        self.Pr0_UD = np.ones((self.NbNMMs)) * 0.56
        self.Pv0_used = np.ones((self.NbNMMs)) * 6.
        self.Pe0_used = np.ones((self.NbNMMs)) * 5.
        self.Pr0_used = np.ones((self.NbNMMs)) * 0.56

        self.tau_d = np.ones((self.NbNMMs)) * 0.2
        self.tau_f = np.ones((self.NbNMMs)) * 0.050
        self.Use = np.ones((self.NbNMMs)) * 0.4
        self.Use_max = np.ones((self.NbNMMs)) * 0.8
        self.tau_Use = np.ones((self.NbNMMs)) * 100
        self.A_ampa = np.ones((self.NbNMMs)) * 5
        self.aa_ampa = np.ones((self.NbNMMs)) * 200
        self.A_nmda = np.ones((self.NbNMMs)) * 1
        self.sigm_NMDApost_Ca_factor = np.ones((self.NbNMMs)) * 200
        self.tauCa = np.ones((self.NbNMMs)) * 0.05
        self.omega_gamma_p = np.ones((self.NbNMMs)) * 5
        self.omega_beta_p = np.ones((self.NbNMMs)) * 80
        self.omega_alpha_p = np.ones((self.NbNMMs)) * 0.4
        self.omega_gamma_d = np.ones((self.NbNMMs)) * 1
        self.omega_beta_d = np.ones((self.NbNMMs)) * 80
        self.omega_alpha_d = np.ones((self.NbNMMs)) * 0.1
        self.tauP = np.ones((self.NbNMMs)) * 50
        self.A_nmda_ext = np.ones((self.NbNMMs)) * 1
        self.aa_nmda_ext = np.ones((self.NbNMMs)) * 25
        self.aa_nmda = np.ones((self.NbNMMs)) * 50
        self.tau_K = np.ones((self.NbNMMs)) * 10
        self.sigm_NMDApost_amplitude = np.ones((self.NbNMMs)) * 1
        self.sigm_NMDApost_slope = np.ones((self.NbNMMs)) * 1
        self.sigm_NMDApost_threshold = np.ones((self.NbNMMs)) * 5
        self.g_kdrive = np.ones((self.NbNMMs)) * 1
        self.campad = np.ones((self.NbNMMs)) * 1
        self.campap = np.ones((self.NbNMMs)) * 2
        self.tau_campa = np.ones((self.NbNMMs)) * 100
        self.cnmda = np.ones((self.NbNMMs)) * 1
        self.V0_init = np.ones((self.NbNMMs)) * 6
        self.stim_sigmoid_rate = np.ones((self.NbNMMs)) * 0.25 # Initialized correctly now

    # Scalar helpers for loop
    def func_omega_p(self, v, i):
        return self.omega_gamma_p[i]/(1+np.exp(-self.omega_beta_p[i]*(v - self.omega_alpha_p[i])))

    def func_omega_d(self, v, i):
        return self.omega_gamma_d[i]/(1+np.exp(-self.omega_beta_d[i]*(v - self.omega_alpha_d[i])))

    def sigmNMDApost(self, v, i):
        return self.sigm_NMDApost_amplitude[i]/(1 + np.exp(-self.sigm_NMDApost_slope[i]*(v - self.sigm_NMDApost_threshold[i])))

        # Vector helper for general use
    def sigmNMDApost_vec(self, v):
        return self.sigm_NMDApost_amplitude/(1 + np.exp(-self.sigm_NMDApost_slope*(v - self.sigm_NMDApost_threshold)))

    def compute_distance(self):
        self.distance = np.sqrt((self.sources[:, 0] - self.position[0]) ** 2 + (self.sources[:, 1]  - self.position[1]) ** 2)

    def random_seeded(self,seed):
        np.random.seed(int(seed))

    def sigmP(self,v):
        return  self.Pe0_used/(1+np.exp( self.Pr0_used*(self.p2[0]-v)))

    def sigmI1(self,v):
        return  self.I1e0/(1+np.exp( self.I1r0*(self.I1v0-v)))

    def sigmI2(self,v):
        return  self.I2e0/(1+np.exp( self.I2r0*(self.I2v0-v)))

    def noiseP(self):
        for i in range(self.NbNMMs):
            self.bruitP[i] = self.Pcoef[i] * np.random.normal(self.Pm[i],self.Ps[i])

    def PSP(self, y0, y1, y2, V, v1, v2):
        return (V * v1 * y0 - 2 * v1 * y2 - v1 ** 2 * y1)

    def PSP_scalar(self, y0, y1, y2, V, v1, v2):
        return (V * v1 * y0 - 2 * v1 * y2 - v1 ** 2 * y1)

    def Eul_Maruyama(self):
        self.bruitP  = self.Pcoef  * self.Pm
        self.dydx1, self.dydP = self.derivT()
        self.y += (self.dydx1 * self.dt)
        self.p += (self.dydP * self.dt)
        self.p *= self.diagonal_mask
        self.y[11] += self.A * self.a1 * self.Pcoef * self.Ps * np.random.normal(loc=0.0, scale=np.sqrt(self.dt),size=self.NbNMMs)
        self.compute_LFP()

    def Eul_Time_Maruyama(self,N, stim):
        self.init_vector()
        lfp = np.zeros((N,self.NbNMMs))

        if np.sum(np.abs(stim)) == 0:
            for k in range(N):
                self.Eul_Maruyama()
                lfp[k,:]= self.EEGoutput
        else:
            for k in range(N):
                self.Stim = stim[:,k]
                self.Eul_Maruyama()
                lfp[k,:]= self.EEGoutput
        return lfp

    def derivT(self , ):
        self.EEGoutput = (self.CP1P * self.y[8,:]
                          + self.CPP * self.y[0,:]
                          - self.CI1P * self.y[2,:]
                          - self.CI2bP * self.y[6,:]
                          - self.CI2aP * self.y[14, :]
                          + self.y[10,:]
                          + np.sum(self.p[2, :, :] * self.p[3, :, :] + self.campad * (self.p[5, :, :].T * self.sigmNMDApost_vec(self.EEGoutput)).T, axis=1))

        self.OutSigmoidEXC = self.sigmP(self.k_P * self.Stim * self.Pre_Post+ self.EEGoutput)

        self.dydx[0,:] = self.y[1,:]
        self.dydx[1,:] =  self.PSP(self.k_P * self.Stim * (not self.Pre_Post)
                                   + self.OutSigmoidEXC,self.y[0,:],self.y[1,:], self.A, self.a1, self.a2)

        self.OutSigmoidI1 = self.sigmI1(self.k_I1 * self.Stim + self.k_I1 * self.Stim * self.Pre_Post
                                        + self.CPI1 * self.y[0,:]
                                        - self.CI1bI1 * self.y[4,:]
                                        - self.CI1I1 * self.y[2,:]
                                        - self.CI2I1 * self.y[6,:])

        self.dydx[2,:] = self.y[3,:]
        self.dydx[3,:] =  self.PSP(self.k_I1 * self.Stim * (not self.Pre_Post)
                                   + self.OutSigmoidI1 ,self.y[2,:],self.y[3,:], self.G, self.g1, self.g2)

        self.OutSigmoidI1b = self.sigmI1(   self.CPI1b*self.y[0,:]
                                            - self.CI1I1b*self.y[2,:])
        self.dydx[4,:] = self.y[5,:]
        self.dydx[5,:] =  self.PSP(self.OutSigmoidI1b,self.y[4,:],self.y[5,:], self.G, self.g1, self.g2)

        self.OutSigmoidI2 = self.sigmI2(  self.k_I2 * self.Stim * self.Pre_Post
                                          + self.CPI2*self.y[0,:])
        self.dydx[6,:] = self.y[7,:]
        self.dydx[7,:] =  self.PSP(self.k_I2 * self.Stim * (not self.Pre_Post)
                                   + self.OutSigmoidI2 ,self.y[6,:],self.y[7,:], self.B, self.b1, self.b2)

        self.OutSigmoidEXC1 = self.sigmP(self.k_Pp * self.Stim * self.Pre_Post + self.CPP1 * self.y[0, :])
        self.dydx[8,:]  =  self.y[9,:]
        self.dydx[9,:]  =   self.PSP(self.k_Pp * self.Stim * (not self.Pre_Post)
                                     + self.OutSigmoidEXC1,self.y[8,:],self.y[9,:], self.A, self.a1, self.a2)

        self.dydx[10,:] = self.y[11,:]
        self.dydx[11,:] = self.PSP(self.bruitP,self.y[10,:],self.y[11,:], self.A, self.a1, self.a2)

        self.dydx[12,:] = self.y[13,:]
        self.dydx[13,:] = self.PSP(self.ExtI_P_P,self.y[12,:],self.y[13,:],self.A, self.a1, self.a2)

        self.dydx[14,:]  =  self.y[15,:]
        self.dydx[15,:]  =   self.PSP(self.k_I2 * self.Stim * (not self.Pre_Post)
                                      + self.OutSigmoidI2, self.y[14,:],self.y[15,:], self.BB, self.bb1, self.bb2)

        # ----------------------------------------------------------------------
        # PLASTICITY LOOP
        # Updates only for EZ->PZ
        # ----------------------------------------------------------------------
        self.dydP[:] = 0.0
        #
        for k in range(self.n_plastic_links):
            post = self.plasticity_indices[k, 0]
            pre = self.plasticity_indices[k, 1]

            # 1. Calcium (p[0])

            nmda_bound = self.p[5, post, pre]
            sigm_eeg = self.sigmNMDApost(self.EEGoutput[post], post)

            drive_ca = nmda_bound * self.campad[post] * self.sigm_NMDApost_Ca_factor[post] * sigm_eeg
            decay_ca = self.p[0, post, pre] / self.tauCa[post]

            self.dydP[0, post, pre] = drive_ca - decay_ca

            # 2. Synaptic Efficacy / Rho (p[1])
            ca = self.p[0, post, pre]
            rho = self.p[1, post, pre]

            term_noise = -1.0 * rho * (1.0 - rho) * (0.5 - rho)
            term_pot = (1.0 - rho) * self.func_omega_p(ca, post)
            term_dep = rho * self.func_omega_d(ca, post)

            self.dydP[1, post, pre] = (term_noise + term_pot - term_dep) / self.tauP[post]

            # 3. Conductance Dynamics (p[2])
            cond = self.p[2, post, pre]
            target = self.campad[post] + rho * (self.campap[post] - self.campad[post])
            self.dydP[2, post, pre] = (target - cond) / self.tau_campa[post]

            # 4. Synaptic Inputs (p[3], p[4], p[5], p[6]) Firing_Pre * Weight_post_pre
            input_drive = self.OutSigmoidEXC[pre] * self.CM_P_P[post, pre]

            # AMPA (p[4] is derivative of p[3])
            self.dydP[4, post, pre] = self.PSP_scalar(input_drive, self.p[3, post, pre], self.p[4, post, pre],
                                                      self.A_ampa[post], self.aa_ampa[post], self.aa_ampa[post])
            self.dydP[3, post, pre] = self.p[4, post, pre]

            # NMDA (p[6] is derivative of p[5])
            self.dydP[6, post, pre] = self.PSP_scalar(input_drive, self.p[5, post, pre], self.p[6, post, pre],
                                                      self.A_nmda[post], self.aa_nmda[post], self.aa_nmda[post])
            self.dydP[5, post, pre] = self.p[6, post, pre]

        # ----------------------------------------------------------------------

        self.PPSEXC = self.CPP*self.y[0,:]
        self.PPSI1 = self.CI1P*self.y[2,:]
        self.PPSI1b = self.y[4,:]
        self.PPSI2basal = self.CI2bP * self.y[6,:]
        self.PPSEXC1 = self.CP1P*self.y[8,:]
        self.PPSI2apical = self.CI2aP*self.y[14,:]
        self.PPSExtI_P_P = self.y[12,:] + self.y[10,:]

        return self.dydx+0.,self.dydP+0.

    def NonNullMat(self):
        if np.max(self.CM_P_P)==0.:
            self.EmptyMat[0]=False
        else:
            self.EmptyMat[0]=True

    def apply_connectivity_Mat(self):
        if self.EmptyMat[0]:
            self.ExtI_P_P	= np.dot(self.CM_P_P  / self.Nb_NMM_m1   ,self.OutSigmoidEXC)

    def compute_LFP(self):
        self.LFPoutput = (1 / (4 * np.pi * self.sigma)) * \
                         (self.PPSEXC1 * (1 / self.distance[0] - 1 / self.distance[1] ) +
                          self.PPSEXC * (1 / self.distance[0] - 1 / self.distance[1] ) +
                          self.PPSI2basal * (1 / self.distance[1] - 1 / self.distance[0] ) +
                          self.PPSI2apical * ( 1 / self.distance[0]  - 1 / self.distance[1]) +
                          self.PPSI1 * (1 / self.distance[2] - 1 / self.distance[1] )+
                          self.PPSExtI_P_P * ( 1 / self.distance[1]  - 1 / self.distance[0]))
