import numpy as np
import importlib as imp
import h5py
# import pylib.Global_variables as GLO
import matplotlib.pyplot as plt
import sys
from numba import jit

from scipy.linalg import expm
from scipy.integrate import RK45

import pylib.mix as mix
import pylib.qucf_oracle as qucf_o
import pylib.qucf_read as qucf_r
import pylib.qucf_matrix_D as qucf_m


def reload():
    mix.reload_module(mix)
    mix.reload_module(qucf_o)
    mix.reload_module(qucf_r)
    mix.reload_module(qucf_m)
    return


def get_dk(norm_Ah_in, norm_Aa_in, t_in, n_dk):
    # dk_res = 0.01 * 2.*np.pi / (norm_Ah_in * t_in)
    # if dk_res >= 2. * k_max_in:
    #     dk_res = 2. * 10 / (1<<(5 + n_dk) - 1)

    # --- fix dk [defined by n_dk] ---
    # REMARK: for very large norm_Ah or t, dk should be inverse to these values, 
    # otherwise, it seems that dk is defined more by the integral discretization error
    # (trapezoidal error: kmax**3/M**2)
    nk = 5 + n_dk
    Nk = 1 << nk
    dk_res = 2. * 10. / (Nk - 1)
    return dk_res


def get_trot(norm_Ah_in, norm_Aa_in, t_in, coef_trot_in, k_max_in):
    # non-normalized time step for the trotterization:
    temp = norm_Ah_in * k_max_in
    if temp >= norm_Aa_in:
        tau_res = coef_trot_in * 1. / temp
        if tau_res >= t_in:
            tau_res = t_in
    else:
        tau_res = coef_trot_in * 1. / norm_Aa_in

    # number of trotterization steps:
    Nt_trot_res = int(t_in / tau_res)
    if Nt_trot_res <= 0:
        Nt_trot_res = 1
    return tau_res, Nt_trot_res


def get_max_err(t_ref, psi_ref, t_LCHS, psi_LCHS, coef_sign):
    max_abs_err = 0
    for id_var in range(2):
        for it in range(len(t_LCHS)):
            t1 = t_LCHS[it]
            v_ref = np.interp(t1, t_ref, psi_ref[:, id_var])  
            err = np.max(np.abs(v_ref - coef_sign * psi_LCHS[it, id_var]))
            if max_abs_err < err:
                max_abs_err = err
    print("max. abs. err: {:0.3e}".format(max_abs_err))
    return