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



def get_initial_state():
    psi_init = np.zeros(2, dtype = complex)
    psi_init[0] = np.sqrt(0.0)
    psi_init[1] = np.sqrt(1.0)

    return psi_init


def compute_angles_initialization(psi_init):
    ay_init = 2*np.arccos(psi_init.real[0])
    print("ay_init  {:0.12e}".format(ay_init))
    return


def get_case_Hi(sel_case):
    if sel_case == 1 or sel_case == 10 or sel_case == 100:
        return case_1()
    if sel_case == 2 or sel_case == 20 or sel_case == 200:
        return case_2()
    if sel_case == 3:
        return case_3()
    if sel_case == 4:
        return case_4()
    if sel_case == 5:
        return case_5()
    
    print("!!! ERROR: sel_case = {:d} is not recognized. !!!".format(sel_case))
    return None


def case_1():
    print("--- Slight non-Hermiticity ---")
    H_orig = np.ones((2, 2), dtype=complex)
    H_orig[0,0] = 1 - 0.0001j
    H_orig[0,1] = 2
    H_orig[1,0] = 2
    H_orig[1,1] = 1
    
    # print("\n*** Original Hamiltonian (without mult. by i)***")
    # mix.print_matrix(H_orig)
    
    Hi = 1j*H_orig
    return Hi


def case_2():
    print("--- Strong non-Hermiticity ---")
    H_orig = np.ones((2, 2), dtype=complex)
    H_orig[0,0] = 1 - 1.0j
    H_orig[0,1] = 2
    H_orig[1,0] = 2
    H_orig[1,1] = 1
    
    # print("\n*** Original Hamiltonian (without mult. by i)***")
    # mix.print_matrix(H_orig)
    
    Hi = 1j*H_orig
    return Hi


def case_3():
    print("--- Case 3: both variables are dissipative ---")
    H_orig = np.ones((2, 2), dtype=complex)
    H_orig[0,0] = 1 - 0.4j
    H_orig[0,1] = 2
    H_orig[1,0] = 2
    H_orig[1,1] = 1 - 0.4j
    
    print("\n*** Original Hamiltonian (without mult. by i)***")
    mix.print_matrix(H_orig)
    
    Hi = 1j*H_orig
    return Hi


def case_4():
    print("--- Case 4: all matrix elements are complex ---")
    H_orig = np.ones((2, 2), dtype=complex)
    H_orig[0,0] = 1 - 0.2j
    H_orig[0,1] = 2 - 0.1j
    H_orig[1,0] = 2 - 0.4j
    H_orig[1,1] = 1 - 0.3j
    
    print("\n*** Original Hamiltonian (without mult. by i)***")
    mix.print_matrix(H_orig)
    
    Hi = 1j*H_orig
    return Hi


def case_5():
    print("--- CASE 5: Growing non-Hermiticity ---")
    H_orig = np.ones((2, 2), dtype=complex)
    H_orig[0,0] = 1 + 0.01j
    H_orig[0,1] = 2
    H_orig[1,0] = 2
    H_orig[1,1] = 1
    
    print("\n*** Original Hamiltonian (without mult. by i)***")
    mix.print_matrix(H_orig)
    
    Hi = 1j*H_orig
    return Hi


# input: Hi = i*H:
def ref_computation(t, Hi, psi_init):
    def calc_y(t,y):
        y = -Hi.dot(y) 
        return y
    
    Nt = len(t)
    dt = np.diff(t)[0]
    
    psi_out = np.zeros((Nt,2), dtype=complex)
    psi_out[0,0] = psi_init[0]
    psi_out[0,1] = psi_init[1]
    
    oo = RK45(calc_y, t[0], psi_out[0,:], t[-1], first_step=dt, max_step=dt)
    Nt_act = 0
    oo.step() # skip one time step
    while mix.compare_two_strings(oo.status, "running"):
        oo.step()
        Nt_act += 1
        psi_out[Nt_act,:] = oo.y
    print()
    print("--- refence computation ---")
    print("sum psi[RK-max-time]**2: {:0.3e}".format(np.sum(np.abs(psi_out[Nt_act,:])**2)))
    return psi_out


def compare_plots_ref_LCHS_py(t, t_plot, psi_ref, psi_LCHS, id_var):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # --- REAL PART ---
    ax.plot(t,      psi_ref.real[:,id_var],  "-b", linewidth = 2, label="Re: ref")
    ax.plot(t,      psi_ref.imag[:,id_var],  "-r", linewidth = 2, label="Im: ref")
    ax.plot(t_plot, psi_LCHS.real[:,id_var], 
        "b", marker = "o", linestyle='None', linewidth = 2, markerfacecolor='None', 
        label="Re: LCHS"
    )
    ax.plot(t_plot, psi_LCHS.imag[:,id_var], 
        "r", marker = "o", linestyle='None', linewidth = 2, markerfacecolor='None', 
        label="Im: LCHS"
    )
    plt.xlabel('$t$')
    plt.ylabel("Re: " + "var[{:d}]".format(id_var))
    ax.legend()
    plt.grid(True)
    plt.show()
    return


def plot_one_sim(t, psi):
    fig = plt.figure()
    ax = fig.add_subplot(111)   
    ax.plot(t, psi.real[:,0], "-b", linewidth = 2, label="Re var1")
    ax.plot(t, psi.real[:,1], "-r", linewidth = 2, label="Re var2")

    ax.plot(t, psi.imag[:,0], "--", color = "gray", linewidth = 2, label="Im var1")
    ax.plot(t, psi.imag[:,1], "--", color = "green", linewidth = 2, label="Im var2")

    plt.xlabel('$t$')
    plt.ylabel("psi")
    ax.legend()
    plt.grid(True)
    plt.show()
    return



def is_Hermitian(A, name):
    return mix.is_Hermitian(A, name)


def h_adj(AA):
    return np.transpose(np.conjugate(AA))
    

def get_herm_aherm_parts(B):
    return mix.get_herm_aherm_parts(B)


def plot_A_structure(
        A, 
        name_A = "", label_A = "", 
        file_save = "", flag_save = False, path_save = None, 
        fontsize = 30, marker_size = 160,
        text_coord_label = [1, 14],
        text_coord_name_A = [12, 1],
):
    def find_rows_columns(A):
        N = A.shape[0]
        rows_plot_1 = np.zeros(N*N)
        cols_plot_1 = np.zeros(N*N)
        N_nz_1 = 0
        for ir in range(N):
            for ic in range(N):
                if np.abs(A[ir, ic]) > 0:
                    N_nz_1 += 1
                    rows_plot_1[N_nz_1-1] = ir
                    cols_plot_1[N_nz_1-1] = ic
        rows_plot_1 = rows_plot_1[0:N_nz_1]
        cols_plot_1 = cols_plot_1[0:N_nz_1]
        return rows_plot_1, cols_plot_1

    N = A.shape[0]
    rows_A, cols_A = find_rows_columns(A)

    fig1 = plt.figure(figsize=(10,10))
    ax = fig1.add_subplot(111)
    ax.scatter(cols_A, rows_A, color="red", s = marker_size) 
    plt.xlim(0, N - 1)
    plt.ylim(0, N - 1)
    plt.gca().invert_yaxis()
    plt.xlabel(r'$\textbf{columns}$', fontsize = fontsize)
    plt.ylabel(r"$\textbf{rows}$", fontsize = fontsize)
    plt.grid()

    plt.yticks(np.arange(0, N, 3))
    plt.xticks(np.arange(0, N, 3))

    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    ax.text(
        int(text_coord_label[0]), int(text_coord_label[1]), 
        r'$\textbf{' + label_A + '}$', fontsize=fontsize
    )
    ax.text(
        int(text_coord_name_A[0]), int(text_coord_name_A[1]), 
        r'$' + name_A + '$', fontsize=fontsize
    )
    plt.show()
    if flag_save:
        if path_save is None:
            print("Error: a path for saving a figure is not given.")
            return
        plt.savefig(path_save + "/" + "{:s}.png".format(file_save))
    return


# ---------------------------------------------------------------------------
def is_Hermitian(A, name):
    return mix.is_Hermitian(A, name)


# ---------------------------------------------------------------------------
def construct_CD_matrix_1D(x, F):
    Nx = len(x)
    dx = np.diff(x)[0]
    H_CD = np.zeros((Nx, Nx), dtype=complex)
    for ii in range(1,Nx-1):
        Fm = F(x[ii-1]) 
        Fc = F(x[ii]) 
        Fp = F(x[ii+1]) 
        H_CD[ii,ii-1] = - (Fm + Fc)
        H_CD[ii,ii+1] =   (Fp + Fc)
    H_CD = -1j/(4.*dx) * H_CD
    H_CD[1,0] = 0.0
    H_CD[Nx-2, Nx-1] = 0.0
    return H_CD


# ---------------------------------------------------------------------------
def construct_UW_matrix_1D(x, F, flag_asin=False, flag_Cheb = True):
    import pylib.Chebyschev_coefs as ch

    def delta_f(i1, i2):
        if i1 == i2:
            return 1
        else:
            return 0
    # ----------------------------------
    Nx = len(x)
    dx = np.diff(x)[0]
    H_UW = np.zeros((Nx, Nx), dtype=complex)

    for ir in range(Nx):
        Fr = F(x[ir])
        if np.abs(Fr) == 0:
            ss_r = 0
        else:
            ss_r = int(Fr/np.abs(Fr))
        H_UW[ir, ir] = - 2. * ss_r * Fr
        ic  = ir - ss_r
        if ic >= 0 and ic < Nx:
            Fc = F(x[ic])
            H_UW[ir, ic] = ss_r * (Fr + Fc)
    H_UW = 1.j/(2.*dx) * H_UW

    # ---------------------------------------------------------
    # --- Compute Hermitian and anti-Hermitian of 1j * H_UW ---

    # *** OPTION 1 ***
    Ah_v1, Aa_v1 = get_herm_aherm_parts(1j * H_UW)

    # *** OPTION 2 ***
    if flag_Cheb:
        x_roots = ch.get_Cheb_roots(Nx)
        x_loc = np.array(x_roots)
    else:
        x_loc = np.array(x)

    Aa_v2 = np.zeros((Nx, Nx), dtype=complex)
    Ah_v2 = np.zeros((Nx, Nx), dtype=complex)
    for ir in range(Nx):
        if flag_asin:
            Fr = F(np.arcsin(x_loc[ir]))
        else:
            Fr = F(x_loc[ir])

        Fr_cc = np.conjugate(Fr)
        ss_r = int(Fr/np.abs(Fr))
        Aa_v2[ir, ir] = - 2. * ss_r * (Fr - Fr_cc)
        Ah_v2[ir, ir] = - 2. * ss_r * (Fr + Fr_cc)

        ic = ir + 1
        if ic >= 0 and ic < Nx:
            if flag_asin:
                Fc = F(np.arcsin(x_loc[ic]))
            else:
                Fc = F(x_loc[ic])
            Fc_cc = np.conjugate(Fc)
            ss_c = int(Fc/np.abs(Fc))

            temp_1 = delta_f(ss_r,-1) * (Fr + Fc)
            temp_2 = delta_f(ss_c, 1) * (Fr_cc + Fc_cc)

            Aa_v2[ir, ic] = - (temp_1 + temp_2)
            Ah_v2[ir, ic] = - temp_1 + temp_2

        ic = ir - 1
        if ic >= 0 and ic < Nx:
            if flag_asin:
                Fc = F(np.arcsin(x_loc[ic]))
            else:
                Fc = F(x_loc[ic])
            Fc_cc = np.conjugate(Fc)
            ss_c = int(Fc/np.abs(Fc))

            temp_1 = delta_f(ss_r, 1) * (Fr + Fc)
            temp_2 = delta_f(ss_c,-1) * (Fr_cc + Fc_cc)

            Aa_v2[ir, ic] = temp_1 + temp_2
            Ah_v2[ir, ic] = temp_1 - temp_2

    Aa_v2 = 1.j/(4.*dx) * Aa_v2
    Ah_v2 = -1./(4.*dx) * Ah_v2

    return H_UW, Aa_v1, Ah_v1, Aa_v2, Ah_v2


# ---------------------------------------------------------------------------
def get_sums_UW_KvN_LCHS_matrices_norm(
        x, F, norm_a, norm_k, dk, 
        flag_asin = False, flag_Cheb = True
):
    import pylib.Chebyschev_coefs as ch

    def delta_f(i1, i2):
        if i1 == i2:
            return 1
        else:
            return 0
    # ----------------------------------
    Nx = len(x)
    dx = np.diff(x)[0]

    if flag_Cheb:
        x_roots = ch.get_Cheb_roots(Nx)
        x_loc = np.array(x_roots)
    else:
        x_loc = np.array(x)

    Aa_v2 = np.zeros((Nx, Nx), dtype=complex)
    Ah_v2 = np.zeros((Nx, Nx), dtype=complex)
    array_ss_h = np.zeros(Nx)
    array_ss_r = np.zeros(Nx-1)
    array_delta_row_R = np.zeros(Nx-1)
    array_delta_col_R = np.zeros(Nx-1)
    array_delta_row_L = np.zeros(Nx-1)
    array_delta_col_L = np.zeros(Nx-1)
    sum_D  = np.zeros(Nx)
    sum_RR = np.zeros(Nx-1)
    # sum_CR = np.zeros(Nx-1)
    sum_RL = np.zeros(Nx-1)
    # sum_CL = np.zeros(Nx-1)

    for ir in range(Nx):
        if flag_asin:
            Fr = F(np.arcsin(x_loc[ir]))
        else:
            Fr = F(x_loc[ir])

        # --- main diagonal ---
        Fr_cc = np.conjugate(Fr)
        ss_r = int(Fr/np.abs(Fr))
        array_ss_h[ir] = ss_r

        Aa_v2[ir, ir] = - 2. * ss_r * (Fr - Fr_cc)
        Ah_v2[ir, ir] = - 2. * ss_r * (Fr + Fr_cc)
        sum_D[ir] = (Fr + Fr_cc)

        # --- right diagonal ---
        ic = ir + 1
        if ic >= 0 and ic < Nx:
            if flag_asin:
                Fc = F(np.arcsin(x_loc[ic]))
            else:
                Fc = F(x_loc[ic])
            Fc_cc = np.conjugate(Fc)
            ss_c = int(Fc/np.abs(Fc))

            array_delta_row_R[ir] = delta_f(ss_r,-1)
            array_delta_col_R[ir] = delta_f(ss_c, 1)

            sum_RR[ir] = (Fr + Fc)
            # sum_CR[ir] = (Fr_cc + Fc_cc)

            temp_1 = delta_f(ss_r,-1) * (Fr + Fc)
            temp_2 = delta_f(ss_c, 1) * (Fr_cc + Fc_cc)

            Aa_v2[ir, ic] = - (temp_1 + temp_2)
            Ah_v2[ir, ic] = - temp_1 + temp_2

            array_ss_r[ir] = ss_c

        # --- left diagonal ---
        ic = ir - 1
        if ic >= 0 and ic < Nx:
            if flag_asin:
                Fc = F(np.arcsin(x_loc[ic]))
            else:
                Fc = F(x_loc[ic])
            Fc_cc = np.conjugate(Fc)
            ss_c = int(Fc/np.abs(Fc))

            array_delta_row_L[ic] = delta_f(ss_r, 1)
            array_delta_col_L[ic] = delta_f(ss_c,-1)

            sum_RL[ic] = (Fr + Fc)
            # sum_CL[ic] = (Fr_cc + Fc_cc)

            temp_1 = delta_f(ss_r, 1) * (Fr + Fc)
            temp_2 = delta_f(ss_c,-1) * (Fr_cc + Fc_cc)

            Aa_v2[ir, ic] = temp_1 + temp_2
            Ah_v2[ir, ic] = temp_1 - temp_2

    Aa_v2 = 1.j/(4.*dx) * Aa_v2
    Ah_v2 = -1./(4.*dx) * Ah_v2

    # --- REMARK: next, we assume that F is real: ---
    # --- because of that, F_cc = F ---
    # sums for normalized matrices Bk and B_kmax:
    coef_k = - dk / (4.* dx * norm_k)
    sum_D_k = -2. * coef_k * sum_D
    sum_R_k = coef_k * sum_RR 
    sum_L_k = coef_k * sum_RL

    # sums for normalized matrices Ba:
    coef_k = 1./(4.* dx * norm_a)
    sum_D_a = coef_k * sum_D
    sum_R_a = coef_k * sum_RR 
    sum_L_a = coef_k * sum_RL
    
    return array_ss_h, array_ss_r, array_delta_row_R, array_delta_col_R, \
        array_delta_row_L, array_delta_col_L, \
        sum_D_k, sum_R_k, sum_L_k, sum_R_a, sum_L_a, sum_D_a


# ---------------------------------------------------------------------------
def get_HALF_sums_UW_KvN_LCHS_matrices_norm_rescaled(
        x, F, norm_a, norm_k, dk, id_b, id_end
):
    import pylib.Chebyschev_coefs as ch

    def delta_f(i1, i2):
        if i1 == i2:
            return 1
        else:
            return 0
    # ----------------------------------
    Nx = len(x)
    dx = np.diff(x)[0]
    sum_RR = np.zeros(Nx)
    sum_RL = np.zeros(Nx)

    # --- here, the x-grid is rescaled for correct computation of Chebyschev angles ---
    f_asin = lambda x1: F(0.5*np.arcsin(x1) - 0.5)

    print("\n here")
    print(f_asin(1.0))

    # for ir in range(id_b, id_end):
    for ir in range(Nx):      

        # --- right diagonal ---
        Fr = f_asin(xr[ir])
        ic = ir + 1
        if ic >= 0 and ic < Nx:
            Fc = f_asin(xr[ic])
            sum_RR[ir] = (Fr + Fc)
            print("ir, ic, v: ", ir, ic, Fc)

        # --- left diagonal ---
        Fr = f_asin(xl[ir])
        ic = ir - 1
        if ic >= 0 and ic < Nx:
            Fc = f_asin(xl[ic])
            sum_RL[ic] = (Fr + Fc)

    # sums for normalized matrices Bk and B_kmax:
    coef_k = - dk / (4.* dx * norm_k)
    sum_R_k = coef_k * sum_RR 
    sum_L_k = coef_k * sum_RL

    # sums for normalized matrices Ba:
    coef_k = 1./(4.* dx * norm_a)
    sum_R_a = coef_k * sum_RR 
    sum_L_a = coef_k * sum_RL

    # --- rescaling because of the D-matrix ---

    # rescaled sums for Bk:
    coef_R_k = 0.250
    sum_R_k /= coef_R_k
    sum_L_k /= coef_R_k

    # rescaled sum for Ba:
    coef_R_a = 0.500
    sum_R_a /= coef_R_a
    sum_L_a /= coef_R_a

    return sum_R_k, sum_L_k, sum_R_a, sum_L_a


# ---------------------------------------------------------------------------
def construct_DI_matrix_1D(x, F, D = 0.001, flag_DI_Z = False, flag_asin = False):
    import pylib.Chebyschev_coefs as ch
    def delta_f(i1, i2):
        if i1 == i2:
            return 1
        else:
            return 0
    # ----------------------------------
    Nx = len(x)
    dx = np.diff(x)[0]
    H = np.zeros((Nx, Nx), dtype=complex)

    if not flag_DI_Z:
        # === OPEN BOUNDARY CONDITIONS ===
        print("--> OPEN boundary conditions")
        ss = 1. / (2.*dx)
        bb = 1. / (dx**2)
        cc = -2.*bb*D
        for ir in range(Nx):
            Fr = F(x[ir])
            if ir == 0:
                F0, F1, F2 = F(x[0]), F(x[1]), F(x[2])
                H[ir, 0] =  2.*cc - 6.*ss*F0
                H[ir, 1] = -5.*cc + 4.*ss*F0 + 4.*ss*F1
                H[ir, 2] =  4.*cc -    ss*F0 -    ss*F2
                H[ir, 3] = -   cc
            elif ir == Nx-1:
                Fq, Fm, Fmm = F(x[ir]), F(x[ir-1]), F(x[ir-2])
                H[ir, ir]   =  2.*cc + 6.*ss*Fq
                H[ir, ir-1] = -5.*cc - 4.*ss*Fm  - 4.*ss*Fq
                H[ir, ir-2] =  4.*cc +    ss*Fmm +    ss*Fq
                H[ir, ir-3] = -   cc
            else:
                Fm, F0, Fp = F(x[ir-1]), F(x[ir]), F(x[ir+1])
                H[ir, ir-1] = cc - ss*Fm - ss*F0
                H[ir, ir]   = -2.*cc
                H[ir, ir+1] = cc + ss*F0 + ss*Fp
        H = - 1.j/2. * H
    else:
        # === ZERO BOUNDARY CONDITIONS ===
        print("--> ZERO boundary conditions")
        ss = 1. / (2.*dx)
        bb = 1. / (dx**2)
        cc = -2.*bb*D
        for ir in range(Nx):
            if ir > 0 and ir < (Nx-1):
                Fm, F0, Fp = F(x[ir-1]), F(x[ir]), F(x[ir+1])
                H[ir, ir-1] = cc - ss*Fm - ss*F0
                H[ir, ir]   = -2.*cc
                H[ir, ir+1] = cc + ss*F0 + ss*Fp
            else:
                H[ir, ir] = 1.0
        H = - 1.j/2. * H

    # ---------------------------------------------------------
    # --- Compute Hermitian and anti-Hermitian of 1j * H_UW ---

    # *** OPTION 1 ***
    Ah_v1, Aa_v1 = get_herm_aherm_parts(1j * H)

    # *** OPTION 2 ***
    def temp_comm():
        # if flag_Cheb:
        #     x_roots = ch.get_Cheb_roots(Nx)
        #     x_loc = np.array(x_roots)
        # else:
        #     x_loc = np.array(x)
        #
        # Aa_v2 = np.zeros((Nx, Nx), dtype=complex)
        # Ah_v2 = np.zeros((Nx, Nx), dtype=complex)
        # for ir in range(Nx):
        #     if flag_asin:
        #         Fr = F(np.arcsin(x_loc[ir]))
        #     else:
        #         Fr = F(x_loc[ir])
        #
        #     Fr_cc = np.conjugate(Fr)
        #     ss_r = int(Fr/np.abs(Fr))
        #     Aa_v2[ir, ir] = - 2. * ss_r * (Fr - Fr_cc)
        #     Ah_v2[ir, ir] = - 2. * ss_r * (Fr + Fr_cc)
        #
        #     ic = ir + 1
        #     if ic >= 0 and ic < Nx:
        #         if flag_asin:
        #             Fc = F(np.arcsin(x_loc[ic]))
        #         else:
        #             Fc = F(x_loc[ic])
        #         Fc_cc = np.conjugate(Fc)
        #         ss_c = int(Fc/np.abs(Fc))
        #
        #         temp_1 = delta_f(ss_r,-1) * (Fr + Fc)
        #         temp_2 = delta_f(ss_c, 1) * (Fr_cc + Fc_cc)
        #
        #         Aa_v2[ir, ic] = - (temp_1 + temp_2)
        #         Ah_v2[ir, ic] = - temp_1 + temp_2
        #
        #     ic = ir - 1
        #     if ic >= 0 and ic < Nx:
        #         if flag_asin:
        #             Fc = F(np.arcsin(x_loc[ic]))
        #         else:
        #             Fc = F(x_loc[ic])
        #         Fc_cc = np.conjugate(Fc)
        #         ss_c = int(Fc/np.abs(Fc))
        #
        #         temp_1 = delta_f(ss_r, 1) * (Fr + Fc)
        #         temp_2 = delta_f(ss_c,-1) * (Fr_cc + Fc_cc)
        #
        #         Aa_v2[ir, ic] = temp_1 + temp_2
        #         Ah_v2[ir, ic] = temp_1 - temp_2
        #
        # Aa_v2 = 1.j/(4.*dx) * Aa_v2
        # Ah_v2 = -1./(4.*dx) * Ah_v2
        return

    return H, Aa_v1, Ah_v1, None, None


# ---------------------------------------------------------------------------
def construct_DI_matrix_1D_OPEN_BNDR(x, F, D = 0.001, flag_asin=False, flag_Cheb = True):
    import pylib.Chebyschev_coefs as ch
    def delta_f(i1, i2):
        if i1 == i2:
            return 1
        else:
            return 0
    # ----------------------------------
    Nx = len(x)
    dx = np.diff(x)[0]
    H = np.zeros((Nx, Nx), dtype=complex)

    ss = 1. / (2.*dx)
    bb = 1. / (dx**2)
    cc = -2.*bb*D
    for ir in range(Nx):
        Fr = F(x[ir])

        if ir == 0:
            F0, F1, F2 = F(x[0]), F(x[1]), F(x[2])
            H[ir, 0] =  2.*cc - 6.*ss*F0
            H[ir, 1] = -5.*cc + 4.*ss*F0 + 4.*ss*F1
            H[ir, 2] =  4.*cc -    ss*F0 -    ss*F2
            H[ir, 3] = -   cc
        elif ir == Nx-1:
            Fq, Fm, Fmm = F(x[ir]), F(x[ir-1]), F(x[ir-2])
            H[ir, ir]   =  2.*cc + 6.*ss*Fq
            H[ir, ir-1] = -5.*cc - 4.*ss*Fm  - 4.*ss*Fq
            H[ir, ir-2] =  4.*cc +    ss*Fmm +    ss*Fq
            H[ir, ir-3] = -   cc
        else:
            Fm, F0, Fp = F(x[ir-1]), F(x[ir]), F(x[ir+1])
            H[ir, ir-1] = cc - ss*Fm - ss*F0
            H[ir, ir]   = -2.*cc
            H[ir, ir+1] = cc + ss*F0 + ss*Fp
    H = - 1.j/2. * H

    # ---------------------------------------------------------
    # --- Compute Hermitian and anti-Hermitian of 1j * H_UW ---

    # *** OPTION 1 ***
    Ah_v1, Aa_v1 = get_herm_aherm_parts(1j * H)

    # *** OPTION 2 ***
    def temp_comm():
        # if flag_Cheb:
        #     x_roots = ch.get_Cheb_roots(Nx)
        #     x_loc = np.array(x_roots)
        # else:
        #     x_loc = np.array(x)
        #
        # Aa_v2 = np.zeros((Nx, Nx), dtype=complex)
        # Ah_v2 = np.zeros((Nx, Nx), dtype=complex)
        # for ir in range(Nx):
        #     if flag_asin:
        #         Fr = F(np.arcsin(x_loc[ir]))
        #     else:
        #         Fr = F(x_loc[ir])
        #
        #     Fr_cc = np.conjugate(Fr)
        #     ss_r = int(Fr/np.abs(Fr))
        #     Aa_v2[ir, ir] = - 2. * ss_r * (Fr - Fr_cc)
        #     Ah_v2[ir, ir] = - 2. * ss_r * (Fr + Fr_cc)
        #
        #     ic = ir + 1
        #     if ic >= 0 and ic < Nx:
        #         if flag_asin:
        #             Fc = F(np.arcsin(x_loc[ic]))
        #         else:
        #             Fc = F(x_loc[ic])
        #         Fc_cc = np.conjugate(Fc)
        #         ss_c = int(Fc/np.abs(Fc))
        #
        #         temp_1 = delta_f(ss_r,-1) * (Fr + Fc)
        #         temp_2 = delta_f(ss_c, 1) * (Fr_cc + Fc_cc)
        #
        #         Aa_v2[ir, ic] = - (temp_1 + temp_2)
        #         Ah_v2[ir, ic] = - temp_1 + temp_2
        #
        #     ic = ir - 1
        #     if ic >= 0 and ic < Nx:
        #         if flag_asin:
        #             Fc = F(np.arcsin(x_loc[ic]))
        #         else:
        #             Fc = F(x_loc[ic])
        #         Fc_cc = np.conjugate(Fc)
        #         ss_c = int(Fc/np.abs(Fc))
        #
        #         temp_1 = delta_f(ss_r, 1) * (Fr + Fc)
        #         temp_2 = delta_f(ss_c,-1) * (Fr_cc + Fc_cc)
        #
        #         Aa_v2[ir, ic] = temp_1 + temp_2
        #         Ah_v2[ir, ic] = temp_1 - temp_2
        #
        # Aa_v2 = 1.j/(4.*dx) * Aa_v2
        # Ah_v2 = -1./(4.*dx) * Ah_v2
        return

    return H, Aa_v1, Ah_v1, None, None


# ---------------------------------------------------------------------------
def select_matrix_norm_nonresc_diags(
        sel_matrix, # sel_matrix = "Ba" or "Bk"
        flag_qsvt, Ba_qsvt, Bk_qsvt, Ba, Bk
    ): 
    # --- Choose the reference matrix ---
    if flag_qsvt:
        if sel_matrix == "Ba":
            print("take Ba_asin.imag")
            A_ref = np.array(Ba_qsvt.imag)
        else:
            print("take Bk_asin.real")
            A_ref = np.array(Bk_qsvt.real)
    else:
        if sel_matrix == "Ba":
            print("take Ba.imag")
            A_ref = np.array(Ba.imag)
        else:
            print("take Bk.real")
            A_ref = np.array(Bk.real)

    # --- Get normalized (non-scaled) diagonals of the reference matrix ---
    diag_D, _ = mix.get_diag(A_ref, i_shift = 0)
    diag_R, _ = mix.get_diag(A_ref, i_shift = 1)
    diag_L, _ = mix.get_diag(A_ref, i_shift = -1)
    return diag_D, diag_R, diag_L


# ---------------------------------------------------------------------------
def solve_KvN_1D_using_Hamiltonian(t, Nx, psi_init, A):
    Nt = len(t)
    dt = np.diff(t)[0]

    psi_tx      = np.zeros((Nt,Nx), dtype = complex)
    psi_tx[0,:] = np.array(psi_init)

    coef_dt = dt * (-1j)
    for it in range(Nt-1):
        Hpsi = A.dot(psi_tx[it])

        for ix in range(1, Nx-1):
            psi_tx[it+1, ix] = psi_tx[it, ix] + coef_dt * Hpsi[ix]
    return psi_tx


# ---------------------------------------------------------------------------
def compute_mean_1D(x, Nt, psi_tx_matrix):
    # x_operator = np.diag(x)
    dx = np.diff(x)[0]
    mean_t = np.zeros(Nt)
    for it in range(Nt):
        psi_t1   = psi_tx_matrix[it,:]
        psi_t1_c = np.conjugate(psi_t1)
        norm = psi_t1_c.dot(psi_t1) * dx
        mean_t[it] = np.real(np.trapz(x*psi_t1_c*psi_t1, dx=dx) / norm)
    return mean_t


# ------------------------------------------------------------------------------------------------
def comp_LCHS_weights(k, flag_Eff = False, arr_theta = None):
    dk = np.diff(k)[0]
    Nk = len(k)

    wk = np.zeros(Nk)
    if not flag_Eff:
        for ik in range(Nk):
            wk[ik] = 1. / (1 + k[ik]*k[ik])
        wk = wk * dk/np.pi
        wk[0]  = 0.5 * wk[0]
        wk[-1] = 0.5 * wk[-1]  
    else:
        kmax = k[-1]
        kmax2 = kmax**2

        d_theta = np.diff(arr_theta)[0]
        # d_theta_2 = np.pi / (Nk-1)
        # print("{:0.12e}".format(d_theta - d_theta_2))

        for ik in range(Nk):
            wk[ik] = np.cos(arr_theta[ik]) / (1 + kmax2 * np.sin(arr_theta[ik])**2)
        wk = wk * kmax * d_theta / np.pi
    return wk


# ------------------------------------------------------------------------------------------------
def LCHS_computation(k, dt, Hi, psi_init, Nt_steps, flag_trotterization, flag_print = False):
    # k-grid:
    dk = np.diff(k)[0]
    k_max = k[-1]
    Nk = len(k)

    # matrices:
    Bh, Ba = get_herm_aherm_parts(Hi)
    wk = comp_LCHS_weights(k)
    N = Hi.shape[0]

    # print("here")
    # print(Ba[1,0], Ba[1,1], Ba[1,2])
    
    exp_max = None
    exp_Ba = None
    if flag_trotterization:
        Prop_Ba = -1.j * dt/2. * Ba
        exp_Ba = expm(Prop_Ba)
        Prop_kmax = 1.j * dt * k_max * Bh
        exp_max = expm(Prop_kmax)
        del Prop_Ba, Prop_kmax
    
    exp_LCHS = np.zeros((N,N), dtype=complex)
    for ik in range(Nk):
        temp = np.identity(N, dtype=complex)

        if flag_trotterization:
            if Nt_steps > 0:
                Prop_k = -1.j * dt * (ik * dk) * Bh
                exp_dt = exp_max.dot(expm(Prop_k))
                exp_dt = exp_dt.dot(exp_Ba)
                exp_dt = exp_Ba.dot(exp_dt)
                    
                for _ in range(Nt_steps):
                    temp = exp_dt.dot(temp)
        else:
            Prop_k = -1.j * dt * Nt_steps * (Ba + k[ik]*Bh)  
            temp = expm(Prop_k)

        exp_LCHS += wk[ik] * temp
         
    # compare the exponentiating matrices:
    if flag_print:
        exp_ref = np.identity(N, dtype=complex)
        exp_dt  = expm(-dt*Hi)
        for it in range(Nt_steps):
            exp_ref = exp_dt.dot(exp_ref)
        del exp_dt
    
        analyse_exp_matrices(exp_ref, exp_LCHS)
        del exp_ref
        
    # compute the output quantum state:
    psi_t = exp_LCHS.dot(psi_init)
    
    if flag_print:
        print()
        print("sum psi_t_max[max-time]**2: {:0.3e}".format(np.sum(np.abs(psi_t)**2)))
    return psi_t



# ------------------------------------------------------------------------------------------------
def LCHS_computation_EFF(k, t, Hi, psi_init):
    k_max = k[-1]
    Nk = len(k)
    arr_theta = np.linspace(-np.pi/2., np.pi/2., Nk)

    # matrices:
    Bh, Ba = get_herm_aherm_parts(Hi)
    wk = comp_LCHS_weights(k, flag_Eff = True, arr_theta = arr_theta)
    N = Hi.shape[0]

    exp_LCHS = np.zeros((N,N), dtype=complex)
    for ik in range(Nk):
        Prop_k = -1.j * t * (Ba + k_max * np.sin(arr_theta[ik]) * Bh)  
        temp = expm(Prop_k)
        exp_LCHS += wk[ik] * temp
         
    # compute the output quantum state:
    psi_t = exp_LCHS.dot(psi_init)
    return psi_t



# ------------------------------------------------------------------------------------------------
def comp_LCHS_weights_OPT(kmax, Nk, beta):
    arr_theta = np.linspace(-np.pi/2., np.pi/2., Nk)
    d_theta = np.diff(arr_theta)[0]
    k = kmax * np.sin(arr_theta)

    v1 = kmax * d_theta * np.cos(arr_theta)
    v2 = 2. * np.pi * np.exp(-(2.**beta)) * np.exp((1+1j*k)**beta) * (1 - 1j*k)
    wk = v1 / v2

    # coef_dk = kmax * d_theta / (2. * np.pi * np.exp(-2.**beta) )
    # fk = 1. / np.exp((1+1j*k)**beta)
    # wk = coef_dk * fk * np.cos(arr_theta) / (1 - 1j*k)
    return wk


# ------------------------------------------------------------------------------------------------
def LCHS_computation_OPT(k, t, beta, Hi, psi_init):
    k_max = k[-1]
    Nk = len(k)
    arr_theta = np.linspace(-np.pi/2., np.pi/2., Nk)

    # matrices:
    Bh, Ba = get_herm_aherm_parts(Hi)
    wk = comp_LCHS_weights_OPT(k_max, Nk, beta)
    N = Hi.shape[0]

    exp_LCHS = np.zeros((N,N), dtype=complex)
    for ik in range(Nk):
        Prop_k = -1.j * t * (Ba + k_max * np.sin(arr_theta[ik]) * Bh)  
        temp = expm(Prop_k)
        exp_LCHS += wk[ik] * temp
         
    # compute the output quantum state:
    psi_t = exp_LCHS.dot(psi_init)
    return psi_t





# ------------------------------------------------------------------------------------------------
def comp_LCMI_weights(k, t):
    dk = np.diff(k)[0]
    Nk = len(k)

    wk = np.zeros(Nk, dtype=complex)
    for ik in range(Nk):
        wk[ik] = np.exp(-1.j * k[ik] * t)
    wk = wk * dk/(2 * np.pi)
    return wk


# ------------------------------------------------------------------------------------------------
def LCMI_computation(k, A, psi_init, t_sim):
    from numpy.linalg import inv as inv_matrix

    # --- weights ---
    wk = comp_LCMI_weights(k, t_sim)

    # --- sum of inverse matrices ---
    N = A.shape[0]
    exp_LCMI = np.zeros((N,N), dtype=complex)
    
    Nk = len(k)
    # print(A)
    # print()
    for ik in range(Nk):
        B = 1j * k[ik] * np.ones(N) + A
        # print(B)
        B_inv = inv_matrix(B)
        exp_LCMI += wk[ik] * B_inv
        
    # --- compute the output quantum state ---
    psi_t = exp_LCMI.dot(psi_init)
    return psi_t


# ------------------------------------------------------------------------------------------------
def analyse_exp_matrices(exp_1, exp_2):
    print("\n--- Exponentiation matrices ---")
    print(exp_1)
    print()
    print(exp_2)

    print("\n --- Difference between the matrix elements ---")
    abs_err_max = 0.0
    for ir in range(exp_1.shape[0]):
        for ic in range(exp_1.shape[1]):
            diff_comp = exp_1[ir,ic] - exp_2[ir,ic]
            abs_err = np.abs(diff_comp)
            if abs_err > abs_err_max:
                abs_err_max = abs_err
            print("[{:d},{:d}]: {:20.3e}".format(ir,ic, diff_comp))
    print()
    print("max. abs. error: {:0.3e}".format(abs_err_max))
    print("- log of max. abs. error: {:0.3f}".format(-np.log10(abs_err_max)))




# ------------------------------------------------------------------------------------------------
def compute_normalized_matrix(A, name_A):
    return mix.compute_normalized_matrix(A, name_A)


# ------------------------------------------------------------------------------------------------
def get_diag(A, i_shift):
    return mix.get_diag(A, i_shift)


# ------------------------------------------------------------------------------------------------
def compute_norm_matrices_LCHS(
        Aa, Ah, kmax, dk, 
        t_step = None,
        factor_global_a    = 1.0,
        factor_global_kmax = 1.0,
        factor_global_k    = 1.0,
):
    Ba     = np.array(Aa)
    B_kmax = - kmax * Ah
    Bk     =     dk * Ah

    # --- Normalize the matrices ---
    Ba_norm,     ncoef_a,    nonsparsity_a_    = mix.compute_normalized_matrix(Ba,     "Ba", True)
    B_kmax_norm, ncoef_kmax, nonsparsity_kmax_ = mix.compute_normalized_matrix(B_kmax, "B_kmax", True)
    Bk_norm,     ncoef_k,    nonsparsity_k_    = mix.compute_normalized_matrix(Bk,     "Bk", True)
    print()
    print("norm of Ba_norm_:     {:0.3f}".format(mix.find_norm_of_matrix(Ba_norm)))
    print("norm of B_kmax_norm_: {:0.3f}".format(mix.find_norm_of_matrix(B_kmax_norm)))
    print("norm of Bk_norm_:     {:0.3f}".format(mix.find_norm_of_matrix(Bk_norm)))

    # --- Time steps ---
    if t_step is not None:
        dt_a    = ncoef_a    * t_step / 2. * factor_global_a
        dt_kmax = ncoef_kmax * t_step      * factor_global_kmax
        dt_k    = ncoef_k    * t_step      * factor_global_k
        print("\n--- Time steps ---")
        print("dt_a, dt_kmax, dt_k: {:0.12e}, {:0.12e}, {:0.12e}".format(dt_a, dt_kmax, dt_k))
    return Ba_norm, B_kmax_norm, Bk_norm


# ------------------------------------------------------------------------------------------------
def compute_norm_matrices_LCHS_with_output_norm(Aa, Ah, kmax, dk):
    Ba     = np.array(Aa)
    B_kmax = - kmax * Ah
    Bk     =     dk * Ah

    # --- Normalize the matrices ---
    Ba_norm,     ncoef_a,    nonsparsity_a_    = mix.compute_normalized_matrix(Ba,     "Ba", True)
    B_kmax_norm, ncoef_kmax, nonsparsity_kmax_ = mix.compute_normalized_matrix(B_kmax, "B_kmax", True)
    Bk_norm,     ncoef_k,    nonsparsity_k_    = mix.compute_normalized_matrix(Bk,     "Bk", True)
    print()
    print("norm of Ba_norm_:     {:0.3f}".format(mix.find_norm_of_matrix(Ba_norm)))
    print("norm of B_kmax_norm_: {:0.3f}".format(mix.find_norm_of_matrix(B_kmax_norm)))
    print("norm of Bk_norm_:     {:0.3f}".format(mix.find_norm_of_matrix(Bk_norm)))
    return Ba_norm, B_kmax_norm, Bk_norm, ncoef_a, ncoef_kmax, ncoef_k


# ------------------------------------------------------------------------------------------------
def plot_save_diagonals(A_plot, A_name, flag_save, flag_save_real, path_save):
    def save_data(rows_loc, diag_loc, sh_loc):
        full_name = path_save + "//" + A_name + "_diag_{:d}".format(sh_loc)
        if flag_save_real:
            yx_loc = diag_loc.real
        else:
            yx_loc = diag_loc.imag
        mix.save_dat_plot_1d_file(full_name, rows_loc, yx_loc)
        return
    # ----------------------------------------

    sh_1 = 0
    diag_1, rows_1 = get_diag(A_plot, i_shift = sh_1)

    sh_2 = 1
    diag_2, rows_2 = get_diag(A_plot, i_shift = sh_2)

    sh_3 = -1
    diag_3, rows_3 = get_diag(A_plot, i_shift = sh_3)

    # --- Real parts ---
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(rows_1, diag_1.real, color='b', marker = "o", linewidth = 2, linestyle='-', label = "shift = {:d}".format(sh_1))
    ax.plot(rows_2, diag_2.real, color='r', marker = "o", linewidth = 2, linestyle='--', label = "shift = {:d}".format(sh_2))
    ax.plot(rows_3, diag_3.real, color='g', marker = "o", linewidth = 2, linestyle='--', label = "shift = {:d}".format(sh_3))
    plt.xlabel('row')
    plt.ylabel("Re")
    plt.title("Real parts of {:s}".format(A_name))
    plt.legend()
    plt.grid(True)
    plt.show()

    # --- Imaginary parts ---
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(rows_1, diag_1.imag, color='b', marker = "o", linewidth = 2, linestyle='-',  label = "shift = {:d}".format(sh_1))
    ax.plot(rows_2, diag_2.imag, color='r', marker = "o", linewidth = 2, linestyle='--', label = "shift = {:d}".format(sh_2))
    ax.plot(rows_3, diag_3.imag, color='g', marker = "o", linewidth = 2, linestyle='--', label = "shift = {:d}".format(sh_3))
    plt.xlabel('row')
    plt.ylabel("Im")
    plt.title("Imag parts of {:s}".format(A_name))
    plt.legend()
    plt.grid(True)
    plt.show()

    # --- Saving ---
    if flag_save:
        save_data(rows_1, diag_1, sh_1)
        save_data(rows_2, diag_2, sh_2)
        save_data(rows_3, diag_3, sh_3)
    return


# ------------------------------------------------------------------------------------------------
def get_rescaled_diags_BE(Ba_norm, B_kmax_norm, Bk_norm):
    # --- get diagonals ---
    diag_A_m1, range_A_m1 = get_diag(Ba_norm, -1)
    diag_A_p1, range_A_p1 = get_diag(Ba_norm,  1)

    diag_kmax_m1, range_kmax_m1 = get_diag(B_kmax_norm, -1)
    diag_kmax_00, range_kmax_00 = get_diag(B_kmax_norm,  0)
    diag_kmax_p1, range_kmax_p1 = get_diag(B_kmax_norm,  1)

    diag_k_m1, range_k_m1 = get_diag(Bk_norm, -1)
    diag_k_00, range_k_00 = get_diag(Bk_norm,  0)
    diag_k_p1, range_k_p1 = get_diag(Bk_norm,  1)

    # --- rescaling ---
    coef_A = 0.500
    diag_A_m1 = diag_A_m1.imag / coef_A
    diag_A_p1 = diag_A_p1.imag / coef_A

    coef_mp, coef_0 = 0.250, 0.500
    diag_kmax_m1 = diag_kmax_m1.real / coef_mp
    diag_kmax_00 = diag_kmax_00.real / coef_0
    diag_kmax_p1 = diag_kmax_p1.real / coef_mp

    diag_k_m1 = diag_k_m1.real / coef_mp
    diag_k_00 = diag_k_00.real / coef_0
    diag_k_p1 = diag_k_p1.real / coef_mp

    dd_diags = {
        "diag_A_m1": diag_A_m1, "range_A_m1": range_A_m1,
        "diag_A_p1": diag_A_p1, "range_A_p1": range_A_p1,
        "diag_kmax_m1": diag_kmax_m1, "range_kmax_m1": range_kmax_m1,
        "diag_kmax_00": diag_kmax_00, "range_kmax_00": range_kmax_00,
        "diag_kmax_p1": diag_kmax_p1, "range_kmax_p1": range_kmax_p1,
        "diag_k_m1": diag_k_m1, "range_k_m1": range_k_m1,
        "diag_k_00": diag_k_00, "range_k_00": range_k_00,
        "diag_k_p1": diag_k_p1, "range_k_p1": range_k_p1,
        "coef_A":    coef_A,  "coef_edge_A": 1./np.sqrt(2.),
        "coef_k_mp": coef_mp, "coef_edge_k": 1./2**(3/2),
        "coef_k_0":  coef_0,  
    }
    return dd_diags






