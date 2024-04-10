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
    if sel_case == 1 or sel_case == 10:
        return case_1()
    if sel_case == 2 or sel_case == 20:
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
    print("--- CASE 1 or 10: Slight non-Hermiticity ---")
    H_orig = np.ones((2, 2), dtype=complex)
    H_orig[0,0] = 1 - 0.0001j
    H_orig[0,1] = 2
    H_orig[1,0] = 2
    H_orig[1,1] = 1
    
    print("\n*** Original Hamiltonian (without mult. by i)***")
    mix.print_matrix(H_orig)
    
    Hi = 1j*H_orig
    return Hi


def case_2():
    print("--- CASE 2 or 20: Strong non-Hermiticity ---")
    H_orig = np.ones((2, 2), dtype=complex)
    H_orig[0,0] = 1 - 1.0j
    H_orig[0,1] = 2
    H_orig[1,0] = 2
    H_orig[1,1] = 1
    
    print("\n*** Original Hamiltonian (without mult. by i)***")
    mix.print_matrix(H_orig)
    
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
    ax.scatter(rows_A, cols_A, color="red", s = marker_size) 
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
def construct_UW_matrix_1D(x, F, flag_asin=False):
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
    x_roots = ch.get_Cheb_roots(Nx)
    x_loc = np.array(x_roots)

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
def comp_LCHS_weights(k):
    dk = np.diff(k)[0]
    Nk = len(k)

    wk = np.zeros(Nk)
    for ik in range(Nk):
        wk[ik] = 1. / (1 + k[ik]*k[ik])
    wk = wk * dk/np.pi
    wk[0]  = 0.5 * wk[0]
    wk[-1] = 0.5 * wk[-1]  
    return wk


# ------------------------------------------------------------------------------------------------
def LCHS_computation(k, dt, Hi, psi_init, Nt_loc, flag_trotterization, flag_print = False):
    # if flag_direct = False, use 2nd order Trotterization.

    # k-grid:
    dk = np.diff(k)[0]
    k_max = k[-1]
    Nk = len(k)

    # matrices:
    Bh, Ba = get_herm_aherm_parts(Hi)
    wk = comp_LCHS_weights(k)
    N = Hi.shape[0]
    
    exp_max = None
    exp_Ba = None
    if flag_trotterization:
        Prop_Ba = -1.j * dt/2. * Ba
        exp_Ba = expm(Prop_Ba)
        Prop_kmax = 1.j * dt * k_max* Bh
        exp_max = expm(Prop_kmax)
        del Prop_Ba, Prop_kmax
    
    exp_LCHS = np.zeros((N,N), dtype=complex)
    for ik in range(Nk):
        temp = np.identity(N, dtype=complex)
        
        exp_dt = None
        if not flag_trotterization:
            Prop_k = -1.j * dt * (Ba + k[ik]*Bh) # here, use Trotterization
            exp_dt = expm(Prop_k)
        else:
            Prop_k = -1.j * dt * (ik * dk) * Bh
            exp_dt = exp_max.dot(expm(Prop_k))
            exp_dt = exp_dt.dot(exp_Ba)
            exp_dt = exp_Ba.dot(exp_dt)
            
        for it in range(Nt_loc):
            temp = exp_dt.dot(temp)
        exp_LCHS += wk[ik] * temp
    del temp, Prop_k, exp_max, exp_Ba, exp_dt, ik
         
    # compare the exponentiating matrices:
    if flag_print:
        exp_ref = np.identity(N, dtype=complex)
        exp_dt  = expm(-dt*Hi)
        for it in range(Nt_loc):
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
def LCHS_computation_FAKE(k, dt, Hi, psi_init, Nt_loc):
    # k-grid:
    dk = np.diff(k)[0]
    Nk = len(k)

    # matrices:
    Bh, _ = get_herm_aherm_parts(Hi)
    wk = comp_LCHS_weights(k)
    N = Hi.shape[0]
    
    exp_LCHS = np.zeros((N,N), dtype=complex)
    for ik in range(Nk):
        temp = np.identity(N, dtype=complex)
        Prop_k = -1.j * dt * (ik * dk) * Bh
        exp_dt = expm(Prop_k)
            
        for _ in range(Nt_loc):
            temp = exp_dt.dot(temp)
        # exp_LCHS += wk[ik] * temp
        exp_LCHS += temp
    del temp, Prop_k, exp_dt, ik
           
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
    Ba     = Aa
    B_kmax = - kmax * Ah
    Bk     =     dk * Ah

    # --- Normalize the matrices ---
    Ba_norm,     ncoef_a,    nonsparsity_a_    = mix.compute_normalized_matrix(Ba,     "Ba")
    B_kmax_norm, ncoef_kmax, nonsparsity_kmax_ = mix.compute_normalized_matrix(B_kmax, "B_kmax")
    Bk_norm,     ncoef_k,    nonsparsity_k_    = mix.compute_normalized_matrix(Bk,     "Bk")
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
        print("dt_a:    {:0.6e}".format(dt_a))
        print("dt_kmax: {:0.6e}".format(dt_kmax))
        print("dt_k:    {:0.6e}".format(dt_k))
    return Ba_norm, B_kmax_norm, Bk_norm


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
    ax.plot(rows_1, diag_1.real, color='b', linewidth = 2, linestyle='-', label = "shift = {:d}".format(sh_1))
    ax.plot(rows_2, diag_2.real, color='r', linewidth = 2, linestyle='--', label = "shift = {:d}".format(sh_2))
    ax.plot(rows_3, diag_3.real, color='g', linewidth = 2, linestyle='--', label = "shift = {:d}".format(sh_3))
    plt.xlabel('row')
    plt.ylabel("Re")
    plt.title("Real parts of {:s}".format(A_name))
    plt.legend()
    plt.grid(True)
    plt.show()

    # --- Imaginary parts ---
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(rows_1, diag_1.imag, color='b', linewidth = 2, linestyle='-', label = "shift = {:d}".format(sh_1))
    ax.plot(rows_2, diag_2.imag, color='r', linewidth = 2, linestyle='--', label = "shift = {:d}".format(sh_2))
    ax.plot(rows_3, diag_3.imag, color='g', linewidth = 2, linestyle='--', label = "shift = {:d}".format(sh_3))
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






