

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy
import h5py

from scipy.integrate import RK45
import random

import pylib.mix as mix

def reload():
    mix.reload_module(mix)
    return


# -------------------------------------------------------------------------------
# --- Solve a system of differential equations with Nx variables ---
def solve_standart(init_cond, t, F):
    def system_to_solve(t, x):
        Nx = len(x)
        F_arr = [None] * Nx
        for ii in range(Nx):
            F_arr[ii] = F[ii](x)
        return F_arr
    # ------------------------------------------------------------------
    Nx = len(F)
    Nt = len(t)
    dt = np.diff(t)[0]
    t_res = np.zeros(Nt)
    t_res[0] = t[0]

    oo = RK45(system_to_solve, t[0], init_cond, t[-1], first_step=dt, max_step=dt)
    sol_ref = np.zeros((Nt, Nx), dtype=float)
    sol_ref[0] = init_cond
    Nt_act = 1
    while mix.compare_two_strings(oo.status, "running"):
        oo.step()
        Nt_act += 1

        if (Nt_act - 1) >= len(t_res):
            print()
            print("WARNING: increase arrays: counter_t = {:d}".format(Nt_act - 1))
            t_res = np.pad(t_res, (0, Nt), 'constant')
            sol_ref = np.pad(sol_ref, ((0, Nt), (0, 0)), mode='constant')
            # print("new size of t array: {:d}".format(len(t_res)))
            print()

        t_res[Nt_act - 1]   = oo.t
        sol_ref[Nt_act - 1] = oo.y

    # --- remove empty cells from the resulting lists ---    
    t_res = t_res[:Nt_act]
    sol_ref = sol_ref[:Nt_act,:]
    return sol_ref, t_res


# -----------------------------------------------------------------------------------------------
# --- Analyse data from modeling a system of differential equations with Nx variables ---
def analyse_classical(init_cond, t, F):
    # --- Solve the system ---
    sol_ref, t_ref = solve_standart(init_cond, t, F)

    Nvar = np.shape(sol_ref)[1]
    flag_3d = False
    if Nvar == 3:
        flag_3d = True

    x = sol_ref[:,0]
    y = sol_ref[:,1]

    z = None
    if flag_3d:
        z = sol_ref[:,2]

    # --- Plotting trajectories ---
    plt.close()
    fig = plt.figure()
    if not flag_3d:
        ax = fig.add_subplot(111)
        ax.plot(
            x, y, 
            "b", linewidth = 1, linestyle='-', 
        )
    else:
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(
            x, y, z,
            "b", linewidth = 1, linestyle='-', 
        )
    
    ax.set_xlabel('$x$')
    ax.set_ylabel("$y$")
    if flag_3d:
        ax.set_zlabel("$z$")
    plt.grid(True)
    plt.show()

    # --- Plotting variables in time ---
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(
        t_ref, x, 
        "b", linewidth = 1, linestyle='-', label = "x", 
    )
    ax.plot(
        t_ref, y, 
        "r", linewidth = 1, linestyle='-', label = "y", 
    )
    if flag_3d:
        ax.plot(
            t_ref, z, 
            "g", linewidth = 1, linestyle='-', label = "z", 
        )
    plt.xlabel('$t$')
    if not flag_3d:
        plt.ylabel("$x,y$")
    else:
        plt.ylabel("$x,y,z$")
    plt.legend()
    plt.grid(True)
    plt.show()
    return sol_ref, t_ref


# -----------------------------------------------------------------------------------------------
# --- Carleman embedding of a nonlinear system with Nx variables ---
def solve_carleman_orig(x_init, N_nl_emb, N_nl_sys, t, F_terms):
    # --------------------------------------------------------------
    # * x_init: initial conditions
    # * N_nl_emb: the number of NL terms used for Carleman embedding 
    # * N_nl_sys: the highest (assumed) nonlinearity in 
    #           the original system of equations
    # --------------------------------------------------------------
    def f_to_RK(t, x):
        f_eqs = np.dot(A, x) + B
        return f_eqs
    # --------------------------------------------------------------
    def form_block(jj_sideband, i_nl):
        jj_sideband = int(jj_sideband)
        Nr, Nc = Nx**(i_nl+1), Nx**(i_nl + 1 + jj_sideband)
        term_sum = np.zeros((Nr, Nc))

        # print("***")
        # print(term_sum.shape)
        for ii_sum in range(i_nl+1):
            one_left  = eyes_list[ii_sum]
            one_right = eyes_list[i_nl - ii_sum]
            temp_prod = np.kron(
                one_left, 
                np.kron(F_terms[1 + jj_sideband], one_right)
            )
            # print()
            # print("ii_sum = {:d}".format(ii_sum))
            # print(one_left.shape)
            # print(F_terms_[1 + jj_sideband].shape)
            # print(one_right.shape)
            # print(temp_prod.shape)
            term_sum += temp_prod
        return term_sum
    # --------------------------------------------------------------

    # --- the number of variables ---
    Nx = np.shape(F_terms[0])[0]

    # --- time parameters ---
    Nt = len(t)
    dt = np.diff(t)[0]
    t_res = np.zeros(Nt)

    # --- The number of elements for each next nonlinear term in embedding ---
    N_terms = np.zeros(N_nl_emb, dtype=int)
    N_tot = 0
    for ii in range(N_nl_emb):
        N_terms[ii] = int(Nx**(ii+1))
        N_tot += N_terms[ii]
    print("Ntot: {:d}".format(N_tot))

    # --- initialize a matrix to save results ---
    sol_carleman = np.zeros((Nt, N_tot))
    Nt_act = 1

    # --- Form the system: d_t u = A u + B ---
    # *** prepare unit matrices ***
    eyes_list = [None] * N_nl_emb
    for ii in range(N_nl_emb):
        eyes_list[ii] = np.eye(int(Nx**ii))

    # *** matrix B ***
    B = np.zeros(N_tot)
    B[:N_terms[0]] = F_terms[0][:,0]

    # *** matrix A ***
    A = np.zeros((N_tot, N_tot))
    i_start_r = 0
    for i_nl in range(N_nl_emb):  # -> consider N_nl_emb rows with blocks where
            # the i_nl-th row contains (non-square) matrices with 
            # the number of rows = N_terms[i_nl];
            # -> correspondingly, the submatrix at ir-th block-row and ic-th block-column has
            # N_terms[ir] rows and N_terms[ic] columns;
        N_loc = N_terms[i_nl]
        i_end_r = i_start_r + N_loc

        # print("\n-------------")
        # print("i_nl, irs, ire: {:d}; {:d}, {:d}".format(i_nl, i_start_r, i_end_r))
 
        # *** diagonal blocks ***
        i_start_c = i_start_r
        i_end_c   = i_end_r
        A[i_start_r:i_end_r, i_start_c:i_end_c] = form_block(0, i_nl)
        del i_start_c, i_end_c

        # *** left-sideband blocks ***
        if i_nl > 0:
            i_start_c = i_start_r - N_terms[i_nl - 1]
            i_end_c   = i_start_c + N_terms[i_nl - 1]
            A[i_start_r:i_end_r, i_start_c:i_end_c] = form_block(-1, i_nl)
            del i_start_c, i_end_c

        # *** blocks in several right sidebands ***  
        i_start_c = i_start_r
        i_end_c = None
        for jj in range(N_nl_sys - 1): # consider (N_nl_sys - 1) right sidebands
            jj_sb = jj + 1
            # print("\n>>> jj_sb = {:d}".format(jj_sb))
            if i_nl < (N_nl_emb - jj_sb): # the block-rows near the righ-hand side of the matrix does not 
                        # include one or several right sidebands
                i_start_c += N_terms[i_nl + jj]
                i_end_c   = i_start_c + N_terms[i_nl + jj_sb]
                A[i_start_r:i_end_r, i_start_c:i_end_c] = form_block(jj_sb, i_nl)
        del i_start_c, i_end_c

        # *** shift the starting row ***
        i_start_r += N_loc
    del i_start_r, i_end_r

    # --- Prepare initial conditions ---
    xs_init = np.zeros(N_tot)
    prod = 1.
    i_start = 0
    for i_nl in range(N_nl_emb):
        i_end = i_start + N_terms[i_nl]

        # print()
        # print("{:d}: {:d}, {:d}, {:d}".format(i_nl, i_start, N_terms[i_nl], i_end))

        # *** i_nl-th Kronecker product ***
        prod = np.kron(x_init, prod)

        # *** set the initial conditions for the terms of the i_nl-th Carleman term ***
        xs_init[i_start:i_end] = prod[:]
        i_start = i_end
    del i_start, i_end, prod

    # --- Runge-Kutta solver ---
    oo = RK45(f_to_RK, t[0], xs_init, t[-1], first_step=dt, max_step=dt)
    sol_carleman[Nt_act - 1,:] = xs_init
    while mix.compare_two_strings(oo.status, "running"):
        oo.step()
        Nt_act += 1

        if (Nt_act - 1) >= len(t_res):
            print()
            print("WARNING: increase arrays: counter_t = {:d}".format(Nt_act - 1))
            t_res = np.pad(t_res, (0, Nt), 'constant')
            sol_carleman = np.pad(sol_carleman, ((0, Nt), (0, 0)), mode='constant')
            # print("new size of t array: {:d}".format(len(t_res)))
            print()

        t_res[Nt_act - 1]          = oo.t
        sol_carleman[Nt_act - 1,:] = oo.y.copy()

    # --- remove empty cells from the resulting lists ---    
    t_res = t_res[:Nt_act]
    sol_carleman = sol_carleman[:Nt_act,:]
    return sol_carleman, t_res, "STD"


# -----------------------------------------------------------------------------------------------
# --- Globalised Carleman embedding of a nonlinear system with Nx variables ---
def solve_carleman_global(
        x_init_cond, N_nl_emb, N_nl_sys, t, 
        sys_coefs, return_sys, shift_coefs, 
        zeta_th = 0.6
    ):
    # --------------------------------------------------------------
    # * x_init_cond: initial conditions;
    # * N_nl_emb: the number of NL terms used for Carleman embedding;
    # * N_nl_sys: the highest (assumed) nonlinearity in the original system of equations;
    # * return_sys: function which returns the system organized as F_terms using 
    #           the system parameters given as a dictionary (map);
    # * zeta_th: the chart size;
    # --------------------------------------------------------------
    def f_to_RK(t, x):
        f_eqs = np.dot(A_chart, x) + B_chart
        return f_eqs
    # --------------------------------------------------------------
    def prepare_init(x_init_loc):
        xs_init = np.zeros(N_tot)
        prod = 1.
        i_start = 0
        for i_nl in range(N_nl_emb):
            i_end = i_start + N_terms[i_nl]

            # *** i_nl-th Kronecker product ***
            prod = np.kron(x_init_loc, prod)

            # *** set the initial conditions for the terms of the i_nl-th Carleman term ***
            xs_init[i_start:i_end] = prod[:]
            i_start = i_end
        return xs_init
    # --------------------------------------------------------------
    def prepare_matrices_AB(F_terms_loc):
        # *** matrix B ***
        B = np.zeros(N_tot)
        B[:N_terms[0]] = F_terms_loc[0][:,0]

        # *** matrix A ***
        A = np.zeros((N_tot, N_tot))
        i_start_r = 0
        for i_nl in range(N_nl_emb):  # -> consider N_nl_emb rows with blocks where
                # the i_nl-th row contains (non-square) matrices with 
                # the number of rows = N_terms[i_nl];
                # -> correspondingly, the submatrix at ir-th block-row and ic-th block-column has
                # N_terms[ir] rows and N_terms[ic] columns;
            N_loc = N_terms[i_nl]
            i_end_r = i_start_r + N_loc
    
            # *** diagonal blocks ***
            i_start_c = i_start_r
            i_end_c   = i_end_r
            A[i_start_r:i_end_r, i_start_c:i_end_c] = form_block(0, i_nl, F_terms_loc)
            del i_start_c, i_end_c

            # *** left-sideband blocks ***
            if i_nl > 0:
                i_start_c = i_start_r - N_terms[i_nl - 1]
                i_end_c   = i_start_c + N_terms[i_nl - 1]
                A[i_start_r:i_end_r, i_start_c:i_end_c] = form_block(-1, i_nl, F_terms_loc)
                del i_start_c, i_end_c

            # *** blocks in several right sidebands ***  
            i_start_c = i_start_r
            i_end_c = None
            for jj in range(N_nl_sys - 1): # consider (N_nl_sys - 1) right sidebands
                jj_sb = jj + 1
                if i_nl < (N_nl_emb - jj_sb): # the block-rows near the righ-hand side of the matrix does not 
                            # include one or several right sidebands
                    i_start_c += N_terms[i_nl + jj]
                    i_end_c   = i_start_c + N_terms[i_nl + jj_sb]
                    A[i_start_r:i_end_r, i_start_c:i_end_c] = form_block(jj_sb, i_nl, F_terms_loc)
            del i_start_c, i_end_c

            # *** shift the starting row ***
            i_start_r += N_loc
        return A, B
    # --------------------------------------------------------------
    def form_block(jj_sideband, i_nl, F_terms_loc):
        jj_sideband = int(jj_sideband)
        Nr, Nc = Nx**(i_nl+1), Nx**(i_nl + 1 + jj_sideband)
        term_sum = np.zeros((Nr, Nc))
        for ii_sum in range(i_nl+1):
            one_left  = eyes_list[ii_sum]
            one_right = eyes_list[i_nl - ii_sum]
            temp_prod = np.kron(
                one_left, 
                np.kron(F_terms_loc[1 + jj_sideband], one_right)
            )
            term_sum += temp_prod
        return term_sum
    # --------------------------------------------------------------
    def shift_x(xs, zeta):
        xs_new = np.array(xs)
        for ix in range(Nx):
            xs_new[ix] = xs_new[ix] + zeta[ix]
        return xs_new
    # --------------------------------------------------------------
    def get_radius(xs):
        sum_of_squares = np.sum(np.square(xs))
        r = np.sqrt(sum_of_squares)
        return r
    # --------------------------------------------------------------
    # def form_zeta_signed(xs):
    #     zeta_sign = np.zeros(Nx)
    #     for ii in range(Nx):
    #         sign_loc = xs[ii] / np.abs(xs[ii])
    #         zeta_sign[ii] = sign_loc*zeta_th
    #     return zeta_sign
    
    # --- the number of variables ---
    F_terms = return_sys(sys_coefs)
    Nx = np.shape(F_terms[0])[0]

    # --- time parameters ---
    Nt = len(t)
    dt = np.diff(t)[0]
    t_res = np.zeros(Nt)
    t_global = 0.

    # --- The number of elements for each next nonlinear term in embedding ---
    N_terms = np.zeros(N_nl_emb, dtype=int)
    N_tot = 0
    for ii in range(N_nl_emb):
        N_terms[ii] = int(Nx**(ii+1))
        N_tot += N_terms[ii]
    print("The total number of variables in the embedding: {:d}".format(N_tot))

    # --- Prepare unit matrices ---
    eyes_list = [None] * N_nl_emb
    for ii in range(N_nl_emb):
        eyes_list[ii] = np.eye(int(Nx**ii))

    # --- Initialize a matrix to save results ---
    sol_carleman = np.zeros((Nt, N_terms[0]))
    Nt_act = 1

    # --- Carleman computation in different charts ---
    x_init_curr = np.array(x_init_cond)
    zeta_gl = np.zeros(Nx)
    while t_global < t[-1]:
        # --- Form the system: d_t u = A u + B ---
        A_chart, B_chart = prepare_matrices_AB(
            return_sys(shift_coefs(sys_coefs, zeta_gl)) 
        )
        xs_init_sys = prepare_init(x_init_curr)

        # --- Runge-Kutta solver ---
        # oo = RK45(f_to_RK, t_global, xs_init_sys, t[-1], first_step=dt, max_step=dt)
        oo = RK45(f_to_RK, t_global, xs_init_sys, t[-1], max_step=dt)
        sol_carleman[Nt_act - 1,:] = shift_x(xs_init_sys[:N_terms[0]], zeta_gl)  
        while mix.compare_two_strings(oo.status, "running"):
            oo.step()
            Nt_act += 1
            if (Nt_act - 1) >= len(t_res):
                print("\nWARNING: increase arrays: counter_t = {:d}".format(Nt_act - 1))
                t_res = np.pad(t_res, (0, Nt), 'constant')
                sol_carleman = np.pad(sol_carleman, ((0, Nt), (0, 0)), mode='constant')


            t_res[Nt_act - 1] = oo.t
            t_global          = t_res[Nt_act - 1]

            x_loc = np.array(oo.y[:N_terms[0]])
            rr = get_radius(x_loc)
            if rr >= zeta_th:

                # --- set the current location as a near-center of a new chart ---
                zeta_gl    += x_loc
                x_init_curr = np.zeros(Nx)

                print("--- Shift to another chart, t = {:0.3f} ---".format(t_global))
                print("radius: {:0.1f}".format(rr))
                print("zeta_gl: ", end="")
                mix.print_array(zeta_gl, ff = [6,2,"f"])
                print("xs: ", end="")
                mix.print_array(x_loc, ff = [14, 3, "e"])
                break
            else:
                sol_carleman[Nt_act - 1,:] = shift_x(x_loc, zeta_gl)

    # --- remove empty cells from the resulting lists ---    
    t_res = t_res[:Nt_act]
    sol_carleman = sol_carleman[:Nt_act,:]
    print("Done")
    return sol_carleman, t_res, "GL"


# -----------------------------------------------------------------------------------------------
# --- Globalised adaptive Carleman embedding of a nonlinear system with Nx variables ---
def solve_carleman_global_adaptive(
        x_init_cond, N_nl_emb, N_nl_sys, t, 
        sys_coefs, return_sys, shift_coefs, 
        zeta_th_init = 0.6, 
        error_threshold = 1.e-1,
        eta_err = 1.e-1,
        N_iter_max = 100.
    ):
    # --------------------------------------------------------------
    # * x_init_cond: initial conditions;
    # * N_nl_emb: the number of NL terms used for Carleman embedding;
    # * N_nl_sys: the highest (assumed) nonlinearity in the original system of equations;
    # * sys_coefs: dictionary with the initial valus of the coefficients describing the target dynamic system; 
    # * return_sys: function which returns the dynamic system organized as matrices F_terms using 
    #           the system parameters given as a dictionary;
    # * zeta_th_init: the initial chart size;
    # * error_threshold: if the signal achieves this limit, zeta decreases;
    # * eta_err: must be 0 < eta_err < 1: this value is multiplied by error_threshold 
    #           to find the threshold when zeta should be increased to speed up simulations;
    # * N_iter_max: the maximum number of reductions of zeta;
    # ---
    # > In the adaptive procedure, one compares simulations with zeta and zeta/2 and changes zeta according to the observed error.
    # > Zeta decreases by 2., if |x - x'|/min(x,x') >= error_threshold, 
    #       where x and x' are computed using zeta and zeta/2., correspondingly.
    # > Zeta increases by 2., if |x - x'|/min(x,x') <= error_threshold * eta_err;
    # > Zeta is always kept < 1.
    # --------------------------------------------------------------
    def f_to_RK(t, x):
        f_eqs = np.dot(A_chart, x) + B_chart
        return f_eqs
    # --------------------------------------------------------------
    def prepare_init(x_init_loc):
        xs_init = np.zeros(N_tot)
        prod = 1.
        i_start = 0
        for i_nl in range(N_nl_emb):
            i_end = i_start + N_terms[i_nl]

            # *** i_nl-th Kronecker product ***
            prod = np.kron(x_init_loc, prod)

            # *** set the initial conditions for the terms of the i_nl-th Carleman term ***
            xs_init[i_start:i_end] = prod[:]
            i_start = i_end
        return xs_init
    # --------------------------------------------------------------
    def prepare_matrices_AB(F_terms_loc):
        # *** matrix B ***
        B = np.zeros(N_tot)
        B[:N_terms[0]] = F_terms_loc[0][:,0]

        # *** matrix A ***
        A = np.zeros((N_tot, N_tot))
        i_start_r = 0
        for i_nl in range(N_nl_emb):  # -> consider N_nl_emb rows with blocks where
                # the i_nl-th row contains (non-square) matrices with 
                # the number of rows = N_terms[i_nl];
                # -> correspondingly, the submatrix at ir-th block-row and ic-th block-column has
                # N_terms[ir] rows and N_terms[ic] columns;
            N_loc = N_terms[i_nl]
            i_end_r = i_start_r + N_loc
    
            # *** diagonal blocks ***
            i_start_c = i_start_r
            i_end_c   = i_end_r
            A[i_start_r:i_end_r, i_start_c:i_end_c] = form_block(0, i_nl, F_terms_loc)
            del i_start_c, i_end_c

            # *** left-sideband blocks ***
            if i_nl > 0:
                i_start_c = i_start_r - N_terms[i_nl - 1]
                i_end_c   = i_start_c + N_terms[i_nl - 1]
                A[i_start_r:i_end_r, i_start_c:i_end_c] = form_block(-1, i_nl, F_terms_loc)
                del i_start_c, i_end_c

            # *** blocks in several right sidebands ***  
            i_start_c = i_start_r
            i_end_c = None
            for jj in range(N_nl_sys - 1): # consider (N_nl_sys - 1) right sidebands
                jj_sb = jj + 1
                if i_nl < (N_nl_emb - jj_sb): # the block-rows near the righ-hand side of the matrix does not 
                            # include one or several right sidebands
                    i_start_c += N_terms[i_nl + jj]
                    i_end_c   = i_start_c + N_terms[i_nl + jj_sb]
                    A[i_start_r:i_end_r, i_start_c:i_end_c] = form_block(jj_sb, i_nl, F_terms_loc)
            del i_start_c, i_end_c

            # *** shift the starting row ***
            i_start_r += N_loc
        return A, B
    # --------------------------------------------------------------
    def form_block(jj_sideband, i_nl, F_terms_loc):
        jj_sideband = int(jj_sideband)
        Nr, Nc = Nx**(i_nl+1), Nx**(i_nl + 1 + jj_sideband)
        term_sum = np.zeros((Nr, Nc))
        for ii_sum in range(i_nl+1):
            one_left  = eyes_list[ii_sum]
            one_right = eyes_list[i_nl - ii_sum]
            temp_prod = np.kron(
                one_left, 
                np.kron(F_terms_loc[1 + jj_sideband], one_right)
            )
            term_sum += temp_prod
        return term_sum
    # --------------------------------------------------------------
    def shift_x(xs, zeta):
        xs_new = np.array(xs)
        for ix in range(Nx):
            xs_new[ix] = xs_new[ix] + zeta[ix]
        return xs_new
    # --------------------------------------------------------------
    def get_radius(xs):
        sum_of_squares = np.sum(np.square(xs))
        r = np.sqrt(sum_of_squares)
        return r
    # --------------------------------------------------------------
    
    # --- the number of variables ---
    F_terms = return_sys(sys_coefs)
    Nx = np.shape(F_terms[0])[0]

    # --- time parameters ---
    Nt = len(t)
    dt = np.diff(t)[0]
    t_res = np.zeros(Nt)
    t_global = 0.

    # --- The number of elements for each next nonlinear term in embedding ---
    N_terms = np.zeros(N_nl_emb, dtype=int)
    N_tot = 0
    for ii in range(N_nl_emb):
        N_terms[ii] = int(Nx**(ii+1))
        N_tot += N_terms[ii]
    print("The total number of variables in the embedding: {:d}".format(N_tot))

    # --- Prepare unit matrices ---
    eyes_list = [None] * N_nl_emb
    for ii in range(N_nl_emb):
        eyes_list[ii] = np.eye(int(Nx**ii))

    # --- Initialize a matrix to save results ---
    sol_carleman = np.zeros((Nt, N_terms[0]))
    Nt_act = 1

    # --- Carleman computation in different charts ---
    curr_sol = [] # reference simulation for some xi at first 
    comp_sol_half   = [] # simulation for xi/2.
    comp_sol_double = []
    x_init_curr = np.array(x_init_cond)

    # might need to set zeta_gl = x_init_curr
    # and then set x_init_curr = 0

    zeta_gl = np.zeros(Nx)
    while t_global < t[-1]:
        # --- Form the system: d_t u = A u + B ---
        A_chart, B_chart = prepare_matrices_AB(
            return_sys(shift_coefs(sys_coefs, zeta_gl)) 
        )
        xs_init_sys = prepare_init(x_init_curr)
        curr_sol = shift_x(xs_init_sys[:N_terms[0]], zeta_gl)  

        sol_carleman[Nt_act - 1,:] = curr_sol

        # --- Runge-Kutta solver ---
        oo = RK45(f_to_RK, t_global, xs_init_sys, t[-1], max_step=dt)
        while mix.compare_two_strings(oo.status, "running"):
            oo.step()
            Nt_act += 1
            if (Nt_act - 1) >= len(t_res):
                print("\nWARNING: increase arrays: counter_t = {:d}".format(Nt_act - 1))
                t_res = np.pad(t_res, (0, Nt), 'constant')
                sol_carleman = np.pad(sol_carleman, ((0, Nt), (0, 0)), mode='constant')

            t_res[Nt_act - 1] = oo.t
            t_global          = t_res[Nt_act - 1]

            x_loc = np.array(oo.y[:N_terms[0]])
            rr = get_radius(x_loc)
            if rr >= zeta_th_init:
                # shift the chart;
                # set the current location as the center of a new chart:
                zeta_gl    += x_loc
                x_init_curr = np.zeros(Nx)

                print("--- Shift to another chart, t = {:0.3f} ---".format(t_global))
                print("radius: {:0.1f}".format(rr))
                print("zeta_gl: ", end="")
                mix.print_array(zeta_gl, ff = [6, 2,"f"])
                print("xs: ", end="")
                mix.print_array(x_loc, ff = [14, 3, "e"])
                
                break
            else:
                sol_carleman[Nt_act - 1,:] = shift_x(x_loc, zeta_gl)

    # --- remove empty cells from the resulting lists (CHANGED) ---    
    t_res = t_res[:Nt_act]
    sol_carleman = sol_carleman[:Nt_act,:]
    print("Done")
    return sol_carleman, t_res, "GL"


# -----------------------------------------------------------------------------------------------
# --- Compare classical with Carleman simulations  ---
def compare_trajectory_cl_and_carleman(
        t_ref, sol_ref, t_emb, sol_emb, 
        flag_save = False, path_save = None, case_title = None, case_emb = None,
        fontsize = 20, fig_size = (10,9)
    ):
    colors_loc     = ["orange", "red", "green", "gray", "black"]
    linestyles_loc = ["-", "--", "--", ":"]
    name_vars      = ["x", "y", "z"]
    fontsize_leg = int(fontsize/4. * 3.)

    Nvar = np.shape(sol_ref)[1]
    s_ref = [None] * Nvar
    s_emb = [None] * Nvar
    s_emb_int = [None] * Nvar

    flag_3d = False
    if Nvar == 3:
        flag_3d = True

    # --- Classical and Carleman results ---
    label_lines = ["x", "y", "z"]
    for ivar in range(Nvar):
        s_ref[ivar] = sol_ref[:,ivar]
        s_emb[ivar] = sol_emb[:,ivar]

    # --- plotting: trajectories ---
    if not flag_3d:
        fig, axs = plt.subplots(3, 1, figsize=fig_size)
        ax_loc = axs[0]
        ax_loc.plot(s_ref[0], s_ref[1], color="b", linewidth = 2, linestyle='-', label = "CL")
        ax_loc.plot(s_emb[0], s_emb[1], color="r", linewidth = 2, linestyle="--", label = "CA")

        # --- Save data ---
        if flag_save:
            mix.save_dat_plot_1d_file(
                path_save + "/REF_{:s}_x.dat".format(case_title), 
                s_ref[0], s_ref[1]
            )
            mix.save_dat_plot_1d_file(
                path_save + "/EMB_{:s}_{:s}_x.dat".format(case_title, case_emb), 
                s_emb[0], s_emb[1]
            )
    else:
        fig = plt.figure(figsize=fig_size)
        ax_loc = fig.add_subplot(111, projection='3d')
        ax_loc.plot(s_ref[0], s_ref[1], s_ref[2], color="b", linewidth = 2, linestyle='-', label = "CL")
        ax_loc.plot(s_emb[0], s_emb[1], s_emb[2], color="r", linewidth = 2, linestyle="--", label = "CA")
        ax_loc.set_zlabel("$z$", fontsize = fontsize) 

        # --- Save data ---
        if flag_save:
            mix.save_dat_plot_3d_trajectory_file(
                path_save + "/REF_{:s}_x.dat".format(case_title), 
                s_ref[0], s_ref[1], s_ref[2]
            )
            mix.save_dat_plot_3d_trajectory_file(
                path_save + "/EMB_{:s}_{:s}_x.dat".format(case_title, case_emb), 
                s_emb[0], s_emb[1], s_emb[2]
            )
    ax_loc.set_xlabel("$x$", fontsize = fontsize)
    ax_loc.set_ylabel("$y$", fontsize = fontsize)       
    offset_text = ax_loc.yaxis.get_offset_text()
    offset_text.set_fontsize(fontsize) 

    ax_loc.legend(fontsize = fontsize_leg)
    ax_loc.grid(True)
    ax_loc.tick_params(axis='both', which='major', labelsize=fontsize)
    # ax_loc.set_xticklabels([])

    # --- separate variables ---
    if flag_3d:
        fig, axs = plt.subplots(2, 1, figsize=fig_size)
        ax_loc = axs[0]
        ax_loc.set_ylabel("$|x|,|y|,|z|$", fontsize = fontsize)
    else:
        ax_loc = axs[1]
        ax_loc.set_ylabel("$|x|,|y|$", fontsize = fontsize)

    for ivar in range(Nvar):
        ax_loc.semilogy(t_ref, np.abs(s_ref[ivar]), color="black",          linewidth = 2, linestyle='-', label = "CL" if ivar == 0 else None)
        ax_loc.semilogy(t_emb, np.abs(s_emb[ivar]), color=colors_loc[ivar], linewidth = 2, linestyle='--', label = "CA: {:s}".format(label_lines[ivar]))
    offset_text = ax_loc.yaxis.get_offset_text()
    offset_text.set_fontsize(fontsize) 

    ax_loc.legend(fontsize = fontsize_leg)
    ax_loc.grid(True)
    ax_loc.tick_params(axis='both', which='major', labelsize=fontsize)
    # ax.set_xticklabels([])

    # --- difference ---
    ax_loc = axs[1] if flag_3d else axs[2]
    for ivar in range(Nvar):
        s_emb_int[ivar] = np.interp(t_ref, t_emb, s_emb[ivar])
        ax_loc.semilogy(
            t_ref, abs(s_ref[ivar] - s_emb_int[ivar]), 
            color=colors_loc[ivar], linewidth = 2, linestyle='-'
        )

        # --- Save data ---
        if flag_save:
            mix.save_dat_plot_1d_file(
                path_save + "/EMB_{:s}_{:s}_err_{:s}.dat".format(case_title, case_emb, name_vars[ivar]), 
                t_ref, abs(s_ref[ivar] - s_emb_int[ivar])
            )
    ax_loc.set_xlabel('$t$', fontsize = fontsize)
    ax_loc.set_ylabel("$|CL - CA|$", fontsize = fontsize)
    offset_text = ax_loc.yaxis.get_offset_text()
    offset_text.set_fontsize(fontsize) 

    # ax_loc.legend(fontsize = fontsize_leg)
    ax_loc.grid(True)
    ax_loc.tick_params(axis='both', which='major', labelsize=fontsize)

    plt.tight_layout()
    return
