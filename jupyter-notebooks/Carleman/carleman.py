

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy
import h5py

from scipy.integrate import RK45
from scipy.integrate import solve_ivp
import random

import pylib.mix as mix

def reload():
    mix.reload_module(mix)
    return


# -------------------------------------------------------------------------------
# --- Solve a system of differential equations with Nx variables ---
# -----------------------------------------------------------------------------------------------
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
# -----------------------------------------------------------------------------------------------
def analyse_classical(init_cond, t, F):

    Nx = len(F)

    # --- Solve the system ---
    sol_ref, t_ref = solve_standart(init_cond, t, F)

    x = sol_ref[:,0]

    y = None
    if Nx > 1:
        y = sol_ref[:,1]

    z = None
    if Nx > 2:
        z = sol_ref[:,2]

    # --- Plotting trajectories ---
    if Nx > 1:
        fig = plt.figure()
        if Nx == 2:
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
        if Nx == 3:
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
    if Nx > 1:
        ax.plot(
            t_ref, y, 
            "r", linewidth = 1, linestyle='-', label = "y", 
        )
    if Nx > 2:
        ax.plot(
            t_ref, z, 
            "g", linewidth = 1, linestyle='-', label = "z", 
        )
    plt.xlabel('$t$')

    if Nx == 1:
       plt.ylabel("$x$") 
    if Nx == 2:
        plt.ylabel("$x,y$")
    if Nx == 3:
        plt.ylabel("$x,y,z$")
    plt.legend()
    plt.grid(True)
    plt.show()
    return sol_ref, t_ref


# -----------------------------------------------------------------------------------------------
# --- Carleman embedding of a nonlinear system with Nx variables ---
# -----------------------------------------------------------------------------------------------
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
# --- Fixed piecewise non-adaptive Carleman embedding of a nonlinear system with Nx variables ---
# -----------------------------------------------------------------------------------------------
def solve_carleman_global_fixed(
        X_init_cond, N_nl_emb, N_nl_sys, t, 
        sys_coefs, return_sys, shift_coefs, 
        X_GC, Zeta_grid
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
    
    # --- the number of variables ---
    F_terms = return_sys(sys_coefs)
    Nx = np.shape(F_terms[0])[0]

    # --- time parameters ---
    Nt = len(t)
    dt = np.diff(t)[0]
    
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

    # --- Initialize arrays to save results ---
    t_res        = np.zeros(Nt)
    sol_carleman = np.zeros((Nt, N_terms[0]))

    # --- Find the tile where the initial conditions are placed ---
    L_grid  = np.zeros(Nx, dtype=np.integer)
    Zeta_gl = np.zeros(Nx)
    x_loc   = np.zeros(Nx)
    for ii in range(Nx):
        d_zeta = 2.*Zeta_grid[ii]
        L_grid[ii]  = int( (X_init_cond[ii] - X_GC[ii]) / d_zeta )
        Zeta_gl[ii] = X_GC[ii] + L_grid[ii] * d_zeta   # the center of the tile where the initial conditions are;
        x_loc[ii]   = X_init_cond[ii] - Zeta_gl[ii]    # the local coordinates of the initial conditions
    
    # --- Carleman computation in different charts ---
    t_global = 0.
    Nt_act   = 0
    t_res[Nt_act]          = t_global
    sol_carleman[Nt_act,:] = np.array(X_init_cond) 
    while t_global < t[-1]:
        A_chart, B_chart = prepare_matrices_AB(
            return_sys(shift_coefs(sys_coefs, Zeta_gl)) 
        )
        xs_init_sys = prepare_init(x_loc)
        oo = RK45(f_to_RK, t_global, xs_init_sys, t[-1], max_step=dt)
        while mix.compare_two_strings(oo.status, "running"):
            oo.step()
            x_loc = np.array(oo.y[:N_terms[0]])

            # --- Save ---
            Nt_act   += 1
            t_global = oo.t
            if Nt_act >= len(t_res):
                print("\nWARNING: increase arrays: t, counter_t: {:0.3e}, {:d}".format(t_global, Nt_act))
                t_res        = np.pad(t_res, (0, Nt), 'constant')
                sol_carleman = np.pad(sol_carleman, ((0, Nt), (0, 0)), mode='constant')
            t_res[Nt_act]          = t_global
            sol_carleman[Nt_act,:] = shift_x(x_loc, Zeta_gl)

            # --- Find dimensions where the trajectory escaped the tile ---
            k_array = np.zeros(Nx, dtype=np.integer)
            Nk = 0
            for ii in range(Nx):
                if np.abs(x_loc[ii]) > Zeta_grid[ii]:
                    Nk += 1
                    k_array[Nk-1] = ii
            k_array = k_array[:Nk]
            del ii

            if Nk > 0:
                # --- Next tile ---     
                for kk in range(Nk):
                    ii = k_array[kk] # the dimension where
                    # shift_i = int(x_loc[ii] / np.abs(x_loc[ii]))
                    shift_i = int(np.sign(x_loc[ii]))
                    L_grid[ii] += shift_i  # the index location of the next tile;

                    d_zeta = 2.*Zeta_grid[ii] 
                    zeta_gl_prev = Zeta_gl[ii]

                    Zeta_gl[ii]  = X_GC[ii] + L_grid[ii] * d_zeta             # the center of the next tile;
                    x_loc[ii]    = zeta_gl_prev + x_loc[ii] - Zeta_gl[ii] # the local coord. in the next tile;
                del kk, ii
                break
                
    # --- remove empty cells from the resulting lists ---    
    t_res        = t_res[:(Nt_act+1)]
    sol_carleman = sol_carleman[:(Nt_act+1),:]
    print("Done")
    return sol_carleman, t_res, "GCE"


# -----------------------------------------------------------------------------------------------
# --- Piecewise non-adaptive Carleman embedding of a nonlinear system with Nx variables ---
# -----------------------------------------------------------------------------------------------
def solve_carleman_global(
        x_init_cond, N_nl_emb, N_nl_sys, t, 
        sys_coefs, return_sys, shift_coefs, 
        zeta_th = 0.1,
        flag_before_boundary = True
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
    
    # --- the number of variables ---
    F_terms = return_sys(sys_coefs)
    Nx = np.shape(F_terms[0])[0]

    # --- time parameters ---
    Nt = len(t)
    dt = np.diff(t)[0]
    
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

    # --- Initialize arrays to save results ---
    t_res        = np.zeros(Nt)
    sol_carleman = np.zeros((Nt, N_terms[0]))

    # --- Carleman computation in different charts ---
    if flag_before_boundary:
        # Save states only before the horizon was hit: a slower version, higher precision.
        # Potenially, can stuck in an infinite loop if xi is too small 
        # (to avoid this, counter_beyond is used).
        zeta_gl  = np.array(x_init_cond) 
        t_global = 0.
        x_prev   = np.zeros(len(x_init_cond)) 
        Nt_act   = 0
        t_res[Nt_act]          = t_global
        sol_carleman[Nt_act,:] = np.array(x_init_cond) 
        counter_beyond = 0
        while t_global < t[-1]:
            A_chart, B_chart = prepare_matrices_AB(
                return_sys(shift_coefs(sys_coefs, zeta_gl)) 
            )
            xs_init_sys = prepare_init(np.zeros(Nx))
            oo = RK45(f_to_RK, t_global, xs_init_sys, t[-1], max_step=dt)
            while mix.compare_two_strings(oo.status, "running"):
                oo.step()
                x_loc = np.array(oo.y[:N_terms[0]])
                rr    = get_radius(x_loc)
                if rr >= zeta_th:
                    counter_beyond += 1
                    if counter_beyond > 4:
                        print("Error: linearization radius is too small")
                        return None, None, None

                    zeta_gl += x_prev # set the current location as a near-center of a new chart
                    break
                else:
                    counter_beyond = 0
                    Nt_act += 1
                    t_global = oo.t
                    if Nt_act >= len(t_res):
                        print("\nWARNING: increase arrays: t, counter_t: {:0.3e}, {:d}".format(t_global, Nt_act))
                        t_res        = np.pad(t_res, (0, Nt), 'constant')
                        sol_carleman = np.pad(sol_carleman, ((0, Nt), (0, 0)), mode='constant')
                    t_res[Nt_act]          = t_global
                    sol_carleman[Nt_act,:] = shift_x(x_loc, zeta_gl)
                    x_prev = x_loc
    else:
        # Save the first state after the horizon: a faster version, but with lower precision
        zeta_gl  = np.array(x_init_cond) 
        t_global = 0.
        Nt_act   = 0
        t_res[Nt_act]          = t_global
        sol_carleman[Nt_act,:] = np.array(x_init_cond) 
        while t_global < t[-1]:
            A_chart, B_chart = prepare_matrices_AB(
                return_sys(shift_coefs(sys_coefs, zeta_gl)) 
            )
            xs_init_sys = prepare_init(np.zeros(Nx))
            oo = RK45(f_to_RK, t_global, xs_init_sys, t[-1], max_step=dt)
            while mix.compare_two_strings(oo.status, "running"):
                oo.step()
                t_global = oo.t
                x_loc = np.array(oo.y[:N_terms[0]])

                Nt_act += 1
                if Nt_act >= len(t_res):
                    print("\nWARNING: increase arrays: t, counter_t: {:0.3e}, {:d}".format(t_global, Nt_act))
                    t_res        = np.pad(t_res, (0, Nt), 'constant')
                    sol_carleman = np.pad(sol_carleman, ((0, Nt), (0, 0)), mode='constant')
                t_res[Nt_act]          = t_global
                sol_carleman[Nt_act,:] = shift_x(x_loc, zeta_gl)
                
                rr    = get_radius(x_loc)
                if rr >= zeta_th:
                    zeta_gl += x_loc # set the current location as a near-center of a new chart
                    break
                
    # --- remove empty cells from the resulting lists ---    
    t_res = t_res[:(Nt_act+1)]
    sol_carleman = sol_carleman[:(Nt_act+1),:]
    print("Done")
    return sol_carleman, t_res, "GL"


# -----------------------------------------------------------------------------------------------
# --- Piecewise adaptive Carleman embedding of a nonlinear system with Nx variables ---
# -----------------------------------------------------------------------------------------------
def solve_carleman_global_adaptive(
        x_init_cond_, N_nl_emb_, N_nl_sys_, t_, 
        sys_coefs_, return_sys_, shift_coefs_, 
        zeta_init_ = 0.6, 
        err_tolerance_ = 1.e-1,
        zeta_min_ = 1e-4,
        zeta_step_ = 0.1,
        zeta_thresh_ = 0.1,
        zeta_max_ = 1.0
    ):
    # --------------------------------------------------------------
    # * x_init_cond: initial conditions;
    # * N_nl_emb: the number of NL terms used for Carleman embedding;
    # * N_nl_sys: the highest (assumed) nonlinearity in the original system of equations;
    # * sys_coefs: dictionary with the initial valus of the coefficients describing the target dynamic system; 
    # * return_sys: function which returns the dynamic system organized as matrices F_terms using 
    #           the system parameters given as a dictionary;
    # * shift_coefs: function describing how the system's coefficients change with the chart shift;
    # * zeta_init: the initial chart size;
    # * err_tolerance: if the difference between two trajectories in two charts is equal or higher than this value, zeta decreases;
    # * zeta_min: minimal possible radius;
    # * zeta_step: additive step to change chart radius;
    # * zeta_thresh: if zeta >= zeta_thresh, zeta is changed by the additive step zeta_step,
    #                if zeta < zeta_thresh, zeta is changed by dividing/multiplying it by 2.
    # --------------------------------------------------------------   
    def change_radius(zeta_ref, flag_reduce):
        coef_mult = 2.

        if flag_reduce:
            if zeta_ref > zeta_thresh_:
                zeta_comp = zeta_ref - zeta_step_
            else:
                zeta_comp = zeta_ref / coef_mult
        else:
            if zeta_ref > zeta_thresh_:
                zeta_comp = zeta_ref + zeta_step_
            else:
                zeta_comp = zeta_ref * coef_mult

        return zeta_comp
    # --------------------------------------------------------------
    def form_block(jj_sideband, i_nl, F_terms_loc):
        jj_sideband = int(jj_sideband)
        Nr, Nc = Nx_**(i_nl+1), Nx_**(i_nl + 1 + jj_sideband)
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
    def f_to_RK_(t, x, A, B):
        f_eqs = np.dot(A, x) + B
        return f_eqs
    # --------------------------------------------------------------
    def get_radius(xs):
        sum_of_squares = np.sum(np.square(xs))
        r = np.sqrt(sum_of_squares)
        return r
    # --------------------------------------------------------------
    def prepare_matrices_AB(F_terms_loc):
        # *** matrix B ***
        B = np.zeros(N_tot_)
        B[:N_terms_[0]] = F_terms_loc[0][:,0]

        # *** matrix A ***
        A = np.zeros((N_tot_, N_tot_))
        i_start_r = 0
        for i_nl in range(N_nl_emb_):  # -> consider N_nl_emb block rows where
                # the i_nl-th row contains (non-square) matrices with 
                # the number of rows = N_terms[i_nl];
                # -> correspondingly, the submatrix at ir-th block-row and ic-th block-column has
                # N_terms[ir] rows and N_terms[ic] columns;
            N_loc = N_terms_[i_nl]
            i_end_r = i_start_r + N_loc
    
            # *** diagonal blocks ***
            i_start_c = i_start_r
            i_end_c   = i_end_r
            A[i_start_r:i_end_r, i_start_c:i_end_c] = form_block(0, i_nl, F_terms_loc)
            del i_start_c, i_end_c

            # *** left-sideband blocks ***
            if i_nl > 0:
                i_start_c = i_start_r - N_terms_[i_nl - 1]
                i_end_c   = i_start_c + N_terms_[i_nl - 1]
                A[i_start_r:i_end_r, i_start_c:i_end_c] = form_block(-1, i_nl, F_terms_loc)
                del i_start_c, i_end_c

            # *** blocks in several right sidebands ***  
            i_start_c = i_start_r
            i_end_c = None
            for jj in range(N_nl_sys_ - 1): # consider (N_nl_sys - 1) right sidebands
                jj_sb = jj + 1
                if i_nl < (N_nl_emb_ - jj_sb): # the block-rows near the righ-hand side of the matrix does not 
                            # include one or several right sidebands
                    i_start_c += N_terms_[i_nl + jj]
                    i_end_c   = i_start_c + N_terms_[i_nl + jj_sb]
                    A[i_start_r:i_end_r, i_start_c:i_end_c] = form_block(jj_sb, i_nl, F_terms_loc)
            del i_start_c, i_end_c

            # *** shift the starting row ***
            i_start_r += N_loc
        return A, B
    # --------------------------------------------------------------
    def prepare_init(x_init_loc):
            xs_init = np.zeros(N_tot_)
            prod = 1.
            i_start = 0
            for i_nl in range(N_nl_emb_):
                i_end = i_start + N_terms_[i_nl]

                # *** i_nl-th Kronecker product ***
                prod = np.kron(x_init_loc, prod)

                # *** set the initial conditions for the terms of the i_nl-th Carleman term ***
                xs_init[i_start:i_end] = prod[:]
                i_start = i_end
            return xs_init
    # --------------------------------------------------------------
    def solve_within_a_single_chart(zeta_chart, zeta_gl_curr, t_init):
        sol_loc = np.zeros((Nt_, Nx_))
        t_array = np.zeros(Nt_)
        A_chart, B_chart = prepare_matrices_AB( 
            return_sys_(shift_coefs_(sys_coefs_, zeta_gl_curr)) 
        )
        xs_init_sys  = prepare_init(np.zeros(Nx_))

        # # save initial conditions:
        # t_array[0]   = t_init
        # sol_loc[0,:] = 0.

        oo = RK45(
            lambda t, x: f_to_RK_(t, x, A_chart, B_chart), 
            t_init, xs_init_sys, t_max_, max_step=dt_
        )
        counter_t = -1
        while mix.compare_two_strings(oo.status, "running"):
            oo.step()
            if (counter_t + 1) >= len(t_array):
                t_array = np.pad(t_array, (0, Nt_), 'constant')
                sol_loc = np.pad(sol_loc, ((0, Nt_), (0, 0)), mode='constant')

            x_loc = np.array(oo.y[:Nx_])
            if get_radius(x_loc) >= zeta_chart:
                break
            else:
                counter_t += 1
                t_array[counter_t]   = oo.t
                sol_loc[counter_t,:] = x_loc
        t_array = t_array[:counter_t+1]
        sol_loc = sol_loc[:counter_t+1,:]
        return t_array, sol_loc
    # --------------------------------------------------------------
    def compare_two_charts(
            t_non_shifted, xs_non_shifted, 
            zeta_non_shifted, zeta_shifted,
            zeta_gl_non_shifted, zeta_gl_shifted,
            is_shifted_reduced, # Whether the shifted chart is the reduced chart?
            t_array = None, sol_non_shifted = None
        ):
        # --- Compare non-shifted and shifted charts ---
        # > The non-shifted chart has a bigger radius, i.e. zeta_non_shifted;
        # > The non-shifted chart starts from a nonzero local coordinate, i.e. xs_non_shifted[-1];
        # > The non-shifted chart is supposed to be more unstable;
        # --------------------------------------------------------------
        if t_array is None:
            # --- The smaller chart was shifted only once ---
            sol_non_shifted = np.zeros((Nt_, Nx_))
            t_array         = np.zeros(Nt_)
            counter_t = -1
            x_non_shifted_next = np.array(xs_non_shifted[-1])
            t_curr = t_non_shifted[-1]
        else:
            # --- The smaller chart was shifted twice or more times ---
            counter_t = len(t_array) - 1
            x_non_shifted_next = np.array(sol_non_shifted[-1])
            t_curr  = t_array[-1]
        x_shifted_next     = np.zeros(Nx_)
        
        A_non_shifted, B_non_shifted = prepare_matrices_AB( 
            return_sys_(shift_coefs_(sys_coefs_, zeta_gl_non_shifted)) 
        )
        A_shifted, B_shifted = prepare_matrices_AB( 
            return_sys_(shift_coefs_(sys_coefs_, zeta_gl_shifted)) 
        )

        t_next = -1
        flag_take_ref = True if is_shifted_reduced else False
        flag_last_time_step = False
        while t_next < t_max_:
            dt_curr = dt_
            if (t_curr + dt_curr) >= t_max_:
                flag_last_time_step = True
                dt_curr = (t_max_ - t_curr)
            t_next = t_curr + dt_curr

            x_non_shifted_EMB = prepare_init(x_non_shifted_next)
            x_shifted_EMB     = prepare_init(x_shifted_next)

            # --- Run the RK45 solver to find result at next time step ---
            oo_non_shifted = RK45(
                lambda t, x: f_to_RK_(t, x, A_non_shifted, B_non_shifted), 
                t_curr, x_non_shifted_EMB, t_next, max_step=dt_curr
            )
            while oo_non_shifted.status == 'running':
                oo_non_shifted.step()
            x_non_shifted_next_EMB = oo_non_shifted.y

            oo_shifted = RK45(
                lambda t, x: f_to_RK_(t, x, A_shifted, B_shifted), 
                t_curr, x_shifted_EMB, t_next, max_step=dt_curr
            )
            while oo_shifted.status == 'running':
                oo_shifted.step()
            x_shifted_next_EMB = oo_shifted.y
            
            # --- Extract solution of the nonlinear problem from the Carleman embedding ---
            x_non_shifted_next = x_non_shifted_next_EMB[:Nx_]
            x_shifted_next     = x_shifted_next_EMB[:Nx_]

            # --- Are there unstable Carleman components? ---
            N_unstable = len(x_non_shifted_next_EMB[np.abs(x_non_shifted_next_EMB) > 1])
            flag_unstable = False
            if N_unstable > 0:
                print(">>> The non-shifted chart is unstable")
                flag_unstable = True

            # --- Compare two trajectories ---
            abs_err = get_radius(x_non_shifted_next - x_shifted_next + zeta_gl_non_shifted - zeta_gl_shifted)
            if is_shifted_reduced:
                # -----------------------------------------------------------------------------
                # --- Comparison between the reference and shifted reduced charts ---
                # -----------------------------------------------------------------------------
                if abs_err >= err_tolerance_ or flag_unstable:
                    return t_non_shifted, xs_non_shifted, False # reduce the chart
                else:
                    if (counter_t + 1) >= len(t_array):
                        t_array         = np.pad(t_array, (0, Nt_), 'constant')
                        sol_non_shifted = np.pad(sol_non_shifted, ((0, Nt_), (0, 0)), mode='constant')

                    if get_radius(x_non_shifted_next) >= zeta_non_shifted:
                        break # the reference chart is taken  
                    else:
                        # --- Save the results computed in the non-shifted chart ---
                        counter_t += 1
                        t_array[counter_t]           = t_next
                        sol_non_shifted[counter_t,:] = x_non_shifted_next 
            else:
                # -----------------------------------------------------------------------------
                # --- Comparison between the enlarged and shifted reference charts ---
                # -----------------------------------------------------------------------------
                if abs_err >= err_tolerance_ or flag_unstable:
                    flag_take_ref = True # the reference chart is taken  
                    break
                else:
                    if (counter_t + 1) >= len(t_array):
                        t_array = np.pad(t_array, (0, Nt_), 'constant')
                        sol_non_shifted = np.pad(sol_non_shifted, ((0, Nt_), (0, 0)), mode='constant')

                    if get_radius(x_non_shifted_next) >= zeta_non_shifted:
                        break # the enlarged chart is taken  
                    else:
                        # --- save the results computed in the non-shifted chart ---
                        counter_t += 1
                        t_array[counter_t]           = t_next
                        sol_non_shifted[counter_t,:] = x_non_shifted_next

            # --- Is the trajectory outside the shifted (smaller) chart? ---
            # --- If yes, shift the shifted chart again ---
            if get_radius(x_shifted_next) >= zeta_shifted and not flag_last_time_step: 
                t_array         = t_array[:counter_t+1]
                sol_non_shifted = sol_non_shifted[:counter_t+1,:]
                return compare_two_charts(
                    t_non_shifted, xs_non_shifted, 
                    zeta_non_shifted, zeta_shifted,
                    zeta_gl_non_shifted, 
                    zeta_gl_shifted + x_shifted_next,
                    is_shifted_reduced,
                    t_array         = t_array, 
                    sol_non_shifted = sol_non_shifted,
                )
            
            # --- consider the next time step ---
            t_curr = t_next
        t_array         = t_array[:counter_t+1]
        sol_non_shifted = sol_non_shifted[:counter_t+1,:]
        return t_array, sol_non_shifted, flag_take_ref
    # --------------------------------------------------------------
    def adaptive(t_init, zeta1, zeta_gl_curr):
        xs_res = np.empty((0, Nx_))
        t_array = np.empty(0)

        zeta_ref  = zeta1
        zeta_comp = change_radius(zeta_ref, flag_reduce=True)
        flag_consider_enlarged = True
        if zeta_comp >= zeta_min_:
            # --- Simulate the reduced chart ---
            t_array_reduc_chart, xs_reduc_chart = solve_within_a_single_chart(zeta_comp, zeta_gl_curr, t_init)
            zeta_gl_red = zeta_gl_curr + xs_reduc_chart[-1]
            if t_array_reduc_chart[-1] >= t_max_:
                return t_array_reduc_chart, xs_reduc_chart, zeta_ref, zeta_gl_red
            
            # --------------------------------------------------------------------
            # --- Compare the reference and shifted reduced charts ---
            # --------------------------------------------------------------------
            while zeta_comp >= zeta_min_:
                t_array_comp, xs_res_from_comp, flag_take_ref = compare_two_charts(
                    t_array_reduc_chart, xs_reduc_chart, 
                    zeta_ref, zeta_comp,
                    zeta_gl_curr, zeta_gl_red,
                    is_shifted_reduced = True 
                )
                
                if flag_take_ref:
                    break  # take the reference chart
                else:
                    print("Take a smaller chart: new zeta = {:0.3e}".format(zeta_comp))

                    # --- consider an even smaller chart ---
                    zeta_ref  = zeta_comp
                    zeta_comp = change_radius(zeta_ref, flag_reduce=True)

                    # --- compute the trajectory in the smaller chart ---
                    t_array_reduc_chart, xs_reduc_chart = solve_within_a_single_chart(zeta_comp, zeta_gl_curr, t_init)
                    zeta_gl_red = zeta_gl_curr + xs_reduc_chart[-1]
                    flag_consider_enlarged = False  # the chart has been reduced once, no need to consider an enlarged chart
            if zeta_comp <= zeta_min_:
                print("\nWarning: the minimal chart radius was achieved, zeta_comp, zeta_min: {:0.3e}, {:0.3e}.\n".format(
                    zeta_comp, zeta_min_
                ))
            xs_res  = np.concatenate((xs_res,  xs_reduc_chart,      xs_res_from_comp), axis = 0)
            t_array = np.concatenate((t_array, t_array_reduc_chart, t_array_comp))
        else:
            # --- Simulate just the reference chart ---
            t_array_ref_chart, xs_ref_chart = solve_within_a_single_chart(zeta_ref, zeta_gl_curr, t_init)
            xs_res  = np.concatenate((xs_res, xs_ref_chart), axis = 0)
            t_array = np.concatenate((t_array, t_array_ref_chart))
            del t_array_ref_chart, xs_ref_chart

        # --------------------------------------------------------------------
        # --- Comparison with an enlarged chart ---
        # --------------------------------------------------------------------
        if flag_consider_enlarged:
            zeta_comp = change_radius(zeta_ref, flag_reduce=False)
            while zeta_comp < zeta_max_ and t_array[-1] < t_max_:
                t_array_comp, xs_res_from_comp, flag_take_ref = compare_two_charts(
                    t_array, xs_res, 
                    zeta_comp, zeta_ref,
                    zeta_gl_curr, zeta_gl_curr + xs_res[-1, :],
                    is_shifted_reduced = False 
                )
                xs_res  = np.concatenate((xs_res,  xs_res_from_comp), axis = 0)
                t_array = np.concatenate((t_array, t_array_comp))
                if flag_take_ref:
                    return t_array, xs_res, zeta_ref, zeta_gl_curr + xs_res[-1, :]  # take the reference chart
                else:
                    # --- consider an even larger chart ---
                    print("Take a larger chart: new zeta = {:0.3e}".format(zeta_comp))
                    zeta_ref  = zeta_comp
                    zeta_comp = change_radius(zeta_ref, flag_reduce=False)
        return t_array, xs_res, zeta_ref, zeta_gl_curr + xs_res[-1, :]  
    # --------------------------------------------------------------

    # --- Check parameters ---
    if zeta_thresh_ <= zeta_step_:
        print("Error: zeta_thresh_ must be > zeta_step_")
        return None, None, None, None
    # if zeta_init_ > zeta_max_:
    #     print("Error: zeta_init_ must be <= zeta_max_")
    #     return None, None, None, None
    if zeta_step_ >= zeta_init_ or zeta_step_ >= zeta_max_:
        print("Error: zeta_step_ must be < zeta_init_ and zeta_max_")
        return None, None, None, None

    
    # --- the number of variables ---
    Nx_ = len(x_init_cond_)

    # --- time parameters ---
    Nt_ = len(t_)
    dt_ = np.diff(t_)[0]
    t_max_ = t_[-1]

    # --- Arrays to store result ---
    t_array_   = np.zeros(1)
    sol_       = np.zeros((1, Nx_))
    zeta_save_ = np.empty(0)

    # --- Save initial conditions ---
    t_array_[0]    = 0.
    sol_[0, :]     = np.array(x_init_cond_)

    # --- The number of elements for each next nonlinear term in embedding ---
    N_terms_ = np.zeros(N_nl_emb_, dtype=int)
    N_tot_ = 0
    for ii in range(N_nl_emb_):
        N_terms_[ii] = int(Nx_**(ii+1))
        N_tot_ += N_terms_[ii]

    # --- Prepare unit matrices ---
    eyes_list = [None] * N_nl_emb_
    for ii in range(N_nl_emb_):
        eyes_list[ii] = np.eye(int(Nx_**ii))

    # --- Computation ---
    t_array_next = [0]
    zeta_gl   = np.array(x_init_cond_)
    zeta_curr = zeta_init_
    while t_array_next[-1] < t_[-1]:
        t_array_next, xs_next, zeta_curr, _ = adaptive(
            t_array_next[-1], zeta_curr, zeta_gl
        )
        sol_       = np.concatenate((sol_,  xs_next + zeta_gl), axis = 0) 
        if len(t_array_) == 1:
            zeta_save_ = np.concatenate((zeta_save_, np.full(len(t_array_next)+1, zeta_curr)))
        else:
            zeta_save_ = np.concatenate((zeta_save_, np.full(len(t_array_next), zeta_curr)))
        t_array_   = np.concatenate((t_array_, t_array_next))

        zeta_gl += xs_next[-1,:]  # must be after saving sol_

        print("\n--- t, zeta: {:0.3e}, {:0.3e} ---".format(t_array_next[-1], zeta_curr))
    print("Done")
    return sol_, t_array_, "PAE", zeta_save_


# -----------------------------------------------------------------------------------------------
# --- Compare classical with Carleman simulations  ---
# -----------------------------------------------------------------------------------------------
def compare_trajectory_cl_and_carleman(
        t_ref, sol_ref, t_emb, sol_emb, 
        flag_save = False, path_save = None, case_title = None, case_emb = None,
        fontsize = 20, fig_size = (10,9),
        radii_emb = None, 
        step_t = 1,
        flag_var_semilogy = True,
        flag_max_error = False
    ):
    colors_loc     = ["orange", "red", "green", "gray", "black"]
    linestyles_loc = ["-", "--", "--", ":"]
    name_vars      = ["x", "y", "z"]
    fontsize_leg = int(fontsize/4. * 3.)

    Nvar = np.shape(sol_ref)[1]
    s_ref     = [None] * Nvar
    s_emb     = [None] * Nvar
    s_emb_int = [None] * Nvar

    flag_3d = False
    if Nvar == 3:
        flag_3d = True

    # --- Classical and Carleman results ---
    label_lines = ["x", "y", "z"]
    for ivar in range(Nvar):
        s_ref[ivar] = sol_ref[:,ivar]
        s_emb[ivar] = sol_emb[:,ivar]

    # --------------------------------------------------------------
    # --- Plotting: trajectories ---
    # --------------------------------------------------------------
    if Nvar > 1:
        if not flag_3d:
            fig, axs = plt.subplots(3, 1, figsize=fig_size)
            ax_loc = axs[0]

            ax_loc.plot(s_ref[0], s_ref[1], color="b", linewidth = 2, linestyle='-', label = "CL")
            ax_loc.plot(s_emb[0], s_emb[1], color="r", linewidth = 2, linestyle="--", label = "CA")

            # # --- Save data ---
            # if flag_save:
            #     mix.save_dat_plot_1d_file(
            #         path_save + "/REF_{:s}_xy.dat".format(case_title), 
            #         s_ref[0], s_ref[1]
            #     )
            #     mix.save_dat_plot_1d_file(
            #         path_save + "/EMB_{:s}_{:s}_xy.dat".format(case_title, case_emb), 
            #         s_emb[0], s_emb[1]
            #     )
        else:
            fig = plt.figure(figsize=fig_size)
            ax_loc = fig.add_subplot(111, projection='3d')
            ax_loc.plot(
                s_ref[0], s_ref[1], s_ref[2], 
                color="b", linewidth = 2, linestyle='-', label = "CL"
            )
            ax_loc.plot(
                s_emb[0][::step_t], s_emb[1][::step_t], s_emb[2][::step_t], 
                color="r", linewidth = 2, linestyle="--", label = "CA"
            )
            ax_loc.set_zlabel("$z$", fontsize = fontsize) 

            # # --- Save data ---
            # if flag_save:
            #     mix.save_dat_plot_3d_trajectory_file(
            #         path_save + "/REF_{:s}_xyz.dat".format(case_title), 
            #         s_ref[0], s_ref[1], s_ref[2]
            #     )
            #     mix.save_dat_plot_3d_trajectory_file(
            #         path_save + "/EMB_{:s}_{:s}_xyz.dat".format(case_title, case_emb), 
            #         s_emb[0][::step_t], s_emb[1][::step_t], s_emb[2][::step_t]
            #     )
        ax_loc.set_xlabel("$x$", fontsize = fontsize)
        ax_loc.set_ylabel("$y$", fontsize = fontsize)       
        offset_text = ax_loc.yaxis.get_offset_text()
        offset_text.set_fontsize(fontsize) 

        ax_loc.legend(fontsize = fontsize_leg)
        ax_loc.grid(True)
        ax_loc.tick_params(axis='both', which='major', labelsize=fontsize)
        # ax_loc.set_xticklabels([])

    # --------------------------------------------------------------
    # --- Separate variables ---
    # --------------------------------------------------------------
    if Nvar == 1:
        fig, axs = plt.subplots(2, 1, figsize=fig_size)
        ax_loc = axs[0]
        str_x = "$|x|$" if(flag_var_semilogy)  else "$x$"
        ax_loc.set_ylabel(str_x, fontsize = fontsize)
    if Nvar == 2:
        ax_loc = axs[1]
        str_x = "$|x|,|y|$" if(flag_var_semilogy)  else "$x,y$"
        ax_loc.set_ylabel(str_x, fontsize = fontsize)
    if Nvar == 3:
        fig, axs = plt.subplots(2, 1, figsize=fig_size)
        ax_loc = axs[0]
        str_x = "$|x|,|y|,|z|$" if(flag_var_semilogy)  else "$x,y,z$"
        ax_loc.set_ylabel(str_x, fontsize = fontsize)
    

    for ivar in range(Nvar):
        if flag_var_semilogy:
            ax_loc.semilogy(
                t_ref, np.abs(s_ref[ivar]),                 
                color="black",          linewidth = 2, linestyle='-', label = "CL" if ivar == 0 else None
            )
            ax_loc.semilogy(
                t_emb[::step_t], np.abs(s_emb[ivar][::step_t]), 
                color=colors_loc[ivar], linewidth = 2, linestyle='--', label = "CA: {:s}".format(label_lines[ivar])
            )
        else:
            ax_loc.plot(
                t_ref, s_ref[ivar],                 
                color="black",          linewidth = 2, linestyle='-', label = "CL" if ivar == 0 else None
            )
            ax_loc.plot(
                t_emb[::step_t], s_emb[ivar][::step_t], 
                color=colors_loc[ivar], linewidth = 2, linestyle='--', label = "CA: {:s}".format(label_lines[ivar])
            )
            
        # # --- Save data ---
        # if flag_save:
        #     mix.save_dat_plot_1d_file(
        #         path_save + "/REF_{:s}_{:s}.dat".format(case_title, label_lines[ivar]), 
        #         t_ref, s_ref[ivar], 
        #     )
        #     mix.save_dat_plot_1d_file(
        #         path_save + "/EMB_{:s}_{:s}_{:s}.dat".format(case_title, case_emb, label_lines[ivar]), 
        #         t_emb[::step_t], s_emb[ivar][::step_t], 
        #     )

    offset_text = ax_loc.yaxis.get_offset_text()
    offset_text.set_fontsize(fontsize) 

    ax_loc.legend(fontsize = fontsize_leg)
    ax_loc.grid(True)
    ax_loc.tick_params(axis='both', which='major', labelsize=fontsize)

    # --------------------------------------------------------------
    # --- Difference between signals ---
    # --------------------------------------------------------------
    ax_loc = axs[1] if Nvar == 1 or Nvar == 3 else axs[2]
    abs_val = 0.
    for ivar in range(Nvar):
        abs_val += abs(s_ref[ivar])**2
    abs_val = np.sqrt(abs_val)

    if not flag_max_error:
        for ivar in range(Nvar):
            s_emb_int[ivar] = np.interp(t_ref, t_emb, s_emb[ivar])

            # err = abs(s_ref[ivar] - s_emb_int[ivar]) 
            # err = abs(s_ref[ivar] - s_emb_int[ivar]) / abs(s_ref[ivar])
            err = abs(s_ref[ivar] - s_emb_int[ivar]) / abs_val

            ax_loc.semilogy(
                t_ref, err, 
                color=colors_loc[ivar], linewidth = 2, linestyle='-'
            )
            ax_loc.set_ylabel("$|CL - CA|/|CL|$", fontsize = fontsize)

            # --- Save data ---
            if flag_save:
                mix.save_dat_plot_1d_file(
                    path_save + "/EMB_{:s}_{:s}_err_{:s}.dat".format(case_title, case_emb, name_vars[ivar]), 
                    t_ref, err
                )
    else:
        errs = np.zeros((Nvar, len(t_ref)))
        for ivar in range(Nvar):
            s_emb_int[ivar] = np.interp(t_ref, t_emb, s_emb[ivar])
            # errs[ivar] = abs(s_ref[ivar] - s_emb_int[ivar]) / abs(s_ref[ivar])
            errs[ivar] = abs(s_ref[ivar] - s_emb_int[ivar]) / abs_val

        max_err = np.max(errs, axis=0)
        ax_loc.semilogy(
            t_ref, max_err, 
            color=colors_loc[ivar], linewidth = 2, linestyle='-'
        )
        ax_loc.set_ylabel("$max(|CL - CA|/|CL|)$", fontsize = fontsize)

        # --- Save data ---
        if flag_save:
            mix.save_dat_plot_1d_file(
                path_save + "/EMB_{:s}_{:s}_max_err.dat".format(case_title, case_emb), 
                t_ref, max_err
            )

    ax_loc.set_xlabel('$t$', fontsize = fontsize)
    offset_text = ax_loc.yaxis.get_offset_text()
    offset_text.set_fontsize(fontsize) 
    # ax_loc.legend(fontsize = fontsize_leg)
    ax_loc.grid(True)
    ax_loc.tick_params(axis='both', which='major', labelsize=fontsize)

    plt.tight_layout()

    # --- Plot the linearization radius separately ---
    if radii_emb is not None:
        plt.figure(figsize=(10, 3))
        plt.plot(t_emb[::step_t], radii_emb[::step_t], color='blue')
        plt.xlabel('t')
        plt.ylabel('lin. radius')
        plt.legend()
        plt.grid(True)

        # --- Save data ---
        if flag_save:
            mix.save_dat_plot_1d_file(
                path_save + "/EMB_{:s}_{:s}_radius.dat".format(case_title, case_emb), 
                t_emb[::step_t], radii_emb[::step_t]
            )
    return



# -----------------------------------------------------------------------------------------------
# --- Printing GCE grid  ---
# -----------------------------------------------------------------------------------------------
def print_GCE_grid(X_GC, Zeta, max_v):
    grid_desc = None
    if X_GC is not None:
        print("\n\n ----------------------------------------------")
        print("--- GCE grid ---")
        print("----------------------------------------------")
        print("center: ")
        mix.print_array(X_GC, ff = [14, 3, "e"])

        print("sizes: ")
        mix.print_array(Zeta, ff = [14, 3, "e"])

        print()

        Nx = len(X_GC)
        str_dims = ["x", "y", "z", "w"]
        grid_desc = []
        for ii in range(Nx):
            xc = X_GC[ii]
            xi = Zeta[ii]
            d_xi = 2. * xi
            l_max = int(np.ceil( (max_v - xc)/d_xi ))
            if l_max < 0:
                print("\n>>> Error: the grid center is beyond max_v")
                return None

            grid_desc_x = np.zeros(2 * l_max + 1 + 1)
            counter_el = -1
            for jj in range(-l_max, l_max + 1):
                counter_el += 1
                grid_desc_x[counter_el] = xc + jj * d_xi - xi # left boundaries of grid tiles
            grid_desc_x[-1] = xc + l_max * d_xi + d_xi # the right boundary of the rightmost tile

            grid_desc.append(np.array(grid_desc_x))

            print("--- {:s} ---".format(str_dims[ii]))
            mix.print_array(grid_desc_x, ff = [14, 3, "e"], n_in_row=len(grid_desc_x))
    return grid_desc