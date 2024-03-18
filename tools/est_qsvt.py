import numpy as np
import h5py
import matplotlib.pyplot as plt
from numba import jit
import cvxpy as cp


import pylib.qsvt_angles as qsvt_a
import pylib.mix as mix


FIG_SIZE_W_ = 8
FIG_SIZE_H_ = 6

SEL_EST_NA_ = 1

   
def reload():
    mix.reload_module(mix)
    mix.reload_module(qsvt_a)
    return

# ----------------------------------------------------------------------------------------
# --- Read reference QSVT angles. ---
def read_ref_QSVT_angles(id_case = 0, Ncoefs = 40):

    # the name of the output file:
    fname_ = "QSVT-MI-estimation-coefs-case{:d}-Nc{:d}.hdf5".format(id_case, Ncoefs)

    # paths where pre-computed QSVT angles are saved and
    #     where resulting parameters for the estimation of the QSVT angles will be saved:
    path_root_       = "./tools/QSVT-angles/inversion/"
    path_root_ref = "./tools/QSVT-angles/inversion/ref-angles"
    path_save_plots_ = path_root_ + "/saved-plots/"

    # -------------------------------------------------------------------------
    if id_case == 0:
        # names of the .hdf5 files where pre-computed QSVT angles are saved:
        path_root_ref = "./tools/QSVT-angles/inversion/ref-angles"
        filenames = \
        [
            "k40_eps13.hdf5",   # id = 0
            "k80_eps13.hdf5",   # id = 1
            "k100_eps13.hdf5",  # id = 2
            "k150_eps13.hdf5",  # id = 3
            "k180_eps13.hdf5",  # id = 4
        ]  

        # the project used to compute the shape of the QSVT angles:
        id_comp_ = 2

    # -------------------------------------------------------------------------
    if id_case == 1:
        path_root_ref = "./tools/QSVT-angles/inversion/ref-angles"
        filenames = \
        [
            "k40_eps13.hdf5",   # id = 0 
            "k80_eps13.hdf5",   # id = 1 
            "k100_eps13.hdf5",  # id = 2
            "k150_eps13.hdf5",  # id = 3
            "k180_eps13.hdf5",  # id = 4
            "k300_eps13.hdf5",  # id = 5
        ]  
        id_comp_ = 5 # one of the smallest error !!!

    # -------------------------------------------------------------------------
    if id_case == 2:
        path_root_ref = "./tools/QSVT-angles/inversion/ref-angles"
        filenames = \
        [
            "k40_eps13.hdf5",   # id = 0
            "k80_eps13.hdf5",   # id = 1
            "k100_eps13.hdf5",  # id = 2
            "k150_eps13.hdf5",  # id = 3
            "k180_eps13.hdf5",  # id = 4
        ]  
        id_comp_ = 4

    # -------------------------------------------------------------------------
    if id_case == 3:  # !!! MAIN !!!
        path_root_ref = "./tools/QSVT-angles/inversion/ref-angles-2"
        filenames = \
        [
            "k20_eps13.hdf5",   # id = 0
            "k40_eps13.hdf5",   # id = 1
            "k60_eps13.hdf5",   # id = 2
            "k80_eps13.hdf5",   # id = 3
            "k100_eps13.hdf5",  # id = 4
            "k120_eps13.hdf5",  # id = 5
        ]  
        id_comp_ = 3

    # -------------------------------------------------------------------------
    if id_case == 4:
        path_root_ref = "./tools/QSVT-angles/inversion/ref-angles-3"
        filenames = \
        [
            "k10_eps11.hdf5",   
            "k20_eps11.hdf5",   
            "k30_eps11.hdf5",   
            "k40_eps11.hdf5",   
            "k50_eps11.hdf5",  
            "k60_eps11.hdf5",  
            "k70_eps11.hdf5",  
            "k80_eps11.hdf5",  
            "k90_eps11.hdf5",  
            "k100_eps11.hdf5", 
            "k110_eps11.hdf5", 
            "k120_eps11.hdf5", 
            "k130_eps11.hdf5", 
            "k140_eps11.hdf5", 
            "k150_eps11.hdf5", 
            "k160_eps11.hdf5",  
        ]  
        id_comp_ = 15 # is not used for this id_case;

    # -------------------------------------------------------------------------
    if id_case == 5:  
        path_root_ref = "./tools/QSVT-angles/inversion/ref-angles-4"
        filenames = \
        [
            "k20_eps6.hdf5",   # id = 0
            "k40_eps6.hdf5",   # id = 1
            "k80_eps6.hdf5",   # id = 2
            "k100_eps6.hdf5",  # id = 3
            "k120_eps6.hdf5",  # id = 4
            "k140_eps6.hdf5",  # id = 5
            "k160_eps6.hdf5",  # id = 6
        ]  
        id_comp_ = 6

    # -------------------------------------------------------------------------
    if id_case == 6:  
        path_root_ref = "./tools/QSVT-angles/inversion/ref-angles-5"
        filenames = \
        [
            "k100_eps8.hdf5",  # id = 0
            "k120_eps8.hdf5",  # id = 1
            "k150_eps8.hdf5",  # id = 2
            "k200_eps8.hdf5",  # id = 3
            "k300_eps8.hdf5",  # id = 4
            "k600_eps8.hdf5",  # id = 5
        ]  
        id_comp_ = 5

    # -------------------------------------------------------------------------
    if id_case == 7:  
        path_root_ref = "./tools/QSVT-angles/inversion/ref-angles-5"
        filenames = \
        [
            "k300_eps8.hdf5",  # id = 0
            "k350_eps8.hdf5",  # id = 1
            "k400_eps8.hdf5",  # id = 2
            "k450_eps8.hdf5",  # id = 3
            "k500_eps8.hdf5",  # id = 4
            "k550_eps8.hdf5",  # id = 5
            "k600_eps8.hdf5",  # id = 6
            "k650_eps8.hdf5",  # id = 7
            "k700_eps8.hdf5",  # id = 8
        ]  
        id_comp_ = 8

    # -------------------------------------------------------------------------
    if id_case == 8:  
        path_root_ref = "./tools/QSVT-angles/inversion/ref-angles-5"
        filenames = []
        for ii in range(10, 660, 10):
            filenames.append(
                "k{:d}_eps8.hdf5".format(ii)
            )
        id_comp_ = len(filenames) - 1

    # read the QSVT angles computed using the L-BFGS approach [Dong-21-DOI:10.1103/PhysRevA.103.042419]:
    dds_ = []
    for ii in range(len(filenames)):
        print("\n----------------------------------------")
        dds_.append(qsvt_a.read_angles(path_root_ref, filenames[ii]))
    del ii
    return dds_, id_comp_, fname_, path_root_, path_save_plots_


# ----------------------------------------------------------------------------------------
def plot_angles(
        dds, ids_plot, 
        xlim = None, 
        flag_save = False, 
        path_save_plots = None,
        flag_shifted = False
    ):
    colors = ["blue", "red", "green", "gray", "black"]

    str_ylabel = "phis"
    if flag_shifted:
        str_ylabel += " - pi/2"

    str_save_start = ""
    if flag_shifted:
        str_save_start = "shifted_"

    fig = plt.figure(figsize=(FIG_SIZE_W_,FIG_SIZE_H_))
    ax = fig.add_subplot(111)
    count_plot = -1
    for ii in ids_plot: 
        count_plot+= 1

        phis_plot = np.array(dds[ii]["phis"])
        Nphis = len(phis_plot) 
        if flag_shifted:
            phis_plot -= np.pi/2.

        x_array = np.array(range(Nphis)) - Nphis/2
        ax.plot(
            x_array, 
            phis_plot, 
            color=colors[count_plot], 
            linewidth = 2, linestyle='-',
            label = "k = {:d}, log_e = {:d}".format(
                int(dds[ii]["function-parameter"]),
                -int(np.log10(dds[ii]["abs-error"]))
            )
        )

        # --- Save the plots if necessary ---
        if flag_save:
            mix.save_dat_plot_1d_file(
                path_save_plots + "/{:s}phis_k{:d}.dat".format(
                    str_save_start,
                    int(dds[ii]['function-parameter'])
                ), 
                x_array, 
                phis_plot
            )
    plt.xlabel('i - N/2')
    plt.ylabel(str_ylabel)
    if xlim is not None:
        plt.xlim(xlim)
    ax.legend()
    plt.grid(True)
    plt.show()
    return 


# ----------------------------------------------------------------------------------------
# Plot maximum positive and negative angles for various kappa.
def plot_max(dds, flag_save, path_save_plots):
    Npr = len(dds)
    array_pos  = np.zeros(Npr)
    array_neg  = np.zeros(Npr)
    kappas     = np.zeros(Npr)
    for count_pr in range(Npr): 
        dd1 = dds[count_pr]
        phis = np.array(dd1["phis"] - np.pi/2.)
        array_pos[count_pr] = np.max(phis)
        array_neg[count_pr] = np.max(-phis)
        kappas[count_pr]    = dd1["function-parameter"]
    array_diff = array_neg - array_pos

    print("maximum difference between pos. and neg. max angles: {:0.3e}".format(
        np.max(np.abs(array_diff))
    ))

    # --- Plot positive and negatives maximums ---
    fig = plt.figure(figsize=(FIG_SIZE_W_,FIG_SIZE_H_))
    ax = fig.add_subplot(111)
    ax.plot(kappas, array_pos, "b:", linewidth = 2, marker = "o", label = "pos")
    ax.plot(kappas, array_neg, "r:", linewidth = 2, marker = "o", label = "abs(neg)")
    plt.xlabel('kappa')
    plt.title("pos = max(phis-pi/2);   neg = max(pi/2 - phis)")
    ax.legend()
    plt.grid(True)
    plt.show()

    # --- Plot the difference between the positive and negatives maximums ---
    fig = plt.figure(figsize=(FIG_SIZE_W_,FIG_SIZE_H_))
    ax = fig.add_subplot(111)
    ax.plot(kappas, np.log10(array_diff), "b:", linewidth = 2, marker = "o")
    plt.xlabel('kappa')
    plt.title("log10(abs(neg) - pos)")
    plt.grid(True)
    plt.show()

    # --- Saving data ---
    if flag_save:
        mix.save_dat_plot_1d_file(
            path_save_plots + "/log_diff_pos_neg.dat", 
            kappas, 
            np.log10(array_diff)
        )
    return


# ---------------------------------------------------------------------------------------------
# --- Compute coefficients-envelope for various kappa ---
def compute_coefs_var_kappa(dds, Ncoefs):
    Nk = len(dds)
    kappas = np.zeros(Nk)
    cns = np.zeros((Nk, Ncoefs))
    cps = np.zeros((Nk, Ncoefs))
    counter_case = -1
    for dd in dds:
        counter_case += 1
        kappas[counter_case] = dd["function-parameter"]
        cn, cp, _, _, _ = compute_coefs_envelop(
            dd, Ncoefs = Ncoefs, 
            flag_plot_envelop = False, 
            flag_plot_shape = False,
            flag_reconstruct = False
        )
        cns[counter_case,:] = cn[:]
        cps[counter_case,:] = cp[:]
    return cns, cps, kappas


# ---------------------------------------------------------------------------------------------
# --- Plot coefficients-envelope for various kappa ---
def plot_coefs_var_kappa(cns, cps, kappas, ids_ch_coef):
    def plot_one_coef(coefs_arr, str_coef):
        colors = ["b", "r", "g", "orange", "magenta", "black"]

        fig = plt.figure(figsize=(FIG_SIZE_W_, FIG_SIZE_H_))
        ax = fig.add_subplot(111)
        coefs = np.zeros(len(coefs_arr))
        counter_1 = -1
        for id_ch_coef in ids_ch_coef:
            counter_1 += 1
            for i_kappa in range(len(coefs_arr)):
                coefs[i_kappa] = coefs_arr[i_kappa][id_ch_coef]
            ax.plot(
                kappas, 
                coefs, 
                color=colors[counter_1], linewidth = 2, linestyle=':', marker = "o",
                label = "coefs-{:s}[{:d}]".format(str_coef, id_ch_coef)

            )
        plt.xlabel('kappa')
        plt.ylabel("coefs-{:s}".format(str_coef))
        plt.grid(True)
        plt.legend()
        plt.show()     
        return
    # ------------------------------------------------------------------
    plot_one_coef(cps, "POS")
    plot_one_coef(cns, "NEG")
    return


# ---------------------------------------------------------------------------------------------
# --- Compute coefficients of series approximating coefs-envelope for various kappa ---
def test_func_FULL(x, a):
    res_pol = 0.
    Ncoefs = a.size
    for ii in range(Ncoefs):
        res_pol += a[ii] * np.cos(ii * np.arccos(x))
    return res_pol


def np_test_func_FULL(x, a):
    Nx = len(x)
    Ncoefs = len(a)
    res_pol = np.zeros(Nx)
    for ix in range(Nx):
        res_pol[ix] = 0.
        for ii in range(Ncoefs):
            res_pol[ix] += a[ii] * np.cos(ii * np.arccos(x[ix]))
    return res_pol


def compute_env_coefs_dependence_kappa(  
    Ncoefs_coefs,
    cns, cps, kappas,
    flag_reco = True
):
    def approx_prof_coef(cs, id_coef):
        prof_coef = cs[:, id_coef]
        x = np.linspace(0.0, 1.0, Nk)
        coefs_coefs = cp.Variable(Ncoefs_coefs)
        objective = cp.Minimize(cp.sum_squares(
            test_func_FULL(x, coefs_coefs) - prof_coef
        ))
        prob = cp.Problem(objective)
        result = prob.solve()
        rec_prof = np_test_func_FULL(x, coefs_coefs.value)
        return coefs_coefs.value, rec_prof
    
    def plot_reco_coef_var_kappa(id_coef, cs_ref, reco_f, str_coef):
        fig = plt.figure(figsize=(FIG_SIZE_W_,FIG_SIZE_H_))
        ax = fig.add_subplot(111)
        ax.plot(
            kappas, cs_ref[:, id_coef],  
            color = "b", linewidth = 2, linestyle='-', 
            label = "ref"
        )
        ax.plot(
            kappas, reco_f[id_coef,:],   
            color="r", linewidth = 2, linestyle=':', 
            label = "reco"
        )
        plt.xlabel('kappa')
        plt.title("{:s}-coef[{:d}](kappa)".format(str_coef, id_coef))
        plt.legend()
        plt.grid(True)
        plt.show()

        fig = plt.figure(figsize=(FIG_SIZE_W_,FIG_SIZE_H_))
        ax = fig.add_subplot(111)
        ax.plot(
            kappas, cs_ref[:, id_coef] - reco_f[id_coef,:],  
            color = "b", linewidth = 2, linestyle='-', 
            label = "ref"
        )
        plt.xlabel('kappa')
        plt.title("error: {:s}-coef[{:d}](kappa)".format(str_coef, id_coef))
        plt.legend()
        plt.grid(True)
        plt.show()
        return

    # --------------------------------------------------------
    Nk, Ncoefs = cns.shape

    # --- Approximate the dependence (with kappa) of each coefficient ---
    # --- by a sequence of Chebyschev coefficients ---
    ccns = np.zeros((Ncoefs, Ncoefs_coefs))
    ccps = np.zeros((Ncoefs, Ncoefs_coefs))
    rfn = np.zeros((Ncoefs, Nk)) # reproduced coefs-neg-envelope for each kappa
    rfp = np.zeros((Ncoefs, Nk)) # reproduced coefs-pos-envelope for each kappa
    for i_coef in range(Ncoefs):
        ccns[i_coef, :], rfn[i_coef, :] = approx_prof_coef(cns, i_coef)
        ccps[i_coef, :], rfp[i_coef, :] = approx_prof_coef(cps, i_coef)

    # --- Plotting reconstructed coefficient for various kappa ---
    if flag_reco:
        id_coef_plot = 10
        plot_reco_coef_var_kappa(id_coef_plot, cns, rfn, "NEG")
        plot_reco_coef_var_kappa(id_coef_plot, cps, rfp, "POS")
    return ccns, ccps


# ---------------------------------------------------------------------------------------------
# --- Reconstruct coefs-envelope for a particular kappa ---
def extrapolate_CEs(kappa_target, cns, cps, kappas):
    from scipy import interpolate
    _, Ncoefs = cns.shape
    coefs_neg = np.zeros(Ncoefs)
    coefs_pos = np.zeros(Ncoefs)
    for i_coef in range(Ncoefs):
        func_extr_neg = interpolate.interp1d(kappas, cns[:, i_coef], fill_value='extrapolate')
        func_extr_pos = interpolate.interp1d(kappas, cps[:, i_coef], fill_value='extrapolate')
        coefs_neg[i_coef] = func_extr_neg(kappa_target)
        coefs_pos[i_coef] = func_extr_pos(kappa_target)
    return coefs_neg, coefs_pos


def reconstruct_CEs_kappa(id_case, dds, cns, cps, kappas):
    def plot_coefs(cs_ref, cs_reco, str_coef):
        x_axis = range(Ncoefs)

        fig = plt.figure(figsize=(FIG_SIZE_W_,FIG_SIZE_H_))
        ax = fig.add_subplot(111)
        ax.plot(x_axis, cs_ref,  color = "b", linewidth = 2, linestyle='-', label = "coefs-ref")
        ax.plot(x_axis, cs_reco, color="r",   linewidth = 2, linestyle=':', label = "coefs-reco")
        plt.xlabel('i')
        plt.title("{:s}-coefs".format(str_coef))
        plt.legend()
        plt.grid(True)
        plt.show()

        fig = plt.figure(figsize=(FIG_SIZE_W_,FIG_SIZE_H_))
        ax = fig.add_subplot(111)
        ax.plot(
            x_axis, cs_ref - cs_reco,  
            color = "b", linewidth = 2, linestyle='-', 
            label = "ref"
        )
        plt.xlabel('i')
        plt.title("{:s}: ref-reco".format(str_coef))
        plt.legend()
        plt.grid(True)
        plt.show()
        return
    # -------------------------------------------------------------
    kappa_target = dds[id_case]["function-parameter"]
    _, Ncoefs = cns.shape
    coefs_neg, coefs_pos = extrapolate_CEs(kappa_target, cns, cps, kappas)

    # --- Plotting coefficients ---
    plot_coefs(cns[id_case, :], coefs_neg, "NEG")
    plot_coefs(cps[id_case, :], coefs_pos, "POS")
    return


# ----------------------------------------------------------------------------------------
# --- COMPUTE the coefficients to describe the change in the maximum and minimum angles ---
def compute_coefs_amplitudes(dds, Ncoefs, flag_save, path_save_plots):
    def test_func(k, coefs):
        res_pol = coefs[0]
        for ii in range(1,Ncoefs):
            res_pol += coefs[ii] / k**ii
        return res_pol 
    
    def est_coefs(ch_maxs, label_max):
        print()
        print("--- Estimation coefs for {:s} amplitudes ---".format(label_max))
        coefs = cp.Variable(Ncoefs)
        objective = cp.Minimize(cp.sum_squares(test_func(kappas, coefs) - ch_maxs))
        prob = cp.Problem(objective)
        result = prob.solve()

        ch_maxs_rec = np.zeros(Npr)
        for ii in range(Npr):
            ch_maxs_rec[ii] = test_func(kappas[ii], coefs.value) 

        max_abs_err = np.max(np.abs(ch_maxs_rec - ch_maxs))
        print("max. abs. err: {:0.3e}".format(max_abs_err))

        # --- Plot the reconstructed amplitudes ---
        fig = plt.figure(figsize=(FIG_SIZE_W_,FIG_SIZE_H_))
        ax = fig.add_subplot(111)
        ax.plot(
            kappas, ch_maxs,     
            "b-", linewidth = 2, marker = "o", 
            label = "- {:s}-maxs-ref".format(label_max)
        )
        ax.plot(
            kappas, ch_maxs_rec, 
            "r:", linewidth = 2, marker = "o", 
            label = "- {:s}-maxs-reco".format(label_max)
        )
        plt.xlabel('kappa')
        ax.legend()
        plt.grid(True)
        plt.show()

        return coefs.value
    # --------------------------------------------------
    Npr = len(dds)
    kappas   = np.zeros(Npr)
    pos_maxs = np.zeros(Npr)
    neg_maxs = np.zeros(Npr)
    count_pr = -1
    for ii in range(Npr):
        count_pr += 1
        dd1 = dds[ii]
        phis = np.array(dd1["phis"] - np.pi/2.)
        kappas[count_pr]    = dd1["function-parameter"]
        pos_maxs[count_pr] = np.max(phis)
        neg_maxs[count_pr] = np.min(phis)
    coefs_neg = est_coefs(neg_maxs, "neg")
    coefs_pos = est_coefs(pos_maxs, "pos")

    # --- Saving data ---
    if flag_save:
        mix.save_dat_plot_1d_file(path_save_plots + "/neg_ampls.dat", kappas, neg_maxs)
        mix.save_dat_plot_1d_file(path_save_plots + "/pos_ampls.dat", kappas, pos_maxs)
    return coefs_neg, coefs_pos


# ----------------------------------------------------------------------------------------
# --- COMPUTE the coefficients to describe the change in the number of angles ---
# flag_save to save the total number of angles versus kappa
# but in python, this procedure draws N-angles-in-half-envelope;
def compute_Na(k, coef_Na_env):
        res_pol = coef_Na_env[0]
        for ii in range(1,len(coef_Na_env)):
            res_pol += coef_Na_env[ii] * k**ii * np.log(k**ii)
        return int(res_pol)


def compute_coefs_Na(dds, Ncoefs, flag_save, path_save_plots):
    def test_func(k, coefs):
        res_pol = coefs[0]
        for ii in range(1,Ncoefs):
            res_pol += coefs[ii] * cp.multiply(
                k**ii, 
                # cp.log(k**ii)/cp.log(10)
                cp.log(k**ii)
            )
        return res_pol 
    
    def est_coefs(Na_env, label_max):
        print()
        print("--- Estimation coefs for {:s} Na ---".format(label_max))
        coefs = cp.Variable(Ncoefs)
        objective = cp.Minimize(cp.sum_squares(test_func(kappas, coefs) - Na_env))
        prob = cp.Problem(objective)
        result = prob.solve()

        Na_env_rec = np.zeros(Npr)
        for ii in range(Npr):
            Na_env_rec[ii] = compute_Na(kappas[ii], coefs.value) 

        max_abs_err = np.max(np.abs(Na_env_rec - Na_env))
        print("max. abs. err: {:0.3e}".format(max_abs_err))

        # --- Plot the reconstructed Na ---
        fig = plt.figure(figsize=(FIG_SIZE_W_,FIG_SIZE_H_))
        ax = fig.add_subplot(111)
        ax.plot(
            kappas, Na_env,     
            "b-", linewidth = 2, marker = "o", 
            label = "- {:s}-maxs-ref".format(label_max)
        )
        ax.plot(
            kappas, Na_env_rec, 
            "r-", linewidth = 3, 
            label = "- {:s}-maxs-reco".format(label_max)
        )
        plt.xlabel('kappa')
        plt.ylabel('Na-env-half')
        ax.legend()
        plt.grid(True)
        plt.show()
        return coefs.value
    # --------------------------------------------------
    Npr = len(dds)
    kappas   = np.zeros(Npr)
    Na_arrays = np.zeros(Npr)
    Na_pos = np.zeros(Npr)
    Na_neg = np.zeros(Npr)
    count_pr = -1
    for ii in range(Npr):
        count_pr += 1
        dd1 = dds[ii]
        kappas[count_pr] = dd1["function-parameter"]

        phis = np.array(dd1["phis"] - np.pi/2.)

        Na = len(phis)
        Na_arrays[count_pr] = Na

        flag_more_neg_peaks = False
        if np.mod(Na//2,2) == 1:
            flag_more_neg_peaks = True

        peaks_ch = extract_env(phis, flag_more_neg_peaks)
        Na_neg[count_pr] = len(peaks_ch["neg-v"])//2
        Na_pos[count_pr] = len(peaks_ch["pos-v"])//2
    coefs_neg = est_coefs(Na_neg, "neg")
    coefs_pos = est_coefs(Na_pos, "pos")

     # --- Saving data ---
    if flag_save:
        mix.save_dat_plot_1d_file(path_save_plots + "/Na_kappa.dat", kappas, Na_arrays)

    return coefs_neg, coefs_pos


# ----------------------------------------------------------------------------------------
# --- Extract envelope of QSVT angles for the matrix inversion. ---
def extract_env(phis_sh, flag_start_neg = True):
    N = len(phis_sh)
    pos_peaks = np.zeros(N)
    neg_peaks = np.zeros(N)
    inds_pos = np.zeros(N)
    inds_neg = np.zeros(N)
    count_pos = 0
    count_neg = 0
    prev_peak = None
    
    for ii in range(N):
        temp = phis_sh[ii]
        if np.abs(temp) <= 1e-12:
            if prev_peak is None: 
                if flag_start_neg: # assume that the angles start from negative angles;
                    prev_peak = -1
                    neg_peaks[count_neg] = temp
                    inds_neg[count_neg]  = ii
                    count_neg += 1
                else:
                    prev_peak = +1
                    pos_peaks[count_pos] = temp
                    inds_pos[count_pos]  = ii
                    count_pos += 1 
            else:
                if prev_peak > 0.0:
                    prev_peak = -1
                    neg_peaks[count_neg] = temp
                    inds_neg[count_neg]  = ii
                    count_neg += 1
                else:
                    prev_peak = +1
                    pos_peaks[count_pos] = temp
                    inds_pos[count_pos]  = ii
                    count_pos += 1 
        else:
            if temp < 0.0:
                prev_peak = -1
                neg_peaks[count_neg] = temp
                inds_neg[count_neg]  = ii
                count_neg += 1
            else:
                prev_peak = +1
                pos_peaks[count_pos] = temp
                inds_pos[count_pos]  = ii
                count_pos += 1 
                         
    pos_peaks = pos_peaks[:count_pos]
    inds_pos  = inds_pos[:count_pos]
    neg_peaks = neg_peaks[:count_neg]
    inds_neg  = inds_neg[:count_neg]
    peaks = {
        "pos-v": pos_peaks,
        "pos-i": inds_pos,
        "neg-v": neg_peaks,
        "neg-i": inds_neg
    }
    return peaks


# ----------------------------------------------------------------------------------------
# --- Construct QSVT angles from positive and negative envelops ---
# Returns angels that can be used in a QSVT circuit.
def construct_angles_from_envelops(
    full_env_neg_APPR, full_env_pos_APPR, flag_start_neg
):
    N_env_half_neg = len(full_env_neg_APPR)//2
    N_env_half_pos = len(full_env_pos_APPR)//2
    phis_appr = np.zeros(N_env_half_neg + N_env_half_pos)
    
    if flag_start_neg:
        # set negative angles:
        for ii in range(N_env_half_neg):
            phis_appr[2*ii] = full_env_neg_APPR[ii]
            
        # set positive angles:
        for ii in range(N_env_half_pos):
            phis_appr[2*ii+1] = full_env_pos_APPR[ii]
    else:
        # set negative angles:
        for ii in range(N_env_half_neg):
            phis_appr[2*ii+1] = full_env_neg_APPR[ii]
            
        # set positive angles:
        for ii in range(N_env_half_pos):
            phis_appr[2*ii] = full_env_pos_APPR[ii]

    # the full sequence of QSVT angles:
    phis_appr = np.concatenate(
        (phis_appr, np.flip(phis_appr))
    )

    # --- Correct the angles to use then in a QSVT circuit ---
    phis_appr += np.pi/2.
    return phis_appr


# ----------------------------------------------------------------------------------------
# --- Compute the coefficients to describe the shape (envelope) of the QSVT angles ---
def compute_coefs_envelop(
    dd,
    Ncoefs = 10, 
    flag_plot_envelop = True, 
    flag_plot_shape = True,
    flag_reconstruct = True
):
    # --------------------------------------------------------
    def test_func_even_Ch(x, a):
        res_pol = 0.
        for ii in range(Ncoefs):
            res_pol += a[ii] * np.cos((2*ii) * np.arccos(x))
        return res_pol

    def np_test_func_even_Ch(x, a):
        Nx = len(x)
        res_pol = np.zeros(Nx)
        for ix in range(Nx):
            res_pol[ix] = 0.
            for ii in range(Ncoefs):
                res_pol[ix] += a[ii] * np.cos((2*ii) * np.arccos(x[ix]))
        return res_pol
    
    def compute_coefs_and_appr(N_half_env, half_norm_env):
        x = np.linspace(0.0, 1.0, N_half_env)

        # x = np.zeros(N_half_env)
        # for ii in range(N_half_env):
        #     x[ii] = np.cos((2*ii + 1)*np.pi / (2.*N_half_env))
        #     # x[ii] = np.cos((2*ii + 1)*np.pi / (2.*2.*N_half_env))
        # x = np.flip(x)

        coefs_env = cp.Variable(Ncoefs)
        objective = cp.Minimize(cp.sum_squares(
            test_func_even_Ch(x,coefs_env) - half_norm_env
        ))
        prob = cp.Problem(objective)
        result = prob.solve()
        half_norm_env_APPR = np_test_func_even_Ch(x, coefs_env.value)
        full_norm_env_APPR = np.concatenate((
            half_norm_env_APPR, np.flip(half_norm_env_APPR)
        ))
        return coefs_env, full_norm_env_APPR

    # --------------------------------------------------------

    Na = len(dd["phis"])
    flag_more_neg_peaks = False
    if np.mod(Na//2,2) == 1:
        flag_more_neg_peaks = True
    
    # chosen QSVT angles
    phis_ch    = np.array(dd["phis"])
    range_full = np.array(range(len(phis_ch)))
    
    # shifted QSVT angles:
    phis_sh_ch = phis_ch - np.pi/2.
    
    # chosen (shifted by pi/2) envelops:
    peaks_ch   = extract_env(phis_sh_ch, flag_more_neg_peaks)
    
    # chosen envelops: 
    full_env_neg = peaks_ch["neg-v"]
    max_v_neg    = np.max(np.abs(full_env_neg))
    
    full_env_pos = peaks_ch["pos-v"]
    max_v_pos    = np.max(np.abs(full_env_pos))
      
    # chosen indices of positive and negative envelops:
    range_env_pos = peaks_ch["pos-i"]
    range_env_neg = peaks_ch["neg-i"]

    # normalized envelops:
    full_norm_env_neg = full_env_neg / max_v_neg
    N_full_env_neg    = len(full_env_neg)
    
    full_norm_env_pos = full_env_pos / max_v_pos
    N_full_env_pos    = len(full_env_pos)
    
    # chosen half envelops:
    half_norm_env_neg  = full_norm_env_neg[0:N_full_env_neg//2]; 
    range_env_half_neg = range_env_neg[0:N_full_env_neg//2] 
    N_half_env_neg     = len(range_env_half_neg)
    
    half_norm_env_pos  = full_norm_env_pos[0:N_full_env_pos//2]; 
    range_env_half_pos = range_env_pos[0:N_full_env_pos//2] 
    N_half_env_pos     = len(range_env_half_pos)
    
    print()
    print("full number of angles: {:d}".format(len(range_full)))
    print("N of NEG. peaks: {:d}".format(N_full_env_neg))
    print("N of POS. peaks: {:d}".format(N_full_env_pos))
    print()
    print("N_env-half-NEG: {:d}".format(N_half_env_neg))
    print("N_env-half-POS: {:d}".format(N_half_env_pos))
    
    # --- Plot the original envelope ---
    print("The envelope for the kappa = {:0.0f} is taken.".format(dd["function-parameter"]))
    if flag_plot_envelop:
        fig = plt.figure(figsize=(FIG_SIZE_W_,FIG_SIZE_H_))
        ax = fig.add_subplot(111)
        ax.plot(range_full,    phis_sh_ch,   color="b", linewidth = 2, linestyle='-',  label = "phis-shifted")
        ax.plot(range_env_pos, full_env_pos, color="r", linewidth = 2, linestyle='-',  label = "pos-envelope")
        ax.plot(range_env_neg, full_env_neg, color="g", linewidth = 2, linestyle='-',  label = "neg-envelope")
        ax.plot()
        plt.xlabel('i')
        plt.ylabel("env")
        ax.legend()
        plt.grid(True)
        plt.show()

    # --- Compute the coefficients ---
    coefs_neg, full_norm_env_neg_APPR = compute_coefs_and_appr(
        N_half_env_neg, half_norm_env_neg
    ) 
    coefs_pos, full_norm_env_pos_APPR = compute_coefs_and_appr(
        N_half_env_pos, half_norm_env_pos
    )    

    # --- Maximum absolute errors ---
    print("Normalized POS. ENV: max. absolute error: {:0.3e}".format(
        np.max(np.abs(full_norm_env_pos_APPR - full_norm_env_pos))
    ))
    print("Normalized NEG. ENV: max. absolute error: {:0.3e}".format(
        np.max(np.abs(full_norm_env_neg_APPR - full_norm_env_neg))
    ))

    # --- Plot the reconstructed shape ---
    if flag_plot_shape:
        fig = plt.figure(figsize=(FIG_SIZE_W_,FIG_SIZE_H_))
        ax = fig.add_subplot(111)
        ax.plot(range_env_pos, full_norm_env_pos,      color="b", linewidth = 2, linestyle='-')
        ax.plot(range_env_pos, full_norm_env_pos_APPR, color="r", linewidth = 2, linestyle=':')
        plt.xlabel('x')
        plt.ylabel("pos. envelope")
        plt.grid(True)
        plt.show()
        
        fig = plt.figure(figsize=(FIG_SIZE_W_,FIG_SIZE_H_))
        ax = fig.add_subplot(111)
        ax.plot(range_env_neg, full_norm_env_neg,      color="b", linewidth = 2, linestyle='-')
        ax.plot(range_env_neg, full_norm_env_neg_APPR, color="r", linewidth = 2, linestyle=':')
        plt.xlabel('x')
        plt.ylabel("neg. envelope")
        plt.grid(True)
        plt.show()

    # --- Reconstruct the QSVT angles ---
    phis_appr = None
    if flag_reconstruct:
        print()
        print("\n--- Reconstructing the QSVT angles for the same kappa ---")
        print("1. The envelope of the QSVT angles are approximated by the computed coefs.")
        print("2. The number of the QSVT angles and their absolute amplitudes are taken from the reference QSVT case.")

        # --- Reconstructing the angles ---
        phis_appr = construct_angles_from_envelops( 
            max_v_neg * full_norm_env_neg_APPR, 
            max_v_pos * full_norm_env_pos_APPR,
            flag_more_neg_peaks
        )
        Nh = len(range_full)//2
        
        # --- Maximum absolute error ---
        abs_err = np.max(np.abs(phis_ch - phis_appr))
        print()
        print("max-abs-err in final reconstructed QSVT angles: {:0.3e}".format(abs_err))
 
        # --- Plot angles ---
        fig = plt.figure(figsize=(FIG_SIZE_W_,FIG_SIZE_H_))
        ax = fig.add_subplot(111)
        ax.plot(range_full - Nh, phis_ch,   color="b", linewidth = 2, linestyle='-', label = "phis-ref")
        ax.plot(range_full - Nh, phis_appr, color="r", linewidth = 2, linestyle='-', label = "phis-appr")
        plt.xlabel('i')
        plt.ylabel("phis")
        # plt.xlim([-50, 50])
        ax.legend()
        plt.grid(True)
        plt.show()

        fig = plt.figure(figsize=(FIG_SIZE_W_,FIG_SIZE_H_))
        ax = fig.add_subplot(111)
        ax.plot(
            range_full - Nh,   
            phis_ch - phis_appr,        
            color="b", linewidth = 2, linestyle='-'
        )
        plt.xlabel('i')
        plt.ylabel("phis_ref - phis_recon")
        plt.grid(True)
        plt.show()
    return coefs_neg.value, coefs_pos.value, N_half_env_neg, N_half_env_pos, phis_appr


# ----------------------------------------------------------------------------------------
# --- Plot coefficients approximating angles' envelope  ---
def plot_env_coefs(dd, Ncoefs_arrs, flag_save, path_save_plots):
    def plot_one(coefs_arr, str_env):
        colors = ["b", "r", "green", "black"]
        markers = ["o", "v", "^", "."]
        fig = plt.figure(figsize=(FIG_SIZE_W_,FIG_SIZE_H_))
        ax = fig.add_subplot(111)
        for ii in range(len(Ncoefs_arrs)):
            ax.plot(
                range(len(coefs_arr[ii])), 
                coefs_arr[ii], 
                color=colors[ii], linewidth = 2, linestyle=':', marker = markers[ii],
                label = "Nc = {:d}".format(Ncoefs_arrs[ii])
            )
        plt.xlabel('i')
        plt.ylabel("coefs-env-{:s}".format(str_env))
        plt.legend()
        plt.grid(True)
        plt.show()
        return
    # -------------------------------------------------------------------------------
    coefs_neg_arr = []
    coefs_pos_arr = []
    for ii in range(len(Ncoefs_arrs)):
        print()
        print("--------------------------------------------------------------------")
        print("N-coefs = {:d}".format(Ncoefs_arrs[ii]))
        Ncoefs = Ncoefs_arrs[ii]
        coefs_shape_neg_1, coefs_shape_pos_1, _, _, _ = compute_coefs_envelop(
            dd,
            Ncoefs = Ncoefs, 
            flag_plot_envelop = False, flag_plot_shape = False, flag_reconstruct = False
        )
        coefs_neg_arr.append(coefs_shape_neg_1)
        coefs_pos_arr.append(coefs_shape_pos_1)
    plot_one(coefs_neg_arr, "NEG")
    plot_one(coefs_pos_arr, "POS")

    # --- Saving data ---
    if flag_save:
        for ii in range(len(Ncoefs_arrs)):
            mix.save_dat_plot_1d_file(
                path_save_plots + "/Ch_coefs_pos_Nc{:d}.dat".format(Ncoefs_arrs[ii]), 
                np.array(range(len(coefs_pos_arr[ii]))), 
                np.log10(np.abs(coefs_pos_arr[ii]))
            )
    return


# ----------------------------------------------------------------------------------------
@jit(nopython=True)
def compute_inverse_function_GPU(U, W, Rphi, Na):
    for ia in range(1,Na):
        U = U.dot(W).dot(Rphi[ia])
    return U[0,0].real


# ----------------------------------------------------------------------------------------
@jit(nopython=True)
def form_Rphi(Rphi, phis, Na):
    for ia in range(Na):
        ephi = np.exp(1j * phis[ia])
        Rphi[ia,0,0] = ephi
        Rphi[ia,1,1] = np.conjugate(ephi)
    return 

# ----------------------------------------------------------------------------------------
# --- Compute 1/x using a sequence of rotations ---
def construct_inverse_function_GPU(
        phis_in, kappa, coef_norm, xlim = None,
        opt_domain = 2,
        flag_save = False, path_save_ = None, fname_save = None 
    ):
    phis_comp = np.array(phis_in)
    Na = len(phis_comp)
    print("kappa: {:0.1f}".format(kappa))
    print("Na: {:d}".format(Na))
    
    # print("coef-norm: {:0.3e}".format(coef_norm))
    # print()
    # print("max. angle - np.pi/2: {:0.3e}".format(np.max(phis_comp - np.pi/2.)))
    # print("min. angle - np.pi/2: {:0.3e}".format(np.min(phis_comp - np.pi/2.)))
    
    # --- Correct the angles for the direct calculation of 1/x ---
    phis_comp     -= np.pi/2.
    phis_comp[0]  += np.pi/4.
    phis_comp[-1] += np.pi/4.

    # --- Plot the angles for the direct calculation of 1/x ---
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.plot(range(Na), phis_comp, color="b", linewidth = 2, linestyle='-')
    # plt.xlabel('i')
    # plt.ylabel("angles")
    # plt.grid(True)
    # plt.show()

    # --- x-grid ---
    Nx = 101
    if opt_domain == 0:
        x_grid_1 = np.linspace(-1.0, -1.0/kappa, Nx)
        x_grid_2 = np.linspace(1.0/kappa, 1.0, Nx)
    if opt_domain == 1:
        x_grid_1 = np.linspace(-1.0+1.0/kappa, -1.0/kappa, Nx)
        x_grid_2 = np.linspace(1.0/kappa, 1.0-1.0/kappa, Nx)
    if opt_domain == 2:
        x_grid_1 = np.linspace(-8.0/kappa, -1.0/kappa, Nx)
        x_grid_2 = np.linspace(1.0/kappa, 8.0/kappa, Nx)
    x_grid = np.concatenate((x_grid_1, x_grid_2))
    Nx = len(x_grid)
    
    # rotation matrices:
    Rphi = np.zeros((Na,2,2), dtype = complex)
    form_Rphi(Rphi, phis_comp, Na)

    # for ia in range(Na):
    #     ephi = np.exp(1j * phis_comp[ia])
    #     Rphi[ia,0,0] = ephi
    #     Rphi[ia,1,1] = np.conjugate(ephi)
    
    # --- reconstruction --- 
    inv_f = np.zeros(Nx)
    for ix in range(Nx):
        x1 = x_grid[ix]
        xs = 1j*np.sqrt(1 - x1**2)
        W = np.array([
            [x1, xs],
            [xs, x1]
        ], dtype = complex)
        U = np.array(Rphi[0])
        inv_f[ix] = compute_inverse_function_GPU(U, W, Rphi, Na)
        
    # --- the reference case ---
    inv_ref = ( 1. - np.exp(-(5*kappa*x_grid)**2) ) / x_grid
    inv_ref *= coef_norm/ kappa

    # --- normalize the functions ---
    norm_coef = np.max(np.abs(inv_ref))
    inv_f   /= norm_coef
    inv_ref /= norm_coef
    
    # --- Error ---
    max_norm_abs_error = np.max(np.abs(inv_ref - inv_f))
    print("max-abs-err: {:0.3e}".format(max_norm_abs_error))

    log_err = np.zeros(Nx)
    for ii in range(Nx):
        temp_err = np.abs(inv_ref[ii] - inv_f[ii])
        if temp_err < 1e-16:
            temp_err = 1e-16
        log_err[ii] = np.log10(temp_err)

    # --- Plotting the computed inverse function ---
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x_grid, inv_ref, color="b", linewidth = 2, linestyle='-', label = "phis-ref")
    ax.plot(x_grid, inv_f,   color="r", linewidth = 2, linestyle='--', label = "appr")
    plt.xlabel('i')
    plt.ylabel("phis")
    if xlim is not None:
        plt.xlim(-5, 5)
    ax.legend()
    plt.grid(True)
    plt.show()

    # --- Plotting the error ---
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x_grid, log_err, color="b", linewidth = 2, linestyle='-')
    plt.xlabel('x')
    plt.ylabel("log10(err)")
    if xlim is not None:
        plt.xlim(-5, 5)
    plt.grid(True)
    plt.show()

    # --- Saving data ---
    if flag_save:
        mix.save_dat_plot_1d_file(
            path_save_ + "/direct_inv_ref_{:s}.dat".format(fname_save), 
            x_grid, inv_ref
        )
        mix.save_dat_plot_1d_file(
            path_save_ + "/direct_inv_reco_{:s}.dat".format(fname_save), 
            x_grid, inv_f
        )
        mix.save_dat_plot_1d_file(
            path_save_ + "/direct_err_{:s}.dat".format(fname_save), 
            x_grid, log_err
        )
    return


# ----------------------------------------------------------------------------------------
# --- Compute 1/x using a sequence of rotations: for a single x ---
def construct_inverse_function_GPU_X1(phis, kappa, coef_norm_qsvt, x_grid):
    # !!! here, do not modify phis !!!
    # here, phis are the version of the QSVT angles used for direct computations:

    # ---
    Na = len(phis)
    Nx = len(x_grid)

    # rotation matrices:
    Rphi = np.zeros((Na,2,2), dtype = complex)
    form_Rphi(Rphi, phis, Na)
    
    # --- reconstruction --- 
    inv_reco = np.zeros(Nx)
    for ii in range(Nx):
        x1 = x_grid[ii]
        xs = 1j*np.sqrt(1 - x1**2)
        W = np.array([
            [x1, xs],
            [xs, x1]
        ], dtype = complex)
        U = np.array(Rphi[0])
        inv_reco[ii] = compute_inverse_function_GPU(U, W, Rphi, Na)
        
    # --- the reference case ---
    inv_ref = ( 1. - np.exp(-(5*kappa*x_grid)**2) ) / x_grid
    inv_ref *= coef_norm_qsvt/ kappa

    # renormalize the signals:
    norm_ampl = np.max(np.abs(inv_ref))
    inv_ref  /= norm_ampl
    inv_reco /= norm_ampl

    # --- Error ---
    max_norm_abs_error = np.max(np.abs(inv_ref - inv_reco))
    return max_norm_abs_error


# ----------------------------------------------------------------------------------------
# --- Store the estimated parameters for the QSVT angles ---
def store_estimation(
        dd, fname, path_root,
        coefs_ampl_neg, coefs_ampl_pos,
        coefs_Na_neg, coefs_Na_pos,
        coefs_shape_neg, coefs_shape_pos,
        N_half_env_neg, N_half_env_pos
    ):
    from datetime import datetime
    from datetime import date

    # --- Current time ---
    curr_time = date.today().strftime("%m/%d/%Y") + ": " + datetime.now().strftime("%H:%M:%S")

    # --- Create the filename ---
    full_fname = path_root + "/" + fname

    # --- Normalization factor used to compute the QSVT angles in the reference L-BFGS calculation:
    factor_norm = dd["factor-norm"]
    kappa_ref   = dd['function-parameter']

    # --- Store data ---
    print("Storing the coefficients to:\n " + full_fname)
    with h5py.File(full_fname, "w") as f:
        grp = f.create_group("basic")
        grp.create_dataset('coef_norm',           data=float(factor_norm))
        grp.create_dataset('date-of-simulation',  data=curr_time)
        grp.create_dataset('descr',  data="for-inversion")
        grp.create_dataset('param-ref',  data=float(kappa_ref))

        grp.create_dataset('N-env-half-neg', data=int(N_half_env_neg))
        grp.create_dataset('N-env-half-pos', data=int(N_half_env_pos))

        grp = f.create_group("coefs-amplitude")
        grp.create_dataset('neg',  data = coefs_ampl_neg)
        grp.create_dataset('pos',  data = coefs_ampl_pos)

        grp = f.create_group("coefs-Na")
        grp.create_dataset('neg',  data = coefs_Na_neg)
        grp.create_dataset('pos',  data = coefs_Na_pos)

        grp = f.create_group("coefs-envelop")
        grp.create_dataset('neg',  data = coefs_shape_neg)
        grp.create_dataset('pos',  data = coefs_shape_pos)
    return


# ----------------------------------------------------------------------------------------
# --- Read the estimated parameters for the QSVT angles ---
def read_estimation(id_case = 0, Ncoefs = 40, path_root = "./tools/QSVT-angles/inversion/"):
    dd = {}
    fname = "QSVT-MI-estimation-coefs-case{:d}-Nc{:d}.hdf5".format(id_case, Ncoefs)
    full_fname = path_root + "/" + fname
    print("Reading the coefficients from:\n " + full_fname)
    with h5py.File(full_fname, "r") as f:
        grp = f["basic"]
        date_comp = grp["date-of-simulation"][()].decode("utf-8")
        dd["factor-norm"] = grp["coef_norm"][()]
        line_descr = grp["descr"][()].decode("utf-8")
        dd["kappa"] = grp["param-ref"][()]
        dd["N-env-half-neg"] = int(grp['N-env-half-neg'][()])
        dd["N-env-half-pos"] = int(grp['N-env-half-pos'][()])

        grp = f["coefs-amplitude"]
        dd["coefs-ampl-neg"] = np.array(grp["neg"])
        dd["coefs-ampl-pos"] = np.array(grp["pos"])

        grp = f["coefs-Na"]
        dd["coefs-Na-neg"] = np.array(grp["neg"])
        dd["coefs-Na-pos"] = np.array(grp["pos"])

        grp = f["coefs-envelop"]
        dd["coefs-env-neg"] = np.array(grp["neg"])
        dd["coefs-env-pos"] = np.array(grp["pos"])
    # ---
    print("When simulated: ", date_comp)
    print("Data: {:s}".format(line_descr))
    print()
    print("kappa-reference: {:0.3f}".format(dd["kappa"]))
    print("factor-norm: {:0.3f}".format(dd["factor-norm"]))
    print("N-env-half-neg: {:d}".format(dd["N-env-half-neg"]))
    print("N-env-half-pos: {:d}".format(dd["N-env-half-pos"]))
    print("N-coefs-envelope-neg: {:d}".format(len(dd["coefs-env-neg"])))
    print("N-coefs-envelope-pos: {:d}".format(len(dd["coefs-env-pos"])))
    return dd


# ----------------------------------------------------------------------------------------
# --- Read the estimated parameters for the QSVT angles ---
def save_estimated_angles(
        kappa_target, dd, phis_approx, id_case, Nc, 
        path_root = "./tools/QSVT-angles/inversion/estimated-angles-Na-k/"
        # path_root = "./tools/QSVT-angles/inversion/estimated-angles-Na-k-logk/"
):
    from datetime import datetime
    from datetime import date

    # --- Current time ---
    curr_time = date.today().strftime("%m/%d/%Y") + ": " + datetime.now().strftime("%H:%M:%S")

    # --- Create the filename ---
    fname = "{:s}_k{:d}_ref{:d}_Nc{:d}.hdf5".format("est_mi", int(kappa_target), id_case, Nc)
    full_fname = path_root + "/" + fname

    # --- Store data ---
    print("write angles to:\n " + full_fname)
    with h5py.File(full_fname, "w") as f:
        grp = f.create_group("basic")
        grp.create_dataset('date-of-simulation',  data=curr_time)
        grp.create_dataset('factor-norm',         data=float(dd["factor-norm"]))
        grp.create_dataset('function-parameter',  data=float(kappa_target))
        grp.create_dataset('function-parity', data=1)
        grp.create_dataset('function-type',   data="inv")
        grp.create_dataset('project-name',    data="est-mi")

        grp = f.create_group("results")
        grp.create_dataset('phis',  data = phis_approx)
    print("Done.")
    return


# ----------------------------------------------------------------------------------------
def reproduce_env(coefs_envelop, coefs_ampl, N_env_half, kappa_goal):
        # - the number of peaks in the half of the envelope -
        x = np.linspace(0.0, 1.0, N_env_half)
        Nx = len(x)
        
        # - construct the half of the normalized envelope --
        Nc = len(coefs_envelop)
        env_half = np.zeros(Nx)
        for ix in range(Nx):
            env_half[ix] = 0.
            for ii in range(Nc):
                env_half[ix] += coefs_envelop[ii] * np.cos((2*ii) * np.arccos(x[ix]))

        # - full envelope -
        full_env = np.concatenate(( env_half, np.flip(env_half) ))

        # - Rescale the envelope angles -
        max_ampl = coefs_ampl[0]
        for ii in range(1, len(coefs_ampl)):
            max_ampl += coefs_ampl[ii] / kappa_goal**ii
        full_env *= np.abs(max_ampl)
        return full_env


# ----------------------------------------------------------------------------------------
def reproduce_Nenv(dd, kappa_goal):
    if SEL_EST_NA_ == 1:
        coef_Na_env_half_neg = dd["N-env-half-neg"]/dd["kappa"]
        coef_Na_env_half_pos = dd["N-env-half-pos"]/dd["kappa"]
        N_env_half_neg = int(coef_Na_env_half_neg * kappa_goal)
        N_env_half_pos = int(coef_Na_env_half_pos * kappa_goal) 

    if SEL_EST_NA_ == 2:
        N_env_half_neg = compute_Na(kappa_goal, dd["coefs-Na-neg"])
        N_env_half_pos = compute_Na(kappa_goal, dd["coefs-Na-pos"])


    N_temp = (N_env_half_neg + N_env_half_pos) // 2 
    N_temp -= int(2*kappa_goal / 100)

    N_env_half_neg = N_temp 
    N_env_half_pos = N_temp
    return N_env_half_neg, N_env_half_pos


# ----------------------------------------------------------------------------------------
def get_flag_peaks(N_env_half_neg, N_env_half_pos):
    Na_rec = (N_env_half_neg + N_env_half_pos) * 2
    flag_more_neg_peaks = False
    if np.mod(Na_rec//2,2) == 1:
        flag_more_neg_peaks = True
    return flag_more_neg_peaks


# ----------------------------------------------------------------------------------------
# --- Estimate the QSVT angles ---
def estimate_angles(
        dd, kappa_goal, 
        flag_variation = False,
        # ---
        N_iter_Na = 10,
        dN_env = None,
        # ---
        N_iter_c = 10,
        coef_dc_init = 0.01,
    ):
    def get_N_neg_pos(N_env):
        N_env_neg = N_env
        N_env_pos = N_env
        return N_env_neg, N_env_pos
    # --------------------------------------------------
    coef_norm = dd["factor-norm"]

    # --- Estimate the number of angles ---
    N_env_half_neg, N_env_half_pos = reproduce_Nenv(dd, kappa_goal)
    flag_peaks = get_flag_peaks(N_env_half_neg, N_env_half_pos)

    # --- Estimation of the QSVT angles ---
    coef_env_neg, coef_ampl_neg = dd["coefs-env-neg"], dd["coefs-ampl-neg"]
    coef_env_pos, coef_ampl_pos = dd["coefs-env-pos"], dd["coefs-ampl-pos"]
    env_neg = reproduce_env(coef_env_neg, coef_ampl_neg, N_env_half_neg, kappa_goal)
    env_pos = reproduce_env(coef_env_pos, coef_ampl_pos, N_env_half_pos, kappa_goal)
    phis_appr_best = construct_angles_from_envelops(env_neg, env_pos, flag_peaks)
    del flag_peaks, env_neg, env_pos

    N_env_PREV = N_env_half_neg
    del N_env_half_neg, N_env_half_pos

    if flag_variation:
        # ----------------------------------------------------------------
        # --- Vary the estimation parameters to get a higher precision ---
        # ----------------------------------------------------------------
        if dN_env is None:
            dN_env = int(kappa_goal / 100)
        dN_env *= -1
        dN_env_init = dN_env

        # --- The x-points where the error is analyzed ---
        Nx = 2
        # x_grid_1 = np.linspace(-2.0/kappa_goal, -1.0/kappa_goal, Nx)
        x_grid_2 = np.linspace(1.0/kappa_goal, 2.0/kappa_goal, Nx)

        # Nx = 4
        # x_grid_1 = np.linspace(-0.9, -1.0/kappa_goal, Nx)
        # x_grid_2 = np.linspace(1.0/kappa_goal, 0.9, Nx)
        
        # x_grid = np.concatenate((x_grid_1, x_grid_2))
        x_grid = np.array(x_grid_2)
        # del x_grid_1
        del Nx, x_grid_2

        # --- Compute 1/x ---
        phis_appr_best[1:-1] -= np.pi/2.
        phis_appr_best[0]    -= np.pi/4.
        phis_appr_best[-1]   -= np.pi/4.
        err_curr = construct_inverse_function_GPU_X1(phis_appr_best, kappa_goal, coef_norm, x_grid)
        
        # --- initial parameters ---
        cn_PREV = np.array(coef_env_neg)
        cp_PREV = np.array(coef_env_pos)

        # --- Print initial data ---
        print()
        print("--- Use the variation of estimation parameters ---")
        print("N-iter: {:d}".format(N_iter_Na))
        print("dN-env: {:d}".format(dN_env))

        print()
        print("iter, Na, N-env, err: {:3d}, {:d}, {:d}, {:0.3e}".format(
            0, len(phis_appr_best), N_env_PREV, err_curr
        ))

        # -----------------------------------
        # --- Variation of Na ---
        print("\n------------------------------------------")
        print("--- Variation of Na ---")
        counter_cycle = 1
        N_env_NEW, Nen_NEW, Nep_NEW, flag_peaks = None, None, None, None
        env_neg, env_pos, phis_appr, err_new    = None, None, None, None
        for ii in range(N_iter_Na):
            # --- new N-env-half ---
            N_env_NEW = N_env_PREV + dN_env
            Nen_NEW, Nep_NEW = get_N_neg_pos(N_env_NEW)

            # --- compute new angles ---
            flag_peaks = get_flag_peaks(Nen_NEW, Nep_NEW)
            env_neg = reproduce_env(cn_PREV, coef_ampl_neg, Nen_NEW, kappa_goal)
            env_pos = reproduce_env(cp_PREV, coef_ampl_pos, Nep_NEW, kappa_goal)
            phis_appr = construct_angles_from_envelops(env_neg, env_pos, flag_peaks)

            # --- compute the error ---
            phis_appr[1:-1] -= np.pi/2.
            phis_appr[0]    -= np.pi/4.
            phis_appr[-1]   -= np.pi/4.
            err_new = construct_inverse_function_GPU_X1(phis_appr, kappa_goal, coef_norm, x_grid)

            # --- Print results ---
            print("iter, Na, N-env, err: {:3d}, {:d}, {:d}, {:0.3e}".format(
                ii+1, len(phis_appr), N_env_NEW, err_new,
            ), end = "")

            # --- modify the estimation parametes ---
            if err_new > err_curr:
                if np.abs(dN_env) == 1:
                    if counter_cycle == 2:
                        break
                    else:
                        print(" ... change direction", end = "")
                        counter_cycle += 1
                        dN_env = dN_env_init//4
                        dN_env *= -1
                else:
                    print(" ... reduce step", end = "")
                    dN_env = dN_env//2
            else:
                print(" >>> new phis", end = "")
                dN_env_init = dN_env
                err_curr = err_new
                N_env_PREV = N_env_NEW
                phis_appr_best = phis_appr
                # if counter_cycle > 1:
                #     print(" ... reduce step", end = "")
                #     dN_env = dN_env//2
                #     if dN_env == 0:
                #         break
            print()

        del N_env_NEW, Nen_NEW, Nep_NEW, flag_peaks
        del env_neg, env_pos, phis_appr, err_new

        # -----------------------------------
        # --- Variation of coefs-envelope ---
        print("\n------------------------------------------")
        print("--- Variation of coefficients ---")
        flag_peaks, cn_NEW, cp_NEW, env_neg, env_pos, phis_appr, err_new = \
            None, None, None, None, None, None, None
        Nen, Nep = get_N_neg_pos(N_env_PREV)
        print("-- err-init: {:14.3e} --".format(err_curr))
        coef_dc = coef_dc_init
        counter_dir = 1
        for i_iter in range(N_iter_c):

            # --- new coefs-envelope ---
            cn_NEW = cn_PREV * (1 - coef_dc)
            cp_NEW = cp_PREV * (1 + coef_dc)

            # --- compute new angles ---
            flag_peaks = get_flag_peaks(Nen, Nep)
            env_neg = reproduce_env(cn_NEW, coef_ampl_neg, Nen, kappa_goal)
            env_pos = reproduce_env(cp_NEW, coef_ampl_pos, Nep, kappa_goal)
            phis_appr = construct_angles_from_envelops(env_neg, env_pos, flag_peaks)

            # --- compute the error ---
            phis_appr[1:-1] -= np.pi/2.
            phis_appr[0]    -= np.pi/4.
            phis_appr[-1]   -= np.pi/4.
            err_new = construct_inverse_function_GPU_X1(phis_appr, kappa_goal, coef_norm, x_grid)

            # --- Print results ---
            print("iter, coef, err: {:3d}, {:0.3e}, {:14.3e}".format(
                i_iter+1, coef_dc, err_new
            ), end = "")

            # --- modify the estimation parametes ---
            if err_new > err_curr:
                if np.abs(coef_dc) < 1e-4:
                    if counter_dir == 2:
                        break
                    else:
                        print(" ... change direction", end = "")
                        counter_dir += 1
                        coef_dc = coef_dc_init/4.
                        coef_dc *= -1
                else:
                    print(" ... reduce step", end = "")
                    coef_dc /= 10.
            else:
                print(" >>> new phis", end = "")
                err_curr = err_new
                coef_dc_init = coef_dc
                cn_PREV = np.array(cn_NEW)
                cp_PREV = np.array(cp_NEW)
                phis_appr_best = phis_appr
                # if counter_dir > 1:
                #     print(" ... *reduce step", end = "")
                #     coef_dc /= 10.
                #     if np.abs(coef_dc) < 1e-4:
                #         break
            print()
        del Nen, Nep, flag_peaks, cn_NEW, cp_NEW, env_neg, env_pos, phis_appr, err_new
        
        # --- Chosen parameters ---
        print()
        print("*** Results ***")
        print("err: {:0.3e}".format(err_curr))
        print("N-env: {:d}".format(N_env_PREV))

        # --- Correct the angles ---
        phis_appr_best[1:-1] += np.pi/2.
        phis_appr_best[0]    += np.pi/4.
        phis_appr_best[-1]   += np.pi/4.
  
        # # --- Compute the best obtained angles ---
        # Nen, Nep = get_N_neg_pos(N_env_PREV)
        # flag_peaks = get_flag_peaks(Nen, Nep)
        # env_neg = reproduce_env(cn_PREV, coef_ampl_neg, Nen, kappa_goal)
        # env_pos = reproduce_env(cp_PREV, coef_ampl_pos, Nep, kappa_goal)
        # phis_appr_best = construct_angles_from_envelops(env_neg, env_pos, flag_peaks)

    # --- Return the estimated QSVT angles ---
    print("Na: {:d}".format(len(phis_appr_best)))
    print("Done.")
    return phis_appr_best, coef_norm









# # ----------------------------------------------------------------------------------------
# # --- Estimate the QSVT angles ---
# def estimate_angles(
#         dd, kappa_goal, 
#         flag_variation = False,
#         # ---
#         N_iter_Na = 10,
#         dN_env = None,
#         # ---
#         N_coefs_cycles = 0,
#         N_coefs_to_vary = 0,
#         N_iter_c = 10,
#         coef_dc_init = 0.01,
#     ):
#     def get_N_neg_pos(N_env):
#         N_env_neg = N_env
#         N_env_pos = N_env
#         return N_env_neg, N_env_pos
#     # --------------------------------------------------
#     coef_norm = dd["factor-norm"]

#     # --- Estimate the number of angles ---
#     N_env_half_neg, N_env_half_pos = reproduce_Nenv(dd, kappa_goal)
#     flag_peaks = get_flag_peaks(N_env_half_neg, N_env_half_pos)

#     # --- Estimation of the QSVT angles ---
#     coef_env_neg, coef_ampl_neg = dd["coefs-env-neg"], dd["coefs-ampl-neg"]
#     coef_env_pos, coef_ampl_pos = dd["coefs-env-pos"], dd["coefs-ampl-pos"]
#     env_neg = reproduce_env(coef_env_neg, coef_ampl_neg, N_env_half_neg, kappa_goal)
#     env_pos = reproduce_env(coef_env_pos, coef_ampl_pos, N_env_half_pos, kappa_goal)
#     phis_appr_best = construct_angles_from_envelops(env_neg, env_pos, flag_peaks)
#     del flag_peaks, env_neg, env_pos

#     N_env_PREV = N_env_half_neg
#     del N_env_half_neg, N_env_half_pos

#     if flag_variation:
#         # ----------------------------------------------------------------
#         # --- Vary the estimation parameters to get a higher precision ---
#         # ----------------------------------------------------------------
#         if dN_env is None:
#             dN_env = int(kappa_goal / 100)
#         dN_env *= -1
#         dN_env_init = dN_env

#         # --- The x-points where the error is analyzed ---
#         Nx = 3
#         x_grid_1 = np.linspace(-2.0/kappa_goal, -1.0/kappa_goal, Nx)
#         x_grid_2 = np.linspace(1.0/kappa_goal, 2.0/kappa_goal, Nx)

#         # Nx = 4
#         # x_grid_1 = np.linspace(-0.9, -1.0/kappa_goal, Nx)
#         # x_grid_2 = np.linspace(1.0/kappa_goal, 0.9, Nx)
        
#         x_grid = np.concatenate((x_grid_1, x_grid_2))
#         del Nx, x_grid_1, x_grid_2

#         # --- Compute 1/x ---
#         phis_appr_best[1:-1] -= np.pi/2.
#         phis_appr_best[0]    -= np.pi/4.
#         phis_appr_best[-1]   -= np.pi/4.
#         err_curr = construct_inverse_function_GPU_X1(phis_appr_best, kappa_goal, coef_norm, x_grid)
        
#         # --- initial parameters ---
#         cn_PREV = np.array(coef_env_neg)
#         cp_PREV = np.array(coef_env_pos)

#         # --- Print initial data ---
#         print()
#         print("--- Use the variation of estimation parameters ---")
#         print("N-iter: {:d}".format(N_iter_Na))
#         print("dN-env: {:d}".format(dN_env))

#         print()
#         print("iter, Na, N-env, err: {:3d}, {:d}, {:d}, {:0.3e}".format(
#             0, len(phis_appr_best), N_env_PREV, err_curr
#         ))

#         # -----------------------------------
#         # --- Variation of Na ---
#         print("\n------------------------------------------")
#         print("--- Variation of Na ---")
#         counter_cycle = 1
#         N_env_NEW, Nen_NEW, Nep_NEW, flag_peaks = None, None, None, None
#         env_neg, env_pos, phis_appr, err_new    = None, None, None, None
#         for ii in range(N_iter_Na):
#             # --- new N-env-half ---
#             N_env_NEW = N_env_PREV + dN_env
#             Nen_NEW, Nep_NEW = get_N_neg_pos(N_env_NEW)

#             # --- compute new angles ---
#             flag_peaks = get_flag_peaks(Nen_NEW, Nep_NEW)
#             env_neg = reproduce_env(cn_PREV, coef_ampl_neg, Nen_NEW, kappa_goal)
#             env_pos = reproduce_env(cp_PREV, coef_ampl_pos, Nep_NEW, kappa_goal)
#             phis_appr = construct_angles_from_envelops(env_neg, env_pos, flag_peaks)

#             # --- compute the error ---
#             phis_appr[1:-1] -= np.pi/2.
#             phis_appr[0]    -= np.pi/4.
#             phis_appr[-1]   -= np.pi/4.
#             err_new = construct_inverse_function_GPU_X1(phis_appr, kappa_goal, coef_norm, x_grid)

#             # --- Print results ---
#             print("iter, Na, N-env, err: {:3d}, {:d}, {:d}, {:0.3e}".format(
#                 ii+1, len(phis_appr), N_env_NEW, err_new,
#             ), end = "")

#             # --- modify the estimation parametes ---
#             if err_new > err_curr:
#                 if np.abs(dN_env) == 1:
#                     if counter_cycle == 2:
#                         break
#                     else:
#                         print(" ... change direction", end = "")
#                         counter_cycle += 1
#                         dN_env = dN_env_init//2
#                         dN_env *= -1
#                 else:
#                     print(" ... reduce step", end = "")
#                     dN_env = dN_env//2
#             else:
#                 print(" >>> new phis", end = "")
#                 dN_env_init = dN_env
#                 err_curr = err_new
#                 N_env_PREV = N_env_NEW
#                 phis_appr_best = phis_appr
#                 if counter_cycle > 1:
#                     print(" ... reduce step", end = "")
#                     dN_env = dN_env//2
#                     if dN_env == 0:
#                         break
#             print()

#         del N_env_NEW, Nen_NEW, Nep_NEW, flag_peaks
#         del env_neg, env_pos, phis_appr, err_new

#         # -----------------------------------
#         # --- Variation of coefs-envelope ---
#         print("\n------------------------------------------")
#         print("--- Variation of coefficients ---")
#         flag_peaks, cn_NEW, cp_NEW, env_neg, env_pos, phis_appr, err_new = \
#             None, None, None, None, None, None, None
#         Nen, Nep = get_N_neg_pos(N_env_PREV)

#         for i_cycle in range(N_coefs_cycles):
#             print()
#             print("*************************************************")
#             print("*** Cycle {:d} ***".format(i_cycle + 1))
#             for i_coef in range(N_coefs_to_vary):
#                 print()
#                 print("-- i-coef, neg, pos, err-init: {:d}, {:0.3e}, {:0.3e}, {:14.3e} --".format(
#                     i_coef, cn_PREV[i_coef], cp_PREV[i_coef], err_curr
#                 ))
#                 counter_cycle = 1
#                 coef_dc = coef_dc_init/(2+2*i_coef)
#                 for i_iter in range(N_iter_c):

#                     # --- new coefs-envelope ---
#                     cn1, cp1 = cn_PREV[i_coef], cp_PREV[i_coef]

#                     cn1 = cn1 * (1 - coef_dc)
#                     cp1 = cp1 * (1 + coef_dc)

#                     cn_NEW = np.array(cn_PREV)
#                     cp_NEW = np.array(cp_PREV)

#                     cn_NEW[i_coef] = cn1
#                     cp_NEW[i_coef] = cp1

#                     # --- compute new angles ---
#                     flag_peaks = get_flag_peaks(Nen, Nep)
#                     env_neg = reproduce_env(cn_NEW, coef_ampl_neg, Nen, kappa_goal)
#                     env_pos = reproduce_env(cp_NEW, coef_ampl_pos, Nep, kappa_goal)
#                     phis_appr = construct_angles_from_envelops(env_neg, env_pos, flag_peaks)

#                     # --- compute the error ---
#                     phis_appr[1:-1] -= np.pi/2.
#                     phis_appr[0]    -= np.pi/4.
#                     phis_appr[-1]   -= np.pi/4.
#                     err_new = construct_inverse_function_GPU_X1(phis_appr, kappa_goal, coef_norm, x_grid)

#                     # --- Print results ---
#                     print("iter, neg, pos, err: {:3d}, {:0.3e}, {:0.3e}, {:14.3e}".format(
#                         i_iter+1, cn1, cp1, err_new
#                     ))

#                     # --- modify the estimation parametes ---
#                     if err_new > err_curr:
#                         if counter_cycle == 2:
#                             break
#                         else:
#                             counter_cycle += 1
#                             coef_dc *= -1
#                     else:
#                         err_curr = err_new
#                         cn_PREV = np.array(cn_NEW)
#                         cp_PREV = np.array(cp_NEW)
#                         phis_appr_best = phis_appr
#         del Nen, Nep, flag_peaks, cn_NEW, cp_NEW, env_neg, env_pos, phis_appr, err_new
        
#         # --- Chosen parameters ---
#         print()
#         print("*** Results ***")
#         print("err: {:0.3e}".format(err_curr))
#         print("N-env: {:d}".format(N_env_PREV))

#         # --- Correct the angles ---
#         phis_appr_best[1:-1] += np.pi/2.
#         phis_appr_best[0]    += np.pi/4.
#         phis_appr_best[-1]   += np.pi/4.
  
#         # # --- Compute the best obtained angles ---
#         # Nen, Nep = get_N_neg_pos(N_env_PREV)
#         # flag_peaks = get_flag_peaks(Nen, Nep)
#         # env_neg = reproduce_env(cn_PREV, coef_ampl_neg, Nen, kappa_goal)
#         # env_pos = reproduce_env(cp_PREV, coef_ampl_pos, Nep, kappa_goal)
#         # phis_appr_best = construct_angles_from_envelops(env_neg, env_pos, flag_peaks)

#     # --- Return the estimated QSVT angles ---
#     print("Na: {:d}".format(len(phis_appr_best)))
#     print("Done.")
#     return phis_appr_best, coef_norm



