import numpy as np
import h5py
import matplotlib.pyplot as plt
import sys
import cmath
import cvxpy as cp
import pylib.mix as mix


def reload():
    mix.reload_module(mix)
    return


class Ch_:
    # where to save the coefficients:
    path_root_ = None

    # the chosen function:
    func_ch_ = None

    # the parameter of the approximated function:
    par_ = None

    # the function parity:
    parity_  = None

    # normalization factor to ensure that the max. function amplitude < 1:
    coef_norm_ = None

    # the name of the function:
    line_f_ = None

    # the string representation of the function parameter:
    line_par_ = None

    # ID of the chosen function:
    id_fun_ = None

    # the test function which is used for the minimization:
    test_func_ = None

    # the procedure used to reconstruct the chosen function using
    # the sequence of Chebyschev polynomials:
    rec_func_ = None

    # the degree of the polynomial used to approximate the chosen function:
    Nd_ = None

    # the chosen number of Chebyschev coefficients:
    Nc_ = None

    # the number of Chebyschev roots used as x-points where 
    # the chosen function and the test polynomial will be evaluated: 
    Nx_ = None

    # x-grid:
    x_ = None

    # the grid with the evaluated test function:
    y_ref_ = None

    # Chebyschev coefficients:
    coefs_ = None

    # Chebyschev coefficients computed by the direct method 
    #   if sel_method_ = 2
    coefs_2_ = None

    # the grid with the reconstructed function:
    y_rec_ = None

    # maximum absolute error:
    max_abs_err_ = None

    # 0: use the minimization procedure to compute the coefficients.
    # 1: use the direct computation.
    # 2: use both methods.
    sel_method_ = 0


    # absolute errors in the reconstructed signal:
    log_err_ = None


    # sel_method = 0: use the minimization procedure to compute the coefficients.
    # sel_method = 1: use the direct computation.
    # sel_method = 2: use both methods.
    def choose_method(self, sel_method):
        self.sel_method_ = sel_method
        return


    def reproduce_even(self, x, coefs):
        Nx = len(x)
        res_pol = np.zeros(Nx)
        for ix in range(Nx):
            res_pol[ix] = 0.
            for ii in range(self.Nc_):
                res_pol[ix] += coefs[ii] * np.cos((2*ii) * np.arccos(x[ix]))
        return res_pol


    def test_even_Ch(self, x, coefs):
        res_pol = 0.
        for ii in range(self.Nc_):
            res_pol += coefs[ii] * np.cos((2*ii) * np.arccos(x))
        return res_pol


    def reproduce_odd(self, x, coefs):
        Nx = len(x)
        res_pol = np.zeros(Nx)
        for ix in range(Nx):
            res_pol[ix] = 0.
            for ii in range(self.Nc_):
                res_pol[ix] += coefs[ii] * np.cos((2*ii+1) * np.arccos(x[ix]))
        return res_pol


    def test_odd_Ch(self, x, coefs):
        res_pol = 0.
        for ii in range(self.Nc_):
            res_pol += coefs[ii] * np.cos((2*ii+1) * np.arccos(x))
        return res_pol
    

    # - id_fun_ = 0 -
    def func_xG(self, x):
        Nx = len(x)
        y = np.zeros(Nx)
        for ii in range(Nx):
            y[ii] = -x[ii] * np.exp(-x[ii]**2/(2.*self.par_**2))
        return y

    # - id_fun_ = 1 -
    def func_inv(self, x):
        Nx = len(x)
        y = np.zeros(Nx)
        for ii in range(Nx):
            y[ii] = ( 1. - np.exp(-(5*self.par_*x[ii])**2) ) / x[ii]
        y *= 1.0/self.par_
        return y

    # - id_fun_ = 2 -
    def func_acos(self, x):
        Nx = len(x)
        y = np.zeros(Nx)
        for ii in range(Nx):
            y[ii] = np.arccos(x[ii])/np.pi - 0.5
        return y

    # - id_fun_ = 3 -
    def func_gauss(self, x):
        Nx = len(x)
        y = np.zeros(Nx)
        for ii in range(Nx):
            y[ii] = np.exp(- np.arcsin(x[ii])**2 / (2 * self.par_**2))
        return y

    # - id_fun_ = 4 -
    def func_LCHS_weights(self, x):
        Nx = len(x)
        y = np.zeros(Nx)
        for ii in range(Nx):
            y[ii] = 1. / np.sqrt(1 + self.par_**2 * np.arcsin(x[ii])**2)
        return y   
    
    # - id_fun_ = 10 -
    def series_x(self, x):
        Nx = len(x)
        y = np.zeros(Nx)
        for ix in range(Nx):
            y[ix] = 0.
            for i_coef in range(len(self.series_coefs_)):
                # y[ix] += self.series_coefs_[i_coef] * np.arcsin(x[ix]) ** i_coef
                y[ix] += self.series_coefs_[i_coef] * np.cos(np.arcsin(x[ix]) * i_coef)
        return y
    
    # --- Choose the function for which Chebyschev coefficients should be computed. ---
    def choose_func(
            self, id_func, par_in, 
            profile_in = None, 
            name_prof = None,
            parity_in = None, 
            path_root_in = None,
            series_coefs = None,
            x_grid = None
    ):
        self.par_ = par_in
        self.id_fun_ = id_func

        if self.id_fun_ == -1:
            self.y_ref_ = np.array(profile_in)
            self.path_root_ = path_root_in
            self.coef_norm_ = 1.0
            self.parity_ = parity_in
            self.line_f_ = name_prof
            self.line_par_ = ""
            if x_grid is not None:
                self.x_ = np.array(x_grid)

        if self.id_fun_ == 0:
            self.path_root_ ="./tools/QSVT-angles/xG/coefs/"
            self.coef_norm_ = 1.0
            self.func_ch_ = self.func_xG
            self.parity_ = 1
            self.line_f_ = "xG"
            self.line_par_ = "{:d}".format(int(self.par_*100))
            
        if self.id_fun_ == 1:
            self.path_root_ ="./tools/QSVT-angles/inversion/coefs/"
            self.coef_norm_ = 0.125
            self.func_ch_ = self.func_inv
            self.parity_ = 1
            self.line_f_ = "inv"
            self.line_par_ = "{:d}".format(int(self.par_))
            
        if self.id_fun_ == 2:
            self.path_root_ ="./tools/QSVT-angles/acos/coefs/"
            self.coef_norm_ = 1.0
            self.func_ch_ = self.func_acos
            self.parity_ = 1
            self.line_f_ = "arccos"
            self.line_par_ = "{:d}".format(None)
            
        if self.id_fun_ == 3:
            self.path_root_ ="./tools/QSVT-angles/Gaussian/coefs/"
            self.coef_norm_ = 1.0 - 1.e-3
            self.func_ch_ = self.func_gauss
            self.parity_ = 0
            self.line_f_ = "gauss"
            self.line_par_ = "{:d}".format(int(self.par_*100))
            
        if self.id_fun_ == 4:
            self.path_root_ ="./tools/QSVT-angles/LCHS-weights/coefs/"
            self.coef_norm_ = 1.0 - 1.e-2
            # self.coef_norm_ = 0.98
            self.func_ch_ = self.func_LCHS_weights
            self.parity_ = 0
            self.line_f_ = "LCHS-weights"
            self.line_par_ = "{:d}".format(int(self.par_))

        if self.id_fun_ == 10:
            self.series_coefs_ = series_coefs
            self.path_root_ = path_root_in
            self.coef_norm_ = 1.0
            self.parity_ = parity_in
            self.func_ch_ = self.series_x
            self.line_f_ = name_prof
            self.line_par_ = "Nc{:d}".format(len(self.series_coefs_))


        # --- Take the test and reconstruction functions appropriate to the chosen parity ---
        self.reproduce_eventest_func_ = None
        self.rec_func_ = None
        if self.parity_ == 0:
            self.test_func_ = self.test_even_Ch
            self.rec_func_  = self.reproduce_even
        else:
            self.test_func_ = self.test_odd_Ch
            self.rec_func_  = self.reproduce_odd

        # --- Print the chosen parameters and functions ---
        if self.id_fun_ >= 0:
            print("Function parameter:\t\t {:0.3e}".format(self.par_))
            print("Chosen function, parity:\t {:s}, {:d}".format(
                self.line_f_, self.parity_
            ))
        if self.id_fun_ == -1:
            print("Chosen profile: {:s}".format(self.line_f_))
            print("Chosen parity: {:d}".format(self.parity_))
        return 
    

    # --- Compute the Chebyschev coefficients ---
    def compute_Ch(self, Nd):
        self.Nd_ = Nd
        self.Nc_ = self.Nd_ // 2

        if self.id_fun_ >= 0: 
            self.Nx_ = self.Nd_*4
        if self.id_fun_ == -1:
            self.Nx_ = len(self.y_ref_)

        # x-grid: Chebyschev roots:
        if self.x_ is None:
            self.x_ = get_Cheb_roots(self.Nx_)    

        # - Evaluate the chosen function -
        if self.id_fun_ >= 0:
            self.y_ref_ = self.func_ch_(self.x_)
            self.y_ref_ *= self.coef_norm_

        # Computation:
        print()
        if self.sel_method_ == 0:
            print("Minimization method is used.")
            self.coefs_ = self.compute_Ch_via_minimization()
        elif self.sel_method_ == 1:
            print("Direct method is used.")
            self.coefs_ = self.compute_Ch_direct()
        elif self.sel_method_ == 2:
            print("Direct and minimization methods are used.")
            self.coefs_   = self.compute_Ch_via_minimization()
            self.coefs_2_ = self.compute_Ch_direct()

            print()
            print("max. diff. between coefs: {:0.3e}".format(
                np.max(np.abs(self.coefs_ - self.coefs_2_))
            ))
            print()
        else:
            print("Error: unknown method selector.")

        print("x[0] = ", self.x_[0])
        print("x[-1] = ", self.x_[-1])

        # reconstruct the function:
        self.y_rec_ = self.rec_func_(self.x_, self.coefs_)

        # maximum absolute error:
        self.max_abs_err_ = np.max(np.abs(self.y_rec_ - self.y_ref_))
        
        print("Chosen polynomial's degree:\t {:d}".format(self.Nd_))
        print("Number of coefficients:\t\t {:d}".format(self.Nc_))
        print("max. abs. error: {:0.3e}".format(self.max_abs_err_))
        return


    # --- Compute Chebyschev coefficients using the minimization procedure ---
    def compute_Ch_via_minimization(self):
        coefs = cp.Variable(self.Nc_)
        objective = cp.Minimize(cp.sum_squares(
            self.test_func_(self.x_, coefs) - self.y_ref_
        ))
        prob = cp.Problem(objective)
        result = prob.solve()
        print("Computation status: ", result)
        print()
        return coefs.value


    # --- Direct computation of Chebyschev coefficients ---
    def compute_Ch_direct(self):
        def compute_coef(n_order):
            c1 = 0.0
            c1 = np.sum(self.y_ref_ * np.cos(n_order * np.arccos(self.x_)))
            if n_order == 0:
                c1 *= 1./self.Nx_
            else:
                c1 *= 2./self.Nx_
            return c1
        # -------------------------------------
        coefs_Ch = np.zeros(self.Nc_)
        if self.parity_ == 0:
            for ii in range(self.Nc_):
                coefs_Ch[ii] = compute_coef(2*ii)
        else:
            for ii in range(self.Nc_):
                coefs_Ch[ii] = compute_coef(2*ii+1)
        return coefs_Ch  


    # --- Plot the original and the reconstructed functions ---
    def plot_reconstructed_function(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(self.x_, self.y_ref_, color="b", linewidth = 2, linestyle='-', label = "ref")
        ax.plot(self.x_, self.y_rec_,  color="r", linewidth = 2, linestyle=':', label = "reco")
        plt.xlabel('x')
        plt.ylabel("y")
        # plt.xlim(-5, 5)
        plt.legend()
        plt.grid(True)
        plt.show()
        return 


    # --- Plot errors --- 
    def plot_errors(self):
        self.log_err_ = np.zeros(len(self.y_rec_))
        for ii in range(len(self.y_rec_)):
            err1 = np.abs(self.y_rec_[ii] - self.y_ref_[ii])
            if err1 < 1e-17:
                err1 = 1e-17
            self.log_err_[ii] = np.log10(err1)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(self.x_, self.log_err_, color="b", linewidth = 2, linestyle='-')
        plt.xlabel('x')
        plt.ylabel("log10(yrec - yref)")
        # plt.xlim(-5, 5)
        # plt.legend()
        plt.grid(True)
        plt.show()
        return  


    # --- Plot the Chebyschev coefficients ---
    def plot_coefficients(self):
        if self.parity_ == 0:
            rx = np.array(range(0, self.Nd_,2))
        else:
            rx = np.array(range(1, self.Nd_+1,2))

        label_coef = None
        if self.sel_method_ == 0 or self.sel_method_ == 2:
            label_coef = "min"
        if self.sel_method_ == 1:
            label_coef = "direct"

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(
            rx, np.log10(np.abs(self.coefs_)),  
            color="b", linewidth = 2, linestyle='-', marker = "o",
            label = label_coef
        )
        if self.sel_method_ == 2:
            ax.plot(
                rx, np.log10(np.abs(self.coefs_2_)),  
                color="r", linewidth = 2, linestyle=':', marker = "s",
                label = "direct"
            )
        plt.xlabel('i')
        plt.ylabel("log10(coefs)")
        if self.sel_method_ == 2:
            plt.legend()
        plt.title("Computed Chebyschev coefs.")
        plt.grid(True)
        plt.show()
        return


    # --- Save the computed Chebyschev coefficients into the .hdf5 file ---
    def save_coefficients(self):
        from datetime import datetime
        from datetime import date

        # --- Current time ---
        curr_time = date.today().strftime("%m/%d/%Y") + ": " + datetime.now().strftime("%H:%M:%S")

        # --- Create the filename ---
        fname_ = "{:s}_{:s}_eps{:d}.hdf5".format(
            self.line_f_, self.line_par_, -int(np.log10(self.max_abs_err_))
        )
        full_fname = self.path_root_ + "/" + fname_

        # --- Store data ---
        print("write angles to:\n " + full_fname)
        with h5py.File(full_fname, "w") as f:
            grp = f.create_group("basic")
            grp.create_dataset('coef_norm',           data=float(self.coef_norm_))
            grp.create_dataset('date-of-simulation',  data=curr_time)
            grp.create_dataset('descr',  data=self.line_f_)
            grp.create_dataset('eps',    data=float(self.max_abs_err_))
            grp.create_dataset('param',  data=float(self.par_))
            grp.create_dataset('parity', data=self.parity_)

            grp = f.create_group("coefs")
            grp.create_dataset('real',  data = self.coefs_)  
        return  


    # --- Reproduce the function using sin(x)-grid ---
    def get_rec_y_sin_x(self, x = None): 
        if x is None:
            x = np.linspace(-1, 1, self.Nx_)
        return self.rec_func_(np.sin(x), self.coefs_)
    
    def get_rec_y_x(self, x = None): 
        if x is None:
            x = np.linspace(-1, 1, self.Nx_)
        return self.rec_func_(x, self.coefs_)


# **********************************************************************************************  
# ********************************************************************************************** 
    
# -------------------------------------------------------------------------------
def get_Cheb_roots(Nx_loc):
    x_roots = np.zeros(Nx_loc)
    for ii in range(Nx_loc):
        x_roots[ii] = np.cos((2*ii + 1)*np.pi / (2.*Nx_loc))
    x_roots = np.flip(x_roots)
    return x_roots


# -------------------------------------------------------------------------------
def approx_by_series(y_ref, Nc_loc, parity):
    def test_x(x, coefs):
        res_pol = 0.
        for i_coef in range(Nc_loc):
            # res_pol += coefs[i_coef] * x**(i_coef)
            res_pol += coefs[i_coef] * np.cos(x*(i_coef))
        return res_pol
    

    def reprod_x(x, coefs):
        Nx = len(x)
        res_pol = np.zeros(Nx)
        for ix in range(Nx):
            res_pol[ix] = 0.
            for i_coef in range(len(coefs)):
                # res_pol[ix] += coefs[i_coef] * x[ix]**(i_coef)
                res_pol[ix] += coefs[i_coef] * np.cos(x[ix]*(i_coef))
        return res_pol
    # -----------------------------------

    def test_ch_even(x, coefs):
        res_pol = 0.
        for i_coef in range(Nc_loc):
            res_pol += coefs[i_coef] * np.cos((2*i_coef) * np.arccos(x))
        return res_pol
    
    def reprod_ch_even(x, coefs):
        Nx = len(x)
        res_pol = np.zeros(Nx)
        for ix in range(Nx):
            res_pol[ix] = 0.
            for i_coef in range(len(coefs)):
                res_pol[ix] += coefs[i_coef] * np.cos((2*i_coef) * np.arccos(x[ix]))
        return res_pol
    
    def test_ch_odd(x, coefs):
        res_pol = 0.
        for i_coef in range(Nc_loc):
            res_pol += coefs[i_coef] * np.cos((2*i_coef+1) * np.arccos(x))
        return res_pol
    
    def reprod_ch_odd(x, coefs):
        Nx = len(x)
        res_pol = np.zeros(Nx)
        for ix in range(Nx):
            res_pol[ix] = 0.
            for i_coef in range(len(coefs)):
                res_pol[ix] += coefs[i_coef] * np.cos((2*i_coef+1) * np.arccos(x[ix]))
        return res_pol

    # -----------------------------------------------------------------------

    # --- x-grid ---
    x_roots = get_Cheb_roots(len(y_ref))

    # --- choose the series ---
    test_func, reprod_func = test_x, reprod_x

    # if parity == 0:
    #     test_func, reprod_func = test_ch_even, reprod_ch_even
    # else:
    #     test_func, reprod_func = test_ch_odd, reprod_ch_odd


    # --- Find the approximation ---
    coefs = cp.Variable(Nc_loc)
    objective = cp.Minimize(cp.sum_squares(
        test_func(x_roots, coefs) - y_ref
    ))
    prob = cp.Problem(objective)
    result = prob.solve()

    # --- reconstructed function ---
    y_rec = reprod_func(x_roots, coefs.value)

    max_abs_err = np.max(np.abs(y_rec - y_ref))
    print("Number of coefficients:\t\t {:d}".format(Nc_loc))
    print("max. abs. error: {:0.3e}".format(max_abs_err))

    # --- Plot the reconstructed function ---
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x_roots, y_ref, color="b", linewidth = 2, linestyle='-', label = "ref")
    ax.plot(x_roots, y_rec,  color="r", linewidth = 2, linestyle=':', label = "reco")
    plt.xlabel('x')
    plt.ylabel("y")
    # plt.xlim(-5, 5)
    plt.legend()
    plt.grid(True)
    plt.show()

    # --- Plot the errors ---
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(
        x_roots, 
        np.log10(np.abs(y_rec - y_ref)), 
        color="b", linewidth = 2, linestyle='-'
    )
    plt.xlabel('x')
    plt.ylabel("log10(yrec - yref)")
    plt.grid(True)
    plt.show()

    # --- Plot the coefficients ---
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(
        range(Nc_loc), np.log10(np.abs(coefs.value)), 
        color="b", linewidth = 2, linestyle='-', marker = "o"
    )
    plt.xlabel('i')
    plt.ylabel("log10(coefs)")
    plt.grid(True)
    plt.show()

    return y_rec, coefs.value