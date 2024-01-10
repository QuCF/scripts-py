import numpy as np
import importlib as imp
import h5py
import matplotlib.pyplot as plt
import sys
from numba import jit
import cmath
import pylib.mix as mix
from scipy.io import savemat

   
def reload():
    mix.reload_module(mix)
    return



def read_Fourier_coefs(path_coef, file_name):
    dd = {}
    filename = path_coef + file_name + ".hdf5"
    with h5py.File(filename, "r") as f:
        gr = f["basic"]
        dd["parity"] = gr["parity"][()]
        dd["param"]  = gr["param"][()]
        dd["eps"]    = gr["eps"][()]
        dd["factor_norm"]   = gr["coef_norm"][()]
        dd["sim_time"]      = gr["date-of-simulation"][()].decode("utf-8")
        dd["function_type"] = gr["descr"][()].decode("utf-8")

        gr = f["coefs"]
        dd["coefs_real"] = np.array(gr["real"])
        dd["coefs_imag"] = np.array(gr["imag"])
        
        gr = f["functions"]
        dd["x"]         = np.array(gr["x"]) # = psi if "gaussian-arcsin"
        dd["ref_funct"] = np.array(gr["pol"])
        dd["orig_func"] = np.array(gr["orig"])   
    dd["coefs"] = dd["coefs_real"] + 1j * dd["coefs_imag"]
    print("when simulated: ", dd["sim_time"])
    print()
    print("function-type: \t\t{:s}".format(dd["function_type"]))
    print("function-parity: \t\t{:d}".format(dd["parity"]))
    print("param: \t\t\t{:f}".format(dd["param"]))
    print("absolute error: \t{:0.3e}".format(dd["eps"]))
    print("norm. factor: \t\t{:0.3e}".format(dd["factor_norm"]))
    print("number of coefficients: {:d}".format(len(dd["coefs"])))
    return dd


def save_coef_mat(path_coef, file_name, data):
    mdic = {
        "function_type":    data["function_type"],
        "parity":           data["parity"],
        "coef":             data["coefs"],
        "param":            data["param"],
        "eps":              data["eps"],
        "rescaling_factor": data["factor_norm"]
    }
    full_name = path_coef + file_name + ".mat"
    savemat(full_name, mdic)
    print("file name: ", full_name)
    return


def read_angles(path_coef, filename):
    full_filename = path_coef + "/" + filename

    dd = {}
    print("Reading angles from the file \n{:s}".format(full_filename))
    with h5py.File(full_filename, "r") as f:
        gr = f["basic"]
        dd["date-of-simulation"] = gr["date-of-simulation"][()].decode("utf-8")
        
        dd["function-type"]       = gr["function-type"][()].decode("utf-8")
        dd["function-parity"]     = gr["function-parity"][()]
        dd["function-parameter"]  = gr["function-parameter"][()]
        dd["abs-error"]           = gr["abs-error"][()]
        dd["factor-norm"]         = gr["factor-norm"][()]
        
        gr = f["results"]
        dd["x"] = np.array(gr["x"]) # Chebyschev roots (nodes);
        dd["pol-coefs"] = np.array(gr["pol-coefs"]) # the polynomial constructed using coefficients;
        dd["pol-angles"] = np.array(gr["pol-angles"]) # the polynomial constructed using computed angles;
        dd["phis"] = np.array(gr["phis"])

    # print some basic results:
    print("when simulated: ", dd["date-of-simulation"])
    print()
    print("function-type: \t\t{:s}".format(dd["function-type"]))
    print("function-parity: \t\t{:d}".format(dd["function-parity"]))
    print("param: \t\t\t{:f}".format(dd["function-parameter"]))
    print("absolute error: \t{:0.3e}".format(dd["abs-error"]))
    print("norm. factor: \t\t{:0.3e}".format(dd["factor-norm"]))
    print("Number of angles: \t{:d}".format(len(dd["phis"])))

    return dd