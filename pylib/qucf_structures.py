import numpy as np
import importlib as imp
import h5py
import matplotlib.pyplot as plt
import sys
from numba import jit
import cmath
import pylib.mix as mix

   
def reload():
    mix.reload()
    return


class GridSection__:
    # matrix column index with which the section is associated:
    column_id_ = None

    # the number of nonempty rows in the section:
    N_rows_ = None

    # row indices representated as integers in input registers:
    ri_ = None

    # values of matrix elements for each row in the section:
    as_ = None

    def __init__(self, circ, N, column_id):
        self.column_id_ = column_id
        nq_input = circ.input_regs_.nq_
        self.ri_ = np.array([N, nq_input], dtype=int)
        self.as_ = np.array(N, dtype=complex)
        self.ri_.fill(np.nan)
        self.N_rows_ = 0
        return

    def add_row(self, integers_in_input_regs, v1):
        self.N_rows_ += 1
        self.ri_[self.N_rows_-1,:] = np.array(integers_in_input_regs)
        self.as_[self.N_rows_-1]   = v1
        return

    def cut(self):
        self.ri_ = self.ri_[:self.N_rows_,:] 
        self.as_ = self.as_[:self.N_rows_,:] 
        return

    # fill ri_ and then cut the part with nan 