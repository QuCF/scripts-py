from asyncio import constants
from pickletools import read_unicodestringnl
import sys
import datetime
from typing import no_type_check
import numpy as np
import time
import h5py
import subprocess
from termcolor import colored
import os

import pylib.mix as mix
import pylib.measurement as mse


def reload():
    mix.reload_module(mix)
    mix.reload_module(mse)
    return


class TESTS__:
    path_gl_ = "../QuCF/tests/"
    path_launch_ = "../QuCF/build_qucf/QuCF"
    __flag_print_all_details_ = None
    __line_PASSED = None
    __line_FAILED = None

    __prec_ = 1e-14


    def __init__(self, flag_print_all_details=False):
        self.__flag_print_all_details_= flag_print_all_details
        self.__line_PASSED = colored("PASSED", 'green', attrs=['reverse', 'blink'])
        self.__line_FAILED = colored("FAILED", 'red', attrs=['reverse', 'blink'])
        return


    def set_precision(self, prec):
        self.__prec_ = prec
        return


    def __launch_test(self, path_test, test_name):
        path_work = self.path_gl_ + path_test
        if self.__flag_print_all_details_:
            subprocess.run([self.path_launch_, test_name, path_work])
        else:
            subprocess.run([self.path_launch_, test_name, path_work], stdout=subprocess.DEVNULL)

        print("\nRead the project: ", test_name)
        print("from the folder: ", self.path_gl_ + path_test)
        oor = mse.MeasOracle__(flag_print=self.__flag_print_all_details_)
        oor.pname_ = test_name
        oor.path_  = self.path_gl_ + path_test
        oor.open()

        return oor

    
    def __cond_print(self, conds):
        if np.all(conds):
            print(self.__line_PASSED + " ", end='')
        else:
            print(self.__line_FAILED + " ", end='')


    def __get_state(self, oor, i_state):
        state_superposition_inp = oor.init_states_[i_state]["state"]
        state_superposition_out = oor.output_all_states_[i_state]["state"]

        ampls_inp = oor.init_states_[i_state]["ampls"]
        ampls_out = oor.output_all_states_[i_state]["ampls"]

        return state_superposition_inp, state_superposition_out, ampls_inp, ampls_out

    # --- Run tests with basic gates ---
    def run_basic_tests(self):
        self.__test_X()
        self.__test_Y()
        self.__test_Z()
        self.__test_H()
        self.__test_SWAP()
        self.__test_Rx()
        self.__test_Ry()
        self.__test_Rz()
        self.__test_Rc()
        return
    

    def run_arithmetic_tests(self):
        self.__test_incrementor()
        self.__test_decrementor()
        self.__test_adder_arb_fixed()
        self.__test_subtractor_arb_fixed()
        self.__test_comparator_arb_fixed()

        return


    # --- Run tests with large circuits ---
    def run_large_circuit_tests(self):
        self.__test_high_n_ancilla()
        return

    def run_sin_test(self):
        self.__test_sin()
        return

    def run_qsvt_tests(self):
        self.__test_qsvt_gauss()
        return


    def __test_X(self):
        print("Testing X gate:... ", end='')
        
        path_test = "basic-gates"
        test_name = "X"
        oor = self.__launch_test(path_test, test_name)

        reg_inp_states, reg_out_states = oor.form_reg_input_output_states()

        i_state = 0
        ss_inp, ss_out, ampls_inp, ampls_out = self.__get_state(oor, i_state)
        self.__cond_print([
            len(ss_out) == 1,  # output superposition of states has only a single state
            np.all(ss_inp[0] == ss_out[0]), # input and output coincide
            ampls_out[0][0] == 1.0 # real-ampl = 1
        ])

        i_state = 1
        ss_inp, ss_out, ampls_inp, ampls_out = self.__get_state(oor, i_state)
        self.__cond_print([
            len(ss_out) == 1,
            reg_out_states["y"][i_state][0] == 1,
            ampls_out[0][0] == 1.0 # real-ampl = 1
        ])

        i_state = 2
        ss_inp, ss_out, ampls_inp, ampls_out = self.__get_state(oor, i_state)
        self.__cond_print([
            len(ss_out) == 1,
            reg_out_states["y"][i_state][0] == 2,
            ampls_out[0][0] == 1.0 # real-ampl = 1
        ])

        print()
        return
    

    def __test_Y(self):
        print("Testing Y gate:... ", end='')
        
        path_test = "basic-gates"
        test_name = "Y"
        oor = self.__launch_test(path_test, test_name)

        reg_inp_states, reg_out_states = oor.form_reg_input_output_states()

        i_state = 0
        ss_inp, ss_out, ampls_inp, ampls_out = self.__get_state(oor, i_state)
        self.__cond_print([
            len(ss_out) == 1,  # output superposition of states has only a single state
            np.all(ss_inp[0] == ss_out[0]), # input and output coincide
            ampls_out[0][0] == 1.0 # real-ampl = 1
        ])

        i_state = 1
        ss_inp, ss_out, ampls_inp, ampls_out = self.__get_state(oor, i_state)
        self.__cond_print([
            len(ss_out) == 1,
            reg_out_states["y"][i_state][0] == 4,
            ampls_out[0][1] == 1.0 # imag-ampl = 1
        ])

        i_state = 2
        ss_inp, ss_out, ampls_inp, ampls_out = self.__get_state(oor, i_state)
        self.__cond_print([
            len(ss_out) == 1,
            reg_out_states["y"][i_state][0] == 0,
            ampls_out[0][1] == -1.0 # imag-ampl = -1
        ])

        print()
        return
    

    def __test_Z(self):
        print("Testing Z gate:... ", end='')
        
        path_test = "basic-gates"
        test_name = "Z"
        oor = self.__launch_test(path_test, test_name)

        reg_inp_states, reg_out_states = oor.form_reg_input_output_states()

        i_state = 0
        ss_inp, ss_out, ampls_inp, ampls_out = self.__get_state(oor, i_state)
        self.__cond_print([
            len(ss_out) == 1,  # output superposition of states has only a single state
            np.all(ss_inp[0] == ss_out[0]), # input and output coincide
            ampls_out[0][0] == 1.0
        ])

        i_state = 1
        ss_inp, ss_out, ampls_inp, ampls_out = self.__get_state(oor, i_state)
        self.__cond_print([
            len(ss_out) == 1,  # output superposition of states has only a single state
            np.all(ss_inp[0] == ss_out[0]), # input and output coincide
            ampls_out[0][0] == 1.0
        ])

        i_state = 2
        ss_inp, ss_out, ampls_inp, ampls_out = self.__get_state(oor, i_state)
        self.__cond_print([
            len(ss_out) == 1,  # output superposition of states has only a single state
            np.all(ss_inp[0] == ss_out[0]), # input and output coincide
            ampls_out[0][0] == -1.0
        ])

        i_state = 3
        ss_inp, ss_out, ampls_inp, ampls_out = self.__get_state(oor, i_state)
        self.__cond_print([
            len(ss_out) == 1,  # output superposition of states has only a single state
            np.all(ss_inp[0] == ss_out[0]), # input and output coincide
            ampls_out[0][0] == -1.0
        ])

        print()
        return
    

    def __test_H(self):
        print("Testing H gate:... ", end='')
        
        path_test = "basic-gates"
        test_name = "H"
        oor = self.__launch_test(path_test, test_name)

        coef_d = 1./np.sqrt(2.)

        reg_inp_states, reg_out_states = oor.form_reg_input_output_states()

        i_state = 0
        ss_inp, ss_out, ampls_inp, ampls_out = self.__get_state(oor, i_state)
        self.__cond_print([
            len(ss_out) == 1,  # output superposition of states has only a single state
            np.all(ss_inp[0] == ss_out[0]), # input and output coincide
            ampls_out[0][0] == 1.0 # real-ampl = 1
        ])

        i_state = 1
        ss_inp, ss_out, ampls_inp, ampls_out = self.__get_state(oor, i_state)
        self.__cond_print([
            len(ss_out) == 2,
            reg_out_states["y"][i_state][0] == 0,
            reg_out_states["y"][i_state][1] == 2,
            ampls_out[0][0] == coef_d, 
            ampls_out[1][0] == coef_d 
        ])

        i_state = 2
        ss_inp, ss_out, ampls_inp, ampls_out = self.__get_state(oor, i_state)
        self.__cond_print([
            len(ss_out) == 2,
            reg_out_states["y"][i_state][0] == 0,
            reg_out_states["y"][i_state][1] == 1,
            ampls_out[0][0] == coef_d, 
            ampls_out[1][0] == coef_d 
        ])

        print()
        return
    

    def __test_SWAP(self):
        print("Testing SWAP gate:... ", end='')
        
        path_test = "basic-gates"
        test_name = "SWAP"
        oor = self.__launch_test(path_test, test_name)

        coef_d = 1./np.sqrt(2.)

        reg_inp_states, reg_out_states = oor.form_reg_input_output_states()

        i_state = 0
        ss_inp, ss_out, ampls_inp, ampls_out = self.__get_state(oor, i_state)
        self.__cond_print([
            len(ss_out) == 1,  # output superposition of states has only a single state
            np.all(ss_inp[0] == ss_out[0]), # input and output coincide
            ampls_out[0][0] == 1.0 # real-ampl = 1
        ])

        i_state = 1
        ss_inp, ss_out, ampls_inp, ampls_out = self.__get_state(oor, i_state)
        self.__cond_print([
            len(ss_out) == 1,
            reg_out_states["y"][i_state][0] == 9,
            reg_out_states["w"][i_state][0] == 3,
            reg_out_states["x"][i_state][0] == 0,
            ampls_out[0][0] == 1.0, 
        ])

        i_state = 2
        ss_inp, ss_out, ampls_inp, ampls_out = self.__get_state(oor, i_state)
        self.__cond_print([
            len(ss_out) == 1,
            reg_out_states["y"][i_state][0] == 5,
            reg_out_states["w"][i_state][0] == 2,
            reg_out_states["x"][i_state][0] == 1,
            ampls_out[0][0] == 1.0, 
        ])

        print()
        return


    def __test_Rx(self):
        print("Testing Rx gate:... ", end='')
        
        path_test = "basic-gates"
        test_name = "Rx"
        oor = self.__launch_test(path_test, test_name)

        theta = 2.64
        theta_h = theta/2.
        coef_ul = np.cos(theta_h)
        coef_ur = -1j*np.sin(theta_h)
        coef_bl = -1j*np.sin(theta_h)
        coef_br = np.cos(theta_h)

        reg_inp_states, reg_out_states = oor.form_reg_input_output_states()

        i_state = 0
        ss_inp, ss_out, ampls_inp, ampls_out = self.__get_state(oor, i_state)
        self.__cond_print([
            len(ss_out) == 1,  # output superposition of states has only a single state
            np.all(ss_inp[0] == ss_out[0]), # input and output coincide
            ampls_out[0][0] == 1.0 # real-ampl = 1
        ])

        i_state = 1
        ss_inp, ss_out, ampls_inp, ampls_out = self.__get_state(oor, i_state)
        self.__cond_print([
            len(ss_out) == 2,
            reg_out_states["y"][i_state][0] == 0,
            reg_out_states["y"][i_state][1] == 1,
            np.abs(ampls_out[0][0] - coef_ul) < self.__prec_, 
            np.abs(ampls_out[1][1] - np.imag(coef_bl)) < self.__prec_
        ])

        i_state = 2
        ss_inp, ss_out, ampls_inp, ampls_out = self.__get_state(oor, i_state)
        self.__cond_print([
            len(ss_out) == 2,
            reg_out_states["y"][i_state][0] == 0,
            reg_out_states["y"][i_state][1] == 2,
            np.abs(ampls_out[0][1] - np.imag(coef_ur)) < self.__prec_, 
            np.abs(ampls_out[1][0] - coef_br) < self.__prec_
        ])

        print()
        return
    

    def __test_Ry(self):
        print("Testing Ry gate:... ", end='')
        
        path_test = "basic-gates"
        test_name = "Ry"
        oor = self.__launch_test(path_test, test_name)

        theta = 0.48
        theta_h = theta/2.
        coef_ul = np.cos(theta_h)
        coef_ur = -np.sin(theta_h)
        coef_bl = np.sin(theta_h)
        coef_br = np.cos(theta_h)

        reg_inp_states, reg_out_states = oor.form_reg_input_output_states()

        i_state = 0
        ss_inp, ss_out, ampls_inp, ampls_out = self.__get_state(oor, i_state)
        self.__cond_print([
            len(ss_out) == 1,  # output superposition of states has only a single state
            np.all(ss_inp[0] == ss_out[0]), # input and output coincide
            reg_out_states["y"][i_state][0] == 0,
            reg_out_states["w"][i_state][0] == 3,
            reg_out_states["x"][i_state][0] == 7,
            ampls_out[0][0] == 1.0 # real-ampl = 1
        ])

        i_state = 1
        ss_inp, ss_out, ampls_inp, ampls_out = self.__get_state(oor, i_state)
        self.__cond_print([
            len(ss_out) == 2,
            reg_out_states["y"][i_state][0] == 0,
            reg_out_states["y"][i_state][1] == 2,
            np.abs(ampls_out[0][0] - coef_ul) < self.__prec_, 
            np.abs(ampls_out[1][0] - coef_bl) < self.__prec_
        ])

        i_state = 2
        ss_inp, ss_out, ampls_inp, ampls_out = self.__get_state(oor, i_state)
        self.__cond_print([
            len(ss_out) == 2,
            reg_out_states["y"][i_state][0] == 0,
            reg_out_states["y"][i_state][1] == 4,
            np.abs(ampls_out[0][0] - coef_ur) < self.__prec_, 
            np.abs(ampls_out[1][0] - coef_br) < self.__prec_
        ])

        print()
        return
    

    def __test_Rz(self):
        print("Testing Rz gate:... ", end='')
        
        path_test = "basic-gates"
        test_name = "Rz"
        oor = self.__launch_test(path_test, test_name)

        theta = 3.02
        theta_h = theta/2.
        coef_ul = np.cos(theta_h) - 1j * np.sin(theta_h)
        coef_br = np.cos(theta_h) + 1j * np.sin(theta_h)

        reg_inp_states, reg_out_states = oor.form_reg_input_output_states()

        i_state = 0
        ss_inp, ss_out, ampls_inp, ampls_out = self.__get_state(oor, i_state)
        self.__cond_print([
            len(ss_out) == 1,  # output superposition of states has only a single state
            np.all(ss_inp[0] == ss_out[0]), # input and output coincide
            reg_out_states["y"][i_state][0] == 0,
            reg_out_states["w"][i_state][0] == 0,
            reg_out_states["x"][i_state][0] == 0,
            ampls_out[0][0] == 1.0 # real-ampl = 1
        ])

        i_state = 1
        ss_inp, ss_out, ampls_inp, ampls_out = self.__get_state(oor, i_state)
        self.__cond_print([
            len(ss_out) == 1,
            reg_out_states["y"][i_state][0] == 0,
            reg_out_states["w"][i_state][0] == 3,
            reg_out_states["x"][i_state][0] == 0,
            np.abs(ampls_out[0][0] - np.real(coef_ul)) < self.__prec_, 
            np.abs(ampls_out[0][1] - np.imag(coef_ul)) < self.__prec_
        ])

        i_state = 2
        ss_inp, ss_out, ampls_inp, ampls_out = self.__get_state(oor, i_state)
        self.__cond_print([
            len(ss_out) == 1,
            reg_out_states["y"][i_state][0] == 8,
            reg_out_states["w"][i_state][0] == 2,
            reg_out_states["x"][i_state][0] == 1,
            np.abs(ampls_out[0][0] - np.real(coef_br)) < self.__prec_, 
            np.abs(ampls_out[0][1] - np.imag(coef_br)) < self.__prec_
        ])

        print()
        return
    

    def __test_Rc(self):
        print("Testing Rc gate:... ", end='')
        
        path_test = "basic-gates"
        test_name = "Rc"
        oor = self.__launch_test(path_test, test_name)

        theta_y = 1.29
        theta_z = 2.64
        theta_yh = theta_y/2.
        theta_zh = theta_z/2.
        coef_ul = np.cos(theta_yh) * np.exp(-1j * theta_zh)
        coef_ur = - np.sin(theta_yh) * np.exp(1j * theta_zh)
        coef_bl = np.sin(theta_yh) * np.exp(-1j * theta_zh)
        coef_br = np.cos(theta_yh) * np.exp(1j * theta_zh)

        reg_inp_states, reg_out_states = oor.form_reg_input_output_states()

        i_state = 0
        ss_inp, ss_out, ampls_inp, ampls_out = self.__get_state(oor, i_state)
        self.__cond_print([
            len(ss_out) == 1,  # output superposition of states has only a single state
            np.all(ss_inp[0] == ss_out[0]), # input and output coincide
            reg_out_states["y"][i_state][0] == 0,
            reg_out_states["w"][i_state][0] == 0,
            reg_out_states["x"][i_state][0] == 0,
            ampls_out[0][0] == 1.0 # real-ampl = 1
        ])

        i_state = 1
        ss_inp, ss_out, ampls_inp, ampls_out = self.__get_state(oor, i_state)
        self.__cond_print([
            len(ss_out) == 2,
            reg_out_states["y"][i_state][0] == 0,
            reg_out_states["w"][i_state][0] == 3,
            reg_out_states["x"][i_state][0] == 0,
            reg_out_states["y"][i_state][1] == 1,
            np.abs(ampls_out[0][0] - np.real(coef_ul)) < self.__prec_, 
            np.abs(ampls_out[0][1] - np.imag(coef_ul)) < self.__prec_, 
            np.abs(ampls_out[1][0] - np.real(coef_bl)) < self.__prec_,
            np.abs(ampls_out[1][1] - np.imag(coef_bl)) < self.__prec_
        ])

        i_state = 2
        ss_inp, ss_out, ampls_inp, ampls_out = self.__get_state(oor, i_state)
        self.__cond_print([
            len(ss_out) == 2,
            reg_out_states["y"][i_state][0] == 0,
            reg_out_states["w"][i_state][0] == 2,
            reg_out_states["x"][i_state][0] == 1,
            reg_out_states["y"][i_state][1] == 2,
            np.abs(ampls_out[0][0] - np.real(coef_ur)) < self.__prec_, 
            np.abs(ampls_out[0][1] - np.imag(coef_ur)) < self.__prec_, 
            np.abs(ampls_out[1][0] - np.real(coef_br)) < self.__prec_,
            np.abs(ampls_out[1][1] - np.imag(coef_br)) < self.__prec_
        ])

        print()
        return
    

    def __test_incrementor(self):
        print("Testing incrementor:... ", end='')
        
        path_test = "arithmetic"
        test_name = "incrementor"
        oor = self.__launch_test(path_test, test_name)

        reg_inp_states, reg_out_states = oor.form_reg_input_output_states()

        i_state = 0
        ss_inp, ss_out, ampls_inp, ampls_out = self.__get_state(oor, i_state)
        self.__cond_print([
            len(ss_out) == 1,  # output superposition of states has only a single state
            reg_out_states["y"][i_state][0] == 1,
            ampls_out[0][0] == 1.0 # real-ampl = 1
        ])

        i_state = 1
        ss_inp, ss_out, ampls_inp, ampls_out = self.__get_state(oor, i_state)
        self.__cond_print([
            len(ss_out) == 1,
            reg_out_states["y"][i_state][0] == 4,
            ampls_out[0][0] == 1.0 # real-ampl = 1
        ])

        i_state = 2
        ss_inp, ss_out, ampls_inp, ampls_out = self.__get_state(oor, i_state)
        self.__cond_print([
            len(ss_out) == 1,
            reg_out_states["y"][i_state][0] == 0,
            ampls_out[0][0] == 1.0 # real-ampl = 1
        ])

        print()
        return
    

    def __test_decrementor(self):
        print("Testing decrementor:... ", end='')
        
        path_test = "arithmetic"
        test_name = "decrementor"
        oor = self.__launch_test(path_test, test_name)

        reg_inp_states, reg_out_states = oor.form_reg_input_output_states()

        i_state = 0
        ss_inp, ss_out, ampls_inp, ampls_out = self.__get_state(oor, i_state)
        self.__cond_print([
            len(ss_out) == 1,  # output superposition of states has only a single state
            reg_out_states["y"][i_state][0] == 15,
            ampls_out[0][0] == 1.0 # real-ampl = 1
        ])

        i_state = 1
        ss_inp, ss_out, ampls_inp, ampls_out = self.__get_state(oor, i_state)
        self.__cond_print([
            len(ss_out) == 1,
            reg_out_states["y"][i_state][0] == 3,
            ampls_out[0][0] == 1.0 # real-ampl = 1
        ])

        i_state = 2
        ss_inp, ss_out, ampls_inp, ampls_out = self.__get_state(oor, i_state)
        self.__cond_print([
            len(ss_out) == 1,
            reg_out_states["y"][i_state][0] == 14,
            ampls_out[0][0] == 1.0 # real-ampl = 1
        ])

        print()
        return


    def __test_adder_arb_fixed(self):
        print("Testing AdderFixed:... ", end='')
        
        path_test = "arithmetic"
        test_name = "adder_arb_fixed"
        oor = self.__launch_test(path_test, test_name)

        reg_inp_states, reg_out_states = oor.form_reg_input_output_states()

        i_state = 0
        ss_inp, ss_out, ampls_inp, ampls_out = self.__get_state(oor, i_state)
        self.__cond_print([
            len(ss_out) == 1,  # output superposition of states has only a single state
            reg_out_states["y"][i_state][0] == 4,
            reg_out_states["w"][i_state][0] == 0,
            np.abs(ampls_out[0][0] - 1.0) < self.__prec_
        ])

        i_state = 1
        ss_inp, ss_out, ampls_inp, ampls_out = self.__get_state(oor, i_state)
        self.__cond_print([
            len(ss_out) == 1,
            reg_out_states["y"][i_state][0] == 0,
            reg_out_states["w"][i_state][0] == 1,
            np.abs(ampls_out[0][0] - 1.0) < self.__prec_ 
        ])

        i_state = 2
        ss_inp, ss_out, ampls_inp, ampls_out = self.__get_state(oor, i_state)
        self.__cond_print([
            len(ss_out) == 1,
            reg_out_states["y"][i_state][0] == 4,
            reg_out_states["w"][i_state][0] == 1,
            np.abs(ampls_out[0][0] - 1.0) < self.__prec_
        ])

        i_state = 3
        ss_inp, ss_out, ampls_inp, ampls_out = self.__get_state(oor, i_state)
        self.__cond_print([
            len(ss_out) == 1,
            reg_out_states["y"][i_state][0] == 0,
            reg_out_states["w"][i_state][0] == 0,
            np.abs(ampls_out[0][0] - 1.0) < self.__prec_
        ])

        print()
        return
    

    def __test_subtractor_arb_fixed(self):
        print("Testing SubtractorFixed:... ", end='')
        
        path_test = "arithmetic"
        test_name = "subtractor_arb_fixed"
        oor = self.__launch_test(path_test, test_name)

        reg_inp_states, reg_out_states = oor.form_reg_input_output_states()

        i_state = 0
        ss_inp, ss_out, ampls_inp, ampls_out = self.__get_state(oor, i_state)
        self.__cond_print([
            len(ss_out) == 1,  # output superposition of states has only a single state
            reg_out_states["y"][i_state][0] == 13,
            reg_out_states["w"][i_state][0] == 1,
            np.abs(ampls_out[0][0] - 1.0) < self.__prec_
        ])

        i_state = 1
        ss_inp, ss_out, ampls_inp, ampls_out = self.__get_state(oor, i_state)
        self.__cond_print([
            len(ss_out) == 1,
            reg_out_states["y"][i_state][0] == 8,
            reg_out_states["w"][i_state][0] == 0,
            np.abs(ampls_out[0][0] - 1.0) < self.__prec_ 
        ])

        i_state = 2
        ss_inp, ss_out, ampls_inp, ampls_out = self.__get_state(oor, i_state)
        self.__cond_print([
            len(ss_out) == 1,
            reg_out_states["y"][i_state][0] == 1,
            reg_out_states["w"][i_state][0] == 0,
            np.abs(ampls_out[0][0] - 1.0) < self.__prec_
        ])

        print()
        return
    

    def __test_comparator_arb_fixed(self):
        print("Testing ComparatorFixed:... ", end='')
        
        path_test = "arithmetic"
        test_name = "comparator_arb_fixed"
        oor = self.__launch_test(path_test, test_name)

        reg_inp_states, reg_out_states = oor.form_reg_input_output_states()

        i_state = 0
        ss_inp, ss_out, ampls_inp, ampls_out = self.__get_state(oor, i_state)
        self.__cond_print([
            len(ss_out) == 1,  # output superposition of states has only a single state
            reg_out_states["y"][i_state][0] == 5,
            reg_out_states["w"][i_state][0] == 2,
            np.abs(ampls_out[0][0] - 1.0) < self.__prec_
        ])

        i_state = 1
        ss_inp, ss_out, ampls_inp, ampls_out = self.__get_state(oor, i_state)
        self.__cond_print([
            len(ss_out) == 1,
            reg_out_states["y"][i_state][0] == 4,
            reg_out_states["w"][i_state][0] == 0,
            np.abs(ampls_out[0][0] - 1.0) < self.__prec_ 
        ])

        i_state = 2
        ss_inp, ss_out, ampls_inp, ampls_out = self.__get_state(oor, i_state)
        self.__cond_print([
            len(ss_out) == 1,
            reg_out_states["y"][i_state][0] == 2,
            reg_out_states["w"][i_state][0] == 0,
            np.abs(ampls_out[0][0] - 1.0) < self.__prec_
        ])

        i_state = 3
        ss_inp, ss_out, ampls_inp, ampls_out = self.__get_state(oor, i_state)
        self.__cond_print([
            len(ss_out) == 1,
            reg_out_states["y"][i_state][0] == 6,
            reg_out_states["w"][i_state][0] == 1,
            np.abs(ampls_out[0][0] - 1.0) < self.__prec_
        ])

        print()
        return


    def __test_high_n_ancilla(self):
        print("Testing a circuit with 28 qubits:... ", end='')
        path_test = "large-circuits"
        test_name = "high-n-qubits"
        oor = self.__launch_test(path_test, test_name)

        coef_d = 1./2.

        reg_inp_states, reg_out_states = oor.form_reg_input_output_states()

        i_state = 0
        ss_inp, ss_out, ampls_inp, ampls_out = self.__get_state(oor, i_state)
        self.__cond_print([
            len(ss_out) == 4,  # output superposition of states has only a single state
            reg_out_states["r"][i_state][0] == 134217724,
            reg_out_states["r"][i_state][1] == 134217725,
            reg_out_states["r"][i_state][2] == 268435452,
            reg_out_states["r"][i_state][3] == 268435453,
            np.abs(ampls_out[0][0] - coef_d) < self.__prec_,
            np.abs(ampls_out[1][0] + coef_d) < self.__prec_,
            np.abs(ampls_out[2][0] + coef_d) < self.__prec_,
            np.abs(ampls_out[3][0] - coef_d) < self.__prec_
        ])
        print()
        return
    

    def __test_sin(self):
        print("Testing the SIN gate:... ", end='')
        path_test = "sin-gate"
        test_name = "sin"
        oor = self.__launch_test(path_test, test_name)

        alpha_0 = oor.constants_["alpha_0"]
        alpha_1 = oor.constants_["alpha"]
        nx = int(oor.constants_["nx"])
        Nx = 1 << nx
        dx = (2. * alpha_1) / Nx
        x = np.zeros(Nx)
        for ix in range(Nx):
            x[ix] = alpha_0 + ix * dx

        # take into account that there is the initialization circuit;
        coef_d = (1./np.sqrt(2.))**nx
        res_check = coef_d * np.sin(x) 

        i_state = 0 # id of the initial state;
        out_za = oor.output_zero_anc_states_[i_state]["state"]

        oor.set_work_states(id_input_state = i_state)
        res_za = oor.get_var_x({"a_be": 0}, "j")

        # check the bitstrings:
        flag_equal_bs = 1
        for ix in range(Nx):
            bs1 = mix.find_bit_array_of_int(ix, nx)
            circ_bs = out_za[ix][1:] # the first qubit is the ancilla;
            if np.all(bs1 == circ_bs):
                flag_equal_bs *= 1
            else:
                flag_equal_bs *= 0

        # check resulting amplitudes:
        flag_equal_values = 1
        for ix in range(Nx):
            if np.abs(res_check[ix] - np.real(res_za[ix])) < self.__prec_:
                flag_equal_values *= 1
            else:
                flag_equal_values *= 0

        # whether all conditions are satified?
        self.__cond_print([
            len(out_za) == Nx,
            flag_equal_bs,
            flag_equal_values
        ])
        return


    def __test_qsvt_gauss(self):
        print("Testing QSVT: Gaussian function:... ", end='')
        path_test = "qsvt-gauss"
        test_name = "gauss"
        oor = self.__launch_test(path_test, test_name)

        
        oor.read_qsvt()

        QSVT_rescaling = 0.98 # coefficient used for the calculation of the QSVT angles:
        mu = oor.dd_["gauss"]["mu"]

        alpha_0 = oor.constants_["alpha_0"]
        alpha_1 = oor.constants_["alpha"]
        x_center = alpha_0 + 1.0

        nx = int(oor.constants_["nx"])
        Nx = 1 << nx
        dx = (2. * alpha_1) / Nx
        x = np.zeros(Nx)
        for ix in range(Nx):
            x[ix] = alpha_0 + ix * dx

        # take into account that there is the initialization circuit;
        coef_d = (1./np.sqrt(2.))**nx
        res_check = coef_d * QSVT_rescaling * np.exp(- (x - x_center)**2/(2*mu**2))

        # read states computed by the QSVT circuit:
        i_state = 0 # id of the initial state;
        out_za = oor.output_zero_anc_states_[i_state]["state"]

        oor.set_work_states(id_input_state = i_state)
        res_za = oor.get_var_x({"a_be": 0, "a_qsvt": 0}, "j")

        # check the bitstrings:
        flag_equal_bs = 1
        for ix in range(Nx):
            bs1 = mix.find_bit_array_of_int(ix, nx)
            circ_bs = out_za[ix][2:] # the first two qubits are ancillae;
            if np.all(bs1 == circ_bs):
                flag_equal_bs *= 1
            else:
                flag_equal_bs *= 0

        # check resulting amplitudes:
        flag_equal_values = 1
        for ix in range(Nx):
            v_ref = res_check[ix]
            v_qsvt = np.real(res_za[ix])
            v_qsvt = np.abs(v_qsvt) # to exclude a global phase;
            if np.abs(v_ref - v_qsvt) < self.__prec_:
                flag_equal_values *= 1
            else:
                flag_equal_values *= 0

        # whether all conditions are satified?
        self.__cond_print([
            len(out_za) == Nx,
            flag_equal_bs,
            flag_equal_values
        ])
        return
    

    def test_AE_qsvt_gauss(self):
        import matplotlib.pyplot as plt

        print("Testing QSVT: Gaussian function:... ", end='')
        path_test = "amplitude-estimation"
        test_name = "AE"
        oor = self.__launch_test(path_test, test_name)

        nx_half = int(oor.constants_["nx"])
        Nx_half = 1 << nx_half

        nx_full = int(oor.constants_["nx_full"])
        Nx_full = 1 << nx_full

        scaling_mu = 1 << (nx_full - nx_half)

        QSVT_rescaling = 0.98 # coefficient used for the calculation of the QSVT angles:

        coef_d = (1. / np.sqrt(2))**nx_full
        F_gauss_asin  = \
            lambda x, param, coef_norm, x0: \
                coef_d*coef_norm * np.exp(- (x - x0)**2/(2*param**2))
        
        x_qc = 2*oor.get_x_grid("j") - 1.0

        oor.read_qsvt()
        probs_meas = oor.probs_
        N_y_points = len(probs_meas)
        int_ys = np.array(range(N_y_points))

        eig_phases = np.pi * int_ys / N_y_points
        probs_est = 1. - np.sin(eig_phases)**2

        En_est_array = probs_est / Nx_half

        id_max_qc = np.where(probs_meas == np.max(probs_meas))[0][0]
        En_est_qc = En_est_array[id_max_qc]

        # classical integral:
        F_cl = F_gauss_asin(x_qc[0:Nx_half], oor.dd_["gauss"]["mu"]/scaling_mu, QSVT_rescaling, -0.5)
        En_cl = np.sum(np.abs(F_cl)**2) / Nx_half

        # analytical error:
        pr_ref = En_cl * Nx_half
        analytical_abs_error = 2*np.pi*np.sqrt(pr_ref*(1-pr_ref))/N_y_points + np.pi**2/(N_y_points**2)
        analytical_abs_error = analytical_abs_error / Nx_half

        # --- form the histogram ---
        print()
        print("classical integration: {:0.3e}".format(En_cl))
        print("   vs   ")
        print("AE integration: {:0.3e}".format(En_est_qc))
        print()
        print("ny: {:d}".format(int(oor.constants_["ny"])))
        print("analytical error (with 0.81 probability): {:0.3e}".format(analytical_abs_error))
        print()
        print("-------------------------------------------------------------------------------")
        print(
            "The test is passed if\n" + 
            "the absolute difference between the classical and AE integration\n" + 
            "is less than the analytical error."
        )
        print("-------------------------------------------------------------------------------")
        self.__cond_print([
            np.abs(En_cl - En_est_qc) < analytical_abs_error
        ])

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(
            En_est_array, probs_meas, 
            color='b', marker="o", linestyle = ':', label = "AE integration"
        )
        ax.axvline(
            x=En_cl, 
            color='red', 
            linestyle = '--', 
            label = "classical integration"
        )
        plt.xlabel("results of the integration in x = [-0.5,0.5]")
        plt.ylabel("probability measuring each of the results")
        ax.legend()
        plt.grid(True)
        # plt.xlim(0., 200.)
        plt.show()
        return
    

    def test_quantum_fourier(self):
        import matplotlib.pyplot as plt
        from scipy.fft import fft

        print("Testing Quantum Fourier Transform:... ", end='')
        path_test = "quantum-fourier"
        test_name = "fourier"
        oor = self.__launch_test(path_test, test_name)
        oor.read_qsvt()

        QSVT_rescaling = 0.98 # coefficient used for the calculation of the QSVT angles:
        mu = oor.dd_["gauss"]["mu"]

        alpha_0 = oor.constants_["alpha_0"]
        alpha_1 = oor.constants_["alpha"]
        x_center = alpha_0 + 1.0

        nx = int(oor.constants_["nx"])
        Nx = 1 << nx
        dx = (2. * alpha_1) / Nx
        x = np.zeros(Nx)
        for ix in range(Nx):
            x[ix] = alpha_0 + ix * dx

        # --- classical Fourier  of a Gaussian---
        coef_d = (1./np.sqrt(2.))**nx # take into account that there is the initialization circuit;
        gauss_signal = coef_d * QSVT_rescaling * np.exp(- (x - x_center)**2/(2*mu**2))
        fft_gaussian = fft(gauss_signal)
        fft_gaussian = fft_gaussian / np.sqrt(np.sum(np.abs(fft_gaussian)**2)) # normalization;
        fft_gaussian_rr = np.concatenate( (fft_gaussian[Nx//2:Nx], fft_gaussian[0:Nx//2]) ) 
        y_work_cl = np.abs(fft_gaussian_rr)**2

        # ---- Quantum Fourier Transform ---
        i_state = 0 # id of the initial state;
        qft_signal = mse.get_complex(oor.output_zero_anc_states_[i_state]["ampls"])
        qft_signal = qft_signal / np.sqrt(np.sum(np.abs(qft_signal)**2))
        qft_signal_rr = np.concatenate( (qft_signal[Nx//2:Nx], qft_signal[0:Nx//2]) )
        y_work_qc = np.abs(qft_signal_rr)**2

        # check resulting amplitudes:
        flag_equal_values = 1
        for ix in range(Nx):
            v_cl = y_work_cl[ix]
            v_qc = y_work_qc[ix]
            if np.abs(np.real(v_cl) - np.real(v_qc)) < self.__prec_:
                flag_equal_values *= 1
            else:
                flag_equal_values *= 0

        print("Max diff. between the classical and quantum Fourier: {:0.3e}".format(
            np.max(y_work_cl - y_work_qc))
        )

        # whether all conditions are satified?
        self.__cond_print([
            len(y_work_qc) == Nx,
            flag_equal_values
        ])

        # --- Plot the classical and quantum Fourier Transform ---
        dx = 1.0 # arbitrary
        dk = 2*np.pi / (dx*Nx)
        k_array_pos = np.linspace(0, 2*np.pi/(2*dx) - dk, Nx//2) 
        k_array_neg = np.linspace(-2*np.pi/(2*dx), -dk, Nx//2)
        k_array_rr = np.concatenate((k_array_neg, k_array_pos))

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(k_array_rr, y_work_cl, 
                color='b',  marker='s', linewidth = 2, linestyle=':', label = "classical fourier")
        ax.plot(k_array_rr, y_work_qc, 
                color='r',  marker='o', linewidth = 2, linestyle=':', label = "quantum fourier")
        plt.xlabel('frequency')
        plt.ylabel("prob. distribution")
        ax.legend()
        plt.grid(True)
        plt.show()
        return