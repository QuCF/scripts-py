import marimo

__generated_with = "0.1.69"
app = marimo.App(width="full")


@app.cell
def __():
    import marimo as mo
    import cvxpy as cp
    import numpy as np
    return cp, mo, np


@app.cell
def __(np):
    # -------------------------------------------------------------
    # --- Parameters for the estimation of the matrix inversion ---
    # -------------------------------------------------------------

    # chosen condition number:
    kappa_qsvt_ = 100

    # normalization coefficient:
    coef_norm_ = 4.

    # parameters for the reconstruction of the QSVT angles:
    coefs_ = np.array([
        4.357722e-01,  5.219540e-01,  6.798121e-02,
        -2.393112e-02, -2.410376e-03,  8.836340e-04 
    ])

    # the coefficient to compute the half of the positive peak of QSVT angles:
    coef_Na  = 112./20.

    # the coefficient to comput the maximum amplitude of the QSVT angles:
    coef_A  = 3.117e-03 * 40.
    return coef_A, coef_Na, coef_norm_, coefs_, kappa_qsvt_


@app.cell
def __(
    coef_alpha_Na,
    coef_beta_Na,
    coefs_,
    include_sign,
    kappa_qsvt_,
    np,
    reproduce_env,
):



    # ------------------------------------------------------------------------------
    # --- Estimate the QSVT angles ---

    # - the number of the QSVT angles -
    Na = int(coef_alpha_Na + coef_beta_Na * kappa_qsvt_)
    range_full = np.array(range(Na))

    Na_h = Na//2
    grid_xa = np.linspace(0.0, 1.0, Na_h)

    # - Absolute amplitudes of the QSVT angles -
    half_env = reproduce_env(grid_xa, coefs_)
    del grid_xa

    # - Include the sign of the QSVT angels -
    phis_half = include_sign(half_env)
    phis_appr = np.concatenate(( phis_half, np.flip(phis_half) ))

    # # - Rescale the QSVT angles -
    # max_ampl = coef_A / kappa_qsvt_
    # phis_appr *= max_ampl

    # # --- Compute the inverse function using the sequence of 2x2 rotations ---
    # # - Correct the angles to compute the inverse function -
    # phis_appr[0]  += np.pi/4.
    # phis_appr[-1] += np.pi/4.

    # # - the refence inverse function -
    # inv_qsvt, inv_ref, x_grid = compute_inverse_function(
    #     phis_appr, kappa_qsvt_, coef_norm_
    # )

    # # - Plot the inverse function -
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.plot(x_grid, inv_ref,  color="b", linewidth = 2, linestyle='-',  label = "ref")
    # ax.plot(x_grid, inv_qsvt, color="r", linewidth = 2, linestyle='--', label = "qsvt")
    # plt.xlabel('i')
    # plt.ylabel("phis")
    # # plt.xlim(-5, 5)
    # ax.legend()
    # plt.grid(True)
    # plt.show()

    # del inv_qsvt, inv_ref, x_grid
    return Na, Na_h, grid_xa, half_env, phis_appr, phis_half, range_full


@app.cell
def __(np, phis_appr):
    # ----------------------------------------------------------------
    # --- Store the QSVT angles to .hdf5 file ---
    # ----------------------------------------------------------------
    # The stored angles can be use to compute the matrix inversion
    # using the QSVT circuit.
    # ---
    # The condition number of the target matrix should be of the order of
    # the parameter kappa_qsvt_.
    # ----------------------------------------------------------------

    # - Correct the QSVT angles -
    phis_save = np.array(phis_appr)
    phis_save[0]  -= np.pi/4.
    phis_save[-1] -= np.pi/4.
    phis_save     += np.pi/2.

    # - Store the QSVT angles -
    return phis_save,


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
