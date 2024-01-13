import marimo

__generated_with = "0.1.75"
app = marimo.App()


@app.cell
def __():
    import marimo as mo
    return mo,


@app.cell
def __():
    # ---------------------------------------------------------
    # --- Estimate the QSVT angles for the matrix inversion ---
    # ---------------------------------------------------------

    # chosen condition number:
    kappa_qsvt_ = 100

    # --- Estimate the first half of the QSVT angles


    # --- Construct the whole sequence of the QSVT angles ---

    # --- Correct the angles to compute the inverse function ---

    # - the refence inverse function -

    # --- Plot the inverse function ---

    # --- Correct the QSVT angles to use them in the QSVT circuit ---

    # --- Store the QSVT a
    return kappa_qsvt_,


@app.cell
def __():
    # ----------------------------------------------------------------
    # --- Store the QSVT angles to .hdf5 file ---
    # ----------------------------------------------------------------
    # The stored angles can be use to compute the matrix inversion
    # using the QSVT circuit.
    # ---
    # The condition number of the target matrix should be of the order of
    # the parameter kappa_qsvt_.
    # ----------------------------------------------------------------

    # 
    return


if __name__ == "__main__":
    app.run()
