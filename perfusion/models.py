"""Perfusion models."""

import numpy as np
import scipy.linalg as sio
from perfusion import signal_contrast as sc

def disc(xdata, af, dv, mtt, tau_a, tau_p):
    """Calculates contrast concentration in liver tissue using the dual-input
     single-compartment model for describing liver perfusion.
    """

    times = xdata[:, 0]
    art_contrast = xdata[:, 1]
    pv_contrast = xdata[:, 2]

    k_1a = af * dv / mtt
    k_1p = (1 - af) * dv / mtt
    k_2 = 1 / mtt
    tau_a = np.rint(tau_a).astype(int)
    tau_p = np.rint(tau_p).astype(int)

    dt = times[1] - times[0]
    t = np.size(times, 0)
    contrast = np.zeros(times.size)

    if tau_a > 0:
        art_contrast[tau_a:] = art_contrast[:-tau_a]
    elif tau_a < 0:
        art_contrast[:-tau_a] = art_contrast[tau_a:]
    if tau_p > 0:
        pv_contrast[tau_p:] = pv_contrast[:-tau_p]
    elif tau_p < 0:
        pv_contrast[:-tau_p] = pv_contrast[tau_p:]

    f0 = (k_1a * art_contrast + k_1p * pv_contrast).transpose()
    f1 = np.tile(f0, (t, 1)) * np.tril(sio.toeplitz(np.exp(-k_2 * (times - times[0]))))
    contrast = dt * (f1.sum(1) - 0.5 * (f1[:, 0] + np.diag(f1)))

    return contrast

def disc_fit_wrapper(unfitted_params, fitted_params):
    """Wrapper for curve fitting to use the disc function.
    """

    af, dv, mtt, tau_a, tau_p = fitted_params
    tau_a *= 1000
    tau_p *= 1000
    acquisition_data = unfitted_params
    art_contrast = sc.art_signal_to_contrast(acquisition_data)
    pv_contrast = sc.pv_signal_to_contrast(acquisition_data)
    allts = np.asscalar(acquisition_data["allts"])
    times = allts[:, 0]
    return disc(times, art_contrast, pv_contrast, af, dv, mtt, tau_a, tau_p)