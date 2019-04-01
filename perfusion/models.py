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

def tofts(xdata, k_trans, k_ep):
    """Calculate contrast concentration using the standard Tofts model."""

    times = xdata[:, 0]
    art_contrast = xdata[:, 1]

    dt = times[1] - times[0]
    t = np.size(times, 0)
    contrast = np.zeros(times.size)

    for i, _ in enumerate(times):
        conv = 0
        for i_prime in range(0, i + 1):
            conv += k_trans * art_contrast[i_prime] \
                * np.exp(-(i - i_prime) * dt * k_ep) * dt
        contrast[i] = conv
    
    return contrast
