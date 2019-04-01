"""Optimization tools to find perfusion properties for contrast enhancement
curves.
"""

import numpy as np
import scipy.optimize as sio
from perfusion import models as m
from perfusion import signal_contrast as sc

def fit_curve(curve, acquisition_data, x0):
    """Use non-linear least squares curve fitting to find perfusion properties
    to describe the given contrast enhancement curve.
    """

    bounds = ([0, 0, 0.0001, 0, 0], [1, 1, 100, 0.015, 0.007])
    art_contrast = sc.art_signal_to_contrast(acquisition_data)
    pv_contrast = sc.pv_signal_to_contrast(acquisition_data)
    allts = np.asscalar(acquisition_data["allts"])
    times = allts[:, 0]
    xdata = np.vstack((times, art_contrast, pv_contrast)).transpose()
    xdata = xdata.astype(dtype=np.float32)
    curve = curve.astype(dtype=np.float32)

    return sio.curve_fit(m.disc, xdata, curve, method='trf')