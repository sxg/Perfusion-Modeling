"""Dictionary-based perfusion modeling functions."""

import numpy as np
import scipy.io as sio
import signal_contrast as sc
import models as m

def create_dictionary(acquisition_data, af_range, dv_range, mtt_range, \
    tau_a_range, tau_p_range, name):
    """Creates perfusion dictionary."""

    # Get the times, AIF, and PVIF
    allts = np.asscalar(acquisition_data["allts"])
    times = allts[:, 0]
    art_contrast = sc.art_signal_to_contrast(acquisition_data)
    pv_contrast = sc.pv_signal_to_contrast(acquisition_data)

    # Size variables
    n_af = af_range.size
    n_dv = dv_range.size
    n_mtt = mtt_range.size
    n_tau_a = tau_a_range.size
    n_tau_p = tau_p_range.size
    t = times.size
    n_entries = n_af * n_dv * n_mtt * n_tau_a * n_tau_p

    # Setup variables for the rSVD
    rank = 20
    B = np.full((rank, n_entries), np.nan)
    W = np.zeros((t, rank))
    Z = np.zeros((t, t))
    omega = np.random.randn((n_entries, rank))
    rss = np.full((n_entries, 1), np.nan)
    lut = np.full((n_entries, 5), np.nan)
    lut_index = -1

    # Create the dictionary
    for i_af in range(0, n_af):
        for i_dv in range(0, n_dv):
            for i_mtt in range(0, n_mtt):
                for i_tau_a in range(0, n_tau_a):
                    for i_tau_p in range(0, n_tau_p):
                        lut_index += 1
                        lut[lut_index, :] = [i_af, i_dv, i_mtt, i_tau_a, i_tau_p]
                        d = m.disc(times, art_contrast, pv_contrast,\
                            af_range[i_af], dv_range[i_dv], mtt_range[i_mtt],\
                            tau_a_range[i_tau_a], tau_p_range[i_tau_p])
                        rss[lut_index] = np.sqrt(np.sum(np.square(d)))
                        d = np.linalg.norm(d - np.mean(d))
                        W += d @ omega[lut_index, :]
                        Z += d @ d.T

    # Delete NaN rows
    lut = lut[~np.isnan(lut).any(axis=1), :]
    rss = rss[~np.isnan(rss).any(axis=1), :]

    lut_index = -1
    Y = Z @ W
    Q = np.linalg.qr(Y, mode="reduced")

    for i_af in range(0, n_af):
        for i_dv in range(0, n_dv):
            for i_mtt in range(0, n_mtt):
                for i_tau_a in range(0, n_tau_a):
                    for i_tau_p in range(0, n_tau_p):
                        lut_index += 1
                        lut[lut_index, :] = [i_af, i_dv, i_mtt, i_tau_a, i_tau_p]
                        d = m.disc(times, art_contrast, pv_contrast,\
                            af_range[i_af], dv_range[i_dv], mtt_range[i_mtt],\
                            tau_a_range[i_tau_a], tau_p_range[i_tau_p])
                        rss[lut_index] = np.sqrt(np.sum(np.square(d)))
                        d = np.linalg.norm(d - np.mean(d))
                        B[:, lut_index] = Q.T * d

    B = B[:, ~np.isnan(B).any(axis=0)]
    V, S, U = np.linalg.svd(B.T, full_matrices=False)
    U @= Q
    Uk = U[:, 0:(rank + 1)]
    Sk = S[0:(rank + 1), 0:(rank + 1)]
    Vk = V[:, 0:(rank + 1)]

    dict_data = {
        "U": Uk,
        "S": Sk,
        "V": Vk,
        "rss": rss,
        "lut": lut,
        "af_range": af_range,
        "dv_range": dv_range,
        "mtt_range": mtt_range,
        "tau_a_range": tau_a_range,
        "tau_p_range": tau_p_range,
        "acquisition_data": acquisition_data
    }

    sio.savemat(name, dict_data)
