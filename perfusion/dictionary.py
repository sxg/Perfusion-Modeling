"""Dictionary-based perfusion modeling functions."""

import numpy as np
import scipy.io as sio
import perfusion.signal_contrast as sc
import perfusion.models as m

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
    dictionary = np.full((t, n_entries), np.nan)
    index = -1

    # Create the dictionary
    for i_af in range(0, n_af):
        for i_dv in range(0, n_dv):
            for i_mtt in range(0, n_mtt):
                for i_tau_a in range(0, n_tau_a):
                    for i_tau_p in range(0, n_tau_p):
                        index += 1
                        dictionary[:, index] = m.disc(times, art_contrast,\
                            pv_contrast, af_range[i_af], dv_range[i_dv],\
                            mtt_range[i_mtt], tau_a_range[i_tau_a],\
                            tau_p_range[i_tau_p])

    dictionary_data = {
        "dictionary": dictionary,
        "af_range": af_range,
        "dv_range": dv_range,
        "mtt_range": mtt_range,
        "tau_a_range": tau_a_range,
        "tau_p_range": tau_p_range,
        "acquisition_data": acquisition_data
    }

    sio.savemat(name, dictionary_data)
