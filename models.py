"""Perfusion models."""

import numpy as np

def disc(times, art_contrast, pv_contrast, af, dv, mtt, tau_a, tau_p):
    """Calculates contrast concentration in liver tissue using the dual-input 
    single-compartment model for describing liver perfusion.
    """

    k_1a = af * dv / mtt
    k_1p = (1 - af) * dv / mtt
    k_2 = 1 / mtt

    dt = times[1] - times[0]
    contrast = np.zeros(times.size)

    for i in range(0, contrast.size):
        sum = 0
        for t in range(0, i + 1):
            sum_a, sum_p = 0, 0
            t_a_delayed = np.around(t - tau_a)
            t_p_delayed = np.around(t - tau_p)
            if 0 <= t_a_delayed < art_contrast.size:
                sum_a = k_1a * art_contrast[t_a_delayed]
            if 0 <= t_p_delayed < pv_contrast.size:
                sum_p = k_1p * pv_contrast[t_p_delayed]
            sum += (sum_a + sum_p) * (np.exp(-k_2 * (i - t) * dt) * dt)
            # print(str(sum_a) + " " + str(sum_p) + " " + str(np.exp(-k_2 * (i - t) * dt) * dt))
            # print("Vars: " + str(k_2) + " " + str(i) + " " + str(t) + " " + str(dt))
        # print("Sum: " + str(sum))
        contrast[i] = sum
    
    return contrast