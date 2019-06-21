"""Interconverts signal to contrast."""

import numpy as np

def signal_to_contrast(signal, acquisition_data):
    """Convert signal intensity to contrast concentration."""
    alpha = np.asscalar(np.asscalar(acquisition_data["flipAngle"])) * np.pi / 180
    t_10_liver = np.asscalar(np.asscalar(acquisition_data["T10l"])) * 1000
    tr = np.asscalar(np.asscalar(acquisition_data["TR"]))
    relaxivity = np.asscalar(np.asscalar(acquisition_data["relaxivity"]))
    scale_factor = np.asscalar(np.asscalar(acquisition_data["scaleFactor"]))

    start_frame = np.asscalar(np.asscalar(acquisition_data["startFrame"])) - 1
    add_frames = np.asscalar(np.asscalar(acquisition_data["addFrames"]))
    stop_frame = start_frame + add_frames + 1
    m_0 = np.mean(signal[start_frame:stop_frame])

    contrast = _signal_to_contrast(signal, m_0, alpha, t_10_liver, tr, relaxivity)
    contrast *= scale_factor
    return contrast

def art_signal_to_contrast(acquisition_data):
    """Convert arterial signal intensity to contrast concentration."""
    allts = acquisition_data["allts"]
    art_signal = np.abs(allts[:, 1])
    pv_signal = np.abs(allts[:, 2])
    alpha = acquisition_data["flipAngle"] * np.pi / 180
    t_10_a = acquisition_data["T10b"] * 1000
    tr = acquisition_data["TR"]
    relaxivity = acquisition_data["relaxivity"]
    hematocrit = 0.4

    start_frame = int(acquisition_data["startFrame"]) - 1
    add_frames = int(acquisition_data["addFrames"])
    stop_frame = start_frame + add_frames + 1
    m_0 = np.mean(pv_signal[start_frame:stop_frame])

    art_contrast = _signal_to_contrast(art_signal, m_0, alpha, t_10_a, tr, relaxivity)
    art_contrast = art_contrast / (1 - hematocrit)
    return art_contrast

def pv_signal_to_contrast(acquisition_data):
    """Convert portal venous signal intensity to contrast concentration."""
    allts = acquisition_data["allts"]
    pv_signal = np.abs(allts[:, 2])
    alpha = acquisition_data["flipAngle"] * np.pi / 180
    t_10_pv = acquisition_data["T10p"] * 1000
    tr = acquisition_data["TR"]
    relaxivity = acquisition_data["relaxivity"]
    hematocrit = 0.4

    start_frame = int(acquisition_data["startFrame"]) - 1
    add_frames = int(acquisition_data["addFrames"])
    stop_frame = start_frame + add_frames + 1
    m_0 = np.mean(pv_signal[start_frame:stop_frame])

    pv_contrast = _signal_to_contrast(pv_signal, m_0, alpha, t_10_pv, tr, relaxivity)
    pv_contrast = pv_contrast / (1 - hematocrit)
    return pv_contrast

def _signal_to_contrast(signal, m_0, alpha, t_10, tr, relaxivity):
    r_10 = 1 / t_10
    m = m_0 \
        * (1 - np.exp(-r_10 * tr) * np.cos(alpha)) \
        / (1 - np.exp(-r_10 * tr)) / np.sin(alpha)
    r_1 = np.abs(np.log(np.divide((m * np.sin(alpha) \
        - (signal * np.cos(alpha))) \
        , (m * np.sin(alpha) - signal))) / tr)
    contrast = (r_1 - r_10) * 1000 / relaxivity
    return contrast
