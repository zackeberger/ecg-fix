import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import butter, sosfiltfilt


FS = 500
TARGET_FS = 500


def ecg_is_dead(ecg: np.ndarray, fs: int = FS, window_sec: float = 2.0, tol: float = 1e-6, n_leads=12) -> bool:
    """
    Return True if any lead in the ECG has a flat segment of at least `window_sec` seconds.
    """
    if ecg.shape[0] == n_leads and ecg.shape[1] == 5000:
        ecg = ecg.T
    assert ecg.shape[1] == n_leads, "ECG must have shape (n_samples, n_leads)"
    assert ecg.shape[0] == 5000, "ECG must have 5000 samples"

    win = int(fs * window_sec)
    windows = sliding_window_view(ecg, window_shape=win, axis=0)
    windows = np.moveaxis(windows, 0, 1)
    w_range = np.ptp(windows, axis=2)
    return bool((w_range <= tol).any())


def _process_single_ecg(ecg, target_fs, n_leads):
    """
    ecg: (12, T) or (T, 12)
    returns: (12, T_new)
    """
    if ecg.shape[0] == n_leads:
        ecg = ecg.T

    T = ecg.shape[0]
    t_old = np.linspace(0, 10, T)
    t_new = np.linspace(0, 10, int(10 * target_fs))

    cols = [np.interp(t_new, t_old, ecg[:, i]) for i in range(n_leads)]
    ecg_interpolated = np.stack(cols, axis=1)
    ecg_normalized = (ecg_interpolated - np.mean(ecg_interpolated)) / (np.std(ecg_interpolated) + 1e-8)

    ecg_filtered = butter_bandpass_filter(
        ecg_normalized,
        lowcut=0.5,
        highcut=40.0,
        fs=target_fs,
        order=5,
    )
    return ecg_filtered.T


def process_ecg(ecg, target_fs=TARGET_FS, n_leads=12):
    """
    Supports:
    - (12, T)
    - (T, 12)
    - (B, 12, T)
    - (B, T, 12)
    """
    if ecg.ndim == 3:
        outputs = [
            _process_single_ecg(ecg[b], target_fs=target_fs, n_leads=n_leads)
            for b in range(ecg.shape[0])
        ]
        return np.stack(outputs, axis=0)

    return _process_single_ecg(ecg, target_fs=target_fs, n_leads=n_leads)


def butter_bandpass_filter(data, lowcut, highcut, fs, order=3):
    """
    Apply a Butterworth bandpass filter to a 2D signal.
    """
    sos = butter(order, [lowcut, highcut], fs=fs, btype="band", output="sos")
    return sosfiltfilt(sos, data, axis=0)


def resample_ecg_linear(ecg: np.ndarray, target_len: int = 5000) -> np.ndarray:
    """
    Linearly resample ECG to target length.
    ecg: (leads, samples)
    """
    leads, n = ecg.shape

    if n == target_len:
        return ecg

    x_old = np.linspace(0, 1, n, endpoint=False)
    x_new = np.linspace(0, 1, target_len, endpoint=False)

    ecg_rs = np.zeros((leads, target_len), dtype=ecg.dtype)
    for i in range(leads):
        ecg_rs[i] = np.interp(x_new, x_old, ecg[i])

    return ecg_rs


def crop_ecg(ecg: np.ndarray, target_len: int = 5000) -> np.ndarray:
    """
    Crop ECG to target length.
    ecg: (leads, samples)
    """
    _leads, n = ecg.shape

    if n <= target_len:
        return ecg

    start = (n - target_len) // 2
    return ecg[:, start:start + target_len]
