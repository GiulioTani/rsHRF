import pywt
import numpy as np
from scipy.signal.windows import gaussian
from scipy.signal import convolve
import warnings


def rsHRF_iterative_wiener_deconv(
    y,
    h,
    TR=None,
    MaxIter=None,
    Tol=1e-4,
    Mode="rest",
    Smooth=None,
    LowPass=None,
    Iterations=None,
):
    """
    Iterative Wiener-like deconvolution with wavelet-based noise estimation.

    Implementation follows the MATLAB v2.5 logic by Guorong Wu (2025-09).
    """

    # ============ PARAMETER PARSING ============
    if Iterations is not None and MaxIter is None:
        warnings.warn(
            "Parameter 'Iterations' is deprecated. Use 'MaxIter' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        MaxIter = Iterations

    if MaxIter is None:
        MaxIter = 50

    # ============ PREPROCESSING ============
    # Mean-center to remove DC offset (MATLAB v2.5 line 28)
    y_mean = np.nanmean(y)
    y = y - y_mean

    y = y.flatten()
    h = h.flatten()

    N = y.shape[0]
    nh = h.shape[0]

    # Pad HRF to signal length
    if nh < N:
        h = np.pad(h, (0, N - nh), mode="constant", constant_values=0)
    elif nh > N:
        h = h[:N]

    # Calculate sampling rate
    if TR is not None:
        fs = 1.0 / TR
        nyquist = fs / 2.0
    else:
        fs = 1.0
        nyquist = 0.5

    # ============ AUTO-RECOMMENDATIONS ============
    if Smooth is None or LowPass is None:
        if Mode.lower() == "rest":
            if TR is not None:
                smooth_rec = max(int(np.round(4.0 / TR)), 3)
                lowpass_rec = min(0.2, 0.8 * nyquist)
            else:
                smooth_rec = 3
                lowpass_rec = 0.2
        elif Mode.lower() == "task":
            if TR is not None:
                smooth_rec = max(int(np.round(2.0 / TR)), 2)
                lowpass_rec = min(0.35, 0.9 * nyquist)
            else:
                smooth_rec = 2
                lowpass_rec = 0.35
        else:
            raise ValueError(f"Unknown Mode: {Mode}. Use 'rest' or 'task'.")

    if Smooth is None:
        Smooth = smooth_rec
    if LowPass is None:
        LowPass = lowpass_rec

    # ============ FFT PREPROCESSING ============
    H = np.fft.fft(h)
    Y = np.fft.fft(y)

    # Initial estimate
    xhat = y.copy()
    Pxx = np.abs(Y) ** 2

    # ============ INITIAL NOISE ESTIMATION ============
    coeffs = pywt.wavedec(y, "db2", level=1)
    detail_coeffs = coeffs[-1]
    # MAD-based noise estimation
    sigma = np.median(np.abs(detail_coeffs)) / 0.6745
    Nf = sigma**2 * N

    # ============ ITERATIVE PROCESS ============
    for iteration in range(MaxIter):
        M = (np.conj(H) * Pxx * Y) / (np.abs(H) ** 2 * Pxx + Nf)
        PxxY = (Pxx * Nf) / (np.abs(H) ** 2 * Pxx + Nf)
        Pxx_new = PxxY + np.abs(M) ** 2

        WienerFilterEst = (np.conj(H) * Pxx_new) / (np.abs(H) ** 2 * Pxx_new + Nf)
        xhat_new = np.real(np.fft.ifft(WienerFilterEst * Y))

        # ============ GAUSSIAN SMOOTHING ============
        if Smooth > 1:
            # MATLAB gausswin(N) uses alpha=2.5 by default.
            # Equivalent SciPy std = (N-1)/(2*alpha)
            std_val = (Smooth - 1) / (2 * 2.5)
            g = gaussian(int(Smooth), std=std_val)
            g = g / np.sum(g)
            xhat_new = convolve(xhat_new, g, mode="same")

        # ============ LOW-PASS FILTERING ============
        if LowPass < nyquist:
            f = np.arange(N) / N * fs
            Xf = np.fft.fft(xhat_new)
            # Kill frequencies above cutoff (Matches MATLAB's blunt approach)
            Xf[f > LowPass] = 0
            xhat_new = np.real(np.fft.ifft(Xf))

        # ============ DYNAMIC NOISE UPDATE ============
        residual = y - convolve(xhat_new, h, mode="same")

        coeffs = pywt.wavedec(residual, "db2", level=1)
        detail_coeffs = coeffs[-1]
        sigma = np.median(np.abs(detail_coeffs)) / 0.6745
        Nf = sigma**2 * N

        # ============ CONVERGENCE CHECK ============
        norm_diff = np.linalg.norm(xhat_new - xhat)
        norm_xhat = np.linalg.norm(xhat)

        if norm_xhat > 0 and (norm_diff / norm_xhat) < Tol:
            xhat = xhat_new
            break

        xhat = xhat_new
        Pxx = Pxx_new

    return xhat
