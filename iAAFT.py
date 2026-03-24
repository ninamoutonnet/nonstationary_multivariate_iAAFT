"""Univariate iAAFT surrogate generation for multichannel EEG data."""

from __future__ import annotations
from typing import List, Sequence, Tuple
import numpy as np
from scipy.fft import fft, ifft


def _iaaft_single(
    x: np.ndarray,
    margin: int = 0,
    max_it: int = 3000,
    error_threshold: float = 1e-6,
) -> np.ndarray:
    
    """
    Generate one iAAFT surrogate for a real-valued 1-D signal.

    Parameters
    ----------
    x : ndarray, shape (n_samples,)
    margin : int
        Number of edge samples to keep fixed (default 0).
    max_it : int
        Maximum number of iterations (default 3000).
    error_threshold : float
        Convergence criterion on the mean spectral error (default 1e-6).

    Returns
    -------
    surrogate : ndarray, shape (n_samples,)
    
    """
    if x.ndim != 1:
        raise ValueError("x must be one-dimensional.")
    if margin < 0 or 2 * margin >= len(x):
        raise ValueError("margin must satisfy 0 <= margin < len(x) / 2.")

    x_amp = np.abs(fft(x))
    x_sorted = np.sort(x)
    x_start = x[:margin]  if margin > 0 else np.array([], dtype=x.dtype)
    x_end = x[-margin:] if margin > 0 else np.array([], dtype=x.dtype)

    core = x[margin: len(x) - margin] if margin > 0 else x
    r = np.random.permutation(core)
    if margin > 0:
        r = np.concatenate((x_start, r, x_end))

    mse, mse_prev = 1000.0, 0.0
    for _ in range(max_it):
        if abs(mse - mse_prev) <= error_threshold:
            break
        mse_prev = mse

        s     = np.real(ifft(x_amp * np.exp(1j * np.angle(fft(r)))))
        r_new = np.empty_like(s)
        r_new[np.argsort(s)] = x_sorted

        if margin > 0:
            r = np.concatenate((x_start, r_new[margin:-margin], x_end))
        else:
            r = r_new

        mse = np.mean(np.abs(x_amp - np.abs(fft(r))))

    return r


def iAAFT(x: np.ndarray) -> np.ndarray:
    """Return one iAAFT surrogate for a 1-D signal."""
    return _iaaft_single(np.asarray(x, dtype=float))


def iAAFT_keep_edges(
    x: np.ndarray,
    margin_exact: float = 0.05,
) -> Tuple[np.ndarray, int]:
    """
    Return one iAAFT surrogate with fixed edges.

    Parameters
    ----------
    x : ndarray, shape (n_samples,)
    margin_exact : float
        Fraction of signal length to keep fixed at each edge (default 0.05).

    Returns
    -------
    surrogate : ndarray, shape (n_samples,)
    margin : int
        Actual number of samples kept fixed.  
    """
    x = np.asarray(x, dtype=float)
    margin = int(np.floor(margin_exact * len(x)))
    return _iaaft_single(x, margin=margin), margin

def filter_changepoints_min_distance(
    changepoints: Sequence[int],
    min_distance: int = 256,
) -> List[int]:
    """
    Filter changepoints to ensure a minimum gap between consecutive ones.

    Parameters
    ----------
    changepoints : sequence of int
    min_distance : int
        Minimum distance in samples between retained changepoints (default 256).

    Returns
    -------
    filtered : list of int
    """
    if min_distance < 0:
        raise ValueError("min_distance must be non-negative.")
    if not changepoints:
        return []
    sorted_cp = sorted(int(cp) for cp in changepoints)
    filtered = [sorted_cp[0]]
    for cp in sorted_cp[1:]:
        if cp - filtered[-1] >= min_distance:
            filtered.append(cp)
    return filtered


def segment_eeg_data(
    eeg_data: np.ndarray,
    changepoints: Sequence[int],
) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
    """
    Segment (n_channels, n_samples) EEG data at the given changepoints.

    Parameters
    ----------
    eeg_data : ndarray, shape (n_channels, n_samples)
    changepoints : sequence of int
        Sample indices at which to split; 0 and n_samples are added automatically.

    Returns
    -------
    segments : list of ndarray, each (n_channels, segment_length)
    segment_indices : list of (start, end) tuples
    """
    if eeg_data.ndim != 2:
        raise ValueError("eeg_data must have shape (n_channels, n_samples).")
    _, n_samples = eeg_data.shape

    boundaries = sorted({0, *map(int, changepoints), n_samples})
    segments, indices = [], []
    for start, end in zip(boundaries, boundaries[1:]):
        if 0 <= start < end <= n_samples:
            segments.append(eeg_data[:, start:end])
            indices.append((start, end))
    return segments, indices

def generate_uv_surrogate(
    eeg_data: np.ndarray,
    fs: float,
    changepoints: Sequence[int],
    min_distance: int = 256,
    margin_exact: float = 0.05,
    verbose: bool = False,
) -> Tuple[np.ndarray, List[int]]:
    """
    Generate a univariate iAAFT surrogate for multichannel EEG.

    Each channel in each segment is processed independently.  Cross-channel
    phase relationships are *not* preserved; see ``mv_iAAFT.generate_mv_surrogate``
    for the multivariate alternative.

    Parameters
    ----------
    eeg_data : ndarray, shape (n_channels, n_samples)
    fs : float
        Sampling frequency in Hz (unused; kept for API consistency).
    changepoints : sequence of int
        Sample indices at which to split the signal.
    min_distance : int
        Minimum gap between retained changepoints (default 256).
    margin_exact : float
        Fraction of each segment to keep fixed at the edges (default 0.05).
    verbose : bool
        Print progress information (default False).

    Returns
    -------
    surrogate_eeg : ndarray, shape (n_channels, n_samples)
    filtered_changepoints : list of int
    """
    eeg_data = np.asarray(eeg_data, dtype=float)
    if eeg_data.ndim != 2:
        raise ValueError("eeg_data must have shape (n_channels, n_samples).")
    del fs 

    filtered_cp = filter_changepoints_min_distance(changepoints, min_distance)
    segments, _ = segment_eeg_data(eeg_data, filtered_cp)

    if verbose:
        lengths = [s.shape[1] for s in segments]
        print(f"UV surrogate | {eeg_data.shape[0]} channels, "
              f"{eeg_data.shape[1]} samples, {len(segments)} segments {lengths}")

    surrogate_segments = []
    for seg in segments:
        if seg.shape[1] < 50: # segment too short — keep original
            surrogate_segments.append(seg)
            continue
        surr = np.stack([iAAFT_keep_edges(ch, margin_exact=margin_exact)[0] for ch in seg])
        surrogate_segments.append(surr)

    return np.concatenate(surrogate_segments, axis=1), filtered_cp