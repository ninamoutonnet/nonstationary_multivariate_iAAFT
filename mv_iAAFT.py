"""Multivariate iAAFT surrogate generation for multichannel EEG data."""

from __future__ import annotations

import warnings
from typing import List, Literal, Sequence, Tuple, Union

import numpy as np
from scipy.fft import fft, ifft

from iAAFT import filter_changepoints_min_distance, segment_eeg_data

# Type alias for the fixed_edges_iterations parameter
FixedEdgesPolicy = Union[Literal["all", "first", "none"], int]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _should_apply_fixed_edges(n_iter: int, policy: FixedEdgesPolicy) -> bool:
    """Return True if edge constraints should be enforced on iteration n_iter."""
    if policy == "all":
        return True
    if policy == "none":
        return False
    if policy == "first":
        return n_iter == 0
    if isinstance(policy, int):
        return n_iter < policy
    raise ValueError("fixed_edges_iterations must be 'all', 'first', 'none', or an int.")


# ---------------------------------------------------------------------------
# Core algorithm
# ---------------------------------------------------------------------------

def _mv_iaaft_single(
    X: np.ndarray,
    margin: int = 0,
    max_it: int = 3000,
    error_threshold: float = 1e-6,
    fixed_edges_iterations: FixedEdgesPolicy = "all",
    reference_channel: int = 0,
) -> np.ndarray:
    """Generate one multivariate iAAFT surrogate.

    Cross-channel phase relationships are preserved by:
    1. Running a standard iAAFT on the reference channel.
    2. Computing the phase shift induced on that channel.
    3. Seeding every other channel's iAAFT with the same phase shift.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_channels)
    margin : int
        Number of edge samples to keep fixed at each end (default 0).
    max_it : int
        Maximum iterations per channel (default 3000).
    error_threshold : float
        Convergence criterion on the mean spectral error (default 1e-6).
    fixed_edges_iterations : 'all' | 'first' | 'none' | int
        When to enforce edge constraints during iteration (default 'all').
    reference_channel : int
        Channel used to derive the shared phase transformation (default 0).

    Returns
    -------
    surrogate : ndarray, shape (n_samples, n_channels)
    """
    N, n_channels = X.shape
    if not (0 <= reference_channel < n_channels):
        raise ValueError(
            f"reference_channel must be in [0, {n_channels - 1}], got {reference_channel}."
        )

    X_amp    = np.abs(fft(X, axis=0))   # (N, n_channels)
    X_sorted = np.sort(X, axis=0)       # (N, n_channels)
    X_start  = X[:margin,  :] if margin > 0 else None
    X_end    = X[-margin:, :] if margin > 0 else None

    r_all = np.empty_like(X)
    ref   = reference_channel

    # ----- Step 1: iAAFT for reference channel -----
    core = X[margin: N - margin, ref] if margin > 0 else X[:, ref]
    r_ref = (
        np.concatenate((X_start[:, ref], np.random.permutation(core), X_end[:, ref]))
        if margin > 0
        else np.random.permutation(core)
    )

    mse, mse_prev = 1000.0, 0.0
    for n_iter in range(max_it):
        if abs(mse - mse_prev) <= error_threshold:
            break
        mse_prev = mse

        s     = np.real(ifft(X_amp[:, ref] * np.exp(1j * np.angle(fft(r_ref)))))
        r_new = np.empty_like(s)
        r_new[np.argsort(s)] = X_sorted[:, ref]

        if margin > 0 and _should_apply_fixed_edges(n_iter, fixed_edges_iterations):
            r_ref = np.concatenate((X_start[:, ref], r_new[margin:-margin], X_end[:, ref]))
        else:
            r_ref = r_new

        mse = np.mean(np.abs(X_amp[:, ref] - np.abs(fft(r_ref))))

    r_all[:, ref] = r_ref

    # ----- Step 2: Phase transformation from reference -----
    phase_transform = np.angle(
        np.exp(1j * (np.angle(fft(r_ref)) - np.angle(fft(X[:, ref]))))
    )

    # ----- Step 3: iAAFT for every other channel, seeded with reference shift -----
    for ch in range(n_channels):
        if ch == ref:
            continue

        r_ch = np.real(ifft(X_amp[:, ch] * np.exp(1j * (np.angle(fft(X[:, ch])) + phase_transform))))
        if margin > 0:
            r_ch = np.concatenate((X_start[:, ch], r_ch[margin:-margin], X_end[:, ch]))

        mse_ch, mse_prev_ch = 1000.0, 0.0
        for n_iter in range(max_it):
            if abs(mse_ch - mse_prev_ch) <= error_threshold:
                break
            mse_prev_ch = mse_ch

            s_ch  = np.real(ifft(X_amp[:, ch] * np.exp(1j * np.angle(fft(r_ch)))))
            r_new = np.empty_like(s_ch)
            r_new[np.argsort(s_ch)] = X_sorted[:, ch]

            if margin > 0 and _should_apply_fixed_edges(n_iter, fixed_edges_iterations):
                r_ch = np.concatenate((X_start[:, ch], r_new[margin:-margin], X_end[:, ch]))
            else:
                r_ch = r_new

            mse_ch = np.mean(np.abs(X_amp[:, ch] - np.abs(fft(r_ch))))

        r_all[:, ch] = r_ch

    return r_all


# ---------------------------------------------------------------------------
# High-level pipeline
# ---------------------------------------------------------------------------

def generate_mv_surrogate(
    eeg_data: np.ndarray,
    fs: float,
    changepoints: Sequence[int],
    min_distance: int = 256,
    margin_exact: float = 0.05,
    fixed_edges_iterations: FixedEdgesPolicy = "all",
    reference_channel: int = 0,
    verbose: bool = False,
) -> Tuple[np.ndarray, List[int], dict]:
    """Generate a multivariate iAAFT surrogate for multichannel EEG.

    Cross-channel phase relationships are preserved within each segment via a
    shared phase transformation derived from the reference channel.

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
    fixed_edges_iterations : 'all' | 'first' | 'none' | int
        When to enforce edge constraints during iteration (default 'all').
    reference_channel : int
        Channel index used to derive the shared phase transformation (default 0).
    verbose : bool
        Print progress information (default False).

    Returns
    -------
    surrogate_eeg : ndarray, shape (n_channels, n_samples)
    filtered_changepoints : list of int
    segment_info : dict
        Keys: n_segments, segment_lengths, segment_indices, total_samples.
    """
    eeg_data = np.asarray(eeg_data, dtype=float)
    if eeg_data.ndim != 2:
        raise ValueError("eeg_data must have shape (n_channels, n_samples).")
    del fs  # retained in signature for API compatibility only

    filtered_cp = filter_changepoints_min_distance(changepoints, min_distance)
    segments, segment_indices = segment_eeg_data(eeg_data, filtered_cp)

    if verbose:
        lengths = [s.shape[1] for s in segments]
        print(f"MV surrogate | {eeg_data.shape[0]} channels, "
              f"{eeg_data.shape[1]} samples, {len(segments)} segments {lengths}")

    surrogate_segments = []
    for seg in segments:
        if seg.shape[1] < 50:          # segment too short — keep original
            surrogate_segments.append(seg)
            continue
        margin = int(np.floor(margin_exact * seg.shape[1]))
        try:
            surr = _mv_iaaft_single(
                seg.T,
                margin=margin,
                fixed_edges_iterations=fixed_edges_iterations,
                reference_channel=reference_channel,
            ).T  # back to (n_channels, segment_length)
            surrogate_segments.append(surr)
        except Exception as exc:
            warnings.warn(
                f"Surrogate generation failed for segment of length {seg.shape[1]}: {exc}. "
                "Falling back to original data."
            )
            surrogate_segments.append(seg)

    surrogate_eeg = np.concatenate(surrogate_segments, axis=1)

    segment_info = {
        "n_segments":      len(segments),
        "segment_lengths": [s.shape[1] for s in segments],
        "segment_indices": segment_indices,
        "total_samples":   surrogate_eeg.shape[1],
    }
    return surrogate_eeg, filtered_cp, segment_info