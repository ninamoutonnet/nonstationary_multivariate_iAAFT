import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import spectrogram

def plot_signal_cp_analysis(signal, cp_list=None, fixed_edges=0.0, fs=1):
    """Plot a signal with changepoint/fixed-edge annotations and spectrogram.

    Parameters
    ----------
    signal : np.ndarray
        One-dimensional signal to visualize.
    cp_list : list[int] | None
        Changepoint sample indices.
    fixed_edges : float
        Fraction of each segment edge highlighted as fixed margins (0 to 1).
    fs : float
        Sampling frequency in Hz.
    """
    cp_list = [] if cp_list is None else cp_list

    fig = plt.figure(figsize=(20, 3))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.5, 1])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    title_fontsize = 16
    label_fontsize = 16
    ticks_fontsize = 12
    plt.rcParams['xtick.labelsize'] = ticks_fontsize
    plt.rcParams['ytick.labelsize'] = ticks_fontsize

    time = np.arange(len(signal)) / fs

    # Time-domain panel
    if fixed_edges == 0 or not cp_list:
        ax1.plot(time, signal, color='#377eb8', linewidth=1.5)
    else:
        segments = [0] + list(cp_list) + [len(signal)]
        for i in range(len(segments) - 1):
            start_idx = segments[i]
            end_idx = segments[i + 1]
            segment_length = end_idx - start_idx
            margin_samples = int(segment_length * fixed_edges)

            if margin_samples > 0:
                left_end = min(start_idx + margin_samples, end_idx)
                ax1.plot(
                    time[start_idx:left_end + 1],
                    signal[start_idx:left_end + 1],
                    color='red',
                    linewidth=1.5,
                )

                middle_start = start_idx + margin_samples
                middle_end = max(start_idx + margin_samples, end_idx - margin_samples)
                if middle_start < middle_end:
                    ax1.plot(
                        time[middle_start:middle_end],
                        signal[middle_start:middle_end],
                        color='#377eb8',
                        linewidth=1.5,
                    )

                right_start = max(start_idx, end_idx - margin_samples)
                ax1.plot(
                    time[right_start:end_idx],
                    signal[right_start:end_idx],
                    color='red',
                    linewidth=1.5,
                )
            else:
                ax1.plot(time[start_idx:end_idx], signal[start_idx:end_idx], color='#377eb8', linewidth=1.5)

    for cp in cp_list:
        ax1.axvline(x=cp / fs, color='black', linestyle='--', alpha=0.7, linewidth=1)

    ax1.set_xlabel('Time (s)', fontsize=label_fontsize)
    ax1.set_ylabel('Amplitude', fontsize=label_fontsize)
    ax1.set_title('Time-Domain Signal', fontsize=title_fontsize)
    ax1.set_xlim(0, time[-1])
    ax1.grid(True, alpha=0.3)

    for spine in ax1.spines.values():
        spine.set_color('darkgrey')
        spine.set_linewidth(1)

    # Spectrogram panel
    nperseg = 1024
    noverlap = 1023
    nfft = 1024
    f, t, sxx = spectrogram(signal, fs=fs, nperseg=nperseg, noverlap=noverlap, nfft=nfft)

    im = ax2.pcolormesh(t, f, 10 * np.log10(sxx), shading='gouraud', cmap='jet', vmin=-50, vmax=5)
    ax2.set_xlabel('Time (s)', fontsize=label_fontsize)
    ax2.set_ylabel('Frequency (Hz)', fontsize=label_fontsize)
    ax2.set_title('Spectrogram', fontsize=title_fontsize)
    ax2.set_ylim(0, 15)

    cbar = plt.colorbar(im, ax=ax2, shrink=0.8, pad=0.02)
    cbar.set_label('Power (dB)', fontsize=label_fontsize)
    cbar.ax.tick_params(labelsize=ticks_fontsize)

    for spine in ax2.spines.values():
        spine.set_color('darkgrey')
        spine.set_linewidth(1)

    plt.tight_layout()
    plt.show()



def plot_mv_signal_cp_analysis(
    eeg_original,
    eeg_surrogate,
    cp_list=None,
    margin_exact=0.0,
    fs=1,
    channel_names=None,
    title=None,
):
    """Plot original vs MV iAAFT surrogate side-by-side for every channel.

    Each row corresponds to one channel: the left panel shows the original
    signal and the right panel shows the surrogate.  Changepoints are marked
    with dashed vertical lines; fixed-edge margins are highlighted in red;
    the free region is plotted in blue — consistent with plot_signal_cp_analysis.

    Parameters
    ----------
    eeg_original : np.ndarray, shape (n_channels, n_samples)
    eeg_surrogate : np.ndarray, shape (n_channels, n_samples)
    cp_list : list[int] | None
        Changepoint sample indices (0 and n_samples are added automatically).
    margin_exact : float
        Fraction of each segment's length to highlight as fixed edges (0–1).
    fs : float
        Sampling frequency in Hz.
    channel_names : list[str] | None
        Labels for each channel row.  Defaults to 'Ch 0', 'Ch 1', …
    title : str | None
        Overall figure title.
    """
    eeg_original  = np.asarray(eeg_original,  dtype=float)
    eeg_surrogate = np.asarray(eeg_surrogate, dtype=float)

    if eeg_original.shape != eeg_surrogate.shape:
        raise ValueError("eeg_original and eeg_surrogate must have the same shape.")
    if eeg_original.ndim != 2:
        raise ValueError("Arrays must have shape (n_channels, n_samples).")

    n_channels, n_samples = eeg_original.shape
    cp_list = [] if cp_list is None else list(cp_list)
    channel_names = (
        channel_names if channel_names is not None
        else [f"Ch {i}" for i in range(n_channels)]
    )

    title_fontsize = 15
    label_fontsize = 13
    ticks_fontsize = 10
    plt.rcParams['xtick.labelsize'] = ticks_fontsize
    plt.rcParams['ytick.labelsize'] = ticks_fontsize

    fig, axes = plt.subplots(
        n_channels, 2,
        figsize=(20, 1.8 * n_channels),
        sharex=True,
        gridspec_kw=dict(hspace=0, wspace=0.06),
    )

    # Ensure axes is always 2-D even for a single channel
    if n_channels == 1:
        axes = axes[np.newaxis, :]

    if title:
        fig.suptitle(title, fontsize=title_fontsize + 2, y=0.9)

    # Column headers on the very first row
    axes[0, 0].set_title("Original",  fontsize=title_fontsize, pad=6)
    axes[0, 1].set_title("Surrogate", fontsize=title_fontsize, pad=6)

    time = np.arange(n_samples) / fs
    seg_boundaries = [0] + cp_list + [n_samples]

    def _draw_signal(ax, signal):
        """Paint one signal onto ax with margin colouring."""
        if margin_exact == 0 or not cp_list:
            ax.plot(time, signal, color='#377eb8', linewidth=1.0)
        else:
            for i in range(len(seg_boundaries) - 1):
                s_start = seg_boundaries[i]
                s_end   = seg_boundaries[i + 1]
                margin  = int((s_end - s_start) * margin_exact)

                if margin > 0:
                    left_end    = min(s_start + margin, s_end)
                    right_start = max(s_start, s_end - margin)
                    mid_start   = s_start + margin
                    mid_end     = max(mid_start, s_end - margin)

                    ax.plot(time[s_start:left_end + 1],
                            signal[s_start:left_end + 1],
                            color='red', linewidth=1.0)
                    if mid_start < mid_end:
                        ax.plot(time[mid_start:mid_end],
                                signal[mid_start:mid_end],
                                color='#377eb8', linewidth=1.0)
                    ax.plot(time[right_start:s_end],
                            signal[right_start:s_end],
                            color='red', linewidth=1.0)
                else:
                    ax.plot(time[s_start:s_end],
                            signal[s_start:s_end],
                            color='#377eb8', linewidth=1.0)

        for cp in cp_list:
            ax.axvline(x=cp / fs, color='black', linestyle='--',
                       alpha=0.7, linewidth=0.9)

    for ch in range(n_channels):
        for col, data in enumerate([eeg_original, eeg_surrogate]):
            ax = axes[ch, col]
            _draw_signal(ax, data[ch])

            ax.set_xlim(0, time[-1])
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='both', labelsize=ticks_fontsize)

            for spine in ax.spines.values():
                spine.set_color('darkgrey')
                spine.set_linewidth(0.8)
            if ch != 0:
                ax.spines['top'].set_visible(False)
            if ch != n_channels - 1:
                ax.spines['bottom'].set_visible(False)

            # x-axis ticks and label only on the bottom row
            if ch < n_channels - 1:
                plt.setp(ax.get_xticklabels(), visible=False)
                ax.tick_params(axis='x', length=0)
            else:
                ax.set_xlabel('Time (s)', fontsize=label_fontsize)

            # Channel label on the left column only
            if col == 0:
                ax.set_ylabel(channel_names[ch], fontsize=label_fontsize,
                              rotation=0, labelpad=40, va='center')

    plt.show()