# iAAFT Surrogate Generation for Nonstationary Multichannel EEG

Python implementation of iterative Amplitude-Adjusted Fourier Transform (iAAFT) surrogate generation for multichannel, non-stationary EEG data augmentation, with changepoint-based segmentation and fixed-edge constraints.

Companion code for:
> **Demystifying Data Augmentation for Multivariate and Nonstationary Time Series: A Fourier Transform Approach**

---

## Overview

Surrogate methods generate synthetic signals that preserve specified statistical properties of the original — here, the amplitude spectrum and amplitude distribution — while randomising phase structure. This repository provides two flavours:

- **Univariate iAAFT** (`iAAFT.py`) — each channel is processed independently. Spectral and distributional properties are preserved per-channel, but cross-channel phase relationships are not.
- **Multivariate iAAFT** (`mv_iAAFT.py`) — a shared phase transformation derived from a reference channel seeds every other channel's iAAFT, preserving cross-channel phase structure within each segment.

Both pipelines support **changepoint-based segmentation**, splitting the signal into quasi-stationary segments before surrogate generation, and **fixed-edge constraints** that hold a configurable fraction of each segment boundary fixed to suppress discontinuities at joins.

## Repository Structure

```
├── iAAFT.py          # Univariate surrogate pipeline + shared segmentation utilities
├── mv_iAAFT.py       # Multivariate surrogate pipeline
├── main.ipynb        # Worked example: load EEG, generate surrogates, visualise
└── README.md
```
