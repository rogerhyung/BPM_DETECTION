"""
BPM Extractor Module
File: bpm_extractor_class3.py
Version: 1.1.2
Author: Junghyun Hyung [형정현]
Last Updated: 2025-06-02

Changelog:
- Added constructor parameters
- Added full pipeline support
- linted overall errors
- Added acf&fft estimation method choose parameter 
"""

import os
import numpy as np
import librosa
import scipy.signal
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from scipy.signal import fftconvolve, find_peaks


class BPMExtractor:
    def __init__(self, num_bands=8, cutoff=3.5, bpm_range=(60, 200)):
        self.num_bands = num_bands
        self.cutoff = cutoff
        self.bpm_range = bpm_range

    # 1. Audio Utilities
    def load_audio(self, filepath,duration=None):
        y, sr = librosa.load(filepath, sr=None, mono=False,duration=duration)
        y_mono = np.mean(y, axis=0) if y.ndim > 1 else y
        return y_mono, sr

    # 2. Filter Banks
    def apply_filterbank(self, signal, sr):
        max_freq = sr / 2 - 1
        band_edges = np.linspace(20, max_freq, self.num_bands + 1)
        ym_list = []
        for i in range(self.num_bands):
            low, high = band_edges[i], band_edges[i + 1]
            sos = scipy.signal.butter(
                N=4, Wn=[low, high], btype='bandpass', fs=sr, output='sos'
            )
            ym = scipy.signal.sosfilt(sos, signal)
            ym_list.append(ym)
        return ym_list, band_edges

    def hz_to_bark(self, f):
        return 13 * np.arctan(0.00076 * f) + 3.5 * np.arctan((f / 7500) ** 2)

    def bark_to_hz(self, z):
        return 650 * np.sinh(z / 7)

    def apply_bark_filterbank(self, signal, sr):
        max_freq = sr / 2 - 1
        bark_edges = np.linspace(
            self.hz_to_bark(20), self.hz_to_bark(max_freq), self.num_bands + 1
        )
        band_edges_hz = self.bark_to_hz(bark_edges)
        ym_list = []
        for i in range(self.num_bands):
            low, high = band_edges_hz[i], band_edges_hz[i + 1]
            sos = scipy.signal.butter(
                N=4, Wn=[low, high], btype='bandpass', fs=sr, output='sos'
            )
            ym = scipy.signal.sosfilt(sos, signal)
            ym_list.append(ym)
        return ym_list, band_edges_hz


    # 3. Envelope Extraction
    def extract_envelope(self, signal, sr, method='lowpass', cutoff=None):
        abs_signal = np.abs(signal)
        if method == 'lowpass':
            actual_cutoff = cutoff if cutoff is not None else self.cutoff
            b, a = scipy.signal.butter(
                N=2, Wn=actual_cutoff / (sr / 2), btype='low'
            )
            envelope = scipy.signal.filtfilt(b, a, abs_signal)
        elif method == 'moving_average':
            window_size = int(sr * 0.1)
            envelope = np.convolve(
                abs_signal, np.ones(window_size) / window_size, mode='same'
            )
        else:
            raise ValueError("Method must be 'lowpass' or 'moving_average'")
        return envelope

    def extract_envelopes(self, ym_list, sr, method='lowpass'):
        return [self.extract_envelope(ym, sr, method) for ym in ym_list]

    # 4. ACF Computation
    def compute_acf_fast(self, signal):
        signal -= np.mean(signal)
        acf = fftconvolve(signal, signal[::-1], mode='full')
        acf = acf[len(acf) // 2:]
        return acf / np.max(acf)

    def compute_all_acfs(self, envelope_list):
        return [self.compute_acf_fast(env) for env in envelope_list]

    # 5. BPM Estimation
    def fast_comb_filter_fft(self, R, sr, plot=False):
        min_bpm, max_bpm = self.bpm_range
        min_period = int(sr * 60 / max_bpm)
        max_period = int(sr * 60 / min_bpm)
        R_crop = R[min_period:max_period]
        length = len(R_crop)

        bpm_candidates = []
        bpm_scores = []

        for tau in range(min_period, max_period):
            comb = np.zeros(length)
            comb[::tau] = 1
            score = np.dot(R_crop[:len(comb)], comb)
            bpm = 60 * sr / tau
            bpm_candidates.append(bpm)
            bpm_scores.append(score)

        estimated_bpm = bpm_candidates[np.argmax(bpm_scores)]

        if plot:
            plt.figure(figsize=(10, 4))
            plt.plot(bpm_candidates, bpm_scores)
            plt.title("BPM Comb Filter Scores")
            plt.xlabel("BPM")
            plt.ylabel("Score")
            plt.grid(True)
            plt.tight_layout()
            plt.show()

        return estimated_bpm, bpm_candidates, bpm_scores

    def refine_bpm_with_harmonics(self, bpm_candidates, bpm_scores):
        max_idx = np.argmax(bpm_scores)
        max_bpm = bpm_candidates[max_idx]
        half_bpm = max_bpm / 2
        idx_close = np.argmin(np.abs(np.array(bpm_candidates) - half_bpm))
        if bpm_scores[idx_close] >= 0.85 * bpm_scores[max_idx]:
            return bpm_candidates[idx_close]
        return max_bpm

    def estimate_bpm_from_acf(self, acf, sr):
        peaks, _ = find_peaks(
            acf, height=0.05, distance=sr * 60 // self.bpm_range[1]
        )
        peak_diffs = np.diff(peaks)
        bpm_values = 60 * sr / peak_diffs
        bpm_values = bpm_values[
            (bpm_values >= self.bpm_range[0]) & (bpm_values <= self.bpm_range[1])
        ]
        bpm_counts = Counter(np.round(bpm_values))
        if not bpm_counts:
            return None
        return bpm_counts.most_common(1)[0][0]

    def plot_bpm_histogram(self, bpm_values):
        sns.histplot(bpm_values, bins=40, kde=True)
        plt.title("Estimated BPM from ACF Peak Distances")
        plt.xlabel("BPM")
        plt.ylabel("Count")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def run_pipeline(
        self, filepath, use_bark=False, method='lowpass',
        cutoff=None, bpm_range=None, plot=False, estimation_method='fft',duration=40
         ):
        """
        Full BPM estimation pipeline from audio file.

        Args:
            filepath (str): Path to audio file.
            use_bark (bool): Whether to use Bark-scale filterbank.
            method (str): Envelope extraction method ('lowpass' or 'moving_average').
            cutoff (float or None): Lowpass cutoff frequency (Hz); uses self.cutoff if None.
            bpm_range (tuple or None): Tuple of (min_bpm, max_bpm); uses self.bpm_range if None.
            plot (bool): Whether to display the comb-filter BPM score plot.
            estimation_method (str): 'fft' or 'acf' — which method to use for BPM detection.
            duration (int): How much part of the song from the start will be computed. 

        Returns:
            float: Refined BPM estimate.
        """
        y, sr = self.load_audio(filepath,duration=duration)
        filter_func = self.apply_bark_filterbank if use_bark else self.apply_filterbank
        ym_list, _ = filter_func(y, sr)

        actual_cutoff = cutoff if cutoff is not None else self.cutoff
        envelopes = [
            self.extract_envelope(ym, sr, method=method)
            if method != 'lowpass'
            else self.extract_envelope(ym, sr, method=method, cutoff=actual_cutoff)
            for ym in ym_list
        ]

        acfs = self.compute_all_acfs(envelopes)
        combined_acf = np.mean(acfs, axis=0)

        actual_bpm_range = bpm_range if bpm_range is not None else self.bpm_range
        self.bpm_range = actual_bpm_range  # temporary override

        if estimation_method == 'fft':
            bpm, bpm_candidates, bpm_scores = self.fast_comb_filter_fft(
                combined_acf, sr, plot=plot
            )
            refined_bpm = self.refine_bpm_with_harmonics(bpm_candidates, bpm_scores)
        elif estimation_method == 'acf':
            refined_bpm = self.estimate_bpm_from_acf(combined_acf, sr)
        else:
            raise ValueError("estimation_method must be 'fft' or 'acf'")

        return refined_bpm
