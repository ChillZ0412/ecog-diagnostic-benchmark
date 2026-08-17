"""
DTCNet preprocessing — faithful reproduction of Wang et al. (2025) Section 2.1

Input: raw .mat files from BCI Competition IV Dataset 4
Steps: load → transpose → bandpass 40-300Hz → notch 60Hz → Morlet → downsample to 100Hz
Output: (ch, freq, T) spectrogram .npy
"""

import numpy as np
import scipy.io as sio
from scipy.signal import butter, filtfilt, iirnotch
from scipy.fft import rfft, irfft, fft, ifft
from scipy.interpolate import interp1d
import os, sys


def load_raw(subject_id, data_dir):
    fname = os.path.join(data_dir, f'sub{subject_id}_comp.mat')
    mat = sio.loadmat(fname)
    tr_e = mat['train_data'].astype(np.float64).T   # (T,ch) → (ch,T)
    te_e = mat['test_data'].astype(np.float64).T
    return tr_e, te_e


def bandpass_filter(data, fs=1000, low=40, high=300, order=4):
    nyq = fs / 2
    b, a = butter(order, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, data, axis=1)


def notch_filter(data, fs=1000, freq=60, q=30):
    b, a = iirnotch(freq, q, fs)
    return filtfilt(b, a, data, axis=1)


def per_channel_normalize(train, test):
    """Paper 2.2.2 Normalization: per-channel z-score + median removal.
    Applied on raw ECoG (ch, T); train statistics normalize both train and test (no test leakage)."""
    mean = train.mean(axis=1, keepdims=True)
    std  = train.std(axis=1, keepdims=True) + 1e-8
    train_n = (train - mean) / std
    test_n  = (test  - mean) / std
    med = np.median(train_n, axis=1, keepdims=True)
    train_n = train_n - med
    test_n  = test_n  - med
    return train_n, test_n


def morlet_spec(data, fs=1000, n_freqs=40, f_min=40, f_max=300, n_cycles=7):
    """Complex Morlet wavelet via FFT convolution"""
    freqs = np.logspace(np.log10(f_min), np.log10(f_max), n_freqs)
    T = data.shape[1]
    spec = np.zeros((data.shape[0], n_freqs, T), dtype=np.float32)
    t = np.arange(T, dtype=np.float64) / float(fs)
    for i, f in enumerate(freqs):
        sigma_t = float(n_cycles) / (2 * np.pi * float(f))
        wavelet = np.exp(2j * np.pi * float(f) * t) * np.exp(-t**2 / (2.0 * sigma_t**2))
        wf = fft(wavelet)
        for ch in range(data.shape[0]):
            conv = ifft(fft(data[ch]) * wf)
            spec[ch, i] = np.abs(conv).astype(np.float32)
    return spec


def process_subject(sid, data_dir, output_dir):
    print(f'Sub{sid}: loading...')
    tr_e, te_e = load_raw(sid, data_dir)

    # Exclude bad channels (amplitude anomaly >10x median, verified on raw data;
    # independently confirmed by Wiener/FingerFlex/SwitchingLM and arXiv:2412.06815)
    bad_ch = {1: [54], 2: [20, 37], 3: [49]}  # 0-indexed: sub1=ch55, sub2=ch21+ch38, sub3=ch50
    if sid in bad_ch:
        tr_e = np.delete(tr_e, bad_ch[sid], axis=0)
        te_e = np.delete(te_e, bad_ch[sid], axis=0)
        print(f'  excluded bad channels: {[c+1 for c in bad_ch[sid]]}')

    # paper order: Normalization (per-channel on raw ECoG) → bandpass → notch → Morlet → downsample
    tr_e, te_e = per_channel_normalize(tr_e, te_e)
    print('  per-channel normalization done')

    for name, eco in [('train', tr_e), ('test', te_e)]:
        print(f'  {name}: bp+notch...')
        eco = bandpass_filter(eco)
        eco = notch_filter(eco)
        print(f'  {name}: Morlet (n_cycles=7)...')
        eco = morlet_spec(eco)
        eco_ds = eco[:, :, ::10]    # 1000→100Hz
        np.save(os.path.join(output_dir, f'sub{sid}_{name}_spec.npy'), eco_ds)
    print(f'Sub{sid} specs done. (finger .npy files already regenerated)')


def main():
    data_dir = 'C:/Users/75060/WorkBuddy/data_raw'
    output_dir = 'C:/Users/75060/WorkBuddy/data_dtcnet'
    os.makedirs(output_dir, exist_ok=True)
    for s in [1, 2, 3]:
        process_subject(s, data_dir, output_dir)
    print(f'\nDone. Spec files in {output_dir}')


if __name__ == '__main__':
    main()
