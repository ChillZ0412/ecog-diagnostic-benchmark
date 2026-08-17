"""Frequency-resolution ablation: Morlet frequency resolution.
Generates spectrogram variants for the frequency-resolution ablation.

B1: bandpass-only (no Morlet)  → (ch, 1, T)   time-domain baseline
B2: Morlet 40 band             → (ch, 40, T)  main experiment (already in data_dtcnet)
B3: Morlet 20 band             → (ch, 20, T)
B4: Morlet 10 band             → (ch, 10, T)

Finger data: ::10 decimation (main-experiment protocol), same across all variants.
"""
import numpy as np
import scipy.io as sio
import os
from scipy.signal import butter, filtfilt, iirnotch
from scipy.fft import fft, ifft


def bandpass(data, fs=1000, low=40, high=300, order=4):
    nyq = fs / 2
    b, a = butter(order, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, data, axis=1)


def notch(data, fs=1000, freq=60, q=30):
    b, a = iirnotch(freq, q, fs)
    return filtfilt(b, a, data, axis=1)


def per_channel_normalize(train, test):
    """Consistent with preprocess_raw.py: per-channel z-score + median removal (train statistics)."""
    mean = train.mean(axis=1, keepdims=True)
    std  = train.std(axis=1, keepdims=True) + 1e-8
    train_n = (train - mean) / std
    test_n  = (test  - mean) / std
    med = np.median(train_n, axis=1, keepdims=True)
    train_n = train_n - med
    test_n  = test_n  - med
    return train_n, test_n


def morlet_spec(data, fs=1000, n_freqs=40, f_min=40, f_max=300, n_cycles=7):
    freqs = np.logspace(np.log10(f_min), np.log10(f_max), n_freqs)
    T = data.shape[1]
    spec = np.zeros((data.shape[0], n_freqs, T), dtype=np.float32)
    t = np.arange(T, dtype=np.float64) / float(fs)
    for i, f in enumerate(freqs):
        sigma_t = float(n_cycles) / (2 * np.pi * float(f))
        wavelet = np.exp(2j * np.pi * float(f) * t) * np.exp(-t**2 / (2.0 * sigma_t**2))
        wf = fft(wavelet)
        for ch in range(data.shape[0]):
            spec[ch, i] = np.abs(ifft(fft(data[ch]) * wf)).astype(np.float32)
    return spec


def process_variant(sid, data_dir, out_dir, n_freqs, use_morlet=True):
    mat = sio.loadmat(os.path.join(data_dir, f'sub{sid}_comp.mat'))
    tr_e = mat['train_data'].astype(np.float64).T
    te_e = mat['test_data'].astype(np.float64).T

    # bad-channel exclusion (0-indexed): sub1=ch55, sub2=ch21+ch38, sub3=ch50
    bad_ch = {1: [54], 2: [20, 37], 3: [49]}
    if sid in bad_ch:
        tr_e = np.delete(tr_e, bad_ch[sid], axis=0)
        te_e = np.delete(te_e, bad_ch[sid], axis=0)

    # paper order: per-channel normalization → bandpass → notch → Morlet → downsample
    tr_e, te_e = per_channel_normalize(tr_e, te_e)

    for name, eco in [('train', tr_e), ('test', te_e)]:
        eco = bandpass(eco)
        eco = notch(eco)
        if use_morlet:
            eco = morlet_spec(eco, n_freqs=n_freqs)      # (ch, n_freqs, T)
        else:
            eco = eco[:, None, :]                        # (ch, 1, T) time-domain
        eco_ds = eco[:, :, ::10]                          # 1000→100Hz
        np.save(os.path.join(out_dir, f'sub{sid}_{name}_spec.npy'), eco_ds)

    # finger ::10 (same across all variants)
    tr_f = mat['train_dg'].astype(np.float64).T
    np.save(os.path.join(out_dir, f'sub{sid}_train_finger.npy'),
            (tr_f[:, ::10]).astype(np.float32))
    tmat = sio.loadmat(os.path.join(data_dir, f'ds4_true_labels/sub{sid}_testlabels.mat'))
    te_f = tmat['test_dg'].astype(np.float64).T
    np.save(os.path.join(out_dir, f'sub{sid}_test_finger.npy'),
            (te_f[:, ::10]).astype(np.float32))


def main():
    data_dir = 'C:/Users/75060/WorkBuddy/data_raw'

    variants = {
        'B1_bandpass': {'n_freqs': 1, 'use_morlet': False},
        'B3_morlet20': {'n_freqs': 20, 'use_morlet': True},
        'B4_morlet10': {'n_freqs': 10, 'use_morlet': True},
    }

    for vname, cfg in variants.items():
        out_dir = f'C:/Users/75060/WorkBuddy/data_ablation/{vname}'
        os.makedirs(out_dir, exist_ok=True)
        print(f'Generating {vname}...')
        for sid in [1, 2, 3]:
            process_variant(sid, data_dir, out_dir, cfg['n_freqs'], cfg['use_morlet'])
            print(f'  sub{sid} done')
        print(f'{vname} complete -> {out_dir}')


if __name__ == '__main__':
    main()
