"""
Stage 2 — Band decomposition (Liang & Bougrain 2012, Section 2.2.1.1).

The paper decomposes each raw ECoG channel into three band-specific signals
using equiripple FIR band-pass filters:
    sub        1 - 60  Hz
    gamma     60 - 100 Hz
    fastgamma 100 - 200 Hz

Implementation notes (important, and worth stating in the report):

1. ZERO-PHASE, SINGLE PASS.
   We use a linear-phase (Type I, odd-length, symmetric) FIR and convolve with
   `fftconvolve(..., mode="same")`. A symmetric FIR of length N has a constant
   group delay of (N-1)/2 samples; `mode="same"` trims exactly that amount, so
   the output is time-aligned with the input -- verified lag = 0 samples.
   This is preferable to `filtfilt`, which filters twice and therefore SQUARES
   the magnitude response (turning a -50 dB stopband into -100 dB and changing
   the effective passband shape). Since the AM feature is a power measure,
   distorting the magnitude response would directly bias the features.

   Caveat: this is non-causal (uses future samples). That is standard and
   acceptable for offline benchmark reproduction, but a real-time BCI would
   need `causal=True`, which applies the filter forward only and leaves a
   (N-1)/2 sample delay.

2. WINDOWED (firwin) RATHER THAN EQUIRIPPLE (remez) BY DEFAULT.
   The paper specifies equiripple filters. In practice `scipy.signal.remez`
   diverges for this filter bank:
     - at 2001+ taps it fails numerically for all three bands
       (passband ripple explodes to >100 dB)
     - at 1001 taps gamma/fastgamma are excellent (stopband -50/-85 dB) but
       the sub band is poor (stopband only -15 dB), because a 1 Hz lower edge
       at fs = 1000 Hz is an extremely narrow transition.
   `firwin` (Hamming-windowed) is unconditionally stable and, at 3301 taps,
   gives the sub band -59 dB DC rejection with 1.5 dB in-band ripple.
   We therefore default to firwin and expose `method="remez"` for comparison.
   This is a documented, deliberate deviation from the paper.

3. FILTER LENGTHS ARE PER BAND (config.FIR_NUMTAPS).
   The sub band needs a much longer filter than the others purely because of
   its 1 Hz lower edge. Different lengths do NOT cause misalignment: linear
   phase + mode="same" keeps every band time-aligned with the raw signal.

4. 60 Hz LINE NOISE.
   These are US recordings, so 60 Hz mains noise is expected -- and 60 Hz sits
   exactly on the sub/gamma boundary (partially suppressed by both transition
   bands), while its harmonics at 120 and 180 Hz land inside fastgamma.
   The paper does not notch. We default to APPLY_NOTCH_60HZ = False to stay
   faithful, and expose the notch as an ablation for the report.
"""
from typing import Iterator, Tuple

import numpy as np
from scipy import signal

import config as C


# ---------------------------------------------------------------------------
# Filter design
# ---------------------------------------------------------------------------
def design_bandpass(band_name: str,
                    fs: float = None,
                    numtaps: int = None,
                    method: str = None) -> np.ndarray:
    """
    Design a linear-phase FIR band-pass filter for one of the configured bands.

    Returns the tap vector (odd length, symmetric -> exactly linear phase).
    """
    fs = C.FS_ECOG if fs is None else fs
    method = C.FIR_METHOD if method is None else method
    numtaps = C.FIR_NUMTAPS[band_name] if numtaps is None else numtaps

    if numtaps % 2 == 0:            # force Type I (odd length) for exact
        numtaps += 1                # (N-1)/2 integer group delay

    lo, hi = C.BANDS[band_name]
    nyq = fs / 2.0

    if method == "firwin":
        taps = signal.firwin(numtaps, [lo, hi], pass_zero=False, fs=fs)

    elif method == "remez":
        lo_trans = max(min(lo * 0.5, 5.0), 0.5)
        hi_trans = max(min((nyq - hi) * 0.5, 10.0), 1.0)
        edges = [0.0, max(lo - lo_trans, 0.01), lo,
                 hi, min(hi + hi_trans, nyq - 0.01), nyq]
        taps = signal.remez(numtaps, edges, [0, 1, 0], fs=fs)

    else:
        raise ValueError(f"unknown FIR method: {method!r}")

    return taps.astype(C.FIR_DTYPE)


def describe_filter(band_name: str, taps: np.ndarray, fs: float = None) -> dict:
    """Measure the realised response: passband ripple, stopband, DC rejection."""
    fs = C.FS_ECOG if fs is None else fs
    lo, hi = C.BANDS[band_name]
    w, h = signal.freqz(taps, worN=16384, fs=fs)
    mag = np.abs(h)

    # evaluate ripple slightly inside the passband to ignore the transition
    inband = (w >= lo * 1.2) & (w <= hi * 0.95)
    stop = (w <= lo * 0.5) | (w >= min(hi * 1.5, fs / 2))

    def db(x):
        return 20.0 * np.log10(np.maximum(x, 1e-12))

    return {
        "band": band_name,
        "range_hz": (lo, hi),
        "numtaps": len(taps),
        "dc_db": float(db(mag[0])),
        "passband_ripple_db": float(db(mag[inband].max()) - db(mag[inband].min())),
        "stopband_db": float(db(mag[stop].max())) if stop.any() else float("nan"),
        "center_gain": float(mag[np.argmin(np.abs(w - (lo + hi) / 2))]),
    }


# ---------------------------------------------------------------------------
# Filter application
# ---------------------------------------------------------------------------
def apply_fir(data: np.ndarray, taps: np.ndarray, causal: bool = False) -> np.ndarray:
    """
    Apply a linear-phase FIR along axis 0 (time) of `data` (time, channels).

    causal=False (default): zero-phase, group delay removed via mode="same".
    causal=True:            forward-only; output retains (N-1)/2 sample delay.
    """
    data = np.asarray(data, dtype=C.FIR_DTYPE)
    taps = np.asarray(taps, dtype=C.FIR_DTYPE)
    if data.ndim == 1:
        data = data[:, None]

    if causal:
        return signal.lfilter(taps, 1.0, data, axis=0).astype(C.FIR_DTYPE)
    return signal.fftconvolve(data, taps[:, None], mode="same", axes=0)


def notch_line_noise(data: np.ndarray,
                     freqs=None,
                     q: float = None,
                     fs: float = None) -> np.ndarray:
    """Optional IIR notch at mains frequency + harmonics (OFF by default)."""
    fs = C.FS_ECOG if fs is None else fs
    freqs = C.NOTCH_FREQS if freqs is None else freqs
    q = C.NOTCH_Q if q is None else q

    out = np.asarray(data, dtype=np.float64)
    for f0 in freqs:
        if f0 >= fs / 2:
            continue
        b, a = signal.iirnotch(f0, q, fs=fs)
        out = signal.filtfilt(b, a, out, axis=0)
    return out.astype(C.FIR_DTYPE)


def decompose_bands(data: np.ndarray,
                    causal: bool = False,
                    apply_notch: bool = None,
                    method: str = None) -> Iterator[Tuple[str, np.ndarray]]:
    """
    Yield (band_name, filtered_array) for each configured band.

    This is a GENERATOR on purpose. Holding all three bands of a 400 s x 62 ch
    recording simultaneously costs ~300 MB in float32; Stage 3 consumes each
    band and immediately reduces it to 25 Hz AM features, so only one band
    needs to exist at a time.
    """
    apply_notch = C.APPLY_NOTCH_60HZ if apply_notch is None else apply_notch

    data = np.asarray(data, dtype=C.FIR_DTYPE)
    if apply_notch:
        data = notch_line_noise(data)

    for band_name in C.BANDS:
        taps = design_bandpass(band_name, method=method)
        yield band_name, apply_fir(data, taps, causal=causal)
