"""
Signal canary — run BEFORE building Stage 3+.

Everything up to here only proves the CODE is correct. This script asks a
different, more important question:

    Does the band-specific power in this data actually carry finger
    information at all?

If the answer is no, no amount of downstream engineering will produce
r = 0.46, and you want to know that now rather than at Stage 7.

Method (a deliberately crude preview of Stage 3):
  * take a slice of the training data
  * band-pass into the 3 bands
  * per channel, compute mean square power in non-overlapping 40 ms windows
    (this IS the AM feature, just without any of the Stage 4-6 machinery)
  * correlate each channel/band power trace against each finger's flexion,
    downsampled the same way
  * report the strongest channel/band per finger

Usage:
    python check_signal.py --subject 1
    python check_signal.py --subject 1 --seconds 100
    python check_signal.py --subject 1 --synthetic
"""
import argparse

import numpy as np

import config as C
from data_io import load_subject, make_synthetic
from filters import decompose_bands


def block_reduce_power(x: np.ndarray, win: int) -> np.ndarray:
    """Sum of squares in non-overlapping windows of `win` samples, along axis 0."""
    n = (x.shape[0] // win) * win
    x = x[:n]
    return (x.astype(np.float64) ** 2).reshape(-1, win, x.shape[1]).sum(axis=1)


def block_reduce_mean(x: np.ndarray, win: int) -> np.ndarray:
    """Mean value in non-overlapping windows (used for the glove target)."""
    n = (x.shape[0] // win) * win
    x = x[:n]
    return x[:n].astype(np.float64).reshape(-1, win, x.shape[1]).mean(axis=1)


def corr_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Pearson r between every column of A and every column of B -> (nA, nB)."""
    A = A - A.mean(0, keepdims=True)
    B = B - B.mean(0, keepdims=True)
    A_sd = np.linalg.norm(A, axis=0)
    B_sd = np.linalg.norm(B, axis=0)
    A_sd[A_sd == 0] = np.inf
    B_sd[B_sd == 0] = np.inf
    return (A.T @ B) / np.outer(A_sd, B_sd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", type=int, default=1, choices=[1, 2, 3])
    ap.add_argument("--seconds", type=int, default=200,
                    help="how much of the training data to use (default 200 s)")
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()

    sd = make_synthetic(args.subject) if args.synthetic else load_subject(args.subject)
    n = min(args.seconds * C.FS_ECOG, sd.train_ecog.shape[0])
    ecog = sd.train_ecog[:n]
    glove = sd.train_glove[:n]
    print(f"subject {sd.subject}: using {n / C.FS_ECOG:.0f} s, "
          f"{ecog.shape[1]} channels")

    win = C.AM_WINDOW_SAMPLES                    # 40 samples = 40 ms
    y = block_reduce_mean(glove, win)            # (T25, 5) targets at 25 Hz
    print(f"target after 40 ms reduction: {y.shape} (expect ~{n // win} x 5)")

    best = {f: (0.0, None) for f in range(C.N_FINGERS)}
    print(f"\n{'band':<11}{'best |r| per finger (channel)':<50}")
    print("-" * 62)

    for band_name, filtered in decompose_bands(ecog):
        X = block_reduce_power(filtered, win)    # (T25, n_ch) AM features
        # log-compress: power is heavy-tailed, correlation is dominated by
        # outliers otherwise. (Diagnostic only -- the paper uses raw power.)
        Xl = np.log1p(np.maximum(X, 0))
        R = corr_matrix(Xl, y)                   # (n_ch, 5)

        row = []
        for f in range(C.N_FINGERS):
            i = int(np.argmax(np.abs(R[:, f])))
            r = float(R[i, f])
            row.append(f"{C.FINGER_NAMES[f][:3]}:{r:+.2f}(ch{i})")
            if abs(r) > abs(best[f][0]):
                best[f] = (r, f"{band_name}/ch{i}")
        print(f"{band_name:<11}" + "  ".join(row))

    print("\nStrongest single feature per finger (across all bands):")
    strong = 0
    for f in range(C.N_FINGERS):
        r, where = best[f]
        flag = "" if abs(r) >= 0.15 else "   <-- weak"
        strong += abs(r) >= 0.15
        print(f"  {C.FINGER_NAMES[f]:<8} r = {r:+.3f}   from {where}{flag}")

    print("\nVERDICT:")
    if strong >= 3:
        print("  Band power clearly carries finger information. The data and the")
        print("  filter bank are healthy -- proceed to Stage 3.")
    elif strong >= 1:
        print("  Weak but non-zero coupling. Plausible for a SINGLE feature with no")
        print("  memory stack and no feature selection; Stages 4-6 are what turn")
        print("  this into r ~ 0.46. Proceed, but re-check here if Stage 7 is poor.")
    else:
        print("  No coupling found. Something is wrong upstream (wrong .mat keys,")
        print("  transposed arrays, or misaligned train/test split). Do NOT build")
        print("  Stage 3 on top of this -- report the output first.")


if __name__ == "__main__":
    main()
