"""25Hz downsampling verification — align with FingerFlex
DTCNet evaluates at 100Hz, FingerFlex at 25Hz. Verify whether the DTCNet 100Hz result,
after block-average downsampling to 25Hz, changes official_r by < 0.03 (teammate equivalence criterion).
"""
import torch, numpy as np, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from model import DTCNet
from train import C

cfg = C()
DATA = 'C:/Users/75060/WorkBuddy/data_dtcnet'

def pearson(a, b):
    a = a - a.mean(0); b = b - b.mean(0)
    denom = np.sqrt((a*a).sum(0) * (b*b).sum(0))
    return ((a*b).sum(0) / np.where(denom > 1e-10, denom, 1.0))

def block_avg(x, k=4):
    """100Hz -> 25Hz block average (mean over every k points)."""
    n = x.shape[0]
    n2 = (n // k) * k
    return x[:n2].reshape(n2 // k, k, x.shape[1]).mean(axis=1)

print("=" * 70)
print("25Hz downsampling verification (100Hz block_average -> 25Hz)")
print("=" * 70)

summary = []
for sid in [1, 2, 3]:
    spec = np.load(f'{DATA}/sub{sid}_test_spec.npy')      # (ch, 40, 20000) @100Hz
    finger = np.load(f'{DATA}/sub{sid}_test_finger.npy')  # (5, 20000) @100Hz
    n_ch = spec.shape[0]
    w = 256
    T = spec.shape[2]
    n_tot = (T - w) + 1  # stride=1

    # load model (best weights)
    m = DTCNet(n_channels=n_ch, n_freqs=40, dropout=0.1, output_mode='trajectory').to(cfg.device)
    m.load_state_dict(torch.load(f'results_final/sub{sid}_model.pt', map_location=cfg.device))
    m.eval()

    # sliding-window inference, take last time step -> 100Hz prediction sequence
    bs = 16
    P = []
    with torch.no_grad():
        for start in range(0, n_tot, bs):
            end = min(start + bs, n_tot)
            xb = np.zeros((end - start, n_ch, 40, w), dtype=np.float32)
            for j, idx in enumerate(range(start, end)):
                xb[j] = spec[..., idx:idx + w]
            pred = m(torch.from_numpy(xb).to(cfg.device))
            P.append(pred[:, :, -1].cpu().numpy())
    p_100 = np.concatenate(P)  # (n_tot, 5) @100Hz

    # true finger @100Hz (window-end aligned, consistent with evaluate)
    t_100 = finger[:, w - 1:w - 1 + n_tot].T  # (n_tot, 5)

    # block-average downsample to 25Hz
    p_25 = block_avg(p_100, 4)
    t_25 = block_avg(t_100, 4)

    r_100 = pearson(p_100, t_100)
    r_25 = pearson(p_25, t_25)

    # official = exclude Ring (index 3)
    off_100 = np.mean([r_100[0], r_100[1], r_100[2], r_100[4]])
    off_25 = np.mean([r_25[0], r_25[1], r_25[2], r_25[4]])
    diff = abs(off_100 - off_25)
    equiv = "OK equivalent" if diff < 0.03 else "X over threshold"

    print(f"sub{sid}: 100Hz official_r={off_100:.4f} | 25Hz official_r={off_25:.4f} | diff={diff:.4f} {equiv}")
    summary.append((sid, off_100, off_25, diff))

print("\n" + "=" * 70)
print("Summary")
print("=" * 70)
for sid, o100, o25, d in summary:
    print(f"sub{sid}: {o100:.4f} -> {o25:.4f} (Δ{d:.4f})")
print(f"\nMax diff: {max(s[3] for s in summary):.4f}")
print("Conclusion: " + ("OK all < 0.03, 100Hz equivalent to 25Hz, can align with FingerFlex directly" if all(s[3] < 0.03 for s in summary) else "X some over threshold, mind protocol difference"))
