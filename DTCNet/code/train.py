"""
DTCNet regression training — reproduction of Wang et al. (2025), Front. Comput. Neurosci.

Usage:
  python train.py --subject 1
  python train.py --subject all --seeds 5
"""

import os, argparse, json, random, time
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import DTCNet


# ======================================================================
# Config — paper hyperparameters
# ======================================================================
class C:
    data_root = "C:/Users/75060/WorkBuddy/data_dtcnet"
    output_dir = "./results"

    window  = 256       # @ 100Hz = 2.56s
    stride  = 1

    lr      = 8.42e-5  # paper: learning rate = 8.42e-5
    wd      = 1e-6      # paper: weight decay = 1e-6
    dropout = 0.1       # paper: dropout = 0.1
    bs      = 32        # reduced batch size to lower VRAM footprint
    epochs  = 100
    patience = 20        # train-loss early stop
    min_delta = 1e-5
    clip    = 1.0
    device  = "cuda" if torch.cuda.is_available() else "cpu"
    output_mode = "trajectory"  # 'trajectory' per-time-step (main) | 'single' end-point (ablation)


class DTCNetLoss(nn.Module):
    """Paper: MSE + CosineSimilarity, equal weights.
    Trajectory: cosine on time dim (dim=2); single: cosine on finger dim (dim=1)."""
    def __init__(self, output_mode='trajectory'):
        super().__init__()
        self.mse = nn.MSELoss()
        self.cos = nn.CosineSimilarity(dim=2 if output_mode == 'trajectory' else 1)

    def forward(self, p, t):
        return self.mse(p, t) + (1 - self.cos(p, t)).mean()


# ======================================================================
# Dataset — memory-mapped + downsampling
# ======================================================================
class SpecDataset(Dataset):
    """Sliding-window dataset over Morlet spectrograms, preprocessed @ 100Hz.

    Normalization: done in preprocess_raw.py (per-channel z-score + median removal),
            no further normalization here.
    """

    def __init__(self, path_spec, path_finger, w=256, s=1, i_start=0, i_end=None,
                 output_mode='trajectory'):
        spec_full = np.load(path_spec, mmap_mode='r')
        fing_full = np.load(path_finger, mmap_mode='r')

        T_in = spec_full.shape[2]
        self.s = spec_full[..., :T_in].copy()
        self.f = fing_full[:, :T_in].copy()
        self.output_mode = output_mode

        self.w, self.st = w, s
        T = self.s.shape[2]
        self.total_n = max(0, (T - w) // s + 1)
        self.i_start = i_start
        self.i_end   = min(i_end, self.total_n) if i_end is not None else self.total_n
        self.n = max(0, self.i_end - self.i_start)

    def __len__(self): return self.n
    def __getitem__(self, i):
        idx = self.i_start + i
        st = idx * self.st
        x = torch.as_tensor(self.s[..., st:st + self.w], dtype=torch.float32)
        if self.output_mode == 'single':
            y = torch.as_tensor(self.f[:, st + self.w - 1], dtype=torch.float32)  # (5,) end-point
        else:
            y = torch.as_tensor(self.f[:, st:st + self.w], dtype=torch.float32)   # (5, w) full trajectory
        return x, y


# ======================================================================
# Train / Eval
# ======================================================================
def run_epoch(m, dl, opt, crit, cfg):
    m.train()
    tot, n = 0.0, 0
    for x, y in dl:
        x, y = x.to(cfg.device), y.to(cfg.device)
        opt.zero_grad()
        L = crit(m(x), y)
        L.backward()
        nn.utils.clip_grad_norm_(m.parameters(), cfg.clip)
        opt.step()
        tot += L.item() * x.size(0)
        n += x.size(0)
    return tot / max(n, 1)


@torch.no_grad()
def _safe_pearson(pred, true):
    """Pearson r with zero-variance protection."""
    p_std = np.std(pred)
    t_std = np.std(true)
    if p_std < 1e-10 or t_std < 1e-10:
        return 0.0  # constant prediction or constant target
    c = np.corrcoef(pred, true)[0, 1]
    return float(c) if np.isfinite(c) else 0.0


def evaluate(m, ds_test, cfg, w=256, s=1):
    """Evaluate model on test set using manual numpy slicing (avoids DataLoader segfault)."""
    m.eval()
    s_data = ds_test.s   # (ch, freq, T)
    f_data = ds_test.f   # (5, T)
    T = s_data.shape[2]
    n_tot = (T - w) // s + 1

    t_inf_start = time.time()
    bs_eval = 16
    P, Tp = [], []
    n_chunks = 20
    chunk = max(1, n_tot // n_chunks)

    for start in range(0, n_tot, bs_eval):
        end = min(start + bs_eval, n_tot)
        xb = np.zeros((end - start, s_data.shape[0], s_data.shape[1], w), dtype=np.float32)
        yb = np.zeros((end - start, 5), dtype=np.float32)
        for j, idx in enumerate(range(start, end)):
            st = idx * s
            xb[j] = s_data[..., st:st + w]
            yb[j] = f_data[:, st + w - 1]
        with torch.no_grad():
            pred = m(torch.from_numpy(xb).to(cfg.device))
            if getattr(cfg, 'output_mode', 'trajectory') == 'trajectory':
                pred = pred[:, :, -1]  # take last time step for trajectory output (same eval protocol as single-point)
        P.append(pred.cpu().numpy())
        Tp.append(yb)
        if start % chunk == 0 and start > 0:
            print(f"  eval {start}/{n_tot}", flush=True)

    p = np.concatenate(P).astype(np.float64)
    t = np.concatenate(Tp).astype(np.float64)

    rr = [_safe_pearson(p[:, i], t[:, i]) for i in range(5)]

    # R² = 1 - SSE/SST (coefficient of determination, NOT r²)
    r2 = []
    for i in range(5):
        sse = np.sum((p[:, i] - t[:, i]) ** 2)
        sst = np.sum((t[:, i] - t[:, i].mean()) ** 2)
        r2.append(1.0 - sse / sst if sst > 1e-10 else 0.0)

    mae = [np.mean(np.abs(p[:, i] - t[:, i])) for i in range(5)]
    
    # Calibration: r² is the mathematical upper bound of R² under affine rescaling
    r_arr = np.array(rr); r2_arr = np.array(r2)
    calib_r2_per = [r_arr[i]**2 for i in range(5)]
    calib_gap_per = [calib_r2_per[i] - r2_arr[i] for i in range(5)]

    return {
        # per-finger r (table format)
        "thumb": rr[0], "index": rr[1], "middle": rr[2],
        "ring": rr[3], "little": rr[4],
        "avg_r": np.mean(rr),  # 5-finger mean
        "official_r": np.mean([rr[0], rr[1], rr[2], rr[4]]),  # exclude ring finger
        # per-finger R²
        "r2_thumb": r2[0], "r2_index": r2[1], "r2_middle": r2[2],
        "r2_ring": r2[3], "r2_little": r2[4],
        "r2_avg": np.mean(r2),
        "r2_official": np.mean([r2[0], r2[1], r2[2], r2[4]]),
        # calibration (r² = optimal affine-rescaled R²; gap = recoverable scale/bias)
        "calib_r2_thumb": calib_r2_per[0], "calib_r2_index": calib_r2_per[1],
        "calib_r2_middle": calib_r2_per[2], "calib_r2_ring": calib_r2_per[3],
        "calib_r2_little": calib_r2_per[4],
        "calib_r2_avg": np.mean(calib_r2_per),
        "calib_r2_official": np.mean([calib_r2_per[0], calib_r2_per[1], calib_r2_per[2], calib_r2_per[4]]),
        "gap_thumb": calib_gap_per[0], "gap_index": calib_gap_per[1],
        "gap_middle": calib_gap_per[2], "gap_ring": calib_gap_per[3],
        "gap_little": calib_gap_per[4],
        "gap_avg": np.mean(calib_gap_per),
        "gap_official": np.mean([calib_gap_per[0], calib_gap_per[1], calib_gap_per[2], calib_gap_per[4]]),
        # per-finger MAE
        "mae_thumb": mae[0], "mae_index": mae[1], "mae_middle": mae[2],
        "mae_ring": mae[3], "mae_little": mae[4],
        "mae_avg": np.mean(mae),
        "mae_official": np.mean([mae[0], mae[1], mae[2], mae[4]]),
        "infer_ms": (time.time() - t_inf_start) * 1000,
    }


# ======================================================================
# Data I/O
# ======================================================================
def load_subject(sid, cfg, val_ratio=0.01):
    base = os.path.join(cfg.data_root, f"sub{sid}")
    for tag in ["train_spec", "train_finger", "test_spec", "test_finger"]:
        if not os.path.exists(f"{base}_{tag}.npy"):
            raise FileNotFoundError(f"Missing file: {base}_{tag}.npy")

    om = getattr(cfg, 'output_mode', 'trajectory')

    ds_full = SpecDataset(f"{base}_train_spec.npy", f"{base}_train_finger.npy",
                          cfg.window, cfg.stride, output_mode=om)
    ds_test = SpecDataset(f"{base}_test_spec.npy", f"{base}_test_finger.npy",
                          cfg.window, cfg.stride, output_mode=om)

    # Validation: last 1% of training time, 256-step gap to avoid window overlap
    n_total = ds_full.n
    n_val = max(int(n_total * val_ratio), 1)
    gap = cfg.window  # 256 steps
    tr_end = n_total - n_val - gap
    val_start = n_total - n_val
    
    ds_train = SpecDataset(f"{base}_train_spec.npy", f"{base}_train_finger.npy",
                           cfg.window, cfg.stride, i_start=0, i_end=tr_end, output_mode=om)
    ds_val   = SpecDataset(f"{base}_train_spec.npy", f"{base}_train_finger.npy",
                           cfg.window, cfg.stride, i_start=val_start, i_end=n_total, output_mode=om)

    tr_ld = DataLoader(ds_train, batch_size=cfg.bs, shuffle=True, drop_last=True,
                       num_workers=0)
    val_ld = DataLoader(ds_val, batch_size=cfg.bs, shuffle=False, num_workers=0)

    x0, _ = ds_train[0]
    return tr_ld, val_ld, ds_test, x0.shape[0], x0.shape[1]


# ======================================================================
# Train one subject
# ======================================================================
def train_subject(sid, cfg, seed=42):
    set_seed(seed)
    tr_ld, val_ld, ds_test, n_ch, n_fr = load_subject(sid, cfg)

    om = getattr(cfg, 'output_mode', 'trajectory')
    m = DTCNet(n_channels=n_ch, n_freqs=n_fr, dropout=cfg.dropout, output_mode=om).to(cfg.device)
    opt = torch.optim.Adam(m.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    crit = DTCNetLoss(output_mode=om)

    print(f"Sub{sid}: {n_ch}ch x {n_fr}freq  |  {m.get_param_count():,} params  "
          f"|  train={len(tr_ld.sampler):,}  val={len(val_ld.sampler)}")

    ckpt_path = os.path.join(cfg.output_dir, f"sub{sid}_ckpt.pt")
    start_ep, best_val_loss, best_w, train_loss_hist = 0, float("inf"), None, []

    # Resume from checkpoint
    if os.path.exists(ckpt_path):
        try:
            ckpt = torch.load(ckpt_path, map_location=cfg.device, weights_only=False)
            if len(ckpt.get('loss_hist', [])) > 1:
                m.load_state_dict(ckpt['model'])
                opt.load_state_dict(ckpt['optimizer'])
                start_ep  = ckpt['epoch'] + 1
                best_val_loss = ckpt.get('best_val_loss', ckpt['best_loss'])
                best_w    = ckpt['best_weights']
                train_loss_hist = ckpt['loss_hist']
                print(f"  resumed from epoch {start_ep-1} (best val loss={best_val_loss:.6f})")
        except: pass
    t0 = time.time()

    patience_left = cfg.patience

    for ep in range(start_ep, cfg.epochs):
        train_loss = run_epoch(m, tr_ld, opt, crit, cfg)
        train_loss_hist.append(train_loss)
        
        # Train-loss based best model selection (val for monitoring only)
        if train_loss < best_val_loss - cfg.min_delta:
            best_val_loss = train_loss
            best_w = {k: v.cpu().clone() for k, v in m.state_dict().items()}
            patience_left = cfg.patience
        else:
            patience_left -= 1

        # Val loss (monitoring only, does not control early stop or model selection)
        m.eval()
        val_loss, val_n = 0.0, 0
        with torch.no_grad():
            for x, y in val_ld:
                x, y = x.to(cfg.device), y.to(cfg.device)
                val_loss += crit(m(x), y).item() * x.size(0)
                val_n += x.size(0)
        val_loss = val_loss / max(val_n, 1)

        if ep % 10 == 0 or patience_left <= 0:
            print(f"  ep {ep:3d}  tr={train_loss:.6f}  val={val_loss:.6f}  best_tr={best_val_loss:.6f}  patience={patience_left}")

        if (ep + 1) % 5 == 0:
            torch.save({'model': m.state_dict(), 'optimizer': opt.state_dict(),
                        'epoch': ep, 'best_val_loss': best_val_loss,
                        'best_weights': best_w, 'loss_hist': train_loss_hist}, ckpt_path)

        if patience_left <= 0:
            print(f"  early stop @ ep {ep} (train_loss={train_loss:.6f})")
            break

    m.load_state_dict(best_w)
    torch.save(m.state_dict(), os.path.join(cfg.output_dir, f"sub{sid}_model.pt"))
    np.save(os.path.join(cfg.output_dir, f"sub{sid}_loss.npy"), np.array(train_loss_hist))
    print(f"  model saved -> sub{sid}  best_val_loss={best_val_loss:.6f}", flush=True)

    r = evaluate(m, ds_test, cfg)
    elapsed = time.time() - t0

    print(f"  >> | Sub{sid} | T={r['thumb']:.3f} | I={r['index']:.3f} | M={r['middle']:.3f} | R={r['ring']:.3f} | L={r['little']:.3f} | Avg_r={r['avg_r']:.4f} | Official_r={r['official_r']:.4f} |")
    print(f"     | Sub{sid} | T={r['r2_thumb']:.3f} | I={r['r2_index']:.3f} | M={r['r2_middle']:.3f} | R={r['r2_ring']:.3f} | L={r['r2_little']:.3f} | Avg_R²={r['r2_avg']:.3f} | Official_R²={r['r2_official']:.3f} |")
    print(f"     | Sub{sid} | T={r['calib_r2_thumb']:.3f} | I={r['calib_r2_index']:.3f} | M={r['calib_r2_middle']:.3f} | R={r['calib_r2_ring']:.3f} | L={r['calib_r2_little']:.3f} | Avg_cR²={r['calib_r2_avg']:.3f} | Official_cR²={r['calib_r2_official']:.3f} |")
    print(f"     | Sub{sid} | T={r['gap_thumb']:.3f} | I={r['gap_index']:.3f} | M={r['gap_middle']:.3f} | R={r['gap_ring']:.3f} | L={r['gap_little']:.3f} | Avg_gap={r['gap_avg']:.3f} | Official_gap={r['gap_official']:.3f} |")
    print(f"     | Sub{sid} | T={r['mae_thumb']:.3f} | I={r['mae_index']:.3f} | M={r['mae_middle']:.3f} | R={r['mae_ring']:.3f} | L={r['mae_little']:.3f} | Avg_MAE={r['mae_avg']:.3f} | Official_MAE={r['mae_official']:.3f} |")
    print(f"     infer={r['infer_ms']:.0f}ms  [{elapsed/60:.1f} min]")

    try: os.remove(ckpt_path)
    except OSError: pass
    return r, best_val_loss


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)


# ======================================================================
# Main
# ======================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="1")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--data-root", default=C.data_root)
    ap.add_argument("--output-dir", default=C.output_dir)
    a = ap.parse_args()

    cfg = C()
    cfg.data_root  = a.data_root
    cfg.output_dir = a.output_dir
    os.makedirs(cfg.output_dir, exist_ok=True)
    print(f"PyTorch {torch.__version__}  device={cfg.device}  bs={cfg.bs}  lr={cfg.lr}")

    subs = [1, 2, 3] if a.subject == "all" else [int(a.subject)]

    all_r = {}
    for s in subs:
        print(f"\n{'='*60}\nSubject {s}\n{'='*60}")
        r, _ = train_subject(s, cfg, seed=42)

        all_r[f"sub{s}"] = {"results": r}
        with open(os.path.join(cfg.output_dir, "results.json"), "w") as f:
            json.dump(all_r, f, indent=2)

    print(f"\nAll saved -> {cfg.output_dir}/results.json")


if __name__ == "__main__":
    main()
