"""DTCNet training launcher — direct Python, no PowerShell buffering issues.

Usage:
  python run_direct.py              # all 3 subjects, trajectory output (main experiment)
  python run_direct.py --subject 1  # single subject
  python run_direct.py --output single  # single-point output (output-layer ablation)
  python run_direct.py --data-root C:/path/to/data_ablation/B3_morlet20  # frequency-resolution ablation
"""
import sys, os, json
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from train import C, train_subject, set_seed

cfg = C()
cfg.output_dir = 'results_final'
os.makedirs(cfg.output_dir, exist_ok=True)

# parse --output (trajectory / single)
if '--output' in sys.argv:
    idx = sys.argv.index('--output')
    cfg.output_mode = sys.argv[idx + 1]

# parse --data-root (for frequency-resolution ablation)
if '--data-root' in sys.argv:
    idx = sys.argv.index('--data-root')
    cfg.data_root = sys.argv[idx + 1]

# parse --out (results directory)
if '--out' in sys.argv:
    idx = sys.argv.index('--out')
    cfg.output_dir = sys.argv[idx + 1]
    os.makedirs(cfg.output_dir, exist_ok=True)

# redirect stdout to log file (line-buffered, UTF-8)
log_path = os.path.join(cfg.output_dir, 'train_log.txt')
log_fh = open(log_path, 'a', encoding='utf-8', buffering=1)

class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj); f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

sys.stdout = Tee(sys.stdout, log_fh)

# parse --subject
subs = [1, 2, 3]
if '--subject' in sys.argv:
    idx = sys.argv.index('--subject')
    arg = sys.argv[idx + 1]
    subs = [1, 2, 3] if arg == 'all' else [int(arg)]

print(f"Config: output_mode={cfg.output_mode}  data_root={cfg.data_root}  out={cfg.output_dir}")

# load existing results (resume-safe)
rf = os.path.join(cfg.output_dir, 'results.json')
all_r = {}
if os.path.exists(rf):
    with open(rf) as f:
        all_r = json.load(f)

for s in subs:
    if f"sub{s}" in all_r:
        print(f"Subject {s} already done, skipping.")
        continue

    print(f"\n{'='*60}\nSubject {s}\n{'='*60}")
    try:
        r, val_loss = train_subject(s, cfg, seed=42)
        all_r[f"sub{s}"] = {"mean": r, "std": {}, "n": 1, "best_val_loss": val_loss}
        with open(rf, "w") as f:
            json.dump(all_r, f, indent=2)
        print(f"  [SAVED] sub{s}")
    except Exception as e:
        print(f"  [ERROR] sub{s}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        torch.cuda.empty_cache()  # GPU cleanup between subjects

print(f"\nDone. Subjects completed: {[k for k in all_r]}")
sys.stdout.files = [f for f in sys.stdout.files if f != log_fh]
log_fh.close()
