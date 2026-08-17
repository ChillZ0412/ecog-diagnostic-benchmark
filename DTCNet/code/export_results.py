"""Export benchmark results to FingerFlex-aligned tables (Markdown + CSV).

Reads results.json produced by run_direct.py and emits:
  - benchmark_r.md / benchmark_r.csv        (per-finger r, aligned with FingerFlex Table 3)
  - benchmark_r2.md / benchmark_r2.csv      (R²)
  - benchmark_calib.md / benchmark_calib.csv (calibrated R² = r²)
  - benchmark_gap.md / benchmark_gap.csv    (calibration gap = r² - R²)
  - benchmark_mae.md / benchmark_mae.csv    (MAE)

Usage:
  python export_results.py --input results_new3/results.json --out benchmark/
"""
import os, json, csv, argparse

FINGERS = ["Thumb", "Index", "Middle", "Ring", "Little"]
METRIC_GROUPS = {
    "r":       {"label": "Pearson r",      "per": ["thumb", "index", "middle", "ring", "little"],
                "avg": "avg_r", "official": "official_r"},
    "r2":      {"label": "R² (SSE/SST)",   "per": ["r2_thumb", "r2_index", "r2_middle", "r2_ring", "r2_little"],
                "avg": "r2_avg", "official": "r2_official"},
    "calib":   {"label": "Calibrated R² (=r²)", "per": ["calib_r2_thumb", "calib_r2_index", "calib_r2_middle",
                "calib_r2_ring", "calib_r2_little"], "avg": "calib_r2_avg", "official": "calib_r2_official"},
    "gap":     {"label": "Calibration gap (r²-R²)", "per": ["gap_thumb", "gap_index", "gap_middle",
                "gap_ring", "gap_little"], "avg": "gap_avg", "official": "gap_official"},
    "mae":     {"label": "MAE",            "per": ["mae_thumb", "mae_index", "mae_middle", "mae_ring", "mae_little"],
                "avg": "mae_avg", "official": "mae_official"},
}


def load_results(path):
    with open(path) as f:
        return json.load(f)


def build_table(results, group):
    """Build rows: [subject, per-finger..., avg, official] + a Mean row."""
    cfg = METRIC_GROUPS[group]
    header = ["Method", "Subject"] + FINGERS + [cfg["label"] + " (Avg_r)", cfg["label"] + " (Official)"]
    rows = [header]
    per_vals = {f: [] for f in FINGERS}
    avg_vals, off_vals = [], []
    for sub in ["sub1", "sub2", "sub3"]:
        if sub not in results:
            continue
        m = results[sub]["mean"]
        per = [m[k] for k in cfg["per"]]
        avg = m[cfg["avg"]]
        off = m[cfg["official"]]
        rows.append(["DTCNet", sub.upper()] + per + [avg, off])
        for i, f in enumerate(FINGERS):
            per_vals[f].append(per[i])
        avg_vals.append(avg); off_vals.append(off)
    # Mean row
    n = len(avg_vals)
    if n > 0:
        mean_per = [sum(per_vals[f]) / n for f in FINGERS]
        rows.append(["DTCNet", "Mean"] + mean_per + [sum(avg_vals) / n, sum(off_vals) / n])
    return rows


def fmt(v):
    return f"{v:.4f}"


def to_markdown(rows):
    lines = []
    lines.append("| " + " | ".join(rows[0]) + " |")
    lines.append("|" + "|".join(["---"] * len(rows[0])) + "|")
    for r in rows[1:]:
        cells = [r[0], r[1]] + [fmt(v) if isinstance(v, float) else str(v) for v in r[2:]]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def to_csv(rows):
    import io
    buf = io.StringIO()
    w = csv.writer(buf)
    for r in rows:
        w.writerow([fmt(v) if isinstance(v, float) else v for v in r])
    return buf.getvalue()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="results_new3/results.json")
    ap.add_argument("--out", default="benchmark")
    a = ap.parse_args()

    if not os.path.exists(a.input):
        print(f"[ERROR] {a.input} not found. Run run_direct.py first.")
        return

    results = load_results(a.input)
    os.makedirs(a.out, exist_ok=True)

    for group in METRIC_GROUPS:
        rows = build_table(results, group)
        md = to_markdown(rows)
        csv_txt = to_csv(rows)
        with open(os.path.join(a.out, f"benchmark_{group}.md"), "w", encoding="utf-8") as f:
            f.write(md + "\n")
        with open(os.path.join(a.out, f"benchmark_{group}.csv"), "w", encoding="utf-8", newline="") as f:
            f.write(csv_txt)
        print(f"  -> benchmark_{group}.md / .csv")

    # Print the primary r table to console
    print("\n" + to_markdown(build_table(results, "r")))


if __name__ == "__main__":
    main()
