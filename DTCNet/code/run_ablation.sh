#!/bin/bash
# Frequency-resolution ablation (4 groups) + output-layer ablation (1 group), 5 groups total, train 3 subjects in sequence.
# ~20h total. Outputs to results_ablation_*/ directories.
cd "C:/Users/75060/WorkBuddy/2026-07-20-12-42-08/dtcnet_regression"
PY="C:/Users/75060/miniconda3/python.exe"

echo "=== Frequency-resolution ablation 1/4: bandpass only (no Morlet, 1 freq) ==="
$PY -u run_direct.py --subject all --data-root C:/Users/75060/WorkBuddy/data_ablation/B1_bandpass --out results_ablation_B1

echo "=== Frequency-resolution ablation 2/4: Morlet 40 band (main experiment, skipped) ==="
echo "(40 band already in results_final/, no rerun needed)"

echo "=== Frequency-resolution ablation 3/4: Morlet 20 band ==="
$PY -u run_direct.py --subject all --data-root C:/Users/75060/WorkBuddy/data_ablation/B3_morlet20 --out results_ablation_B3

echo "=== Frequency-resolution ablation 4/4: Morlet 10 band ==="
$PY -u run_direct.py --subject all --data-root C:/Users/75060/WorkBuddy/data_ablation/B4_morlet10 --out results_ablation_B4

echo "=== Output-layer ablation: end-point single output (output_mode=single) ==="
$PY -u run_direct.py --subject all --output single --out results_ablation_single

echo "=== All ablations done ==="