# Toy Bimodal Composition Example

This directory contains a self-contained reproduction of the toy bimodal composition example used in the paper to illustrate the mode-averaging behavior of compositional diffusion samplers and the fix introduced by RCD (Figure 1 of the main paper).

The setup uses three local diffusion models trained on overlapping length-3 chunks of trajectories drawn from a bimodal distribution with modes at $+1$ and $-1$, and composes them at inference time over horizons $L \in \{4, \ldots, 20\}$.

---

## Contents

```
toy_example/
├── README.md
├── train.py        # Train the three local diffusion models
├── inference.py    # Run CompDiffuser and RCD samplers
├── plot.py         # Helpers shared by make_figure.py
└── make_figure.py  # Assemble the standalone panels for the paper figure
```

After training and inference, two extra directories are created:

```
toy_example/
├── checkpoints/    # model_start.pth, model_bridge.pth, model_end.pth
└── results/        # L{H}_{method}.npy and L{H}_{method}.json
```

---

## Reproduction Workflow

All commands are run from this `toy_example/` directory.

### 1. Train the three local diffusion models

```bash
python train.py
```

This trains the start / bridge / end models on length-3 segments and saves `model_start.pth`, `model_bridge.pth`, `model_end.pth` under `checkpoints/`. Training takes only a few minutes per model on a single GPU.

### 2. Run inference for CompDiffuser and RCD at each horizon

For each horizon `H in {4, 5, 6, ..., 20}` and each method `M in {base, rcd}`, run:

```bash
python inference.py --method $M --horizon $H --save-dir results
```

For RCD specifically, we use the same hyperparameters as in the paper (probe ratio 0.35, 4 Monte-Carlo reconstruction samples, intermittent rate 1) and a horizon-dependent guidance scale: `0.5` for `H=4` (the baseline case where mode-averaging is not yet an issue and a gentle guidance suffices) and `5.0` for `H>=5`:

```bash
# Default (H >= 5)
python inference.py --method rcd --horizon $H --save-dir results \
    --rcd-guidance-scale 5.0 \
    --rcd-probe-ratio 0.35 \
    --rcd-n-mc-samples 4 \
    --rcd-overlap-weight 1.0 \
    --rcd-recon-weight 1.0 \
    --rcd-inter-rate 1

# Special case: H = 4 (use a smaller guidance scale to avoid over-guidance)
python inference.py --method rcd --horizon 4 --save-dir results \
    --rcd-guidance-scale 0.5 \
    --rcd-probe-ratio 0.35 \
    --rcd-n-mc-samples 4 \
    --rcd-overlap-weight 1.0 \
    --rcd-recon-weight 1.0 \
    --rcd-inter-rate 1
```

Each invocation writes `L{H}_{method}.npy` (samples) and `L{H}_{method}.json` (success rate and other statistics) into the chosen `--save-dir`.

### 3. Build the paper figure

```bash
python make_figure.py --results-dir results
```

This produces `standalone_panels/`, one PDF/PNG per panel (training segments, CompDiffuser samples, RCD samples, valid-rate vs.\ horizon).

---

## Reproducing the Component Ablation in the Appendix

The appendix also contains a three-panel ablation that isolates the two RCD guidance components (overlap-only, reconstruction-only, both). To reproduce, run inference at horizon `L=10` with three different `(recon_weight, overlap_weight)` combinations and tag the outputs:

```bash
# Overlap only
python inference.py --method rcd --horizon 10 --save-dir results_component_ablation \
    --method-tag overlap_only \
    --rcd-guidance-scale 5.0 --rcd-probe-ratio 0.35 --rcd-n-mc-samples 4 --rcd-inter-rate 1 \
    --rcd-recon-weight 0.0 --rcd-overlap-weight 1.0

# Reconstruction only
python inference.py --method rcd --horizon 10 --save-dir results_component_ablation \
    --method-tag recon_only \
    --rcd-guidance-scale 5.0 --rcd-probe-ratio 0.35 --rcd-n-mc-samples 4 --rcd-inter-rate 1 \
    --rcd-recon-weight 1.0 --rcd-overlap-weight 0.0

# Both (full RCD)
python inference.py --method rcd --horizon 10 --save-dir results_component_ablation \
    --method-tag both \
    --rcd-guidance-scale 5.0 --rcd-probe-ratio 0.35 --rcd-n-mc-samples 4 --rcd-inter-rate 1 \
    --rcd-recon-weight 1.0 --rcd-overlap-weight 1.0
```

Each call writes `L10_rcd_{tag}.npy` and `L10_rcd_{tag}.json` into `results_component_ablation/`.
