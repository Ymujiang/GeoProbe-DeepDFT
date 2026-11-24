# GeoProbe-DeepDFT

**Symmetry-Preserving $O(3)$ Geometric Probe Gating and Multi-Scale Enriched Radial Filters for Real-Space Electron Density Prediction**

GeoProbe-DeepDFT resolves the directional blindness and single-basis radial bottleneck of one-way PaiNN message passing (`PainnDensityModel` from Jørgensen & Bhowmik, *npj Comput. Mater.* **2022**, *8*, 183) while preserving exact $E(3) \times S_N \times \text{PBC}$ symmetry and $C^2$ differentiability wrt probe coordinates $\mathbf{r}$.

## Key Architectural Innovations

1. **Low-Rank $O(3)$-Invariant Directional Probe Gating (`PaiNNInteractionOneWay`)**:
   Projects atom ($\mathbf{v}_j$) and probe ($\mathbf{v}_p$) equivariant vector channels to $D = 32$ latent modes and constructs four parity-even contractions along edge unit vector $\hat{\mathbf{r}}_{jp}$:
   $$\langle \mathbf{v}_j, \hat{\mathbf{r}}_{jp} \rangle, \quad \langle \mathbf{v}_p, \hat{\mathbf{r}}_{jp} \rangle, \quad \langle \mathbf{v}_j, \mathbf{v}_p \rangle, \quad \langle \mathbf{U}\mathbf{v}_j, \hat{\mathbf{r}}_{jp} \rangle^2$$
   A ReZero-initialized MLP outputs a multiplicative factor $(1 + \tanh(\cdot)) \in (0, 2)$ applied symmetrically to both scalar and sender-vector messages, together with a vector-magnitude-aware update gate (`state_scalar`, `msg_scalar`, `||msg_vec||`).
2. **Multi-Scale `EnrichedRadialFilter` ($30 \to 58 \to 64 \to 3F$)**:
   Augments the canonical 30 spherical-sinc radial functions with 16 trainable-frequency Bessel waves ($\sin(\omega_k d)/(d + 10^{-5})$), 4 short-range inverse-distance terms ($(1 + d)^{-k},\, k=1\dots 4$), and 8 learnable Gaussian radial envelopes.
3. **Chunked Edge Streaming (65,536 edges/chunk)**:
   Eliminates transient $(E, 3, F)$ allocation spikes during inference, cutting peak VRAM by **43.1%** relative to canonical `h=128` (`1,596.5 MB` vs `2,807.1 MB` at 8,192 probes/chunk on RTX 5080).

## Headline Benchmark Summary

| Benchmark / Split | Canonical (`h=128`, 1.49M) | Matched (`h=144`, 1.87M) | GeoProbe (`h=128`, 1.87M) | Rel. Change vs `h=128` |
|---|---|---|---|---|
| **NMC Solid (3-seed, 5k steps, 50 test)** | $1.0195 \pm 0.0165\%$ | $0.9715 \pm 0.0640\%$ | **$0.7045 \pm 0.0198\%$** | **-30.9%** ($p = 3.84 \times 10^{-5}$) |
| **NMC Full-500 Holdout (seed 42)** | $0.9951\%$ ($\text{SD } 0.2206\%$) | $0.9719\%$ ($\text{SD } 0.2122\%$) | **$0.6571\%$** ($\text{SD } 0.1527\%$) | **-34.0%** (**500/500 wins**, $p = 1.26 \times 10^{-83}$) |
| **NMC Extended 15k Steps (100 test)** | $0.5139\%$ | — | **$0.3524\%$** | **-31.4%** |
| **5-Fold Stoichiometry OOD CV** | $1.4961 \pm 0.6718\%$ | — | **$0.7502 \pm 0.1909\%$** | **-49.9%** (**5/5 folds**, peak **-76.6%**) |
| **EC Liquid (5k steps, 50 test)** | $2.0972\%$ | $1.9760\%$ | **$1.1881\%$** | **-43.3%** |
| **QM9 Gas (5k steps, 50 test)** | $5.5244\%$ | $7.5983\%$ | **$2.6895\%$** | **-51.3%** |
| **NMC Autograd $\nabla\hat{\rho}$ $L_2$ MAE / Cos** | $0.1362$ / $0.9034$ | — | **$0.0843$ / $0.9556$** | **-38.1%** MAE |

## Installation

```bash
git clone https://github.com/Ymujiang/GeoProbe-DeepDFT.git
cd GeoProbe-DeepDFT
pip install -r requirements.txt
pip install -e .
```

## Quick Verification & Reproducibility

### 1. Symmetry ($S_N$, Translation + PBC Wrap, $O(3)$ Proper & Improper) and $C^2$ Autograd Tests

```bash
pytest -v
```

Observed output (`pytest -v`):
```text
tests/test_invariance_and_autograd.py::test_parameter_counts PASSED      [ 33%]
tests/test_invariance_and_autograd.py::test_exact_symmetries_loaded_checkpoint PASSED [ 66%]
tests/test_invariance_and_autograd.py::test_c2_smoothness_and_autograd PASSED [100%]
======================== 3 passed, 2 warnings in 2.73s =========================
```

### 2. Evaluate Pretrained Checkpoints on Held-Out NMC Sample Grids

```bash
python evaluate.py --model geoprobe
python evaluate.py --model canonical
```

Observed output (`sample_data/nmc_sample_2structs.pt`, RTX 5080):
- `geoprobe` (`1,871,617` params): `mean_eps_mae_pct = 0.7469%` (`1.CHGCAR.lz4`: `0.4798%`, `3.CHGCAR.lz4`: `1.0140%`)
- `canonical` (`1,487,233` params): `mean_eps_mae_pct = 1.1531%` (`1.CHGCAR.lz4`: `0.7800%`, `3.CHGCAR.lz4`: `1.5262%`)

### 3. Parameter & Ablation Ledger

```bash
python run_ablation.py --list_params
```

### 4. Hardware Throughput & Peak Memory Profiler

```bash
python scripts/profile_throughput.py
```

Observed output (4,096 probes, 121,667 edges, RTX 5080):
- `canonical_h128`: `17.372 ms/chunk`, `235,783 probes/s`, `1,258.76 MB` peak VRAM
- `param_matched_h144`: `19.208 ms/chunk`, `213,250 probes/s`, `1,407.77 MB` peak VRAM
- `geoprobe_h128`: `27.644 ms/chunk`, `148,169 probes/s`, `1,160.09 MB` peak VRAM

## Datasets & Attribution

Full VASP `CHGCAR` electron density archives for NMC ($\text{Li}_x(\text{Ni,Mn,Co})\text{O}_2$), liquid ethylene carbonate (EC), and QM9 follow the open DTU Data releases of Jørgensen & Bhowmik (*npj Comput. Mater.* **2022**, *8*, 183; DOI: `10.1038/s41524-022-00863-y`). Pre-computed `results/*.json` ledgers and trained seed-42 NMC checkpoints (`checkpoints/`) are included for immediate auditability under the MIT License.
