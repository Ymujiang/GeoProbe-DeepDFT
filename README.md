# GeoProbe-DeepDFT

**Symmetry-Preserving $O(3)$ Geometric Probe Gating and Multi-Scale Enriched Radial Filters for Real-Space Electron Density Prediction**

GeoProbe-DeepDFT resolves the first-hop directional blindness and single-basis radial bottleneck of one-way PaiNN message passing (`PainnDensityModel` from Jørgensen & Bhowmik, *npj Comput. Mater.* **2022**, *8*, 183) while preserving exact $E(3) \times S_N \times \text{PBC}$ symmetry and globally $C^1$-smooth ($C^\infty$ on $(0, r_{\mathrm{cut}})$) differentiability with respect to probe coordinates $\mathbf{r}$.

## Key Architectural Innovations

1. **Low-Rank $O(3)$-Invariant Directional Probe Gating (`PaiNNInteractionOneWay`)**:
   Projects atom ($\mathbf{v}_j$) and probe ($\mathbf{v}_p$) equivariant vector channels to $D = 32$ latent modes and constructs four parity-even contractions along edge unit vector $\hat{\mathbf{r}}_{jp}$:
   $$\langle \mathbf{v}_j, \hat{\mathbf{r}}_{jp} \rangle, \quad \langle \mathbf{v}_p, \hat{\mathbf{r}}_{jp} \rangle, \quad \langle \mathbf{v}_j, \mathbf{v}_p \rangle, \quad \langle \mathbf{U}\mathbf{v}_j, \hat{\mathbf{r}}_{jp} \rangle^2$$
   A ReZero-initialized MLP outputs a multiplicative factor $(1 + \tanh(\cdot)) \in (0, 2)$ applied symmetrically to both scalar and sender-vector messages, together with a vector-magnitude-aware update gate (`state_scalar`, `msg_scalar`, `||msg_vec||`).
2. **Multi-Scale `EnrichedRadialFilter` ($30 \to 58 \to 64 \to 3F$)**:
   Augments the canonical 30 spherical-sinc radial functions with 16 trainable-frequency Bessel waves ($\sin(\omega_k d)/(d + 10^{-5})$), 4 short-range inverse-distance terms ($(1 + d)^{-k},\, k=1\dots 4$), and 8 learnable Gaussian radial envelopes.
3. **Chunked Edge Streaming (65,536 edges/chunk)**:
   Bounds transient $(E_{\text{chunk}}, 3, F)$ allocations during inference for any PaiNN model (`1,601.1 MB` peak VRAM at 8,192 probes/chunk on RTX 5080 vs `1,237.3 MB` for chunked canonical `h=128`, `2,806.4 MB` for unchunked official DeepDFT, and `9,238.6 MB` for `ChargE3Net` $l_{\max}=2, \text{mul}=256$).

## Headline Benchmark Summary

| Benchmark / Split | Canonical (`h=128`, 1.49M) | Matched (`h=144`, 1.87M) | GeoProbe (`h=128`, 1.87M) | Rel. Change vs `h=128` |
|---|---|---|---|---|
| **NMC Solid (3-seed, 5k steps, 50 test)** | $1.0195 \pm 0.0165\%$ | $0.9715 \pm 0.0640\%$ | **$0.7045 \pm 0.0198\%$** | **-30.9%** ($p = 3.84 \times 10^{-5}$) |
| **NMC vs `ChargE3Net` ($l_{\max}=2$, 5k, s42)** | $1.0302\%$ (`ChargE3Net`: $1.4283\%$) | $1.0049\%$ | **$0.6825\%$** | **-33.8%** (**-52.2%** vs `ChargE3Net`) |
| **NMC Full-500 Holdout (5k, seed 42)** | $0.9951\%$ ($\text{SD } 0.2206\%$) | $0.9719\%$ ($\text{SD } 0.2122\%$) | **$0.6571\%$** ($\text{SD } 0.1527\%$) | **-34.0%** (**500/500 wins**, $p = 1.26 \times 10^{-83}$) |
| **NMC Extended 15k Steps (100 test)** | $0.5139\%$ | — | **$0.3524\%$** | **-31.4%** |
| **NMC Converged 100k Steps (100 test)** | $0.1296 \pm 0.0426\%$ | $0.1187 \pm 0.0413\%$ | **$0.1084 \pm 0.0385\%$** | **-16.4%** / **-8.7%** vs `h=144` (**100/100 wins vs both**) |
| **5-Fold Stoichiometry OOD CV (NMC)** | $1.4961 \pm 0.6718\%$ | — | **$0.7502 \pm 0.1909\%$** | **-49.9%** (**5/5 folds**, peak **-76.6%**) |
| **EC Liquid (5k steps, 50 test, `pbc=True`)** | $2.0972\%$ | $1.9760\%$ | **$1.1881\%$** | **-43.3%** (RMSE **-50.6%**) |
| **QM9 Gas (15k steps, 50 test, `pbc=False`)** | $2.8159\%$ (`0.916 D`) | $2.6820\%$ (`0.637 D`) | **$2.5782\%$** (`0.693 D`) | **-8.4%** ($|\Delta N_e|$ **-34.0%**, dipole **-24.4%**) |
| **NMC Bader Charge MAE (5k / 100k, 25 cells)** | $15.17$ / $1.68\text{ m}e$ (`E3`: $28.55\text{ m}e$) | — | **$10.50$ / $1.54\text{ m}e$** | **-30.8%** / **-8.0%** (**-63.2%** vs `ChargE3Net`) |
| **EC Bader Charge MAE / Hartree $V_H$ (5k)** | $78.47\text{ m}e$ / $311.7\text{ mV}$ | — | **$52.87\text{ m}e$ / $194.8\text{ mV}$** | **-32.6%** / **-37.5%** ($E_H[\Delta\rho]$ **-65.8%**) |
| **NMC Autograd $\nabla\hat{\rho}$ $L_2$ / Cos (5k)** | $0.1362$ / $0.9034$ | — | **$0.0843$ / $0.9556$** | **-38.1%** MAE (100k cos: **0.9983** vs 0.9974) |

## Hardware Profiling on NVIDIA RTX 5080 (Fair Chunked Comparison + `ChargE3Net`)

| Model | Edge Chunk | Train Step (`4×512`) | Train VRAM | Inf. Chunk (`8,192`) | Inf. Throughput | Inf. VRAM |
|---|---|---|---|---|---|---|
| Canonical `h=128` (official) | None | `36.19 ms` | `2,242.8 MB` | `34.33 ms` | `238.7k pts/s` | `2,806.4 MB` |
| Canonical `h=128` (chunked) | `65,536` | `36.44 ms` | `2,243.1 MB` | `34.13 ms` | `240.0k pts/s` | `1,237.3 MB` |
| Matched `h=144` (chunked) | `65,536` | `40.60 ms` | `2,514.9 MB` | `38.60 ms` | `212.2k pts/s` | `1,379.5 MB` |
| **GeoProbe `h=128` (chunked)** | **`65,536`** | **`58.04 ms`** | **`3,794.5 MB`** | **`63.57 ms`** | **`128.9k pts/s`** | **`1,601.1 MB`** |
| `ChargE3Net` ($l_{\max}=2, \text{mul}=256$) | Native | `238.49 ms` | `4,005.3 MB` | `164.31 ms` | `49.9k pts/s` | `9,238.6 MB` |
| `ChargE3Net` ($l_{\max}=4, \text{mul}=500$)* | Native | `631.37 ms` (`2×200`) | `4,389.4 MB` | `142.36 ms` (`1,024`) | `7.2k pts/s` | `5,450.7 MB` |

*\*Official `ChargE3Net` ($l_{\max}=4, \text{mul}=500$, 1.91M params) exceeds 16 GB VRAM at `batch=4×512` and is profiled at its native `batch=2×200` and `1,024`-probe inference chunk.*

## Installation

```bash
git clone https://github.com/Ymujiang/GeoProbe-DeepDFT.git
cd GeoProbe-DeepDFT
pip install -r requirements.txt
pip install -e .
```

## Quick Verification & Reproducibility

### 1. Symmetry ($S_N$, Translation + PBC Wrap, $O(3)$ Proper & Improper) and Autograd Smoothness Tests

```bash
pytest -v
```

### 2. Evaluate Pretrained Checkpoints on Held-Out NMC Sample Grids

```bash
python evaluate.py --model geoprobe
python evaluate.py --model canonical
```

### 3. Parameter & Ablation Ledger

```bash
python run_ablation.py --list_params
```

### 4. Hardware Throughput & Peak Memory Profiler

```bash
python scripts/profile_throughput.py
```

## Datasets & Attribution

Full VASP/PBE `CHGCAR` electron density archives for NMC ($\text{Li}_x(\text{Ni,Mn,Co})\text{O}_2$), liquid ethylene carbonate (EC), and QM9 (`pbc=False`) follow the open DTU Data releases of Jørgensen & Bhowmik (*npj Comput. Mater.* **2022**, *8*, 183; DOI: `10.1038/s41524-022-00863-y`). Complete raw experiment ledgers (`results/*.json`, including 100k convergence, `ChargE3Net`, Bader/Hartree/dipole observables, and fair chunked profiling) and trained seed-42 NMC checkpoints (`checkpoints/`) are archived here under the MIT License.
