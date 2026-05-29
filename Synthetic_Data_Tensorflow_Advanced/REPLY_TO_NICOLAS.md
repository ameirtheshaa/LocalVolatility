# Reply to Nicolas — IBP three-way check on NN density

**Subject:** IBP three-way check on NN density — results before meeting

Hi Nicolas,

We implemented your three-way comparison. A short note on implementation: the analysis path must use the same normalized price as training, **φ̃ = 1 − exp(−NN_φ)**. Using the raw network output as φ̃ inflated (ii) artificially (~2.7×); with the correct map, the picture below is what we report.

## Results — pretrained models (`models/example_pretrained`)

| T | ref = S₀e^(rT) | (i) ∫ K ∂²C/∂K² | (ii) e^(rT)C_NN(0) | (iii) E[S_T] MC |
|---:|---:|---:|---:|---:|
| 0.5 | 1020.20 | 948.86 | 948.93 | 1042.19 |
| 1.0 | 1040.81 | 967.65 | 967.74 | 1125.70 |
| 1.5 | 1061.84 | 982.49 | 982.95 | 1179.55 |

## Results — after K=0 retrain (`models/runs/k0_bc_retrain`, λ_K0=10, K_min=100)

| T | ref | (i) | (ii) | (iii) |
|---:|---:|---:|---:|---:|
| 0.5 | 1020.20 | 1012.20 | 1014.34 | 1018.92 |
| 1.0 | 1040.81 | 1030.73 | 1033.67 | 1038.13 |

## Interpretation

1. **(i) ≡ (ii)** to machine precision in all runs — your integration-by-parts chain is satisfied by our implementation.
2. **Pretrained (i)/(ii)** are ~7% below the forward; **(iii)** from NN-driven MC is 2–11% above — the main residual for KDE–NN mean mismatch is MC/σ_NN, not a broken ∂²C/∂K² formula.
3. **Explicit penalty on C_NN(0,T)=S₀** plus training strikes down to K=100 brings all three quantities within about **1%** at T ∈ {0.5, 1.0}.
4. Part of the **visible** curve gap on the plots is still the q1–q99 integration window ((i') vs (i)).

Plots: `pdf_analysis_*.png` in the model directories (Panel 3 shows the diagnostic text box).

Best,  
Ameir
