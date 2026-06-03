# IBP presentation (Nicolas Privault meeting)

Beamer deck adapted from the NTU template in `HW/presentation` (`talk_mz_local_volatility.tex` style).

## Build

```bash
cd presentation
pdflatex main.tex
pdflatex main.tex
```

Output: `main.pdf`

## Figures

Sources: `../plots/ibp_comparison/` (regenerate with `python3 examples/run_ibp_comparison_plots.py`).

Stable copies live in `figures/` including T=1.0 crops and the Gaussian triptych.

## Contents

- Nicolas's IBP three-way check (i), (ii), (iii)
- Legacy vs corrected $\tilde\varphi$ mapping
- K=0 retrain results
- Full three-panel comparison plots
