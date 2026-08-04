# Caveat: MC provenance in `pdf_analysis_*` figures generated before the provenance fix

Timestamped `pdf_analysis_*.png` / `.pdf` figures created **before the MC-provenance fix**
(`Label PDF-analysis subtitles from recorded MC provenance`) carry a subtitle that may not
describe the Monte Carlo paths actually plotted.

## What is wrong

The subtitle was hardcoded to

    Monte Carlo with $dS_t = rS_t dt + \sigma_{NN}(t,S)S_t dW_t$

i.e. it always asserted the paths were simulated under the **learned** volatility. But with
`reuse_training_mc = True` (the default) the analysis reuses `training_data.npz`, whose paths were
generated under the **ground-truth** volatility. In that case the subtitle names the wrong
surface.

The error is one-directional: the string only ever claimed `sigma_NN`, so figures whose paths
genuinely were `sigma_NN` are labelled correctly. Affected figures claim `sigma_NN` while showing
ground-truth paths.

Confirmed numerically on `synthetic_paper_large_dataset_dupire_exact`: the figure's own reported
`sigma_MC` (0.24 / 0.38 / 0.50 / 0.60 at T = 0.25 / 0.50 / 0.75 / 1.00) matches the `S_matrix` in
`training_data.npz` (0.241 / 0.381 / 0.496 / 0.585) — ground truth, not `sigma_NN`. Separately, the
default `n_paths_analysis = 10**6` made the pre-fix NN-volatility simulator unrunnable (it
allocated a 10**6 x 10**6 array, ~7.3 TiB; fixed alongside), so figures produced with the default
configuration cannot have come from the simulate branch at all.

## What is NOT wrong

**Only the subtitle text.** The plotted curves, histograms, KDEs and every number in the
accompanying `diagnostics.json` are unaffected. This was checked by rendering the same data and
model through the pre-fix and post-fix code in one environment: all plotted arrays were
bit-identical, all in-panel annotations identical, and `diagnostics.json` differed only by the
addition of a `mc_provenance` key. Read the data in these figures as valid; disregard the SDE line
in the subtitle.

## Why these are not regenerated

The filenames encode the generation timestamp. Regenerating would write today's output into a file
named for an earlier date, overwriting the historical record with something it is not — a worse
outcome than a wrong caption. Figures generated after the fix state their MC provenance from a
recorded tag (`ground_truth`, `nn_repriced`, or an explicit "surface unrecorded"), and
`diagnostics.json` records it.

## Affected files

22 tracked files on this branch:

- `Synthetic_Data_Tensorflow_Advanced/models/example_pretrained/pdf_analysis_*` (4)
- `Synthetic_Data_Tensorflow_Advanced/models/runs/synthetic_paper_large_dataset_constant_vol/pdf_analysis_*` (14)
- `Synthetic_Data_Tensorflow_Advanced/models/runs/synthetic_paper_large_dataset_constant_vol_HPC/pdf_analysis_*` (4)

On branches that also carry a tracked `docs/` directory there are a further 7 timestamped
`docs/pdf_analysis_*` files under the same caveat.

Non-timestamped outputs under `plots/` are regenerable and are **not** covered by this caveat —
the four sets under `plots/reproduce_user_regen/` were re-run under the fixed pipeline and carry
correct provenance tags.
