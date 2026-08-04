# Note on the `pdf_analysis_*` figures here

The `pdf_analysis_*` figures in this directory predate the MC-provenance fix, and their subtitle
asserts the Monte Carlo was run under the learned `sigma_NN`. Where the analysis reused
`training_data.npz` — the default — those paths were generated under the **ground-truth**
volatility instead, so the SDE line names the wrong surface.

**Only the caption is affected.** The plotted curves and every number in `diagnostics.json` are
unchanged; this was verified by rendering the same data through the pre- and post-fix code in one
environment.

These are deliberately **not** regenerated: the filenames encode their generation timestamp, and
rewriting them with today's output would destroy that record.

Full detail, including the numerical confirmation: [`../../../../FIGURE_PROVENANCE_CAVEAT.md`](../../../../FIGURE_PROVENANCE_CAVEAT.md)
