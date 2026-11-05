# Extrapolation disclosure: held-out interpolation cases against the realised training span

All test_interp parameters lie inside the pre-registered sampling ranges (PLAN.md section 4); this table shows which lie outside the span actually realised by the 28 training draws.

| case | family | parameter | test value | realised training span | outside |
|---|---|---|---|---|---|
| S2_seed121 | S | c | 0.1236 | [0.0145, 0.1236] | yes |
| S2_seed124 | S | c | 0.1454 | [0.0145, 0.1236] | yes |
| D2_seed220 | D | t_shift | 0.0951 | [0.1070, 0.2530] | yes |
| D2_seed220 | D | x_shift | 0.1813 | [0.0254, 0.1682] | yes |
| D2_seed221 | D | x_shift | 0.1825 | [0.0254, 0.1682] | yes |
| M_seed320 | M | b | 0.3628 | [0.2429, 0.3519] | yes |
| M_seed320 | M | f | 0.7022 | [0.8476, 1.4983] | yes |
| M_seed321 | M | a | 0.2817 | [0.1401, 0.2798] | yes |
| M_seed321 | M | c | 0.0901 | [0.0108, 0.0831] | yes |
| M_seed322 | M | f | 0.7756 | [0.8476, 1.4983] | yes |

10 (case, parameter) pairs lie outside the realised training span; all other test_interp parameters are inside it.
