# Gate 5 — boundary-adapted reduced model vs Fourier model vs refined full solve

Reference: boundary-consistent full solve at n_k = 512 (interpolated to the 256 grid); for the constant-σ control also Black–Scholes. u_grid = interior RMS difference between the 256 and 512 full solves (numerical uncertainty); an improvement is claimed only when it exceeds u_grid.

## C_sig0.30 (C, {'sigma': 0.3})

u_grid = 4.03e-06; density u = 4.32e-06; full512 vs BS = 1.97e-05

| model | start | dof | price int. RMS | price vs BS | density L2 | neg. mass max | neg. mass T_max | inverse-vol invalid | butterfly viol. | stable | reduced abscissa | proj. err | rollout s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A_fourier_m33 | from_payoff_t0 | 33 | 9.85e-02 | 9.85e-02 | 4.42e-02 | 1.53e+00 | 6.49e-01 | 0.154 | 0.134 | yes | -3.84e-14 | 7.78e-03 | 7.214e-03 |
| A_fourier_m33 | from_projected_t0.2 | 33 | 8.71e-02 | 8.71e-02 | 7.50e-02 | 2.05e+01 | 7.30e-01 | 0.152 | 0.133 | yes | -3.84e-14 | 7.78e-03 | 7.271e-03 |
| A_fourier_m65 | from_payoff_t0 | 65 | 1.06e-01 | 1.06e-01 | 4.00e-02 | 1.56e+00 | 6.89e-01 | 0.139 | 0.128 | yes | -5.72e-14 | 3.86e-03 | 6.553e-03 |
| A_fourier_m65 | from_projected_t0.2 | 65 | 9.36e-02 | 9.36e-02 | 8.96e-02 | 4.53e+01 | 7.71e-01 | 0.132 | 0.121 | yes | -5.72e-14 | 3.86e-03 | 6.430e-03 |
| A_full256 | payoff_t0 | 256 | 1.79e-05 | 3.24e-05 | 3.78e-05 | 3.48e-04 | 3.48e-04 | 0.181 | 0.000 | yes | – | – | 4.275e-02 |
| B_fourier_m33 | from_payoff_t0 | 33 | 2.97e-02 | 2.97e-02 | 2.13e-02 | 9.40e-01 | 4.92e-01 | 0.539 | 0.523 | yes | -3.40e-15 | 3.23e-04 | 7.454e-03 |
| B_fourier_m33 | from_projected_t0.2 | 33 | 2.23e-02 | 2.23e-02 | 1.98e-02 | 1.42e+00 | 5.35e-01 | 0.496 | 0.457 | yes | -3.40e-15 | 3.23e-04 | 7.260e-03 |
| B_fourier_m65 | from_payoff_t0 | 65 | 2.58e-02 | 2.58e-02 | 1.90e-02 | 9.42e-01 | 5.16e-01 | 0.514 | 0.498 | yes | -3.12e-13 | 8.83e-05 | 1.395e-02 |
| B_fourier_m65 | from_projected_t0.2 | 65 | 1.91e-02 | 1.91e-02 | 1.68e-02 | 1.43e+00 | 5.60e-01 | 0.468 | 0.445 | yes | -3.12e-13 | 8.83e-05 | 6.456e-03 |
| B_full256 | payoff_t0 | 256 | 4.03e-06 | 2.03e-05 | 4.32e-06 | 6.19e-07 | 9.50e-09 | 0.169 | 0.000 | yes | – | – | 1.315e-01 |
| B_full512_reference |  | 512 | – | – | – | – | – | – | – | yes | – | – | 2.643e-01 |
| B_sine_m17 | from_payoff_t0 | 17 | 1.03e-05 | 2.41e-05 | 1.04e-04 | 6.32e-04 | 0.00e+00 | 0.169 | 0.000 | yes | -2.14e-01 | 3.42e-06 | 7.705e-03 |
| B_sine_m17 | from_projected_t0.2 | 17 | 5.02e-06 | 2.13e-05 | 6.66e-05 | 2.32e-04 | 0.00e+00 | 0.170 | 0.000 | yes | -2.14e-01 | 3.42e-06 | 7.670e-03 |
| B_sine_m33 | from_payoff_t0 | 33 | 4.15e-06 | 2.04e-05 | 1.43e-05 | 1.92e-06 | 0.00e+00 | 0.169 | 0.000 | yes | -2.14e-01 | 2.25e-07 | 7.577e-03 |
| B_sine_m33 | from_projected_t0.2 | 33 | 4.03e-06 | 2.03e-05 | 1.43e-05 | 1.94e-06 | 0.00e+00 | 0.169 | 0.000 | yes | -2.14e-01 | 2.25e-07 | 7.573e-03 |
| B_sine_m65 | from_payoff_t0 | 65 | 4.03e-06 | 2.03e-05 | 5.14e-06 | 1.48e-06 | 0.00e+00 | 0.169 | 0.000 | yes | -2.14e-01 | 1.04e-08 | 4.123e-02 |
| B_sine_m65 | from_projected_t0.2 | 65 | 4.03e-06 | 2.03e-05 | 5.14e-06 | 1.48e-06 | 0.00e+00 | 0.169 | 0.000 | yes | -2.14e-01 | 1.04e-08 | 9.486e-03 |

Uncertainty-aware claims (price interior RMS):

- from_projected_t0.2: sine m=17 vs Fourier A_m33: gain 8.71e-02 (exceeds u_grid); density gain 7.50e-02 (exceeds density u); negative mass 2.05e+01 → 2.32e-04; sine within u of full-256: True; speed-up vs full-256 rollout: 17.1×
- from_projected_t0.2: sine m=17 vs Fourier B_m33: gain 2.23e-02 (exceeds u_grid); density gain 1.97e-02 (exceeds density u); negative mass 1.42e+00 → 2.32e-04; sine within u of full-256: True; speed-up vs full-256 rollout: 17.1×
- from_projected_t0.2: sine m=17 vs Fourier A_m65: gain 9.36e-02 (exceeds u_grid); density gain 8.95e-02 (exceeds density u); negative mass 4.53e+01 → 2.32e-04; sine within u of full-256: True; speed-up vs full-256 rollout: 17.1×
- from_projected_t0.2: sine m=17 vs Fourier B_m65: gain 1.91e-02 (exceeds u_grid); density gain 1.67e-02 (exceeds density u); negative mass 1.43e+00 → 2.32e-04; sine within u of full-256: True; speed-up vs full-256 rollout: 17.1×
- from_projected_t0.2: sine m=33 vs Fourier A_m33: gain 8.71e-02 (exceeds u_grid); density gain 7.50e-02 (exceeds density u); negative mass 2.05e+01 → 1.94e-06; sine within u of full-256: True; speed-up vs full-256 rollout: 17.4×
- from_projected_t0.2: sine m=33 vs Fourier B_m33: gain 2.23e-02 (exceeds u_grid); density gain 1.98e-02 (exceeds density u); negative mass 1.42e+00 → 1.94e-06; sine within u of full-256: True; speed-up vs full-256 rollout: 17.4×
- from_projected_t0.2: sine m=33 vs Fourier A_m65: gain 9.36e-02 (exceeds u_grid); density gain 8.96e-02 (exceeds density u); negative mass 4.53e+01 → 1.94e-06; sine within u of full-256: True; speed-up vs full-256 rollout: 17.4×
- from_projected_t0.2: sine m=33 vs Fourier B_m65: gain 1.91e-02 (exceeds u_grid); density gain 1.67e-02 (exceeds density u); negative mass 1.43e+00 → 1.94e-06; sine within u of full-256: True; speed-up vs full-256 rollout: 17.4×
- from_projected_t0.2: sine m=65 vs Fourier A_m33: gain 8.71e-02 (exceeds u_grid); density gain 7.50e-02 (exceeds density u); negative mass 2.05e+01 → 1.48e-06; sine within u of full-256: True; speed-up vs full-256 rollout: 13.9×
- from_projected_t0.2: sine m=65 vs Fourier B_m33: gain 2.23e-02 (exceeds u_grid); density gain 1.98e-02 (exceeds density u); negative mass 1.42e+00 → 1.48e-06; sine within u of full-256: True; speed-up vs full-256 rollout: 13.9×
- from_projected_t0.2: sine m=65 vs Fourier A_m65: gain 9.36e-02 (exceeds u_grid); density gain 8.96e-02 (exceeds density u); negative mass 4.53e+01 → 1.48e-06; sine within u of full-256: True; speed-up vs full-256 rollout: 13.9×
- from_projected_t0.2: sine m=65 vs Fourier B_m65: gain 1.91e-02 (exceeds u_grid); density gain 1.68e-02 (exceeds density u); negative mass 1.43e+00 → 1.48e-06; sine within u of full-256: True; speed-up vs full-256 rollout: 13.9×
- from_payoff_t0: sine m=17 vs Fourier A_m33: gain 9.85e-02 (exceeds u_grid); density gain 4.41e-02 (exceeds density u); negative mass 1.53e+00 → 6.32e-04; sine within u of full-256: False; speed-up vs full-256 rollout: 17.1×
- from_payoff_t0: sine m=17 vs Fourier B_m33: gain 2.97e-02 (exceeds u_grid); density gain 2.12e-02 (exceeds density u); negative mass 9.40e-01 → 6.32e-04; sine within u of full-256: False; speed-up vs full-256 rollout: 17.1×
- from_payoff_t0: sine m=17 vs Fourier A_m65: gain 1.06e-01 (exceeds u_grid); density gain 3.99e-02 (exceeds density u); negative mass 1.56e+00 → 6.32e-04; sine within u of full-256: False; speed-up vs full-256 rollout: 17.1×
- from_payoff_t0: sine m=17 vs Fourier B_m65: gain 2.58e-02 (exceeds u_grid); density gain 1.89e-02 (exceeds density u); negative mass 9.42e-01 → 6.32e-04; sine within u of full-256: False; speed-up vs full-256 rollout: 17.1×
- from_payoff_t0: sine m=33 vs Fourier A_m33: gain 9.85e-02 (exceeds u_grid); density gain 4.42e-02 (exceeds density u); negative mass 1.53e+00 → 1.92e-06; sine within u of full-256: True; speed-up vs full-256 rollout: 17.4×
- from_payoff_t0: sine m=33 vs Fourier B_m33: gain 2.97e-02 (exceeds u_grid); density gain 2.13e-02 (exceeds density u); negative mass 9.40e-01 → 1.92e-06; sine within u of full-256: True; speed-up vs full-256 rollout: 17.4×
- from_payoff_t0: sine m=33 vs Fourier A_m65: gain 1.06e-01 (exceeds u_grid); density gain 4.00e-02 (exceeds density u); negative mass 1.56e+00 → 1.92e-06; sine within u of full-256: True; speed-up vs full-256 rollout: 17.4×
- from_payoff_t0: sine m=33 vs Fourier B_m65: gain 2.58e-02 (exceeds u_grid); density gain 1.90e-02 (exceeds density u); negative mass 9.42e-01 → 1.92e-06; sine within u of full-256: True; speed-up vs full-256 rollout: 17.4×
- from_payoff_t0: sine m=65 vs Fourier A_m33: gain 9.85e-02 (exceeds u_grid); density gain 4.42e-02 (exceeds density u); negative mass 1.53e+00 → 1.48e-06; sine within u of full-256: True; speed-up vs full-256 rollout: 3.2×
- from_payoff_t0: sine m=65 vs Fourier B_m33: gain 2.97e-02 (exceeds u_grid); density gain 2.13e-02 (exceeds density u); negative mass 9.40e-01 → 1.48e-06; sine within u of full-256: True; speed-up vs full-256 rollout: 3.2×
- from_payoff_t0: sine m=65 vs Fourier A_m65: gain 1.06e-01 (exceeds u_grid); density gain 4.00e-02 (exceeds density u); negative mass 1.56e+00 → 1.48e-06; sine within u of full-256: True; speed-up vs full-256 rollout: 3.2×
- from_payoff_t0: sine m=65 vs Fourier B_m65: gain 2.58e-02 (exceeds u_grid); density gain 1.90e-02 (exceeds density u); negative mass 9.42e-01 → 1.48e-06; sine within u of full-256: True; speed-up vs full-256 rollout: 3.2×

## F_fresh_base0.28_t0.2_x0.05 (F, {'sigma_base': 0.28, 't_shift': 0.2, 'x_shift': 0.05})

u_grid = 3.01e-06; density u = 8.97e-07; step u = 5.77e-08

| model | start | dof | price int. RMS | price vs BS | density L2 | neg. mass max | neg. mass T_max | inverse-vol invalid | butterfly viol. | stable | reduced abscissa | proj. err | rollout s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A_fourier_m33 | from_payoff_t0 | 33 | 1.21e-01 | – | 2.42e-02 | 9.20e-01 | 1.01e-01 | 0.267 | 0.251 | yes | 1.73e-13 | 8.65e-03 | 8.829e+00 |
| A_fourier_m33 | from_projected_t0.2 | 33 | 1.16e-01 | – | 5.28e-02 | 2.04e+01 | 1.39e-01 | 0.244 | 0.229 | yes | 1.73e-13 | 8.65e-03 | 8.111e+00 |
| A_fourier_m65 | from_payoff_t0 | 65 | 1.29e-01 | – | 2.31e-02 | 9.66e-01 | 1.15e-01 | 0.267 | 0.250 | yes | -3.69e-13 | 4.30e-03 | 1.245e+01 |
| A_fourier_m65 | from_projected_t0.2 | 65 | 1.23e-01 | – | 7.24e-02 | 4.53e+01 | 1.55e-01 | 0.231 | 0.222 | yes | -3.69e-13 | 4.30e-03 | 1.115e+01 |
| A_full256 | payoff_t0 | 256 | 1.64e-02 | – | 2.78e-03 | 9.00e-02 | 9.00e-02 | 0.191 | 0.177 | yes | – | – | 4.845e+01 |
| B_fourier_m33 | from_payoff_t0 | 33 | 7.79e-02 | – | 1.77e-02 | 6.85e-01 | 5.92e-02 | 0.425 | 0.409 | yes | 1.20e-13 | 2.46e-04 | 8.638e+00 |
| B_fourier_m33 | from_projected_t0.2 | 33 | 5.81e-02 | – | 1.69e-02 | 1.32e+00 | 7.53e-02 | 0.411 | 0.393 | yes | 1.20e-13 | 2.46e-04 | 7.385e+00 |
| B_fourier_m65 | from_payoff_t0 | 65 | 7.11e-02 | – | 1.68e-02 | 7.15e-01 | 6.99e-02 | 0.404 | 0.395 | yes | -9.53e-13 | 6.68e-05 | 1.466e+01 |
| B_fourier_m65 | from_projected_t0.2 | 65 | 5.30e-02 | – | 1.57e-02 | 1.33e+00 | 8.67e-02 | 0.389 | 0.376 | yes | -9.53e-13 | 6.68e-05 | 1.147e+01 |
| B_full256 | payoff_t0 | 256 | 3.01e-06 | – | 8.97e-07 | 2.74e-08 | 2.69e-08 | 0.004 | 0.000 | yes | – | – | 4.621e+01 |
| B_full512_reference |  | 512 | – | – | – | – | – | – | – | yes | – | – | 1.233e+02 |
| B_sine_m17 | from_payoff_t0 | 17 | 8.10e-06 | – | 1.64e-04 | 6.92e-07 | 0.00e+00 | 0.005 | 0.000 | yes | -7.13e-01 | 8.09e-06 | 8.712e+00 |
| B_sine_m17 | from_projected_t0.2 | 17 | 6.42e-06 | – | 1.61e-04 | 1.01e-05 | 0.00e+00 | 0.005 | 0.000 | yes | -7.13e-01 | 8.09e-06 | 5.816e+00 |
| B_sine_m33 | from_payoff_t0 | 33 | 3.08e-06 | – | 2.55e-05 | 2.29e-07 | 0.00e+00 | 0.004 | 0.000 | yes | -7.13e-01 | 4.11e-07 | 9.400e+00 |
| B_sine_m33 | from_projected_t0.2 | 33 | 2.88e-06 | – | 2.57e-05 | 6.27e-08 | 0.00e+00 | 0.004 | 0.000 | yes | -7.13e-01 | 4.11e-07 | 7.156e+00 |
| B_sine_m65 | from_payoff_t0 | 65 | 3.00e-06 | – | 3.75e-06 | 0.00e+00 | 0.00e+00 | 0.004 | 0.000 | yes | -7.13e-01 | 1.72e-08 | 1.437e+01 |
| B_sine_m65 | from_projected_t0.2 | 65 | 3.01e-06 | – | 3.77e-06 | 0.00e+00 | 0.00e+00 | 0.004 | 0.000 | yes | -7.13e-01 | 1.72e-08 | 1.017e+01 |

Uncertainty-aware claims (price interior RMS):

- from_projected_t0.2: sine m=17 vs Fourier A_m33: gain 1.16e-01 (exceeds u_grid); density gain 5.27e-02 (exceeds density u); negative mass 2.04e+01 → 1.01e-05; sine within u of full-256: False; speed-up vs full-256 rollout: 7.9×
- from_projected_t0.2: sine m=17 vs Fourier B_m33: gain 5.81e-02 (exceeds u_grid); density gain 1.67e-02 (exceeds density u); negative mass 1.32e+00 → 1.01e-05; sine within u of full-256: False; speed-up vs full-256 rollout: 7.9×
- from_projected_t0.2: sine m=17 vs Fourier A_m65: gain 1.23e-01 (exceeds u_grid); density gain 7.22e-02 (exceeds density u); negative mass 4.53e+01 → 1.01e-05; sine within u of full-256: False; speed-up vs full-256 rollout: 7.9×
- from_projected_t0.2: sine m=17 vs Fourier B_m65: gain 5.30e-02 (exceeds u_grid); density gain 1.55e-02 (exceeds density u); negative mass 1.33e+00 → 1.01e-05; sine within u of full-256: False; speed-up vs full-256 rollout: 7.9×
- from_projected_t0.2: sine m=33 vs Fourier A_m33: gain 1.16e-01 (exceeds u_grid); density gain 5.28e-02 (exceeds density u); negative mass 2.04e+01 → 6.27e-08; sine within u of full-256: True; speed-up vs full-256 rollout: 6.5×
- from_projected_t0.2: sine m=33 vs Fourier B_m33: gain 5.81e-02 (exceeds u_grid); density gain 1.69e-02 (exceeds density u); negative mass 1.32e+00 → 6.27e-08; sine within u of full-256: True; speed-up vs full-256 rollout: 6.5×
- from_projected_t0.2: sine m=33 vs Fourier A_m65: gain 1.23e-01 (exceeds u_grid); density gain 7.24e-02 (exceeds density u); negative mass 4.53e+01 → 6.27e-08; sine within u of full-256: True; speed-up vs full-256 rollout: 6.5×
- from_projected_t0.2: sine m=33 vs Fourier B_m65: gain 5.30e-02 (exceeds u_grid); density gain 1.57e-02 (exceeds density u); negative mass 1.33e+00 → 6.27e-08; sine within u of full-256: True; speed-up vs full-256 rollout: 6.5×
- from_projected_t0.2: sine m=65 vs Fourier A_m33: gain 1.16e-01 (exceeds u_grid); density gain 5.28e-02 (exceeds density u); negative mass 2.04e+01 → 0.00e+00; sine within u of full-256: True; speed-up vs full-256 rollout: 4.5×
- from_projected_t0.2: sine m=65 vs Fourier B_m33: gain 5.81e-02 (exceeds u_grid); density gain 1.69e-02 (exceeds density u); negative mass 1.32e+00 → 0.00e+00; sine within u of full-256: True; speed-up vs full-256 rollout: 4.5×
- from_projected_t0.2: sine m=65 vs Fourier A_m65: gain 1.23e-01 (exceeds u_grid); density gain 7.24e-02 (exceeds density u); negative mass 4.53e+01 → 0.00e+00; sine within u of full-256: True; speed-up vs full-256 rollout: 4.5×
- from_projected_t0.2: sine m=65 vs Fourier B_m65: gain 5.30e-02 (exceeds u_grid); density gain 1.57e-02 (exceeds density u); negative mass 1.33e+00 → 0.00e+00; sine within u of full-256: True; speed-up vs full-256 rollout: 4.5×
- from_payoff_t0: sine m=17 vs Fourier A_m33: gain 1.21e-01 (exceeds u_grid); density gain 2.40e-02 (exceeds density u); negative mass 9.20e-01 → 6.92e-07; sine within u of full-256: False; speed-up vs full-256 rollout: 5.3×
- from_payoff_t0: sine m=17 vs Fourier B_m33: gain 7.79e-02 (exceeds u_grid); density gain 1.76e-02 (exceeds density u); negative mass 6.85e-01 → 6.92e-07; sine within u of full-256: False; speed-up vs full-256 rollout: 5.3×
- from_payoff_t0: sine m=17 vs Fourier A_m65: gain 1.29e-01 (exceeds u_grid); density gain 2.29e-02 (exceeds density u); negative mass 9.66e-01 → 6.92e-07; sine within u of full-256: False; speed-up vs full-256 rollout: 5.3×
- from_payoff_t0: sine m=17 vs Fourier B_m65: gain 7.11e-02 (exceeds u_grid); density gain 1.66e-02 (exceeds density u); negative mass 7.15e-01 → 6.92e-07; sine within u of full-256: False; speed-up vs full-256 rollout: 5.3×
- from_payoff_t0: sine m=33 vs Fourier A_m33: gain 1.21e-01 (exceeds u_grid); density gain 2.42e-02 (exceeds density u); negative mass 9.20e-01 → 2.29e-07; sine within u of full-256: True; speed-up vs full-256 rollout: 4.9×
- from_payoff_t0: sine m=33 vs Fourier B_m33: gain 7.79e-02 (exceeds u_grid); density gain 1.77e-02 (exceeds density u); negative mass 6.85e-01 → 2.29e-07; sine within u of full-256: True; speed-up vs full-256 rollout: 4.9×
- from_payoff_t0: sine m=33 vs Fourier A_m65: gain 1.29e-01 (exceeds u_grid); density gain 2.30e-02 (exceeds density u); negative mass 9.66e-01 → 2.29e-07; sine within u of full-256: True; speed-up vs full-256 rollout: 4.9×
- from_payoff_t0: sine m=33 vs Fourier B_m65: gain 7.11e-02 (exceeds u_grid); density gain 1.68e-02 (exceeds density u); negative mass 7.15e-01 → 2.29e-07; sine within u of full-256: True; speed-up vs full-256 rollout: 4.9×
- from_payoff_t0: sine m=65 vs Fourier A_m33: gain 1.21e-01 (exceeds u_grid); density gain 2.42e-02 (exceeds density u); negative mass 9.20e-01 → 0.00e+00; sine within u of full-256: True; speed-up vs full-256 rollout: 3.2×
- from_payoff_t0: sine m=65 vs Fourier B_m33: gain 7.79e-02 (exceeds u_grid); density gain 1.77e-02 (exceeds density u); negative mass 6.85e-01 → 0.00e+00; sine within u of full-256: True; speed-up vs full-256 rollout: 3.2×
- from_payoff_t0: sine m=65 vs Fourier A_m65: gain 1.29e-01 (exceeds u_grid); density gain 2.31e-02 (exceeds density u); negative mass 9.66e-01 → 0.00e+00; sine within u of full-256: True; speed-up vs full-256 rollout: 3.2×
- from_payoff_t0: sine m=65 vs Fourier B_m65: gain 7.11e-02 (exceeds u_grid); density gain 1.68e-02 (exceeds density u); negative mass 7.15e-01 → 0.00e+00; sine within u of full-256: True; speed-up vs full-256 rollout: 3.2×

## S_seed9 (S, {'b': 0.37405, 'c': 0.030157, 's': 0.028682, 'seed': 9})

u_grid = 3.63e-06; density u = 2.52e-06

| model | start | dof | price int. RMS | price vs BS | density L2 | neg. mass max | neg. mass T_max | inverse-vol invalid | butterfly viol. | stable | reduced abscissa | proj. err | rollout s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A_fourier_m33 | from_payoff_t0 | 33 | 1.06e-01 | – | 3.85e-02 | 1.26e+00 | 4.35e-01 | 0.209 | 0.186 | yes | -4.08e-14 | 7.83e-03 | 7.183e-03 |
| A_fourier_m33 | from_projected_t0.2 | 33 | 9.69e-02 | – | 6.85e-02 | 2.05e+01 | 5.12e-01 | 0.185 | 0.159 | yes | -4.08e-14 | 7.83e-03 | 1.477e-02 |
| A_fourier_m65 | from_payoff_t0 | 65 | 1.14e-01 | – | 3.60e-02 | 1.30e+00 | 4.71e-01 | 0.199 | 0.187 | yes | 3.93e-13 | 3.89e-03 | 6.566e-03 |
| A_fourier_m65 | from_projected_t0.2 | 65 | 1.04e-01 | – | 8.54e-02 | 4.53e+01 | 5.49e-01 | 0.166 | 0.154 | yes | 3.93e-13 | 3.89e-03 | 3.060e-02 |
| A_full256 | payoff_t0 | 256 | 7.31e-04 | – | 3.20e-04 | 1.50e-02 | 1.50e-02 | 0.139 | 0.048 | yes | – | – | 3.732e-02 |
| B_fourier_m33 | from_payoff_t0 | 33 | 4.37e-02 | – | 2.22e-02 | 8.70e-01 | 3.08e-01 | 0.523 | 0.513 | yes | 6.76e-15 | 3.10e-04 | 1.913e-02 |
| B_fourier_m33 | from_projected_t0.2 | 33 | 3.25e-02 | – | 2.07e-02 | 1.39e+00 | 3.53e-01 | 0.482 | 0.457 | yes | 6.76e-15 | 3.10e-04 | 1.537e-02 |
| B_fourier_m65 | from_payoff_t0 | 65 | 3.89e-02 | – | 2.04e-02 | 8.82e-01 | 3.30e-01 | 0.499 | 0.488 | yes | 3.86e-13 | 8.45e-05 | 6.736e-03 |
| B_fourier_m65 | from_projected_t0.2 | 65 | 2.86e-02 | – | 1.84e-02 | 1.40e+00 | 3.75e-01 | 0.456 | 0.441 | yes | 3.86e-13 | 8.45e-05 | 6.552e-03 |
| B_full256 | payoff_t0 | 256 | 3.63e-06 | – | 2.52e-06 | 3.95e-07 | 2.55e-08 | 0.076 | 0.000 | yes | – | – | 1.376e-01 |
| B_full512_reference |  | 512 | – | – | – | – | – | – | – | yes | – | – | 2.571e-01 |
| B_sine_m17 | from_payoff_t0 | 17 | 8.58e-06 | – | 1.27e-04 | 6.57e-05 | 0.00e+00 | 0.076 | 0.000 | yes | -3.32e-01 | 6.12e-06 | 1.577e-02 |
| B_sine_m17 | from_projected_t0.2 | 17 | 6.07e-06 | – | 1.16e-04 | 1.80e-04 | 0.00e+00 | 0.075 | 0.000 | yes | -3.32e-01 | 6.12e-06 | 1.574e-02 |
| B_sine_m33 | from_payoff_t0 | 33 | 3.79e-06 | – | 2.07e-05 | 1.84e-06 | 0.00e+00 | 0.075 | 0.000 | yes | -3.32e-01 | 3.34e-07 | 2.368e-02 |
| B_sine_m33 | from_projected_t0.2 | 33 | 3.62e-06 | – | 2.09e-05 | 1.86e-06 | 0.00e+00 | 0.075 | 0.000 | yes | -3.32e-01 | 3.34e-07 | 8.397e-03 |
| B_sine_m65 | from_payoff_t0 | 65 | 3.63e-06 | – | 4.34e-06 | 1.35e-06 | 0.00e+00 | 0.075 | 0.000 | yes | -3.32e-01 | 1.43e-08 | 3.862e-02 |
| B_sine_m65 | from_projected_t0.2 | 65 | 3.63e-06 | – | 4.34e-06 | 1.38e-06 | 0.00e+00 | 0.075 | 0.000 | yes | -3.32e-01 | 1.43e-08 | 4.057e-02 |

Uncertainty-aware claims (price interior RMS):

- from_projected_t0.2: sine m=17 vs Fourier A_m33: gain 9.69e-02 (exceeds u_grid); density gain 6.84e-02 (exceeds density u); negative mass 2.05e+01 → 1.80e-04; sine within u of full-256: True; speed-up vs full-256 rollout: 8.7×
- from_projected_t0.2: sine m=17 vs Fourier B_m33: gain 3.25e-02 (exceeds u_grid); density gain 2.06e-02 (exceeds density u); negative mass 1.39e+00 → 1.80e-04; sine within u of full-256: True; speed-up vs full-256 rollout: 8.7×
- from_projected_t0.2: sine m=17 vs Fourier A_m65: gain 1.04e-01 (exceeds u_grid); density gain 8.53e-02 (exceeds density u); negative mass 4.53e+01 → 1.80e-04; sine within u of full-256: True; speed-up vs full-256 rollout: 8.7×
- from_projected_t0.2: sine m=17 vs Fourier B_m65: gain 2.86e-02 (exceeds u_grid); density gain 1.82e-02 (exceeds density u); negative mass 1.40e+00 → 1.80e-04; sine within u of full-256: True; speed-up vs full-256 rollout: 8.7×
- from_projected_t0.2: sine m=33 vs Fourier A_m33: gain 9.69e-02 (exceeds u_grid); density gain 6.85e-02 (exceeds density u); negative mass 2.05e+01 → 1.86e-06; sine within u of full-256: True; speed-up vs full-256 rollout: 16.4×
- from_projected_t0.2: sine m=33 vs Fourier B_m33: gain 3.25e-02 (exceeds u_grid); density gain 2.07e-02 (exceeds density u); negative mass 1.39e+00 → 1.86e-06; sine within u of full-256: True; speed-up vs full-256 rollout: 16.4×
- from_projected_t0.2: sine m=33 vs Fourier A_m65: gain 1.04e-01 (exceeds u_grid); density gain 8.53e-02 (exceeds density u); negative mass 4.53e+01 → 1.86e-06; sine within u of full-256: True; speed-up vs full-256 rollout: 16.4×
- from_projected_t0.2: sine m=33 vs Fourier B_m65: gain 2.86e-02 (exceeds u_grid); density gain 1.83e-02 (exceeds density u); negative mass 1.40e+00 → 1.86e-06; sine within u of full-256: True; speed-up vs full-256 rollout: 16.4×
- from_projected_t0.2: sine m=65 vs Fourier A_m33: gain 9.69e-02 (exceeds u_grid); density gain 6.85e-02 (exceeds density u); negative mass 2.05e+01 → 1.38e-06; sine within u of full-256: True; speed-up vs full-256 rollout: 3.4×
- from_projected_t0.2: sine m=65 vs Fourier B_m33: gain 3.25e-02 (exceeds u_grid); density gain 2.07e-02 (exceeds density u); negative mass 1.39e+00 → 1.38e-06; sine within u of full-256: True; speed-up vs full-256 rollout: 3.4×
- from_projected_t0.2: sine m=65 vs Fourier A_m65: gain 1.04e-01 (exceeds u_grid); density gain 8.54e-02 (exceeds density u); negative mass 4.53e+01 → 1.38e-06; sine within u of full-256: True; speed-up vs full-256 rollout: 3.4×
- from_projected_t0.2: sine m=65 vs Fourier B_m65: gain 2.86e-02 (exceeds u_grid); density gain 1.83e-02 (exceeds density u); negative mass 1.40e+00 → 1.38e-06; sine within u of full-256: True; speed-up vs full-256 rollout: 3.4×
- from_payoff_t0: sine m=17 vs Fourier A_m33: gain 1.06e-01 (exceeds u_grid); density gain 3.84e-02 (exceeds density u); negative mass 1.26e+00 → 6.57e-05; sine within u of full-256: False; speed-up vs full-256 rollout: 8.7×
- from_payoff_t0: sine m=17 vs Fourier B_m33: gain 4.36e-02 (exceeds u_grid); density gain 2.20e-02 (exceeds density u); negative mass 8.70e-01 → 6.57e-05; sine within u of full-256: False; speed-up vs full-256 rollout: 8.7×
- from_payoff_t0: sine m=17 vs Fourier A_m65: gain 1.13e-01 (exceeds u_grid); density gain 3.58e-02 (exceeds density u); negative mass 1.30e+00 → 6.57e-05; sine within u of full-256: False; speed-up vs full-256 rollout: 8.7×
- from_payoff_t0: sine m=17 vs Fourier B_m65: gain 3.89e-02 (exceeds u_grid); density gain 2.03e-02 (exceeds density u); negative mass 8.82e-01 → 6.57e-05; sine within u of full-256: False; speed-up vs full-256 rollout: 8.7×
- from_payoff_t0: sine m=33 vs Fourier A_m33: gain 1.06e-01 (exceeds u_grid); density gain 3.85e-02 (exceeds density u); negative mass 1.26e+00 → 1.84e-06; sine within u of full-256: True; speed-up vs full-256 rollout: 5.8×
- from_payoff_t0: sine m=33 vs Fourier B_m33: gain 4.37e-02 (exceeds u_grid); density gain 2.21e-02 (exceeds density u); negative mass 8.70e-01 → 1.84e-06; sine within u of full-256: True; speed-up vs full-256 rollout: 5.8×
- from_payoff_t0: sine m=33 vs Fourier A_m65: gain 1.14e-01 (exceeds u_grid); density gain 3.60e-02 (exceeds density u); negative mass 1.30e+00 → 1.84e-06; sine within u of full-256: True; speed-up vs full-256 rollout: 5.8×
- from_payoff_t0: sine m=33 vs Fourier B_m65: gain 3.89e-02 (exceeds u_grid); density gain 2.04e-02 (exceeds density u); negative mass 8.82e-01 → 1.84e-06; sine within u of full-256: True; speed-up vs full-256 rollout: 5.8×
- from_payoff_t0: sine m=65 vs Fourier A_m33: gain 1.06e-01 (exceeds u_grid); density gain 3.85e-02 (exceeds density u); negative mass 1.26e+00 → 1.35e-06; sine within u of full-256: True; speed-up vs full-256 rollout: 3.6×
- from_payoff_t0: sine m=65 vs Fourier B_m33: gain 4.37e-02 (exceeds u_grid); density gain 2.22e-02 (exceeds density u); negative mass 8.70e-01 → 1.35e-06; sine within u of full-256: True; speed-up vs full-256 rollout: 3.6×
- from_payoff_t0: sine m=65 vs Fourier A_m65: gain 1.14e-01 (exceeds u_grid); density gain 3.60e-02 (exceeds density u); negative mass 1.30e+00 → 1.35e-06; sine within u of full-256: True; speed-up vs full-256 rollout: 3.6×
- from_payoff_t0: sine m=65 vs Fourier B_m65: gain 3.89e-02 (exceeds u_grid); density gain 2.04e-02 (exceeds density u); negative mass 8.82e-01 → 1.35e-06; sine within u of full-256: True; speed-up vs full-256 rollout: 3.6×

