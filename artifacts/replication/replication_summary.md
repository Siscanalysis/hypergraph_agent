Late half (episodes 9-16), paired per unit:

| Cell | Role | Contrast | Units | Late ratios (a / b) | Mean difference [95% CI] | a lower in | Sign test p | Verdict |
|---|---|---|---|---|---|---|---|---|
| F1 eps=0 | primary | `focused_sample - maximal` | 50 | 1.51 / 1.87 | -0.357 [-0.449, -0.270] | 42 of 50 | 0.0000 | pass |
| F1 eps=0 | primary | `focused_sample - random_omit` | 50 | 1.51 / 2.46 | -0.947 [-1.036, -0.865] | 50 of 50 | 0.0000 | pass |
| F1 eps=0.1 | secondary | `focused_sample - maximal` | 50 | 1.76 / 2.08 | -0.311 [-0.408, -0.218] | 37 of 50 | 0.0005 | pass |
| F1 eps=0.1 | secondary | `focused_sample - random_omit` | 50 | 1.76 / 2.53 | -0.764 [-0.862, -0.667] | 50 of 50 | 0.0000 | pass |
| F2 eps=0 | secondary | `focused_sample - maximal` | 50 | 2.05 / 1.85 | 0.202 [0.120, 0.280] | 13 of 50 | 0.0014 | fail |
| F2 eps=0 | secondary | `focused_sample - random_omit` | 50 | 2.05 / 2.46 | -0.406 [-0.505, -0.314] | 47 of 50 | 0.0000 | pass |
| F2 eps=0.1 | secondary | `focused_sample - maximal` | 50 | 2.35 / 2.10 | 0.248 [0.195, 0.297] | 5 of 50 | 0.0000 | fail |
| F2 eps=0.1 | secondary | `focused_sample - random_omit` | 50 | 2.35 / 2.48 | -0.126 [-0.185, -0.072] | 37 of 50 | 0.0000 | pass |
| F0 eps=0 | control | `focused_sample - maximal` | 50 | 2.27 / 1.85 | 0.414 [0.375, 0.452] | 1 of 50 | 0.0000 | descriptive |
| F0 eps=0.1 | control | `focused_sample - maximal` | 50 | 2.39 / 2.08 | 0.313 [0.271, 0.355] | 1 of 50 | 0.0000 | descriptive |

Difference-in-differences (late minus early paired difference) and late success difference:

| Cell | Contrast | D [95% CI] | D below 0 | Late success difference [95% CI] |
|---|---|---|---|---|
| F1 eps=0 | `focused_sample - maximal` | -0.571 [-0.655, -0.496] | pass | -0.005 [-0.013, 0.000] |
| F1 eps=0 | `focused_sample - random_omit` | -0.554 [-0.656, -0.456] | pass | 0.335 [0.278, 0.393] |
| F1 eps=0.1 | `focused_sample - maximal` | -0.503 [-0.596, -0.411] | pass | -0.058 [-0.098, -0.020] |
| F1 eps=0.1 | `focused_sample - random_omit` | -0.467 [-0.567, -0.371] | pass | 0.360 [0.307, 0.415] |
| F2 eps=0 | `focused_sample - maximal` | -0.298 [-0.385, -0.219] | pass | -0.388 [-0.465, -0.315] |
| F2 eps=0 | `focused_sample - random_omit` | -0.324 [-0.412, -0.243] | pass | 0.388 [0.302, 0.468] |
| F2 eps=0.1 | `focused_sample - maximal` | -0.059 [-0.114, -0.006] | pass | -0.522 [-0.583, -0.465] |
| F2 eps=0.1 | `focused_sample - random_omit` | -0.081 [-0.132, -0.030] | pass | 0.165 [0.105, 0.225] |
| F0 eps=0 | `focused_sample - maximal` | -0.039 [-0.100, 0.019] | fail | -0.207 [-0.250, -0.165] |
| F0 eps=0.1 | `focused_sample - maximal` | -0.051 [-0.125, 0.020] | fail | -0.290 [-0.340, -0.240] |

Pooled cost ratios (steps / optimal steps), success and omitted candidates:

| Cell | Arm | Runs | Episodes 1-8 | Episodes 9-16 | All | Success (late) | Omitted (late) |
|---|---|---|---|---|---|---|---|
| F1 eps=0 | `focused_sample` | 5 | 2.04 | 1.52 | 1.78 | 0.92 (0.99) | 0.61 |
| F1 eps=0 | `maximal` | 5 | 1.82 | 1.84 | 1.83 | 1.00 (1.00) | 0.00 |
| F1 eps=0 | `random_omit` | 5 | 2.43 | 2.44 | 2.43 | 0.64 (0.66) | 0.65 |
| F1 eps=0 | `reference` | 5 | 1.00 | 1.00 | 1.00 | 1.00 (1.00) | 0.00 |
| F1 eps=0.1 | `focused_sample` | 5 | 2.23 | 1.77 | 2.00 | 0.84 (0.91) | 0.61 |
| F1 eps=0.1 | `maximal` | 5 | 2.03 | 2.05 | 2.04 | 0.97 (0.97) | 0.00 |
| F1 eps=0.1 | `random_omit` | 5 | 2.51 | 2.51 | 2.51 | 0.56 (0.55) | 0.65 |
| F1 eps=0.1 | `reference` | 5 | 1.10 | 1.09 | 1.10 | 1.00 (1.00) | 0.00 |
| F2 eps=0 | `focused_sample` | 5 | 2.33 | 2.05 | 2.19 | 0.50 (0.61) | 0.69 |
| F2 eps=0 | `maximal` | 5 | 1.84 | 1.84 | 1.84 | 1.00 (1.00) | 0.00 |
| F2 eps=0 | `random_omit` | 5 | 2.41 | 2.43 | 2.42 | 0.24 (0.23) | 0.69 |
| F2 eps=0 | `reference` | 5 | 1.00 | 1.00 | 1.00 | 1.00 (1.00) | 0.00 |
| F2 eps=0.1 | `focused_sample` | 5 | 2.41 | 2.33 | 2.37 | 0.28 (0.34) | 0.69 |
| F2 eps=0.1 | `maximal` | 5 | 2.10 | 2.09 | 2.09 | 0.86 (0.86) | 0.00 |
| F2 eps=0.1 | `random_omit` | 5 | 2.44 | 2.45 | 2.45 | 0.17 (0.18) | 0.69 |
| F2 eps=0.1 | `reference` | 5 | 1.10 | 1.10 | 1.10 | 1.00 (1.00) | 0.00 |
| F0 eps=0 | `focused_sample` | 5 | 2.29 | 2.26 | 2.27 | 0.79 (0.79) | 0.61 |
| F0 eps=0 | `maximal` | 5 | 1.83 | 1.84 | 1.84 | 1.00 (1.00) | 0.00 |
| F0 eps=0 | `reference` | 5 | 1.00 | 1.00 | 1.00 | 1.00 (1.00) | 0.00 |
| F0 eps=0.1 | `focused_sample` | 5 | 2.42 | 2.39 | 2.40 | 0.67 (0.69) | 0.61 |
| F0 eps=0.1 | `maximal` | 5 | 2.05 | 2.07 | 2.06 | 0.98 (0.97) | 0.00 |
| F0 eps=0.1 | `reference` | 5 | 1.10 | 1.09 | 1.10 | 1.00 (1.00) | 0.00 |

Break-even episode (descriptive): F1 eps=0 9; F1 eps=0.1 9; F2 eps=0 never; F2 eps=0.1 never; F0 eps=0 never; F0 eps=0.1 never.
Amortization check at eps=0 (pre-registered): F1 D -0.571 [-0.655, -0.496], F0 D -0.039 [-0.100, 0.019]; supported: yes.
Amortization check at eps=0.1 (reported): F1 D -0.503 [-0.596, -0.411], F0 D -0.051 [-0.125, 0.020]; supported: yes.
Primary cell F1 eps=0: pass. General replication (every F1 and F2 cell passes): no.
