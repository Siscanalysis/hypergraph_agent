Late half (episodes 9-16), `focused_sample - maximal` paired per world:

| Family | Kind | Worlds | Late ratios (sampler / maximal) | Mean difference [95% CI] | Sampler lower in | Sign test p | Reading | Break-even |
|---|---|---|---|---|---|---|---|---|
| D | one factor from F1 (primary) | 50 | 1.65 / 1.62 | 0.029 [-0.044, 0.098] | 20 of 50 | 0.2529 | inconclusive | 15 |
| DH | D with pools of 8 (headroom-matched depth) | 50 | 1.81 / 1.82 | -0.005 [-0.100, 0.085] | 26 of 50 | 0.8877 | inconclusive | 14 |
| A | one factor from F1 | 50 | 1.62 / 1.89 | -0.271 [-0.379, -0.162] | 35 of 50 | 0.0038 | advantage | 9 |
| P | one factor from F1 | 50 | 1.60 / 2.24 | -0.639 [-0.746, -0.535] | 49 of 50 | 0.0000 | advantage | 6 |
| I | one factor from F1 (positive control) | 50 | 1.42 / 1.84 | -0.417 [-0.497, -0.341] | 48 of 50 | 0.0000 | advantage | 6 |
| F2-D | one factor removed from F2 | 50 | 1.84 / 2.34 | -0.496 [-0.611, -0.377] | 42 of 50 | 0.0000 | advantage | 7 |
| F2-A | one factor removed from F2 (near-matched deep check) | 50 | 1.88 / 1.80 | 0.074 [-0.019, 0.163] | 18 of 50 | 0.0854 | inconclusive | never |
| F2-P | one factor removed from F2 | 50 | 1.92 / 1.65 | 0.263 [0.175, 0.354] | 13 of 50 | 0.0009 | reversed | never |
| F2 | F2 replica, slack 2 (control) | 50 | 2.09 / 1.87 | 0.215 [0.109, 0.319] | 14 of 50 | 0.0026 | reversed | never |
| F2S6 | F2, slack 6 | 50 | 2.10 / 1.87 | 0.226 [0.121, 0.336] | 14 of 50 | 0.0026 | reversed | 16 |
| F2S12 | F2, slack 12 | 50 | 2.09 / 1.87 | 0.216 [0.099, 0.337] | 17 of 50 | 0.0328 | reversed | 16 |
| DS6 | D, slack 6 | 50 | 1.61 / 1.62 | -0.008 [-0.077, 0.063] | 26 of 50 | 0.7754 | inconclusive | 12 |
| DS12 | D, slack 12 | 50 | 1.58 / 1.62 | -0.033 [-0.109, 0.042] | 28 of 50 | 0.4799 | inconclusive | 13 |

Difference-in-differences, late success difference and the late excess by outcome (pooled; failed episodes count their whole budget):

| Family | DiD [95% CI] | Late success difference [95% CI] | Late excess | from failed (n) | from solved | Attributed to | Solved: sampler / maximal | Failed at deadline after a contradiction |
|---|---|---|---|---|---|---|---|---|
| D | -0.257 [-0.316, -0.197] | -0.163 [-0.212, -0.117] | 0.038 | 0.080 (65 of 400) | -0.042 | failed episodes | 1.57 / 1.63 | 65 of 65 |
| DH | -0.280 [-0.343, -0.217] | -0.318 [-0.407, -0.235] | 0.010 | 0.138 (127 of 400) | -0.128 | failed episodes | 1.68 / 1.87 | 127 of 127 |
| A | -0.554 [-0.645, -0.468] | -0.015 [-0.030, -0.003] | -0.229 | 0.023 (6 of 400) | -0.251 | - | 1.60 / 1.86 | 6 of 6 |
| P | -0.779 [-0.861, -0.699] | -0.028 [-0.048, -0.010] | -0.613 | 0.024 (11 of 400) | -0.637 | - | 1.57 / 2.23 | 11 of 11 |
| I | -0.475 [-0.553, -0.403] | -0.013 [-0.022, -0.003] | -0.397 | 0.011 (5 of 400) | -0.408 | - | 1.40 / 1.82 | 5 of 5 |
| F2-D | -0.637 [-0.733, -0.542] | -0.050 [-0.077, -0.025] | -0.462 | 0.061 (20 of 400) | -0.523 | - | 1.78 / 2.34 | 20 of 20 |
| F2-A | -0.224 [-0.300, -0.155] | -0.410 [-0.500, -0.320] | 0.095 | 0.178 (164 of 400) | -0.083 | failed episodes | 1.72 / 1.88 | 164 of 164 |
| F2-P | -0.317 [-0.396, -0.239] | -0.220 [-0.270, -0.170] | 0.277 | 0.170 (88 of 400) | 0.107 | failed episodes | 1.81 / 1.67 | 88 of 88 |
| F2 | -0.278 [-0.361, -0.199] | -0.407 [-0.490, -0.328] | 0.222 | 0.257 (163 of 400) | -0.035 | failed episodes | 1.86 / 1.92 | 163 of 163 |
| F2S6 | -0.435 [-0.511, -0.356] | -0.190 [-0.255, -0.133] | 0.235 | 0.189 (76 of 400) | 0.047 | failed episodes | 1.96 / 1.90 | 76 of 76 |
| F2S12 | -0.559 [-0.646, -0.474] | -0.062 [-0.090, -0.037] | 0.221 | 0.099 (25 of 400) | 0.121 | solved episodes | 2.01 / 1.88 | 25 of 25 |
| DS6 | -0.394 [-0.482, -0.304] | -0.035 [-0.052, -0.020] | -0.000 | 0.030 (14 of 400) | -0.030 | - | 1.58 / 1.61 | 14 of 14 |
| DS12 | -0.431 [-0.495, -0.366] | -0.013 [-0.022, -0.003] | -0.024 | 0.018 (5 of 400) | -0.042 | - | 1.56 / 1.61 | 5 of 5 |

Pooled cost ratios (steps / optimal steps), success and omitted candidates:

| Family | Arm | Runs | Episodes 1-8 | Episodes 9-16 | All | Late by seed | Success (late) | Omitted (late) | Bounds from |
|---|---|---|---|---|---|---|---|---|---|
| D | `focused_sample` | 5 | 1.91 | 1.64 | 1.78 | 1.71 / 1.57 / 1.64 / 1.63 / 1.65 | 0.72 (0.84) | 0.61 | D |
| D | `maximal` | 5 | 1.62 | 1.61 | 1.61 | 1.55 / 1.69 / 1.57 / 1.61 / 1.61 | 1.00 (1.00) | 0.00 | D |
| D | `reference` | 5 | 1.00 | 1.00 | 1.00 | 1.00 / 1.00 / 1.00 / 1.00 / 1.00 | 1.00 (1.00) | 0.00 | D |
| DH | `focused_sample` | 5 | 2.08 | 1.81 | 1.95 | 1.75 / 1.92 / 1.69 / 1.74 / 1.99 | 0.52 (0.68) | 0.69 | DH |
| DH | `maximal` | 5 | 1.80 | 1.80 | 1.80 | 1.72 / 1.86 / 1.75 / 1.87 / 1.83 | 1.00 (1.00) | 0.00 | DH |
| DH | `reference` | 5 | 1.00 | 1.00 | 1.00 | 1.00 / 1.00 / 1.00 / 1.00 / 1.00 | 1.00 (1.00) | 0.00 | DH |
| A | `focused_sample` | 5 | 2.16 | 1.62 | 1.89 | 1.63 / 1.53 / 1.72 / 1.59 / 1.66 | 0.94 (0.98) | 0.61 | A |
| A | `maximal` | 5 | 1.86 | 1.85 | 1.85 | 1.84 / 1.84 / 1.94 / 1.82 / 1.83 | 1.00 (1.00) | 0.00 | A |
| A | `reference` | 5 | 1.00 | 1.00 | 1.00 | 1.00 / 1.00 / 1.00 / 1.00 / 1.00 | 1.00 (1.00) | 0.00 | A |
| P | `focused_sample` | 5 | 2.34 | 1.60 | 1.98 | 1.59 / 1.73 / 1.63 / 1.47 / 1.58 | 0.90 (0.97) | 0.70 | P |
| P | `maximal` | 5 | 2.18 | 2.21 | 2.20 | 2.30 / 2.17 / 2.22 / 2.30 / 2.08 | 1.00 (1.00) | 0.00 | P |
| P | `reference` | 5 | 1.00 | 1.00 | 1.00 | 1.00 / 1.00 / 1.00 / 1.00 / 1.00 | 1.00 (1.00) | 0.00 | P |
| I | `focused_sample` | 5 | 1.91 | 1.42 | 1.66 | 1.45 / 1.51 / 1.33 / 1.40 / 1.41 | 0.94 (0.99) | 0.62 | I |
| I | `maximal` | 5 | 1.83 | 1.81 | 1.82 | 1.92 / 1.77 / 1.71 / 1.85 / 1.84 | 1.00 (1.00) | 0.00 | I |
| I | `reference` | 5 | 1.00 | 1.00 | 1.00 | 1.00 / 1.00 / 1.00 / 1.00 / 1.00 | 1.00 (1.00) | 0.00 | I |
| F2-D | `focused_sample` | 5 | 2.46 | 1.85 | 2.16 | 1.77 / 1.73 / 1.94 / 1.82 / 1.99 | 0.90 (0.95) | 0.70 | F2-D |
| F2-D | `maximal` | 5 | 2.29 | 2.31 | 2.30 | 2.23 / 2.36 / 2.23 / 2.41 / 2.35 | 1.00 (1.00) | 0.00 | F2-D |
| F2-D | `reference` | 5 | 1.00 | 1.00 | 1.00 | 1.00 / 1.00 / 1.00 / 1.00 / 1.00 | 1.00 (1.00) | 0.00 | F2-D |
| F2-A | `focused_sample` | 5 | 2.06 | 1.87 | 1.97 | 1.85 / 2.03 / 1.77 / 1.86 / 1.82 | 0.48 (0.59) | 0.68 | F2-A |
| F2-A | `maximal` | 5 | 1.76 | 1.77 | 1.77 | 1.75 / 1.73 / 1.82 / 1.83 / 1.74 | 1.00 (1.00) | 0.00 | F2-A |
| F2-A | `reference` | 5 | 1.00 | 1.00 | 1.00 | 1.00 / 1.00 / 1.00 / 1.00 / 1.00 | 1.00 (1.00) | 0.00 | F2-A |
| F2-P | `focused_sample` | 5 | 2.23 | 1.91 | 2.07 | 1.97 / 1.88 / 1.88 / 1.95 / 1.86 | 0.64 (0.78) | 0.60 | F2-P |
| F2-P | `maximal` | 5 | 1.65 | 1.63 | 1.64 | 1.59 / 1.65 / 1.63 / 1.67 / 1.64 | 1.00 (1.00) | 0.00 | F2-P |
| F2-P | `reference` | 5 | 1.00 | 1.00 | 1.00 | 1.00 / 1.00 / 1.00 / 1.00 / 1.00 | 1.00 (1.00) | 0.00 | F2-P |
| F2 | `focused_sample` | 5 | 2.35 | 2.08 | 2.21 | 2.20 / 2.03 / 1.91 / 2.18 / 2.09 | 0.47 (0.59) | 0.69 | F2 |
| F2 | `maximal` | 5 | 1.86 | 1.86 | 1.86 | 1.84 / 1.79 / 1.88 / 1.91 / 1.88 | 1.00 (1.00) | 0.00 | F2 |
| F2 | `reference` | 5 | 1.00 | 1.00 | 1.00 | 1.00 / 1.00 / 1.00 / 1.00 / 1.00 | 1.00 (1.00) | 0.00 | F2 |
| F2S6 | `focused_sample` | 5 | 2.52 | 2.09 | 2.31 | 2.11 / 1.93 / 2.15 / 2.12 / 2.17 | 0.69 (0.81) | 0.69 | F2S6 |
| F2S6 | `maximal` | 5 | 1.86 | 1.86 | 1.86 | 1.84 / 1.79 / 1.88 / 1.91 / 1.88 | 1.00 (1.00) | 0.00 | F2 |
| F2S6 | `reference` | 5 | 1.00 | 1.00 | 1.00 | 1.00 / 1.00 / 1.00 / 1.00 / 1.00 | 1.00 (1.00) | 0.00 | F2 |
| F2S12 | `focused_sample` | 5 | 2.64 | 2.08 | 2.36 | 2.13 / 1.99 / 2.12 / 2.11 / 2.05 | 0.88 (0.94) | 0.69 | F2S12 |
| F2S12 | `maximal` | 5 | 1.86 | 1.86 | 1.86 | 1.84 / 1.79 / 1.88 / 1.91 / 1.88 | 1.00 (1.00) | 0.00 | F2 |
| F2S12 | `reference` | 5 | 1.00 | 1.00 | 1.00 | 1.00 / 1.00 / 1.00 / 1.00 / 1.00 | 1.00 (1.00) | 0.00 | F2 |
| DS6 | `focused_sample` | 5 | 2.01 | 1.60 | 1.81 | 1.62 / 1.63 / 1.58 / 1.61 / 1.58 | 0.86 (0.96) | 0.61 | DS6 |
| DS6 | `maximal` | 5 | 1.62 | 1.61 | 1.61 | 1.55 / 1.69 / 1.57 / 1.61 / 1.61 | 1.00 (1.00) | 0.00 | D |
| DS6 | `reference` | 5 | 1.00 | 1.00 | 1.00 | 1.00 / 1.00 / 1.00 / 1.00 / 1.00 | 1.00 (1.00) | 0.00 | D |
| DS12 | `focused_sample` | 5 | 2.02 | 1.58 | 1.80 | 1.62 / 1.57 / 1.53 / 1.60 / 1.59 | 0.97 (0.99) | 0.61 | DS12 |
| DS12 | `maximal` | 5 | 1.62 | 1.61 | 1.61 | 1.55 / 1.69 / 1.57 / 1.61 / 1.61 | 1.00 (1.00) | 0.00 | D |
| DS12 | `reference` | 5 | 1.00 | 1.00 | 1.00 | 1.00 / 1.00 / 1.00 / 1.00 / 1.00 | 1.00 (1.00) | 0.00 | D |

Late failure of `focused_sample` by goal depth:

| Family | depth 1 | depth 2 | depth 3 | depth 4 | depth 5 |
|---|---|---|---|---|---|
| D | - | - | 0.06 (n=128) | 0.18 (n=138) | 0.24 (n=134) |
| DH | - | - | 0.16 (n=140) | 0.34 (n=118) | 0.46 (n=142) |
| A | 0.00 (n=134) | 0.01 (n=133) | 0.04 (n=133) | - | - |
| P | 0.00 (n=136) | 0.01 (n=136) | 0.08 (n=128) | - | - |
| I | 0.00 (n=108) | 0.01 (n=149) | 0.03 (n=143) | - | - |
| F2-D | 0.00 (n=129) | 0.03 (n=146) | 0.12 (n=125) | - | - |
| F2-A | - | - | 0.28 (n=134) | 0.38 (n=145) | 0.59 (n=121) |
| F2-P | - | - | 0.08 (n=140) | 0.16 (n=133) | 0.44 (n=127) |
| F2 | - | - | 0.28 (n=149) | 0.39 (n=132) | 0.59 (n=119) |
| F2S6 | - | - | 0.08 (n=149) | 0.17 (n=132) | 0.34 (n=119) |
| F2S12 | - | - | 0.02 (n=149) | 0.01 (n=132) | 0.18 (n=119) |
| DS6 | - | - | 0.01 (n=128) | 0.04 (n=138) | 0.05 (n=134) |
| DS12 | - | - | 0.00 (n=128) | 0.01 (n=138) | 0.02 (n=134) |

Slack contrasts, paired per world (looser minus base; same worlds and tasks):

| Contrast | Worlds | Change of the late difference [95% CI] | Lower in | Reading | Change of the early difference | Change of late success |
|---|---|---|---|---|---|---|
| F2S6 vs F2 | 50 | 0.010 [-0.091, 0.117] | 23 of 50 | inconclusive | 0.168 [0.119, 0.215] | 0.217 [0.140, 0.295] |
| F2S12 vs F2 | 50 | 0.001 [-0.095, 0.099] | 26 of 50 | inconclusive | 0.282 [0.203, 0.359] | 0.345 [0.278, 0.417] |
| DS6 vs D | 50 | -0.036 [-0.098, 0.028] | 30 of 50 | inconclusive | 0.100 [0.037, 0.164] | 0.128 [0.087, 0.173] |
| DS12 vs D | 50 | -0.062 [-0.133, 0.009] | 30 of 50 | inconclusive | 0.112 [0.048, 0.179] | 0.150 [0.105, 0.200] |

Primary family D: inconclusive (study R criterion: fail).
Depth against headroom: D inconclusive, DH inconclusive (F2-A inconclusive): D not reversed: no reversal to attribute.
DS6 uses the bounds of D: maximal, reference (maximal: 800 shared, 0 mismatched, 0 missing episodes; reference: 800 shared, 0 mismatched, 0 missing episodes).
DS12 uses the bounds of D: maximal, reference (maximal: 800 shared, 0 mismatched, 0 missing episodes; reference: 800 shared, 0 mismatched, 0 missing episodes).
F2S6 uses the bounds of F2: maximal, reference (maximal: 800 shared, 0 mismatched, 0 missing episodes; reference: 800 shared, 0 mismatched, 0 missing episodes).
F2S12 uses the bounds of F2: maximal, reference (maximal: 800 shared, 0 mismatched, 0 missing episodes; reference: 800 shared, 0 mismatched, 0 missing episodes).
