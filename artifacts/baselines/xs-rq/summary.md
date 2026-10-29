# Study XS: xs-rq

Members: `s2` (xs-rq-s2), `s12` (xs-rq-s12).

## Selection (validation worlds only)

Best learner `s12_sparse` (member `s12`) by cost_ratio_late minus random at the learner's slack at budget 2500000: `s2_sparse` -0.006, `s2_shaped` -0.020, `s12_sparse` -0.376, `s12_shaped` -0.241

## Tests

| Test | Role | a - b | Estimate [CI] | Per training seed | Units | Verdict |
|---|---|---|---|---|---|---|
| XS1 best learner - random (its slack), late cost ratio | primary | s12_sparse - random.s12 | -0.332 [-0.432, -0.231] | s0 -0.297, s1 -0.367 | 50 | holds |
| XS1s best learner - random (its slack), late success | secondary | s12_sparse - random.s12 | +0.279 [+0.223, +0.335] | s0 +0.278, s1 +0.280 | 50 | holds |
| XS2 focused_sample (its slack) - best learner, late cost ratio | secondary | focused_sample.s12 - s12_sparse | -2.585 [-2.759, -2.415] | s0 -2.620, s1 -2.551 (seeds of b) | 50 | holds |
| XS3 best learner - maximal (its slack), late cost ratio | secondary | s12_sparse - maximal.s12 | +2.099 [+1.976, +2.224] | s0 +2.134, s1 +2.064 | 50 | does not hold |
| XS4 best learner, late minus early cost ratio (within-world improvement) | secondary | s12_sparse | +0.018 [-0.099, +0.129] | s0 +0.045, s1 -0.010 | 50 | does not hold |
| XSF1 shaping at slack 2: shaped - sparse, late cost ratio | factor | s2_shaped - s2_sparse | +0.012 [-0.020, +0.053] | s0 +0.031, s1 -0.008 (seeds of b) | 50 | does not hold |
| XSF2 shaping at slack 12: shaped - sparse, late cost ratio | factor | s12_shaped - s12_sparse | +0.054 [-0.023, +0.126] | s0 +0.019, s1 +0.089 (seeds of b) | 50 | does not hold |
| XSF3 slack, sparse: (s12 - random s12) - (s2 - random s2), late cost ratio | factor | s12_sparse~random - s2_sparse~random | -0.311 [-0.420, -0.199] | s0 -0.291, s1 -0.330 (seeds of b) | 50 | holds |
| XSF4 slack, shaped: (s12 - random s12) - (s2 - random s2), late cost ratio | factor | s12_shaped~random - s2_shaped~random | -0.269 [-0.379, -0.164] | s1 -0.251, s0 -0.287 (seeds of b) | 50 | holds |
| XSF5 training length, best learner: largest - first budget, late cost ratio | factor | s12_sparse - s12_sparse#300000 | -0.262 [-0.352, -0.176] | s0 -0.268, s1 -0.256 (seeds of b) | 50 | holds |
| XSF6 training length, best learner: largest - middle budget, late cost ratio | factor | s12_sparse - s12_sparse#1000000 | -0.068 [-0.137, +0.006] | s0 -0.113, s1 -0.023 (seeds of b) | 50 | does not hold |
| XSD1 s2_sparse - random (slack 2) | descriptive | s2_sparse~random | -0.021 [-0.062, +0.016] | s0 -0.041, s1 -0.002 | 50 | does not hold |
| XSD2 s2_shaped - random (slack 2) | descriptive | s2_shaped~random | -0.009 [-0.042, +0.025] | s0 +0.009, s1 -0.028 | 50 | does not hold |
| XSD3 s12_sparse - random (slack 12) | descriptive | s12_sparse~random | -0.332 [-0.432, -0.231] | s0 -0.297, s1 -0.367 | 50 | holds |
| XSD4 s12_shaped - random (slack 12) | descriptive | s12_shaped~random | -0.278 [-0.377, -0.180] | s0 -0.222, s1 -0.335 | 50 | holds |
| XSD5 focused_sample (its slack), late minus early cost ratio | descriptive | focused_sample.s12 | -0.606 [-0.671, -0.543] | - | 50 | holds |

## Learning curves (test worlds)

| Arm | Budget | Late cost ratio | Late success | Late steps | Units |
|---|---|---|---|---|---|
| focused_sample.s12 | agent | 1.422 | 1.000 | 9.1 | 50 |
| focused_sample.s2 | agent | 1.434 | 0.995 | 9.2 | 50 |
| maximal.s12 | agent | 1.908 | 1.000 | 12.1 | 50 |
| maximal.s2 | agent | 1.908 | 1.000 | 12.1 | 50 |
| random.s12 | agent | 4.340 | 0.203 | 27.4 | 50 |
| random.s2 | agent | 2.981 | 0.080 | 18.8 | 50 |
| reference.s12 | agent | 1.000 | 1.000 | 6.4 | 50 |
| reference.s2 | agent | 1.000 | 1.000 | 6.4 | 50 |
| s12_shaped | 300000 | 4.290 | 0.241 | 27.1 | 50 |
| s12_shaped | 1000000 | 4.100 | 0.384 | 25.9 | 50 |
| s12_shaped | 2500000 | 4.061 | 0.436 | 25.6 | 50 |
| s12_sparse | 300000 | 4.269 | 0.250 | 26.8 | 50 |
| s12_sparse | 1000000 | 4.075 | 0.384 | 25.7 | 50 |
| s12_sparse | 2500000 | 4.007 | 0.481 | 25.2 | 50 |
| s2_shaped | 300000 | 2.963 | 0.104 | 18.7 | 50 |
| s2_shaped | 1000000 | 2.961 | 0.144 | 18.6 | 50 |
| s2_shaped | 2500000 | 2.971 | 0.119 | 18.7 | 50 |
| s2_sparse | 300000 | 2.998 | 0.072 | 18.9 | 50 |
| s2_sparse | 1000000 | 3.002 | 0.083 | 18.9 | 50 |
| s2_sparse | 2500000 | 2.959 | 0.125 | 18.6 | 50 |

## Training-task success between checkpoints (both training seeds)

| Learner | Interactions | Success rate | Invalid choices per interaction |
|---|---|---|---|
| s12_shaped | 0-300000 | 0.253 | 0.000 |
| s12_shaped | 300000-1000000 | 0.413 | 0.000 |
| s12_shaped | 1000000-2500000 | 0.445 | 0.000 |
| s12_sparse | 0-300000 | 0.243 | 0.000 |
| s12_sparse | 300000-1000000 | 0.426 | 0.000 |
| s12_sparse | 1000000-2500000 | 0.463 | 0.000 |
| s2_shaped | 0-300000 | 0.074 | 0.000 |
| s2_shaped | 300000-1000000 | 0.124 | 0.000 |
| s2_shaped | 1000000-2500000 | 0.116 | 0.000 |
| s2_sparse | 0-300000 | 0.056 | 0.000 |
| s2_sparse | 300000-1000000 | 0.104 | 0.000 |
| s2_sparse | 1000000-2500000 | 0.109 | 0.000 |
