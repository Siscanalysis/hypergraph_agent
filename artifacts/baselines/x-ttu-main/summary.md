# Study X: x-ttu-main

Setting `tt_u_k3`; runs used 50, incomplete 0.

## Selection

Best learner `mppo_rnd` by steps_late on the validation split at budget 1000000: `mppo` 95.083, `mppo_rnd` 93.125, `rppo` 99.368

## Tests

| Test | Role | a - b | Estimate [CI] | Units | Verdict |
|---|---|---|---|---|---|
| XU1 pooled - best learner, actions to first success | secondary | pooled - mppo_rnd | -700.367 [-853.434, -537.665] | 30 | holds |
| XU2 pooled - best learner, late actions per episode | secondary | pooled - mppo_rnd | -88.447 [-90.722, -85.675] | 30 | holds |
| XU3 best learner - random, actions to first success | secondary | mppo_rnd - random | +542.567 [+354.375, +718.100] | 30 | does not hold |
| XU4 best learner - random, late actions per episode | secondary | mppo_rnd - random | +4.175 [+0.042, +8.256] | 30 | does not hold |
| XU5 best learner, late minus early cost ratio | descriptive | mppo_rnd | +0.117 [-0.247, +0.497] | 30 | does not hold |

## Learning curves (test worlds; agents as reference)

| Arm | Budget | Late cost ratio | Late success | Late steps | First success steps | Units |
|---|---|---|---|---|---|---|
| mppo | 10000 | 14.531 | 0.061 | 97.7 | 902.5 | 30 |
| mppo | 100000 | 14.384 | 0.106 | 96.7 | 927.4 | 30 |
| mppo | 1000000 | 14.373 | 0.081 | 96.7 | 841.4 | 30 |
| mppo_rnd | 10000 | 14.468 | 0.092 | 97.3 | 853.9 | 30 |
| mppo_rnd | 100000 | 13.628 | 0.250 | 91.8 | 639.5 | 30 |
| mppo_rnd | 1000000 | 14.150 | 0.156 | 95.2 | 771.2 | 30 |
| oracle | agent | 1.000 | 1.000 | 6.8 | 6.8 | 30 |
| pooled | agent | 1.000 | 1.000 | 6.8 | 70.9 | 30 |
| random | agent | 13.544 | 0.217 | 91.0 | 228.7 | 30 |
| reference | agent | 1.000 | 1.000 | 6.8 | 6.8 | 30 |
| rppo | 10000 | 14.871 | 0.003 | 100.0 | 1177.8 | 30 |
| rppo | 100000 | 14.873 | 0.000 | 100.0 | 1200.0 | 30 |
| rppo | 1000000 | 14.873 | 0.000 | 100.0 | 1200.0 | 30 |

## Interactions and time

- training `mppo` seed 0: 1,000,000 interactions, 3342.04 s (299.2 steps/s)
- training `mppo` seed 1: 1,000,000 interactions, 3327.76 s (300.5 steps/s)
- training `rppo` seed 0: 1,000,000 interactions, 2121.89 s (471.3 steps/s)
- training `rppo` seed 1: 1,000,000 interactions, 2346.51 s (426.2 steps/s)
- training `mppo_rnd` seed 0: 1,000,000 interactions, 2087.11 s (479.1 steps/s)
- training `mppo_rnd` seed 1: 1,000,000 interactions, 1983.32 s (504.2 steps/s)
- agent `oracle`: 81 interactions per test world over its whole stream
- agent `pooled`: 144 interactions per test world over its whole stream
- agent `random`: 1068 interactions per test world over its whole stream
- agent `reference`: 81 interactions per test world over its whole stream
