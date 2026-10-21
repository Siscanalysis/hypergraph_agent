# Study X: x-ttl-main

Setting `tt_l2_d2`; runs used 70, incomplete 0.

## Selection

Best learner `mppo_rnd` by steps_late on the validation split at budget 1000000: `mppo` 100.000, `mppo_rnd` 96.174, `rppo` 99.500

## Tests

| Test | Role | a - b | Estimate [CI] | Units | Verdict |
|---|---|---|---|---|---|
| XL1 promote - best learner, late actions per episode | secondary | promote - mppo_rnd | -87.969 [-89.008, -86.792] | 32 | holds |
| XL2 remember - best learner, late actions per episode | secondary | remember - mppo_rnd | -76.542 [-81.758, -70.739] | 32 | holds |
| XL3 best learner - random, late actions per episode | secondary | mppo_rnd - random | +5.307 [+2.591, +8.318] | 32 | does not hold |
| XL4 best learner, late minus early cost ratio | descriptive | mppo_rnd | +0.183 [-0.025, +0.406] | 32 | does not hold |

## Learning curves (test worlds; agents as reference)

| Arm | Budget | Late cost ratio | Late success | Late steps | First success steps | Units |
|---|---|---|---|---|---|---|
| mppo | 10000 | 14.245 | 0.010 | 99.7 | 1137.1 | 32 |
| mppo | 100000 | 14.275 | 0.003 | 99.9 | 1181.9 | 32 |
| mppo | 1000000 | 14.279 | 0.003 | 100.0 | 1191.9 | 32 |
| mppo_rnd | 10000 | 13.913 | 0.060 | 97.4 | 918.7 | 32 |
| mppo_rnd | 100000 | 13.889 | 0.052 | 97.2 | 988.4 | 32 |
| mppo_rnd | 1000000 | 14.070 | 0.044 | 98.5 | 924.7 | 32 |
| oracle_library | agent | 1.268 | 1.000 | 8.9 | 8.6 | 32 |
| promote | agent | 1.503 | 1.000 | 10.5 | 33.0 | 32 |
| random | agent | 13.312 | 0.135 | 93.2 | 561.8 | 32 |
| reference | agent | 1.000 | 1.000 | 7.0 | 7.0 | 32 |
| remember | agent | 3.135 | 0.885 | 21.9 | 195.5 | 32 |
| rppo | 10000 | 13.922 | 0.062 | 97.5 | 941.3 | 32 |
| rppo | 100000 | 14.029 | 0.047 | 98.2 | 940.9 | 32 |
| rppo | 1000000 | 14.106 | 0.029 | 98.7 | 1092.2 | 32 |

## Interactions and time

- training `mppo` seed 0: 1,000,000 interactions, 2694.23 s (371.2 steps/s)
- training `mppo` seed 1: 1,000,000 interactions, 2691.11 s (371.6 steps/s)
- training `rppo` seed 0: 1,000,000 interactions, 3246.82 s (308.0 steps/s)
- training `rppo` seed 1: 1,000,000 interactions, 3363.08 s (297.3 steps/s)
- training `mppo_rnd` seed 0: 1,000,000 interactions, 1854.6 s (539.2 steps/s)
- training `mppo_rnd` seed 1: 1,000,000 interactions, 1853.29 s (539.6 steps/s)
- agent `oracle_library`: 106 interactions per test world over its whole stream
- agent `promote`: 161 interactions per test world over its whole stream
- agent `random`: 1126 interactions per test world over its whole stream
- agent `reference`: 84 interactions per test world over its whole stream
- agent `remember`: 423 interactions per test world over its whole stream
