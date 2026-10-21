# Study X: x-rq-main

Setting `rq_f1`; runs used 60, incomplete 0.

## Selection

Best learner `mppo` by cost_ratio_late on the validation split at budget 1000000: `mppo` 2.903, `mppo_rnd` 2.910, `ppo` 2.972, `rppo` 2.972

## Tests

| Test | Role | a - b | Estimate [CI] | Units | Verdict |
|---|---|---|---|---|---|
| X1 focused_sample - best learner, late cost ratio | primary | focused_sample - mppo | -1.437 [-1.567, -1.312] | 50 | holds |
| X1s focused_sample - best learner, late success | secondary | focused_sample - mppo | +0.887 [+0.856, +0.917] | 50 | holds |
| X2 maximal - best learner, late cost ratio | secondary | maximal - mppo | -1.067 [-1.156, -0.980] | 50 | holds |
| X3 best learner - random, late cost ratio | secondary | mppo - random | -0.021 [-0.051, +0.006] | 50 | does not hold |
| X4 best learner, late minus early cost ratio (within-world improvement) | descriptive | mppo | +0.048 [-0.042, +0.137] | 50 | does not hold |
| X4r focused_sample, late minus early cost ratio | descriptive | focused_sample | -0.503 [-0.575, -0.432] | 50 | holds |

## Learning curves (test worlds; agents as reference)

| Arm | Budget | Late cost ratio | Late success | Late steps | First success steps | Units |
|---|---|---|---|---|---|---|
| focused_sample | agent | 1.517 | 0.993 | 10.0 | 17.0 | 50 |
| maximal | agent | 1.886 | 1.000 | 12.2 | 11.7 | 50 |
| mppo | 10000 | 3.035 | 0.000 | 19.5 | 312.0 | 50 |
| mppo | 100000 | 3.012 | 0.035 | 19.4 | 263.4 | 50 |
| mppo | 1000000 | 2.953 | 0.105 | 19.0 | 180.8 | 50 |
| mppo_rnd | 10000 | 3.028 | 0.013 | 19.5 | 269.8 | 50 |
| mppo_rnd | 100000 | 3.013 | 0.031 | 19.4 | 257.9 | 50 |
| mppo_rnd | 1000000 | 2.958 | 0.156 | 19.1 | 131.5 | 50 |
| ppo | 10000 | 3.035 | 0.000 | 19.5 | 312.0 | 50 |
| ppo | 100000 | 3.035 | 0.000 | 19.5 | 312.0 | 50 |
| ppo | 1000000 | 3.035 | 0.000 | 19.5 | 309.5 | 50 |
| random | agent | 2.975 | 0.092 | 19.1 | 190.8 | 50 |
| reference | agent | 1.000 | 1.000 | 6.6 | 6.3 | 50 |
| rppo | 10000 | 3.035 | 0.000 | 19.5 | 312.0 | 50 |
| rppo | 100000 | 3.035 | 0.000 | 19.5 | 312.0 | 50 |
| rppo | 1000000 | 3.035 | 0.000 | 19.5 | 312.0 | 50 |

## Interactions and time

- training `mppo` seed 0: 1,000,000 interactions, 6718.02 s (148.9 steps/s)
- training `mppo` seed 1: 1,000,000 interactions, 6745.9 s (148.2 steps/s)
- training `mppo_rnd` seed 0: 1,000,000 interactions, 4491.77 s (222.6 steps/s)
- training `mppo_rnd` seed 1: 1,000,000 interactions, 4484.33 s (223.0 steps/s)
- training `ppo` seed 0: 1,000,000 interactions, 3785.55 s (264.2 steps/s)
- training `ppo` seed 1: 1,000,000 interactions, 3802.79 s (263.0 steps/s)
- training `rppo` seed 0: 1,000,000 interactions, 5876.28 s (170.2 steps/s)
- training `rppo` seed 1: 1,000,000 interactions, 5889.39 s (169.8 steps/s)
- agent `focused_sample`: 187 interactions per test world over its whole stream
- agent `maximal`: 195 interactions per test world over its whole stream
- agent `random`: 307 interactions per test world over its whole stream
- agent `reference`: 106 interactions per test world over its whole stream
