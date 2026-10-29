# Study X: xs-rq-s2

Setting `rq_f1`; runs used 42, incomplete 0.

## Selection

Best learner `s2_shaped` by cost_ratio_late on the validation split at budget 2500000: `s2_shaped` 2.764, `s2_sparse` 2.777

## Tests

| Test | Role | a - b | Estimate [CI] | Units | Verdict |
|---|---|---|---|---|---|

## Learning curves (test worlds; agents as reference)

| Arm | Budget | Late cost ratio | Late success | Late steps | First success steps | Units |
|---|---|---|---|---|---|---|
| focused_sample | agent | 1.434 | 0.995 | 9.2 | 16.2 | 50 |
| maximal | agent | 1.908 | 1.000 | 12.1 | 11.7 | 50 |
| random | agent | 2.981 | 0.080 | 18.8 | 189.3 | 50 |
| reference | agent | 1.000 | 1.000 | 6.4 | 6.0 | 50 |
| s2_shaped | 300000 | 2.963 | 0.104 | 18.7 | 177.0 | 50 |
| s2_shaped | 1000000 | 2.961 | 0.144 | 18.6 | 146.3 | 50 |
| s2_shaped | 2500000 | 2.971 | 0.119 | 18.7 | 159.1 | 50 |
| s2_sparse | 300000 | 2.998 | 0.072 | 18.9 | 210.8 | 50 |
| s2_sparse | 1000000 | 3.002 | 0.083 | 18.9 | 183.2 | 50 |
| s2_sparse | 2500000 | 2.959 | 0.125 | 18.6 | 146.7 | 50 |

## Interactions and time

- training `s2_shaped` seed 0: 2,500,000 interactions, 22932.15 s (109.0 steps/s)
- training `s2_shaped` seed 1: 2,500,000 interactions, 22878.1 s (109.3 steps/s)
- training `s2_sparse` seed 0: 2,500,000 interactions, 22987.83 s (108.8 steps/s)
- training `s2_sparse` seed 1: 2,500,000 interactions, 23183.94 s (107.8 steps/s)
- agent `focused_sample`: 175 interactions per test world over its whole stream
- agent `maximal`: 194 interactions per test world over its whole stream
- agent `random`: 302 interactions per test world over its whole stream
- agent `reference`: 102 interactions per test world over its whole stream
