# Study X: xs-rq-s12

Setting `rq_f1`; runs used 42, incomplete 0.

## Selection

Best learner `s12_sparse` by cost_ratio_late on the validation split at budget 2500000: `s12_shaped` 3.865, `s12_sparse` 3.730

## Tests

| Test | Role | a - b | Estimate [CI] | Units | Verdict |
|---|---|---|---|---|---|

## Learning curves (test worlds; agents as reference)

| Arm | Budget | Late cost ratio | Late success | Late steps | First success steps | Units |
|---|---|---|---|---|---|---|
| focused_sample | agent | 1.422 | 1.000 | 9.1 | 14.3 | 50 |
| maximal | agent | 1.908 | 1.000 | 12.1 | 11.7 | 50 |
| random | agent | 4.340 | 0.203 | 27.4 | 129.0 | 50 |
| reference | agent | 1.000 | 1.000 | 6.4 | 6.0 | 50 |
| s12_shaped | 300000 | 4.290 | 0.241 | 27.1 | 134.9 | 50 |
| s12_shaped | 1000000 | 4.100 | 0.384 | 25.9 | 69.2 | 50 |
| s12_shaped | 2500000 | 4.061 | 0.436 | 25.6 | 67.4 | 50 |
| s12_sparse | 300000 | 4.269 | 0.250 | 26.8 | 127.9 | 50 |
| s12_sparse | 1000000 | 4.075 | 0.384 | 25.7 | 90.5 | 50 |
| s12_sparse | 2500000 | 4.007 | 0.481 | 25.2 | 56.5 | 50 |

## Interactions and time

- training `s12_shaped` seed 0: 2,500,000 interactions, 23192.75 s (107.8 steps/s)
- training `s12_shaped` seed 1: 2,500,000 interactions, 23192.52 s (107.8 steps/s)
- training `s12_sparse` seed 0: 2,500,000 interactions, 23152.23 s (108.0 steps/s)
- training `s12_sparse` seed 1: 2,500,000 interactions, 23167.6 s (107.9 steps/s)
- agent `focused_sample`: 177 interactions per test world over its whole stream
- agent `maximal`: 194 interactions per test world over its whole stream
- agent `random`: 439 interactions per test world over its whole stream
- agent `reference`: 102 interactions per test world over its whole stream
