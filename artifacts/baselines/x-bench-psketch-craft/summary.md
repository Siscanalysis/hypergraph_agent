# Study X: x-bench-psketch-craft

Setting `bench_psketch_craft`; runs used 12, incomplete 0.

## Selection

Best learner `mppo` by cost_ratio_late on the validation split at budget 1000000: `mppo` 2.903, `mppo_rnd` 2.910, `ppo` 2.972, `rppo` 2.972

## Tests

| Test | Role | a - b | Estimate [CI] | Units | Verdict |
|---|---|---|---|---|---|
| XB psketch_craft: focused_sample - best learner, late cost ratio | secondary | focused_sample - mppo | -0.649 [-0.748, -0.547] | 10 | holds |
| XB psketch_craft: maximal - best learner, late cost ratio | descriptive | maximal - mppo | -0.568 [-0.607, -0.526] | 10 | holds |
| XB psketch_craft: focused_sample - maximal, late cost ratio | descriptive | focused_sample - maximal | -0.082 [-0.148, -0.014] | 10 | holds |

## Learning curves (test worlds; agents as reference)

| Arm | Budget | Late cost ratio | Late success | Late steps | First success steps | Units |
|---|---|---|---|---|---|---|
| focused_sample | agent | 1.537 | 0.912 | 9.4 | 8.8 | 10 |
| maximal | agent | 1.619 | 1.000 | 9.8 | 8.4 | 10 |
| mppo | 1000000 | 2.186 | 0.087 | 13.3 | 110.5 | 10 |
| mppo_rnd | 1000000 | 2.188 | 0.163 | 13.3 | 77.6 | 10 |
| ppo | 1000000 | 2.233 | 0.000 | 13.6 | 214.8 | 10 |
| random | agent | 2.193 | 0.113 | 13.3 | 106.3 | 10 |
| reference | agent | 1.000 | 1.000 | 6.1 | 4.6 | 10 |
| rppo | 1000000 | 2.233 | 0.000 | 13.6 | 214.8 | 10 |

## Conversion record (benchmark worlds)

10 worlds; items per world 11.0; promoted elements per world 0.0; dropped recipes per world 0.0; share of the closure kept as items 1.00.

| World | Goals | Items | Promoted | Dropped recipes | Kept share |
|---|---|---|---|---|---|
| B0c449888 | 7 sink goals | 11 | - | 0 | 1.00 |
| B6191d541 | 7 sink goals | 11 | - | 0 | 1.00 |
| B107c735c | 7 sink goals | 11 | - | 0 | 1.00 |
| B1468a722 | 7 sink goals | 11 | - | 0 | 1.00 |
| B1cdc056f | 7 sink goals | 11 | - | 0 | 1.00 |
| B63953491 | 7 sink goals | 11 | - | 0 | 1.00 |
| B0c0a4642 | 7 sink goals | 11 | - | 0 | 1.00 |
| Bc4d92251 | 7 sink goals | 11 | - | 0 | 1.00 |
| B9199ed22 | 7 sink goals | 11 | - | 0 | 1.00 |
| B11af4679 | 7 sink goals | 11 | - | 0 | 1.00 |

## Interactions and time

- agent `focused_sample`: 154 interactions per test world over its whole stream
- agent `maximal`: 154 interactions per test world over its whole stream
- agent `random`: 211 interactions per test world over its whole stream
- agent `reference`: 94 interactions per test world over its whole stream
