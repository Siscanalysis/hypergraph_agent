# Study X: x-bench-little-alchemy-2

Setting `bench_little_alchemy_2`; runs used 12, incomplete 0.

## Selection

Best learner `mppo` by cost_ratio_late on the validation split at budget 1000000: `mppo` 2.903, `mppo_rnd` 2.910, `ppo` 2.972, `rppo` 2.972

## Tests

| Test | Role | a - b | Estimate [CI] | Units | Verdict |
|---|---|---|---|---|---|
| XB little_alchemy_2: focused_sample - best learner, late cost ratio | secondary | focused_sample - mppo | -1.859 [-2.272, -1.467] | 30 | holds |
| XB little_alchemy_2: maximal - best learner, late cost ratio | descriptive | maximal - mppo | -1.246 [-1.530, -0.973] | 30 | holds |
| XB little_alchemy_2: focused_sample - maximal, late cost ratio | descriptive | focused_sample - maximal | -0.613 [-0.824, -0.404] | 30 | holds |

## Learning curves (test worlds; agents as reference)

| Arm | Budget | Late cost ratio | Late success | Late steps | First success steps | Units |
|---|---|---|---|---|---|---|
| focused_sample | agent | 1.423 | 0.925 | 8.0 | 24.1 | 30 |
| maximal | agent | 2.036 | 1.000 | 10.5 | 10.2 | 30 |
| mppo | 1000000 | 3.282 | 0.110 | 16.3 | 166.7 | 30 |
| mppo_rnd | 1000000 | 3.248 | 0.206 | 16.1 | 101.1 | 30 |
| ppo | 1000000 | 3.411 | 0.000 | 16.7 | 270.0 | 30 |
| random | agent | 3.229 | 0.150 | 16.1 | 135.0 | 30 |
| reference | agent | 1.000 | 1.000 | 5.5 | 5.3 | 30 |
| rppo | 1000000 | 3.411 | 0.000 | 16.7 | 270.0 | 30 |

## Conversion record (benchmark worlds)

30 worlds; items per world 5.2; promoted elements per world 6.0; dropped recipes per world 4.3; share of the closure kept as items 0.14.

| World | Goals | Items | Promoted | Dropped recipes | Kept share |
|---|---|---|---|---|---|
| B5c02c8c5 | omitted | 3 | omitted | 2 | 0.09 |
| Bda508845 | omitted | 2 | omitted | 3 | 0.14 |
| Ba7e74d60 | omitted | 6 | omitted | 5 | 0.17 |
| B0c1c1353 | omitted | 7 | omitted | 12 | 0.17 |
| B25243ba5 | omitted | 6 | omitted | 5 | 0.11 |
| B2abd4e97 | omitted | 3 | omitted | 0 | 0.14 |
| B72700738 | omitted | 2 | omitted | 0 | 0.14 |
| B5dc7cf90 | omitted | 9 | omitted | 5 | 0.23 |
| Ba7245a89 | omitted | 2 | omitted | 0 | 0.06 |
| Bbbcc5ef6 | omitted | 6 | omitted | 0 | 0.11 |
| B89030a92 | omitted | 6 | omitted | 3 | 0.13 |
| B51947f4e | omitted | 1 | omitted | 2 | 0.04 |
| Bba1a3f48 | omitted | 5 | omitted | 5 | 0.06 |
| Bcb9c6b62 | omitted | 1 | omitted | 0 | 0.05 |
| B93eea5fa | omitted | 13 | omitted | 12 | 0.17 |
| Bff9ce51c | omitted | 4 | omitted | 3 | 0.05 |
| B61cf03a6 | omitted | 2 | omitted | 0 | 0.22 |
| Bc891d7d7 | omitted | 7 | omitted | 14 | 0.11 |
| B7c23e2c5 | omitted | 6 | omitted | 7 | 0.13 |
| Bc11108a5 | omitted | 13 | omitted | 14 | 0.22 |
| B74698055 | omitted | 5 | omitted | 4 | 0.09 |
| Bbc2b474b | omitted | 5 | omitted | 0 | 0.10 |
| B2e73897a | omitted | 2 | omitted | 2 | 0.12 |
| Ba2327d47 | omitted | 2 | omitted | 1 | 0.03 |
| Bc2fe7575 | omitted | 7 | omitted | 3 | 0.39 |
| Bbac6d2c2 | omitted | 9 | omitted | 4 | 0.45 |
| B39f47c4a | omitted | 7 | omitted | 13 | 0.12 |
| Be9f7ded7 | omitted | 6 | omitted | 0 | 0.10 |
| Bb8163367 | omitted | 4 | omitted | 2 | 0.13 |
| Bb367717c | omitted | 6 | omitted | 8 | 0.13 |

## Interactions and time

- agent `focused_sample`: 150 interactions per test world over its whole stream
- agent `maximal`: 169 interactions per test world over its whole stream
- agent `random`: 261 interactions per test world over its whole stream
- agent `reference`: 88 interactions per test world over its whole stream
