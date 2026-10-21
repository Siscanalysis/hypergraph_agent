# Study X: x-bench-crafter

Setting `bench_crafter`; runs used 12, incomplete 1.

## Selection

Best learner `mppo` by cost_ratio_late on the validation split at budget 1000000: `mppo` 2.903, `mppo_rnd` 2.910, `ppo` 2.972, `rppo` 2.972

## Tests

| Test | Role | a - b | Estimate [CI] | Units | Verdict |
|---|---|---|---|---|---|
| XB crafter: focused_sample - best learner, late cost ratio | secondary | focused_sample - mppo | -0.812 [-0.911, -0.704] | 10 | holds |
| XB crafter: maximal - best learner, late cost ratio | descriptive | maximal - mppo | -0.502 [-0.566, -0.437] | 10 | holds |
| XB crafter: focused_sample - maximal, late cost ratio | descriptive | focused_sample - maximal | -0.310 [-0.357, -0.252] | 10 | holds |

## Learning curves (test worlds; agents as reference)

| Arm | Budget | Late cost ratio | Late success | Late steps | First success steps | Units |
|---|---|---|---|---|---|---|
| focused_sample | agent | 1.426 | 0.975 | 10.2 | 17.8 | 10 |
| maximal | agent | 1.736 | 1.000 | 12.4 | 11.3 | 10 |
| mppo | 1000000 | 2.239 | 0.100 | 15.9 | 166.8 | 10 |
| mppo_rnd | 1000000 | 2.259 | 0.106 | 16.0 | 142.2 | 10 |
| ppo | 1000000 | 2.305 | 0.000 | 16.3 | 263.2 | 10 |
| random | agent | 2.251 | 0.087 | 16.0 | 123.2 | 10 |
| reference | agent | 1.000 | 1.000 | 7.2 | 6.2 | 10 |
| rppo | 1000000 | 2.305 | 0.000 | 16.3 | 263.2 | 10 |

## Conversion record (benchmark worlds)

10 worlds; items per world 16.0; promoted elements per world 1.0; dropped recipes per world 5.0; share of the closure kept as items 0.94.

| World | Goals | Items | Promoted | Dropped recipes | Kept share |
|---|---|---|---|---|---|
| B66ec281f | 7 sink goals | 16 | wood | 5 | 0.94 |
| Bbe3af325 | 7 sink goals | 16 | wood | 5 | 0.94 |
| Bad112471 | 7 sink goals | 16 | wood | 5 | 0.94 |
| B85de135f | 7 sink goals | 16 | wood | 5 | 0.94 |
| B14d7abad | 7 sink goals | 16 | wood | 5 | 0.94 |
| B915c37fc | 7 sink goals | 16 | wood | 5 | 0.94 |
| B75defd6c | 7 sink goals | 16 | wood | 5 | 0.94 |
| B796987fc | 7 sink goals | 16 | wood | 5 | 0.94 |
| Be01b8f82 | 7 sink goals | 16 | wood | 5 | 0.94 |
| B1353b048 | 7 sink goals | 16 | wood | 5 | 0.94 |

## Interactions and time

- agent `focused_sample`: 187 interactions per test world over its whole stream
- agent `maximal`: 201 interactions per test world over its whole stream
- agent `random`: 258 interactions per test world over its whole stream
- agent `reference`: 120 interactions per test world over its whole stream
