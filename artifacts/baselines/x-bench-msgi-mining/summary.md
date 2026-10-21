# Study X: x-bench-msgi-mining

Setting `bench_msgi_mining`; runs used 12, incomplete 0.

## Selection

Best learner `mppo` by cost_ratio_late on the validation split at budget 1000000: `mppo` 2.903, `mppo_rnd` 2.910, `ppo` 2.972, `rppo` 2.972

## Tests

| Test | Role | a - b | Estimate [CI] | Units | Verdict |
|---|---|---|---|---|---|
| XB msgi_mining: focused_sample - best learner, late cost ratio | secondary | focused_sample - mppo | -0.754 [-0.889, -0.626] | 20 | holds |
| XB msgi_mining: maximal - best learner, late cost ratio | descriptive | maximal - mppo | -0.431 [-0.505, -0.349] | 20 | holds |
| XB msgi_mining: focused_sample - maximal, late cost ratio | descriptive | focused_sample - maximal | -0.323 [-0.471, -0.188] | 20 | holds |

## Learning curves (test worlds; agents as reference)

| Arm | Budget | Late cost ratio | Late success | Late steps | First success steps | Units |
|---|---|---|---|---|---|---|
| focused_sample | agent | 1.174 | 0.944 | 8.5 | 10.7 | 20 |
| maximal | agent | 1.497 | 1.000 | 10.2 | 9.7 | 20 |
| mppo | 1000000 | 1.928 | 0.100 | 13.2 | 142.0 | 20 |
| mppo_rnd | 1000000 | 1.949 | 0.175 | 13.3 | 122.5 | 20 |
| ppo | 1000000 | 1.996 | 0.000 | 13.6 | 218.8 | 20 |
| random | agent | 1.961 | 0.069 | 13.4 | 137.0 | 20 |
| reference | agent | 1.000 | 1.000 | 7.0 | 6.5 | 20 |
| rppo | 1000000 | 1.996 | 0.000 | 13.6 | 218.8 | 20 |

## Conversion record (benchmark worlds)

20 worlds; items per world 6.2; promoted elements per world 0.0; dropped recipes per world 0.0; share of the closure kept as items 1.00.

| World | Goals | Items | Promoted | Dropped recipes | Kept share |
|---|---|---|---|---|---|
| B6eb1f1b0 | Bake pork | 9 | - | 0 | 1.00 |
| B32470ccd | Craft silver diamond earrings | 12 | - | 0 | 1.00 |
| Bf2160e7e | Get coal | 3 | - | 0 | 1.00 |
| B1c596b97 | Get diamond ore | 9 | - | 0 | 1.00 |
| B723a5181 | Get gold ore | 9 | - | 0 | 1.00 |
| B10c8b7d8 | Get iron ore | 3 | - | 0 | 1.00 |
| B3e1f9e2a | Hit pig | 3 | - | 0 | 1.00 |
| B79c9a24e | Light furnace | 5 | - | 0 | 1.00 |
| Bfe2fc6bb | Make arrow | 1 | - | 0 | 1.00 |
| B56846f2b | Make bow | 1 | - | 0 | 1.00 |
| Bda4b89cb | Make electrum bracelet | 13 | - | 0 | 1.00 |
| Ba4e39b3a | Make firewood | 1 | - | 0 | 1.00 |
| B8bdaed3f | Make goldware | 11 | - | 0 | 1.00 |
| B735f9981 | Make iron pickaxe | 8 | - | 0 | 1.00 |
| B76d84144 | Make silverware | 8 | - | 0 | 1.00 |
| Bd4493c6a | Make stick | 1 | - | 0 | 1.00 |
| Bb68c0172 | Make stone pickaxe | 2 | - | 0 | 1.00 |
| B850afc07 | Smelt gold | 10 | - | 0 | 1.00 |
| B93a62615 | Smelt iron | 7 | - | 0 | 1.00 |
| Be5ac8d41 | Smelt silver | 7 | - | 0 | 1.00 |

## Interactions and time

- agent `focused_sample`: 152 interactions per test world over its whole stream
- agent `maximal`: 164 interactions per test world over its whole stream
- agent `random`: 216 interactions per test world over its whole stream
- agent `reference`: 115 interactions per test world over its whole stream
