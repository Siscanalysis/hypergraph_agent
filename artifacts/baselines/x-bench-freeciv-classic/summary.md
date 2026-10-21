# Study X: x-bench-freeciv-classic

Setting `bench_freeciv_classic`; runs used 12, incomplete 0.

## Selection

Best learner `mppo` by cost_ratio_late on the validation split at budget 1000000: `mppo` 2.903, `mppo_rnd` 2.910, `ppo` 2.972, `rppo` 2.972

## Tests

| Test | Role | a - b | Estimate [CI] | Units | Verdict |
|---|---|---|---|---|---|
| XB freeciv_classic: focused_sample - best learner, late cost ratio | secondary | focused_sample - mppo | -1.196 [-1.501, -0.896] | 20 | holds |
| XB freeciv_classic: maximal - best learner, late cost ratio | descriptive | maximal - mppo | -0.449 [-0.549, -0.344] | 20 | holds |
| XB freeciv_classic: focused_sample - maximal, late cost ratio | descriptive | focused_sample - maximal | -0.747 [-1.004, -0.497] | 20 | holds |

## Learning curves (test worlds; agents as reference)

| Arm | Budget | Late cost ratio | Late success | Late steps | First success steps | Units |
|---|---|---|---|---|---|---|
| focused_sample | agent | 1.095 | 0.975 | 7.9 | 11.8 | 20 |
| maximal | agent | 1.841 | 1.000 | 11.5 | 11.4 | 20 |
| mppo | 1000000 | 2.291 | 0.100 | 14.4 | 195.8 | 20 |
| mppo_rnd | 1000000 | 2.247 | 0.237 | 14.2 | 112.7 | 20 |
| ppo | 1000000 | 2.359 | 0.000 | 14.7 | 236.4 | 20 |
| random | agent | 2.223 | 0.131 | 14.1 | 132.6 | 20 |
| reference | agent | 1.000 | 1.000 | 7.0 | 6.7 | 20 |
| rppo | 1000000 | 2.359 | 0.000 | 14.7 | 236.4 | 20 |

## Conversion record (benchmark worlds)

20 worlds; items per world 4.8; promoted elements per world 3.6; dropped recipes per world 0.0; share of the closure kept as items 0.49.

| World | Goals | Items | Promoted | Dropped recipes | Kept share |
|---|---|---|---|---|---|
| B71592bf4 | Amphibious Warfare | 9 | The Wheel, Literacy, Iron Working, University, Democracy, Leadership, Navigation | 0 | 0.30 |
| Bca88a296 | Automobile | 11 | Astronomy, Literacy, Banking, Invention, Chemistry, Bridge Building, Economics, Explosives, Steel | 0 | 0.29 |
| Bab5307c5 | Chivalry | 4 | - | 0 | 1.00 |
| B7d7d1f15 | Code of Laws | 1 | - | 0 | 1.00 |
| B02b78d6e | Communism | 8 | Astronomy, Literacy, Invention, Bridge Building, Banking, Philosophy | 0 | 0.32 |
| Bbe1126e3 | Currency | 1 | - | 0 | 1.00 |
| B3599f8dc | Democracy | 5 | The Wheel, Literacy, Banking | 0 | 0.42 |
| Bf8d35f47 | Economics | 5 | Code of Laws, Mysticism, Mathematics, Banking | 0 | 0.42 |
| B850493c3 | Electronics | 9 | Astronomy, Literacy, Banking, Invention, Bridge Building, Economics, Electricity | 0 | 0.28 |
| B0280388a | Environmentalism | 8 | Chemistry, Electricity, Explosives, Electronics, Theory of Gravity, The Corporation, Machine Tools, Computers, Recycling | 0 | 0.14 |
| B71857de5 | Fusion Power | 3 | Mass Production, Nuclear Power | 0 | 0.06 |
| Bd075355a | Guerilla Warfare | 5 | Iron Working, Invention, University, Democracy, Leadership, Communism | 0 | 0.14 |
| Ba14c05f4 | Iron Working | 1 | - | 0 | 1.00 |
| B02ee1b08 | Medicine | 4 | Code of Laws, Mysticism, Trade | 0 | 0.50 |
| Bd70965b8 | Metallurgy | 6 | The Wheel, Literacy, Iron Working, University | 0 | 0.40 |
| B98da19eb | Monarchy | 2 | - | 0 | 1.00 |
| B3375cc0f | Plastics | 7 | Electricity, Explosives, Electronics, Theory of Gravity, Machine Tools, Computers, Refining | 0 | 0.12 |
| B3405419b | Sanitation | 5 | Code of Laws, Mysticism, Trade, Engineering | 0 | 0.42 |
| B405f17a2 | Superconductors | 2 | Mass Production, Nuclear Power | 0 | 0.04 |
| B5034b972 | The Wheel | 1 | - | 0 | 1.00 |

## Interactions and time

- agent `focused_sample`: 142 interactions per test world over its whole stream
- agent `maximal`: 187 interactions per test world over its whole stream
- agent `random`: 229 interactions per test world over its whole stream
- agent `reference`: 115 interactions per test world over its whole stream
