# Study X: x-ttl-tune

Setting `tt_l2_d2`; runs used 24, incomplete 0.

## Selection

Best learner `mppo_rnd@stochastic` by steps_late on the test split at budget 100000: `mppo.ent@deterministic` 100.000, `mppo.ent@stochastic` 100.000, `mppo.wide@deterministic` 100.000, `mppo.wide@stochastic` 100.000, `mppo@deterministic` 100.000, `mppo@stochastic` 94.750, `mppo_rnd.b01@deterministic` 100.000, `mppo_rnd.b01@stochastic` 98.819, `mppo_rnd@deterministic` 96.875, `mppo_rnd@stochastic` 85.292, `rppo.ent@deterministic` 100.000, `rppo.ent@stochastic` 100.000, `rppo.long@deterministic` 100.000, `rppo.long@stochastic` 100.000, `rppo@deterministic` 100.000, `rppo@stochastic` 99.569

## Tuning selection (per learner group)

| Group | Selected | Mode | Score | Candidates |
|---|---|---|---|---|
| mppo | `mppo` | stochastic | 94.750 | mppo@deterministic 100.000, mppo@stochastic 94.750, mppo.ent@deterministic 100.000, mppo.ent@stochastic 100.000, mppo.wide@deterministic 100.000, mppo.wide@stochastic 100.000 |
| mppo_rnd | `mppo_rnd` | stochastic | 85.292 | mppo_rnd@deterministic 96.875, mppo_rnd@stochastic 85.292, mppo_rnd.b01@deterministic 100.000, mppo_rnd.b01@stochastic 98.819 |
| rppo | `rppo` | stochastic | 99.569 | rppo@deterministic 100.000, rppo@stochastic 99.569, rppo.ent@deterministic 100.000, rppo.ent@stochastic 100.000, rppo.long@deterministic 100.000, rppo.long@stochastic 100.000 |

## Tests

| Test | Role | a - b | Estimate [CI] | Units | Verdict |
|---|---|---|---|---|---|

## Learning curves (test worlds; agents as reference)

| Arm | Budget | Late cost ratio | Late success | Late steps | First success steps | Units |
|---|---|---|---|---|---|---|
| mppo.ent@deterministic | 100000 | 14.286 | 0.000 | 100.0 | 1200.0 | 12 |
| mppo.ent@stochastic | 100000 | 14.286 | 0.000 | 100.0 | 1200.0 | 12 |
| mppo.wide@deterministic | 100000 | 14.286 | 0.000 | 100.0 | 1200.0 | 12 |
| mppo.wide@stochastic | 100000 | 14.286 | 0.000 | 100.0 | 1200.0 | 12 |
| mppo@deterministic | 100000 | 14.286 | 0.000 | 100.0 | 1200.0 | 12 |
| mppo@stochastic | 100000 | 13.536 | 0.083 | 94.8 | 707.5 | 12 |
| mppo_rnd.b01@deterministic | 100000 | 14.286 | 0.000 | 100.0 | 1200.0 | 12 |
| mppo_rnd.b01@stochastic | 100000 | 14.117 | 0.069 | 98.8 | 903.0 | 12 |
| mppo_rnd@deterministic | 100000 | 13.839 | 0.097 | 96.9 | 870.4 | 12 |
| mppo_rnd@stochastic | 100000 | 12.185 | 0.250 | 85.3 | 605.5 | 12 |
| rppo.ent@deterministic | 100000 | 14.286 | 0.000 | 100.0 | 1200.0 | 12 |
| rppo.ent@stochastic | 100000 | 14.286 | 0.000 | 100.0 | 1200.0 | 12 |
| rppo.long@deterministic | 100000 | 14.286 | 0.000 | 100.0 | 1200.0 | 12 |
| rppo.long@stochastic | 100000 | 14.286 | 0.000 | 100.0 | 1137.1 | 12 |
| rppo@deterministic | 100000 | 14.286 | 0.000 | 100.0 | 1200.0 | 12 |
| rppo@stochastic | 100000 | 14.224 | 0.014 | 99.6 | 1024.8 | 12 |

## Interactions and time

- training `mppo` seed 0: 100,000 interactions, 261.6 s (382.3 steps/s)
- training `mppo.ent` seed 0: 100,000 interactions, 259.67 s (385.1 steps/s)
- training `mppo.wide` seed 0: 100,000 interactions, 327.95 s (304.9 steps/s)
- training `mppo_rnd` seed 0: 100,000 interactions, 286.79 s (348.7 steps/s)
- training `mppo_rnd.b01` seed 0: 100,000 interactions, 287.37 s (348.0 steps/s)
- training `rppo` seed 0: 100,000 interactions, 265.53 s (376.6 steps/s)
- training `rppo.ent` seed 0: 100,000 interactions, 199.45 s (501.4 steps/s)
- training `rppo.long` seed 0: 100,000 interactions, 226.57 s (441.4 steps/s)
