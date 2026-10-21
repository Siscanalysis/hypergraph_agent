# Study X: x-ttu-tune

Setting `tt_u_k3`; runs used 24, incomplete 0.

## Selection

Best learner `mppo_rnd@stochastic` by steps_late on the test split at budget 100000: `mppo.ent@deterministic` 100.000, `mppo.ent@stochastic` 97.458, `mppo.wide@deterministic` 100.000, `mppo.wide@stochastic` 96.069, `mppo@deterministic` 100.000, `mppo@stochastic` 99.472, `mppo_rnd.b01@deterministic` 100.000, `mppo_rnd.b01@stochastic` 98.278, `mppo_rnd@deterministic` 100.000, `mppo_rnd@stochastic` 88.639, `rppo.ent@deterministic` 100.000, `rppo.ent@stochastic` 98.181, `rppo.long@deterministic` 100.000, `rppo.long@stochastic` 100.000, `rppo@deterministic` 100.000, `rppo@stochastic` 98.764

## Tuning selection (per learner group)

| Group | Selected | Mode | Score | Candidates |
|---|---|---|---|---|
| mppo | `mppo.wide` | stochastic | 96.069 | mppo@deterministic 100.000, mppo@stochastic 99.472, mppo.ent@deterministic 100.000, mppo.ent@stochastic 97.458, mppo.wide@deterministic 100.000, mppo.wide@stochastic 96.069 |
| mppo_rnd | `mppo_rnd` | stochastic | 88.639 | mppo_rnd@deterministic 100.000, mppo_rnd@stochastic 88.639, mppo_rnd.b01@deterministic 100.000, mppo_rnd.b01@stochastic 98.278 |
| rppo | `rppo.ent` | stochastic | 98.181 | rppo@deterministic 100.000, rppo@stochastic 98.764, rppo.ent@deterministic 100.000, rppo.ent@stochastic 98.181, rppo.long@deterministic 100.000, rppo.long@stochastic 100.000 |

## Tests

| Test | Role | a - b | Estimate [CI] | Units | Verdict |
|---|---|---|---|---|---|

## Learning curves (test worlds; agents as reference)

| Arm | Budget | Late cost ratio | Late success | Late steps | First success steps | Units |
|---|---|---|---|---|---|---|
| mppo.ent@deterministic | 100000 | 14.683 | 0.000 | 100.0 | 1200.0 | 12 |
| mppo.ent@stochastic | 100000 | 14.319 | 0.097 | 97.5 | 791.2 | 12 |
| mppo.wide@deterministic | 100000 | 14.683 | 0.000 | 100.0 | 1200.0 | 12 |
| mppo.wide@stochastic | 100000 | 14.121 | 0.083 | 96.1 | 866.5 | 12 |
| mppo@deterministic | 100000 | 14.683 | 0.000 | 100.0 | 1200.0 | 12 |
| mppo@stochastic | 100000 | 14.607 | 0.042 | 99.5 | 1069.1 | 12 |
| mppo_rnd.b01@deterministic | 100000 | 14.683 | 0.000 | 100.0 | 1200.0 | 12 |
| mppo_rnd.b01@stochastic | 100000 | 14.437 | 0.056 | 98.3 | 1010.3 | 12 |
| mppo_rnd@deterministic | 100000 | 14.683 | 0.014 | 100.0 | 1175.0 | 12 |
| mppo_rnd@stochastic | 100000 | 12.996 | 0.264 | 88.6 | 511.2 | 12 |
| rppo.ent@deterministic | 100000 | 14.683 | 0.000 | 100.0 | 1200.0 | 12 |
| rppo.ent@stochastic | 100000 | 14.423 | 0.042 | 98.2 | 1153.4 | 12 |
| rppo.long@deterministic | 100000 | 14.683 | 0.000 | 100.0 | 1200.0 | 12 |
| rppo.long@stochastic | 100000 | 14.683 | 0.000 | 100.0 | 1200.0 | 12 |
| rppo@deterministic | 100000 | 14.683 | 0.000 | 100.0 | 1200.0 | 12 |
| rppo@stochastic | 100000 | 14.506 | 0.028 | 98.8 | 1104.8 | 12 |

## Interactions and time

- training `mppo` seed 0: 100,000 interactions, 250.85 s (398.7 steps/s)
- training `mppo.ent` seed 0: 100,000 interactions, 252.7 s (395.7 steps/s)
- training `mppo.wide` seed 0: 100,000 interactions, 311.31 s (321.2 steps/s)
- training `mppo_rnd.b01` seed 0: 100,000 interactions, 273.38 s (365.8 steps/s)
- training `rppo` seed 0: 100,000 interactions, 194.24 s (514.8 steps/s)
- training `rppo.ent` seed 0: 100,000 interactions, 224.74 s (444.9 steps/s)
- training `rppo.long` seed 0: 100,000 interactions, 193.58 s (516.6 steps/s)
- training `mppo_rnd` seed 0: 100,000 interactions, 276.04 s (362.3 steps/s)
