# Study X: x-rq-tune

Setting `rq_f1`; runs used 33, incomplete 0.

## Selection

Best learner `mppo_rnd@stochastic` by cost_ratio_late on the test split at budget 100000: `mppo.ent@deterministic` 3.143, `mppo.ent@stochastic` 3.125, `mppo.wide@deterministic` 3.121, `mppo.wide@stochastic` 3.135, `mppo@deterministic` 3.143, `mppo@stochastic` 3.122, `mppo_rnd.b01@deterministic` 3.139, `mppo_rnd.b01@stochastic` 3.128, `mppo_rnd@deterministic` 3.143, `mppo_rnd@stochastic` 3.111, `ppo.ent@deterministic` 3.143, `ppo.ent@stochastic` 3.143, `ppo.wide@deterministic` 3.143, `ppo.wide@stochastic` 3.143, `ppo@deterministic` 3.143, `ppo@stochastic` 3.143, `rppo.ent@deterministic` 3.143, `rppo.ent@stochastic` 3.143, `rppo.long@deterministic` 3.143, `rppo.long@stochastic` 3.143, `rppo@deterministic` 3.143, `rppo@stochastic` 3.143

## Tuning selection (per learner group)

| Group | Selected | Mode | Score | Candidates |
|---|---|---|---|---|
| mppo | `mppo.wide` | deterministic | 3.121 | mppo@deterministic 3.143, mppo@stochastic 3.122, mppo.ent@deterministic 3.143, mppo.ent@stochastic 3.125, mppo.wide@deterministic 3.121, mppo.wide@stochastic 3.135 |
| mppo_rnd | `mppo_rnd` | stochastic | 3.111 | mppo_rnd@deterministic 3.143, mppo_rnd@stochastic 3.111, mppo_rnd.b01@deterministic 3.139, mppo_rnd.b01@stochastic 3.128 |
| ppo | `ppo` | deterministic | 3.143 | ppo@deterministic 3.143, ppo@stochastic 3.143, ppo.ent@deterministic 3.143, ppo.ent@stochastic 3.143, ppo.wide@deterministic 3.143, ppo.wide@stochastic 3.143 |
| rppo | `rppo` | deterministic | 3.143 | rppo@deterministic 3.143, rppo@stochastic 3.143, rppo.ent@deterministic 3.143, rppo.ent@stochastic 3.143, rppo.long@deterministic 3.143, rppo.long@stochastic 3.143 |

## Tests

| Test | Role | a - b | Estimate [CI] | Units | Verdict |
|---|---|---|---|---|---|

## Learning curves (test worlds; agents as reference)

| Arm | Budget | Late cost ratio | Late success | Late steps | First success steps | Units |
|---|---|---|---|---|---|---|
| mppo.ent@deterministic | 100000 | 3.143 | 0.000 | 19.8 | 314.6 | 20 |
| mppo.ent@stochastic | 100000 | 3.125 | 0.050 | 19.7 | 259.3 | 20 |
| mppo.wide@deterministic | 100000 | 3.121 | 0.019 | 19.7 | 279.9 | 20 |
| mppo.wide@stochastic | 100000 | 3.135 | 0.013 | 19.8 | 237.2 | 20 |
| mppo@deterministic | 100000 | 3.143 | 0.000 | 19.8 | 298.0 | 20 |
| mppo@stochastic | 100000 | 3.122 | 0.031 | 19.7 | 255.9 | 20 |
| mppo_rnd.b01@deterministic | 100000 | 3.139 | 0.006 | 19.8 | 311.1 | 20 |
| mppo_rnd.b01@stochastic | 100000 | 3.128 | 0.013 | 19.8 | 276.3 | 20 |
| mppo_rnd@deterministic | 100000 | 3.143 | 0.000 | 19.8 | 314.6 | 20 |
| mppo_rnd@stochastic | 100000 | 3.111 | 0.044 | 19.6 | 250.5 | 20 |
| ppo.ent@deterministic | 100000 | 3.143 | 0.000 | 19.8 | 314.6 | 20 |
| ppo.ent@stochastic | 100000 | 3.143 | 0.000 | 19.8 | 314.6 | 20 |
| ppo.wide@deterministic | 100000 | 3.143 | 0.000 | 19.8 | 314.6 | 20 |
| ppo.wide@stochastic | 100000 | 3.143 | 0.000 | 19.8 | 314.6 | 20 |
| ppo@deterministic | 100000 | 3.143 | 0.000 | 19.8 | 314.6 | 20 |
| ppo@stochastic | 100000 | 3.143 | 0.000 | 19.8 | 314.6 | 20 |
| rppo.ent@deterministic | 100000 | 3.143 | 0.000 | 19.8 | 314.6 | 20 |
| rppo.ent@stochastic | 100000 | 3.143 | 0.000 | 19.8 | 314.6 | 20 |
| rppo.long@deterministic | 100000 | 3.143 | 0.000 | 19.8 | 314.6 | 20 |
| rppo.long@stochastic | 100000 | 3.143 | 0.000 | 19.8 | 314.6 | 20 |
| rppo@deterministic | 100000 | 3.143 | 0.000 | 19.8 | 314.6 | 20 |
| rppo@stochastic | 100000 | 3.143 | 0.000 | 19.8 | 314.6 | 20 |

## Interactions and time

- training `mppo` seed 0: 100,000 interactions, 367.73 s (271.9 steps/s)
- training `mppo.ent` seed 0: 100,000 interactions, 366.77 s (272.7 steps/s)
- training `mppo.wide` seed 0: 100,000 interactions, 712.66 s (140.3 steps/s)
- training `ppo` seed 0: 100,000 interactions, 322.27 s (310.3 steps/s)
- training `ppo.ent` seed 0: 100,000 interactions, 326.18 s (306.6 steps/s)
- training `ppo.wide` seed 0: 100,000 interactions, 685.79 s (145.8 steps/s)
- training `rppo` seed 0: 100,000 interactions, 615.0 s (162.6 steps/s)
- training `rppo.ent` seed 0: 100,000 interactions, 603.05 s (165.8 steps/s)
- training `rppo.long` seed 0: 100,000 interactions, 458.77 s (218.0 steps/s)
- training `mppo_rnd` seed 0: 100,000 interactions, 375.09 s (266.6 steps/s)
- training `mppo_rnd.b01` seed 0: 100,000 interactions, 351.4 s (284.6 steps/s)
