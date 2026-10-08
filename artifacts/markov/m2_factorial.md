| Arm | Worlds | Late cost ratio | Late success | All episodes | Evaluations / world | Evaluations / replan | Seconds / replan (run) | Choice seconds / replan |
|---|---|---|---|---|---|---|---|---|
| `focused_sample` | 50 | 1.46 | 1.00 | 1.69 | 2071 | 52.2 | 0.0200 | - |
| `hyper_rank` | 50 | 1.49 | 0.98 | 1.72 | 69998 | 1737.2 | 0.3592 | 0.3436 |
| `pair_sample` | 50 | 1.56 | 0.99 | 1.79 | 0 | 0.0 | 0.0074 | 0.0052 |
| `pair_rank` | 50 | 1.68 | 0.99 | 1.87 | 0 | 0.0 | 0.0056 | 0.0035 |
| `hyper_sample_ppr` | 50 | 1.52 | 0.98 | 1.73 | 2091 | 50.7 | 0.0231 | 0.0205 |
| `maximal` | 50 | 1.83 | 1.00 | 1.82 | 0 | 0.0 | 0.0080 | - |
| `reference` | 50 | 1.00 | 1.00 | 1.00 | 0 | 0.0 | - | - |

| Effect | Late cost [95% CI] | Late cost [97.5% CI] | Late success [95% CI] |
|---|---|---|---|
| representation (hyper - pair) | -0.146 [-0.194, -0.097] | -0.146 [-0.202, -0.090] | 0.000 [-0.009, 0.009] |
| selector (rank - sample) | 0.077 [0.034, 0.118] | 0.077 [0.028, 0.124] | -0.010 [-0.019, -0.003] |
| interaction | -0.079 [-0.172, 0.011] | -0.079 [-0.185, 0.026] | -0.010 [-0.030, 0.007] |

Primary (representation main effect on late cost): hyper lowers late cost.
Secondary, `hyper_sample_ppr` - `focused_sample` (50 worlds): evaluations per world 19.4 [-77.4, 113.6], per replan -1.53 [-2.18, -0.97], late cost 0.068 [0.013, 0.126], late success -0.015 [-0.028, -0.005].
Selected levels: {'representation': 'hyper', 'selector': 'sample'}; winning cell `focused_sample` (descriptive against `hyper_rank`: -0.037 [-0.094, 0.020]); complete: True (50 paired worlds).
