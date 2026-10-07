"""Duration-aware (semi-Markov) returns and advantages.

For a decision k lasting tau_k primitive steps with rewards r_{t_k..t_k+tau_k-1}:

    R_k     = sum_{j<tau_k} gamma**j r_{t_k+j}
    Gamma_k = gamma**tau_k
    delta_k = R_k + Gamma_k * bootstrap_k * V_next_k - V_k
    A_k     = delta_k + Gamma_k * lambda**tau_k * continuation_k * A_{k+1}

``lambda`` decays per primitive step (declared convention). ``bootstrap_k`` is
0 after a genuine terminal (success, failure, task deadline) and 1 otherwise;
``continuation_k`` is 0 at the last stored decision of a segment, so an
administrative cutoff can bootstrap from V_next without chaining an advantage
that was never collected. Primitive decisions have tau = 1.
"""

from __future__ import annotations

import numpy as np


def option_return(rewards, gamma: float) -> float:
    return float(sum((gamma ** j) * r for j, r in enumerate(rewards)))


def smdp_gae(R, tau, values, next_values, bootstrap, continuation, gamma: float, lam: float):
    R = np.asarray(R, dtype=float)
    tau = np.asarray(tau, dtype=float)
    V = np.asarray(values, dtype=float)
    Vn = np.asarray(next_values, dtype=float)
    boot = np.asarray(bootstrap, dtype=float)
    cont = np.asarray(continuation, dtype=float)
    if np.any(tau < 1):
        raise ValueError("every decision lasts at least one primitive step")
    adv = np.zeros_like(R)
    nxt = 0.0
    for k in range(len(R) - 1, -1, -1):
        G = gamma ** tau[k]
        delta = R[k] + G * boot[k] * Vn[k] - V[k]
        adv[k] = delta + G * (lam ** tau[k]) * cont[k] * nxt
        nxt = adv[k]
    return adv, adv + V


def segment_gae(R, tau, values, final_value: float, terminal: bool, gamma: float, lam: float):
    """Advantages for one contiguous segment of decisions.

    ``terminal`` means the segment ended in a genuine task terminal; otherwise
    ``final_value`` is V at the true next decision boundary.
    """
    k = len(R)
    next_values = list(values[1:]) + [final_value]
    bootstrap = [1.0] * k
    continuation = [1.0] * k
    if k:
        bootstrap[-1] = 0.0 if terminal else 1.0
        continuation[-1] = 0.0
    return smdp_gae(R, tau, values, next_values, bootstrap, continuation, gamma, lam)
