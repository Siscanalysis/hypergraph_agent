"""Recurrent PPO over whole-episode sequences.

Minibatches are sets of complete episodes; each episode is recomputed from
its zero initial state in order, so recurrent state is never shuffled across
time. Padding does not exist (one task per sequence); incomplete trailing
options are excluded from the losses but kept in the record. Advantage
normalization, when enabled, uses valid samples of the whole batch.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch


@dataclass
class PPOConfig:
    lr: float = 1e-3
    epochs: int = 4
    minibatch_episodes: int = 8
    clip: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    gamma: float = 1.0
    lam: float = 0.95
    # "none": raw advantages (rewards are 0/1, so they are already on a fixed scale).
    # "standard" divides by the batch std, which turns pure value noise into
    # unit-scale gradients when a sparse-reward batch has no success.
    adv_norm: str = "none"

    def to_dict(self) -> dict:
        return asdict(self)


class StaleBatchError(RuntimeError):
    """Raised when data were collected under a different topology/library snapshot."""


def check_snapshots(records, expected) -> None:
    for r in records:
        exp = expected(r)
        if (r.snapshot_id, r.library_id) != exp:
            raise StaleBatchError(
                f"episode {r.task_key} collected under {(r.snapshot_id, r.library_id)}, "
                f"current {exp}")


def ppo_update(policy, optimizer, records, cfg: PPOConfig, rng: np.random.Generator,
               expected=None) -> dict:
    if expected is not None:
        check_snapshots(records, expected)
    data = []
    for r in records:
        adv, tgt, mask = r.advantages(cfg.gamma, cfg.lam)
        if mask.any():
            data.append((r, torch.tensor(adv, dtype=torch.float32),
                         torch.tensor(tgt, dtype=torch.float32), torch.tensor(mask)))
    if not data:
        return {"n_samples": 0, "n_updates": 0}
    all_adv = torch.cat([a[m] for _, a, _, m in data])
    if cfg.adv_norm == "standard":
        mean = all_adv.mean()
        std = all_adv.std().clamp_min(1e-6) if len(all_adv) > 1 else torch.tensor(1.0)
    elif cfg.adv_norm == "none":
        mean, std = torch.tensor(0.0), torch.tensor(1.0)
    else:
        raise ValueError(f"unknown adv_norm {cfg.adv_norm!r}")
    stats = {"pg_loss": 0.0, "v_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0, "clipfrac": 0.0}
    n_terms, n_updates = 0, 0
    for _ in range(cfg.epochs):
        order = rng.permutation(len(data))
        for s in range(0, len(order), cfg.minibatch_episodes):
            pg_sum = v_sum = ent_sum = 0.0
            n_valid = 0
            kl_sum = clip_sum = 0.0
            for i in order[s:s + cfg.minibatch_episodes]:
                r, adv, tgt, mask = data[int(i)]
                logits, values, _, _ = policy.forward_sequence(r.struct, r.present, r.mem)
                logp_all = torch.log_softmax(logits, -1)
                logp = logp_all.gather(1, r.actions.view(-1, 1)).squeeze(1)
                a = (adv - mean) / std
                ratio = torch.exp(logp - r.logp)
                pg = -torch.min(ratio * a, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * a)
                ent = -(logp_all.exp() * logp_all).sum(-1)
                vl = (values - tgt) ** 2
                pg_sum = pg_sum + pg[mask].sum()
                v_sum = v_sum + vl[mask].sum()
                ent_sum = ent_sum + ent[mask].sum()
                n_valid += int(mask.sum())
                with torch.no_grad():
                    kl_sum += float((r.logp - logp)[mask].sum())
                    clip_sum += float(((ratio - 1).abs() > cfg.clip)[mask].float().sum())
            if n_valid == 0:
                continue
            loss = (pg_sum + cfg.vf_coef * v_sum - cfg.ent_coef * ent_sum) / n_valid
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite PPO loss")
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
            optimizer.step()
            n_updates += 1
            stats["pg_loss"] += float(pg_sum.detach()) / n_valid
            stats["v_loss"] += float(v_sum.detach()) / n_valid
            stats["entropy"] += float(ent_sum.detach()) / n_valid
            stats["approx_kl"] += kl_sum / n_valid
            stats["clipfrac"] += clip_sum / n_valid
            n_terms += 1
    out = {k: v / max(n_terms, 1) for k, v in stats.items()}
    out.update({"n_samples": int(len(all_adv)), "n_updates": n_updates,
                "adv_mean": float(mean), "adv_std": float(std)})
    return out
