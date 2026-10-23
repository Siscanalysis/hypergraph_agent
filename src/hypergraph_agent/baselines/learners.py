"""Off-the-shelf learners of study X: Stable-Baselines3 and sb3-contrib.

The libraries are imported only when a learner is built or loaded, so the
package imports without them. Algorithms (``ALGOS``):

- ``maskable_ppo``: sb3-contrib ``MaskablePPO`` with ``MlpPolicy``; reads the
  wrapper's ``action_masks()``, so it never chooses a masked action.
- ``ppo``: Stable-Baselines3 ``PPO`` with ``MlpPolicy``; no masks: a masked
  action executes ``wait`` (one step, see ``wrappers``).
- ``recurrent_ppo``: sb3-contrib ``RecurrentPPO`` with ``MlpLstmPolicy``; no
  masks (the library has none for recurrent policies), invalid actions as for
  ``ppo``. Run in trial mode its LSTM state persists over the episodes of a
  world (RL^2-style), so it can adapt within a world.
- ``dqn``: Stable-Baselines3 ``DQN`` with ``MlpPolicy``; no masks.

Parameters not given in a learner's ``params`` are the library defaults; the
config key ``features_dim`` puts ``MlpFeatures`` (one linear layer with ReLU)
in front of the policy instead of the flattened observation. The study's
recurrent learner uses it (64 features, one LSTM of 64 units shared by actor
and critic, 512-sample minibatches): with the default architecture, an LSTM of
256 units per head reading the whole 1,339-wide RecipeQuest observation,
sb3-contrib's per-step sequence loop ran at about 3-10 training interactions
per second in development, too slow for the planned budgets. A
learner may add ``rnd``: random network distillation (Burda et al., 2018) as a
reward wrapper around the vectorized environment, since neither library
ships a maintained implementation. A fixed random target network and a
trained predictor (two-layer MLPs on observations normalized by running
statistics and clipped to [-5, 5]); the intrinsic reward of a transition is the
predictor's squared error on the next observation divided by the running
standard deviation of that error, and the learner receives the environment's
reward plus ``beta`` times it. The predictor is trained with Adam on the
observations of every ``update_every`` vectorized steps. It differs from the
original in using one combined (episodic) reward and value head rather than
two value heads with a non-episodic intrinsic return. The bonus exists only
during training.

Checkpoints: the policy acting at the B-th training interaction is saved for
every budget B; the run stops at the largest budget, so it executes exactly
that many interactions (budgets are multiples of the number of environments).
"""

from __future__ import annotations

import importlib
import importlib.metadata
from pathlib import Path

import numpy as np
import torch

ALGOS = {
    "maskable_ppo": ("sb3_contrib", "MaskablePPO", "MlpPolicy"),
    "ppo": ("stable_baselines3", "PPO", "MlpPolicy"),
    "recurrent_ppo": ("sb3_contrib", "RecurrentPPO", "MlpLstmPolicy"),
    "dqn": ("stable_baselines3", "DQN", "MlpPolicy"),
}
MASKED = ("maskable_ppo",)
RECURRENT = ("recurrent_ppo",)
RND_DEFAULTS = {"beta": 0.01, "hidden": 128, "out": 64, "lr": 1e-4, "update_every": 128, "batch": 256,
                "epochs": 1}


def algo_class(algo: str):
    if algo not in ALGOS:
        raise KeyError(f"unknown algorithm {algo!r}; known: {sorted(ALGOS)}")
    module, name, _ = ALGOS[algo]
    return getattr(importlib.import_module(module), name)


def library_versions() -> dict:
    out = {}
    for dist in ("stable-baselines3", "sb3-contrib", "gymnasium", "torch", "numpy"):
        try:
            out[dist] = importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            out[dist] = None
    return out


class MlpFeatures(torch.nn.Module):
    """A one-layer feature extractor (linear, ReLU) for the policy: with ``params.features_dim`` a learner
    reads this ``features_dim``-wide projection of the observation instead of the flattened
    observation (the library default). Used for the recurrent learner, whose LSTM would otherwise
    take the whole observation as input."""

    def __init__(self, observation_space, features_dim: int = 64):
        super().__init__()
        self.features_dim = int(features_dim)
        n = int(np.prod(observation_space.shape))
        self.net = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(n, self.features_dim),
                                       torch.nn.ReLU())

    def forward(self, x):
        return self.net(x)


def library_kwargs(params: dict) -> dict:
    """Config params -> library keyword arguments (``features_dim`` selects ``MlpFeatures``)."""
    kw = dict(params)
    dim = kw.pop("features_dim", None)
    if dim is not None:
        pk = dict(kw.get("policy_kwargs") or {})
        pk.update(features_extractor_class=MlpFeatures, features_extractor_kwargs={"features_dim": int(dim)})
        kw["policy_kwargs"] = pk
    return kw


def build_model(algo: str, venv, params: dict, seed: int):
    cls = algo_class(algo)
    return cls(ALGOS[algo][2], venv, seed=int(seed), device="cpu", verbose=0, **library_kwargs(params))


def load_model(algo: str, path: str | Path):
    return algo_class(algo).load(str(path), device="cpu")


# ------------------------------------------------------------------ RND
def rnd_vec_env(venv, seed: int, **kw):
    """Wrap a VecEnv so that rewards include the RND bonus (see the module docstring)."""
    from stable_baselines3.common.running_mean_std import RunningMeanStd
    from stable_baselines3.common.vec_env import VecEnvWrapper

    cfg = {**RND_DEFAULTS, **kw}
    unknown = set(cfg) - set(RND_DEFAULTS)
    if unknown:
        raise KeyError(f"unknown rnd keys {sorted(unknown)}")

    def mlp(d_in):
        return torch.nn.Sequential(torch.nn.Linear(d_in, cfg["hidden"]), torch.nn.ReLU(),
                                   torch.nn.Linear(cfg["hidden"], cfg["hidden"]), torch.nn.ReLU(),
                                   torch.nn.Linear(cfg["hidden"], cfg["out"]))

    class RNDVecEnv(VecEnvWrapper):
        def __init__(self, inner):
            super().__init__(inner)
            dim = int(np.prod(inner.observation_space.shape))
            with torch.random.fork_rng():
                torch.manual_seed(seed)
                self.target, self.predictor = mlp(dim), mlp(dim)
            for p in self.target.parameters():
                p.requires_grad_(False)
            self.opt = torch.optim.Adam(self.predictor.parameters(), lr=cfg["lr"])
            self.obs_rms = RunningMeanStd(shape=inner.observation_space.shape)
            self.err_rms = RunningMeanStd(shape=())
            self.buffer: list[np.ndarray] = []
            self.gen = np.random.default_rng(seed)
            self.stats = {"updates": 0, "intrinsic_sum": 0.0, "steps": 0}

        def _norm(self, x):
            z = (x - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + 1e-8)
            return torch.as_tensor(np.clip(z, -5, 5), dtype=torch.float32)

        def reset(self):
            obs = self.venv.reset()
            self.obs_rms.update(obs)
            return obs

        def step_wait(self):
            obs, rew, dones, infos = self.venv.step_wait()
            nxt = obs.copy()
            for i, d in enumerate(dones):
                if d and "terminal_observation" in infos[i]:
                    nxt[i] = infos[i]["terminal_observation"]
            x = self._norm(nxt)
            with torch.no_grad():
                err = ((self.predictor(x) - self.target(x)) ** 2).mean(dim=1).numpy()
            self.err_rms.update(err)
            bonus = err / np.sqrt(self.err_rms.var + 1e-8)
            self.obs_rms.update(nxt)
            self.buffer.append(x.numpy())
            if len(self.buffer) >= cfg["update_every"]:
                self._train()
            self.stats["intrinsic_sum"] += float(bonus.sum())
            self.stats["steps"] += len(bonus)
            for i in range(len(infos)):
                infos[i] = {**infos[i], "intrinsic_reward": float(bonus[i])}
            return obs, (rew + cfg["beta"] * bonus).astype(np.float32), dones, infos

        def _train(self):
            data = torch.as_tensor(np.concatenate(self.buffer), dtype=torch.float32)
            self.buffer = []
            for _ in range(cfg["epochs"]):
                for idx in np.array_split(self.gen.permutation(len(data)), max(1, len(data) // cfg["batch"])):
                    xb = data[idx]
                    loss = ((self.predictor(xb) - self.target(xb)) ** 2).mean()
                    self.opt.zero_grad()
                    loss.backward()
                    self.opt.step()
            self.stats["updates"] += 1

    return RNDVecEnv(venv)


# ------------------------------------------------------------ callbacks
def checkpoint_callback(budgets, save, log=None, log_every: int = 10_000):
    """Saves the acting policy at every budget (``save(budget, model, num_timesteps)``) and stops
    training at the largest; ``log(record)`` receives training-task outcomes every ``log_every``
    (and, for a shaped learner, the sum of the shaping terms of the window)."""
    from stable_baselines3.common.callbacks import BaseCallback

    class CheckpointCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self.pending = sorted(int(b) for b in budgets)
            self.window = {"tasks": 0, "success": 0, "invalid": 0}
            self.next_log = log_every

        def _on_step(self) -> bool:
            for info in self.locals.get("infos", ()):
                self.window["invalid"] += int(bool(info.get("invalid")))
                if "shaping" in info:  # study XS: the shaping term received in the window
                    self.window["shaping"] = self.window.get("shaping", 0.0) + float(info["shaping"])
                if info.get("task_end"):
                    self.window["tasks"] += 1
                    self.window["success"] += int(bool(info.get("task_success")))
            if log is not None and self.num_timesteps >= self.next_log:
                log({"type": "training", "interactions": self.num_timesteps, **self.window})
                self.window = {"tasks": 0, "success": 0, "invalid": 0}
                self.next_log += log_every
            while self.pending and self.num_timesteps >= self.pending[0]:
                save(self.pending.pop(0), self.model, self.num_timesteps)
            return bool(self.pending)

    return CheckpointCallback()


# ------------------------------------------------------------- acting
class Actor:
    """A frozen policy acting in one wrapper (evaluation). ``reset()`` at the start of every learner
    episode (a task, or a world block in trial mode)."""

    def __init__(self, model, algo: str, deterministic: bool):
        self.model, self.algo, self.deterministic = model, algo, deterministic
        self.reset()

    def reset(self):
        self.state, self.start = None, np.ones((1,), dtype=bool)

    def act(self, obs: np.ndarray, mask: np.ndarray) -> int:
        if self.algo in MASKED:
            a, _ = self.model.predict(obs, deterministic=self.deterministic, action_masks=mask)
        elif self.algo in RECURRENT:
            a, self.state = self.model.predict(obs[None], state=self.state, episode_start=self.start,
                                               deterministic=self.deterministic)
            a = a[0]
            self.start = np.zeros((1,), dtype=bool)
        else:
            a, _ = self.model.predict(obs, deterministic=self.deterministic)
        return int(a)
