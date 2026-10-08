"""Study X: the repository's agents against open-source reinforcement-learning baselines.

``encoders`` and ``wrappers`` give RecipeQuest and TechTree fixed observation
and action spaces built from public information only, so off-the-shelf
learners (``learners``: Stable-Baselines3 and sb3-contrib, imported lazily) can
train across tasks and worlds; ``sources`` builds the task streams (evaluator
side); ``agents`` runs the repository's agents on the same streams; ``runner``
charges every interaction to the study ledger and records runs; ``analysis``
computes the endpoints of docs/studies/X_open_source_baselines.md;
``python -m hypergraph_agent.baselines`` is the command line.

External benchmarks: ``setting.loader`` names a function that returns the
tasks of one world block (``sources``); ``benchmark_sets`` is study X's loader
for RecipeQuest worlds converted from published recipe graphs
(``hypergraph_agent.benchmarks``), evaluated with the checkpoints of another
config (``run.checkpoints_from``). Wrappers, learners, runner and analysis need
no change as long as the tasks fit the fixed layouts
(``encoders.RQLayout(max_variants)``, ``encoders.TTLayout``).
"""
