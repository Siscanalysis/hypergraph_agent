# MSGI precondition inference (ILP): source and license

**Citation.** Sungryull Sohn, Hyunjae Woo, Jongwook Choi, Honglak Lee. Meta
Reinforcement Learning with Autonomous Inference of Subtask Dependencies.
ICLR 2020. arXiv:2001.00248.

**Source.** https://github.com/srsohn/msgi at commit
`e665861f2d08f41b7dad16588447203e5010145a` (default branch `master`, commit
date 2020-07-16, retrieved 2026-10-08), the grid-world code base used for the
Mining and Playground domains:

| File here | File in the repository | Bytes | SHA-256 |
|---|---|---|---|
| `graph/ILP.py` | `grid-world/graph/ILP.py` | 20294 | `762f425014fd5b9be8dd90962ab24caaa911dc1c3ae32a1ef0ee6c010d5d70d2` |
| `graph/graph_utils.py` | `grid-world/graph/graph_utils.py` | 1948 | `51ae8ebe73e52335abafca3851872c5471c8aaf200d101e20aded636e40b3736` |
| `LICENSE` | `LICENSE` | 1068 | `7b0746fd903d431ba39707885557deed87e4214f71a337413ad4ff500b0edbc2` |

The checksums are of the bytes stored in the repository at that commit
(`git show <commit>:<path> | sha256sum`); `.gitattributes` keeps the files
byte-exact. The files are unchanged. The repository's StarCraft II code base
(`sc2/agents/ILP.py`) has a different precondition learner (scikit-learn
decision trees) and is not used.

**License.** MIT, Copyright (c) 2020 Hyunjae Woo, verified on 2026-10-08 from
the repository's `LICENSE` file (the same text and checksum as
`benchmarks/msgi_mining/LICENSE`, recorded from the GitHub license API as SPDX
`MIT` in `benchmarks/msgi_mining/SOURCE.md`). The license text is kept in
`LICENSE`.

## What is used, and how

`src/hypergraph_agent/inference_baselines/msgi_adapter.py` imports `ILP.py`
under its own package name `graph` (as in the original layout) and calls three
methods of the `ILP` class on each recipe's attempts, unchanged:
`cart_train` (the authors' CART with their Gini rule), `decode_cart` (tree to
a sum of products, one row per AND term, +1 positive and -1 negated literal)
and `simplify_Kmap` (the reductions A + AB -> A and A + A'B -> A + B, called
as in `ILP._infer_precondition`, which keeps the in-place reductions and
discards the returned tensor). Duplicate AND terms are merged as in
`ILP._fillout_mat`. The adapter also applies `ILP._locate_node_layer`'s two
rules for subtasks that need no tree: always eligible means no precondition,
never eligible means unknown.

What is not used: `ILP.insert` and `ILP.infer_graph`, which take the
eligibility of every subtask at every step (MSGI's environment reveals it);
RecipeQuest reveals only the outcome of the attempted craft, so the adapter
gives each recipe its attempts as labelled examples and the recipe's public
candidate pool and item inputs as features, in place of the layer search over
all subtasks. The reward-inference, exploration-bonus and GRProp parts of
MSGI are not used.
