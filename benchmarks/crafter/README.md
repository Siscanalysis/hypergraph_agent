# Crafter as an environment

Crafter is also a pip package with a 2D survival environment whose 22
achievements form the tech tree described by `data.yaml`. This repository
does not depend on it: the loader reads only the committed `data.yaml`, and
no package was installed to build or test the benchmark files.

To run the environment itself (for example to compare against published
Crafter baselines), install the release whose `data.yaml` is committed here:

```bash
python -m pip install crafter==1.8.3
```

Version 1.8.3 (PyPI, uploaded 2023-12-13) requires `numpy`, `imageio`,
`pillow`, `opensimplex` and `ruamel.yaml`; `pygame` is needed only for the
interactive viewer (`crafter[gui]`). `crafter.Env(area=(64, 64), view=(9, 9),
size=(64, 64), reward=True, length=10000, seed=None)` follows the older Gym
interface (`step` returns four values; it uses `gym` spaces when `gym` is
installed) and reports the count of every achievement in
`info["achievements"]`.

How the repository uses Crafter:

1. As a recipe graph (`loaders.load("crafter")`), for statistics and for the
   RecipeQuest conversion: `to_recipequest.recipequest_world(graph,
   sink_goals(graph))` gives one world with all 16 items Crafter's data
   defines beyond wood (wood is promoted to a base fact because the pickaxe
   and sword recipes need no terrain material). Agents of this repository
   can then play Crafter's dependency structure with hidden requirements,
   without its navigation, survival and combat.
2. As an external reference: agents that solve the converted world are not
   thereby comparable with published Crafter scores, which measure
   achievement success rates in the pixel environment over 1M steps. Such a
   comparison needs the package above and an agent that acts in the pixel
   environment.
