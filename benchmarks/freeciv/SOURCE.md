# Freeciv technology trees: source and license

**Citation.** The Freeciv project, Freeciv 3.2.6 (Git tag `R3_2_6`),
https://github.com/freeciv/freeciv, https://www.freeciv.org.

**Source.** Ruleset files at commit `2c4304c742aa81f1d73a5d604ed83237d57804a1`
(tag `R3_2_6`), retrieved 2026-10-08: `data/classic/` (the Civilization II
style ruleset) and `data/civ2civ3/` (the default ruleset of Freeciv 3.x),
three files each: `techs.ruleset`, `units.ruleset`, `buildings.ruleset`.

**License and redistribution decision: fetch script only.** Freeciv is
released under the GNU General Public License, version 2 or (at the user's
option) any later version: `doc/README` at the pinned commit, the `COPYING`
file (GPL v2 text) and the GitHub license API (`GPL-2.0`), verified
2026-10-08. The ruleset files are part of that release. Copyleft data is not
committed to this MIT repository: `fetch.py` downloads the six files into
`data/` (not tracked) and verifies their SHA-256. `stats.json` holds only
counts and summary statistics.

## Files

| File | Bytes | SHA-256 |
|---|---|---|
| `data/classic/techs.ruleset` | 20738 | `cd0ebfae9dfa2ff51f8827c8205d542fe845e169d2537f262e3387cc009b10a1` |
| `data/classic/units.ruleset` | 78346 | `3ac6fed02dc964f31a0676393d81b524fb9f1147dd2d72cdc729e11c993b2ac0` |
| `data/classic/buildings.ruleset` | 51912 | `73f5654ef0eb16aaed9f64313de2a03a4827b1b7bfaa0dda9847b9adf97cd5d9` |
| `data/civ2civ3/techs.ruleset` | 21450 | `45cd775adf9d61ecbfd93ad35a137ef59b5c3f5c0baca5f0249def0a4c925bd4` |
| `data/civ2civ3/units.ruleset` | 90442 | `0884174c09389bf5f0a6de7550e6eec03d791b6250bca1d85f92256f51557636` |
| `data/civ2civ3/buildings.ruleset` | 62454 | `e80e320b816be012239137580e4a62d2e39a8f223edf8bb67022488ba18b4476` |

```bash
python benchmarks/freeciv/fetch.py          # download and verify
python benchmarks/freeciv/fetch.py --check  # verify only
```

## What the loader reads

`loaders.load_freeciv(ruleset)` parses the `[advance_*]` sections of
`techs.ruleset`: a technology with `req1` and `req2` both `"None"` is a base
fact (seven in both rulesets); every other technology has one recipe that
requires `req1`, `req2` (when not `"None"`) and its `root_req`, if any.
Technologies that require `"Never"` would be left out (none in these two
rulesets). With `unlockables=True` (default), every `[unit_*]` and
`[building_*]` section whose `reqs` vector names technologies becomes an
element made available, through an unlock of kind `all`, when those
technologies are known. Non-technology requirements (buildings, terrain,
governments, city size) are not represented; units and buildings without a
technology requirement are left out (both counted in the graph's notes). Help
texts and other strings are not read.
