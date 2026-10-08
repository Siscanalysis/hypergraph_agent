# Little Alchemy 1 and 2 combination tables: source and license

**Citation.** Franziska Brändle, Lena J. Stocks, Joshua B. Tenenbaum, Samuel J.
Gershman, Eric Schulz. Empowerment contributes to exploration behaviour in a
creative video game. Nature Human Behaviour 7(9):1481-1489, 2023.
https://doi.org/10.1038/s41562-023-01661-2

**Source.** The paper's data and code availability statements point to
https://github.com/franziskabraendle/alchemy_empowerment; its README points
to the data archive
https://keeper.mpdl.mpg.de/d/28c50dc3a6bf4d10995d/ (Max Planck Digital
Library, archive metadata dated 2022), folder
`empowermentexploration/resources/littlealchemy/data/`, which holds the game
trees `alchemy1Gametree.json` and `alchemy2Gametree.json`. Retrieved
2026-10-08.

**License and redistribution decision: fetch script only.** The GitHub
repository is MIT-licensed (Copyright (c) 2022 Lena J. Stocks, Franziska
Brändle; verified 2026-10-08 from its `LICENSE` file at commit
`7b1ef89cd3bcebf78daf2c5fae6f58f162670321`, the head of `main` at retrieval,
and the GitHub license API). The Keeper archive states no license (its `archive-metadata.md` lists
title, authors, description, year and institute only). The combination tables
and element descriptions are content of the commercial games Little Alchemy
and Little Alchemy 2 (Recloak), which the research authors do not claim to
license. Nothing from these files is committed here: `fetch.py` downloads the
two game trees from the archive into `data/` (not tracked) and verifies their
SHA-256. The loader reads only element names, combination pairs and unlock
conditions, never the descriptions. `stats.json` holds only counts and
summary statistics.

## Files

| File | Bytes | SHA-256 | Source |
|---|---|---|---|
| `data/alchemy1Gametree.json` | 109410 | `a40d3e3008cb7022c610a4fcc2e0664338ac4eafcc103b30acdf9df2f7e8881b` | Keeper archive, `.../littlealchemy/data/alchemy1Gametree.json` |
| `data/alchemy2Gametree.json` | 370891 | `897e36c0354a1ddad607bcb556da9fd5f3bb6464ae395642281471a74366a16a` | Keeper archive, `.../littlealchemy/data/alchemy2Gametree.json` |

```bash
python benchmarks/little_alchemy/fetch.py          # download and verify
python benchmarks/little_alchemy/fetch.py --check  # verify only
```

The exact download URLs are in `manifest.json`. If the archive moves, the
same files can be requested from the authors (contact in the repository
README); the checksums identify them.

## What the loader reads

`loaders.load_little_alchemy(version)`: every element is identified by its
name (names are unique in both files). Each pair in `parents` is one recipe
(`AND` of the two elements); an element has as many recipes as pairs
(alternatives). A pair of an element with itself requires that element once
(amount 2 kept as metadata). Base: the four `prime` elements of version 2;
version 1 has no flags, and its base is air, earth, fire and water. Version 2
marks nine elements as conditional and they become unlocks: six are granted
after a number of discoveries (`progress`, for example `time` after 100) and
three after `min` of a listed set of elements (`k_of_n`, for example `light`
after 5 of 16). Five elements
flagged `hidden` are labelled `secret`. Recipes that produce a base element or
require their own product, and duplicate pairs, are dropped (counted in the
graph's notes).
