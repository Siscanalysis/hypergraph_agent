"""External benchmarks as recipe graphs, with converters to this repository's environments.

``graph`` defines the neutral ``RecipeGraph``; ``loaders`` read each
benchmark under ``benchmarks/`` (repository root) into it, after ``sources``
has verified the files' SHA-256; ``to_recipequest`` builds RecipeQuest worlds
from a graph and ``to_techtree`` builds TechTree-like concept graphs (and,
where the data has that shape, TechTree worlds); ``stats`` computes the
statistics written to ``benchmarks/<name>/stats.json``. Sources, licenses and
checksums are documented in ``benchmarks/README.md`` and in each folder's
``SOURCE.md``. Evaluator side: converted worlds hold hidden truth, and agent
modules must receive only ``public_view`` of their tasks.
"""
