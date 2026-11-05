"""Learned priors over hidden requirements (studies HP1 and HP2).

Agent side: everything here reads public data and the agent's own evidence only.

``size``       requirement-size priors and their empirical Bayes estimate from pinned recipes (HP1)
``posterior``  the exact posterior under any per-recipe prior: weighted enumeration of complete
               hypergraphs for goal-only logs, the factorized updater for items observable (HP1)
``schema``     lifted rules over public families and classes: exact enumeration of schemas, clingo
               for hidden classes, averaged against a no-schema model (HP2)
"""
