"""Study RL (docs/studies/RL_rule_learners.md): exact search for the smallest
rule consistent with all successes and failures against the exact posterior,
on study X2's logs and on new logs with slips and with requirements outside
the posterior's class.

Modules:

``dnf``         smallest monotone DNF consistent with a recipe's attempts: clingo, with a
                MaxSAT (RC2) cross-check; hard, and soft for slips
``classical``   RIPPER, CART and FIGS, and exploratory Boolean symbolic regression (gplearn),
                each decoded into a rule
``learners``    every learner on the scored recipes of one world at one checkpoint
``collection``  the study's own config schema and trace collection on X2's machinery
                (charges the ledger ``runs/RL-ledger.json``)
``analysis``    the offline benchmark: X2's metrics, the decisions, the tests, the cache
                and the report

Agents never import this package. Heavy libraries (clingo, python-sat,
wittgenstein, scikit-learn, imodels, gplearn) are imported only when a
learner runs.
"""
