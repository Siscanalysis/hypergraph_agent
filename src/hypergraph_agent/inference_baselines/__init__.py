"""Study X2 (docs/studies/X2_inference_baselines.md): the repository's inference
of hidden prerequisites against open-source structure-inference baselines on
identical episode logs.

Modules:

``traces``        episode logs in the representations the methods need (public data only)
``msgi_adapter``  MSGI's precondition inference, the authors' code (``third_party/msgi``)
``macq_adapter``  macq's Observer (in process) and ARMS (in a worker process, ``arms_worker``)
``arms_worker``   the ARMS worker process and its documented run-time changes to macq and nnf
``learners``      the factorized posterior and a SAM-style safe learner (reimplementation)
``collection``    the study's own config schema and the trace collection (charges the ledger)
``benchmark``     the offline benchmark: scores, metrics, acting cost, the primary decision,
                  the secondary tests, the cache and the exploratory ARMS hash-seed check
``provenance``    code hash, vendored-file hash, library versions and the cache key

Agents never import this package; ``benchmark`` reads the hidden requirements
that the collection stores, evaluator side, in each evidence file.
"""
