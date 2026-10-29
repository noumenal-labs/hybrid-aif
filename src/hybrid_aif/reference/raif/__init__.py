"""R-AIF, replicated: NACLab/robust-active-inference @ ``9b1864e8`` (the oracle).

"Their" is that code, "the oracle" its logged runs, "the CRSPP" its prior-preference model,
and "addendum §n" is ``docs/design/amortization-p0-addendum.md``.

``world.py``    their ``tiny`` RSSM and the CRSPP as declared Systems
``agent.py``    the assembled agent and the canonical training step
``crspp.py``    the preference model's loss and no-action scans
``revise.py``   self-revision: ρ labels
``metrics.py``  ACR, SR and R-S
``columns.py``  the episode's task columns
``seed.py``     seeding M⁺ in their driver's row convention
``run.py``      the resumable multi-seed Mountain Car harness and CLI
``fixtures/``   the oracle's exported expert episodes and resolved config
"""

from hybrid_aif.reference.raif import crspp as _crspp  # noqa: E402, F401  (registers the CRSPP row)
