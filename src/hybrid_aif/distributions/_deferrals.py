"""The shared deferral message for parameter posteriors without a novelty estimator."""


def deferred_novelty(family: str) -> NotImplementedError:
    """The one wording for "this family's novelty estimator does not exist yet".

    ``family`` is the noun phrase naming the posterior ("an LG noise posterior",
    "an MNIW joint posterior", ...); everything else is shared so the message
    stays consistent across families and greppable as one string.
    """
    return NotImplementedError(
        "continuous parameter novelty is deferred: the objectives registry has no "
        f"estimator for {family}, so `epistemic` naming this slot "
        "raises (design: parameter-learning.md D6/step 3, per-variable-infogain.md V4)"
    )
