"""Reference implementation for the alert-budgeted early warning paper.

Modules
-------
eaai.data      cohort loading, harmonisation, splits, synthetic generator
eaai.scores    clinical instruments and score-model plumbing
eaai.policy    stateful alarm policy and blocked conformal calibration
eaai.eval      metrics and statistical tests
eaai.experiments  runnable experiment drivers
"""

__version__ = "0.1.0"
