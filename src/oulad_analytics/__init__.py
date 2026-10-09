"""oulad-analytics: early prediction of at-risk students from OULAD data.

The package is organised as a small, testable pipeline:

    loader     -> read and validate the raw OULAD CSV tables
    preprocess -> clean demographic data and build the target variable
    features   -> aggregate assessments and VLE clicks up to a cut-off day
    model      -> train and evaluate baseline classifiers
    viz        -> produce figures for reports
    synthetic  -> generate a small fake dataset in OULAD format (for tests/demo)
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
