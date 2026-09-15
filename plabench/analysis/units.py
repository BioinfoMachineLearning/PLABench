"""Affinity unit conversions.

The CASP16 label files report ``binding_affinity`` as a free energy in kcal/mol, while
every model in the benchmark emits a p-scale value (pKd / pIC50). The two are related by
dG = -RT ln(10) * pX.

CASP16 built its files at T = 300 K, giving RT ln(10) = 1.372571 kcal/mol per log unit.
That is not a guess: both label files also carry the raw IC50 they were derived from, and
dividing -binding_affinity by this constant reproduces 6 - log10(IC50 / uM) to within
1e-5 for all 140 compounds. The 1.36423 that appears in older scripts is the 298.15 K
value and leaves a systematic 0.047 log-unit error.
"""
from __future__ import annotations

# RT ln(10) in kcal/mol at the temperature CASP16 used (300 K).
DG_KCAL_PER_LOG_UNIT = 1.372571


def dg_to_pkd(dg_kcal_per_mol):
    """Free energy in kcal/mol -> p-scale affinity. Works on scalars, Series and arrays."""
    return -dg_kcal_per_mol / DG_KCAL_PER_LOG_UNIT


def pkd_to_dg(pkd):
    """p-scale affinity -> free energy in kcal/mol."""
    return -pkd * DG_KCAL_PER_LOG_UNIT
