"""Unit tests for the case-duration lognormal estimation (synthetic data only).

Data-dependent parity vs the Stata xlsx and real pulls lives in cadaster-algo (which holds
those files); here we pin the logic: the cleaning.do issue flags, the closed-form MLE, the
sparse-type proxying + guard, and the shared taxonomy (including that VA's reference label is
just a relabel of the same grouping).
"""
import numpy as np
import pandas as pd
import pytest

from smart_mediator_assignment.algorithm.case_types import simplify_case_types
from smart_mediator_assignment.algorithm.duration_estimation import (
    _lognormal_mle,
    clean_hazard_sample,
    estimate_lognormal_duration_params,
)

DATAPULL = pd.Timestamp("2024-01-01")


def _clean_case(**overrides):
    """A single case that passes every cleaning.do filter; override one field per test."""
    base = {
        "case_type": "Matrimonial Property Cases",
        "case_status": "CONCLUDED",
        "mediator_id": 42,
        "referral_date": "2022-01-01",
        "mediator_appointment_date": "2022-02-01",
        "conclusion_date": "2022-04-01",
        "created_at": "2022-01-01",
        "outcome_name": "Settlement Agreement",
        "case_outcome_agreement": 1,
    }
    base.update(overrides)
    return base


def _df(*cases):
    return pd.DataFrame(list(cases))


def test_clean_case_is_kept():
    assert len(clean_hazard_sample(_df(_clean_case()), DATAPULL)) == 1


@pytest.mark.parametrize("override", [
    {"mediator_id": np.nan},                                              # issue 1
    {"referral_date": "2022-03-01"},                                      # issue 2: appt before referral
    {"mediator_appointment_date": np.nan},                               # issue 3
    {"conclusion_date": "2022-01-15"},                                    # issue 4: concl before appt
    {"conclusion_date": "2025-01-01"},                                    # issue 5: concl after datapull
    {"mediator_appointment_date": "2023-12-15",                          # issue 6: too new (<300d before pull)
     "conclusion_date": "2023-12-20", "referral_date": "2023-12-01"},
    {"referral_date": "2020-06-01", "mediator_appointment_date": "2020-07-01",
     "conclusion_date": "2020-09-01"},                                    # issue 7: pandemic referral
    {"outcome_name": "Terminated"},                                       # caseoutcome == 4
    {"case_outcome_agreement": np.nan},                                   # no observed outcome
])
def test_each_exclusion_drops_the_case(override):
    assert len(clean_hazard_sample(_df(_clean_case(**override)), DATAPULL)) == 0


def test_lognormal_mle_is_normal_fit_to_log_duration():
    logd = np.log(pd.Series([10.0, 20.0, 30.0, 40.0, 50.0]))
    fit = _lognormal_mle(logd)
    assert fit["Intercept"] == pytest.approx(logd.mean())
    assert fit["Sigma"] == pytest.approx(logd.std(ddof=0))   # ML (population) SD, matching streg
    assert fit["Number of observations"] == 5


def test_sparse_type_is_proxied_not_dropped():
    fittable = [_clean_case(case_type="Criminal Cases", case_outcome_agreement=(i % 2))
                for i in range(500)]
    sparse = [_clean_case(case_type="Judicial Review", case_outcome_agreement=(i % 2))
              for i in range(4)]  # <1% of arrivals, below the proxy-share guard
    est = estimate_lognormal_duration_params(_df(*fittable, *sparse), DATAPULL, min_fit_n=30)
    assert est["Judicial Review"]["agreement"].get("proxied") is True
    assert not est["Criminal Cases"]["agreement"].get("proxied")


def test_oversized_proxied_type_warns():
    fittable = [_clean_case(case_type="Criminal Cases", case_outcome_agreement=(i % 2))
                for i in range(80)]
    sparse = [_clean_case(case_type="Judicial Review", case_outcome_agreement=(i % 2))
              for i in range(20)]  # 20% of arrivals, well over MAX_PROXY_SHARE
    with pytest.warns(UserWarning, match="proxied case type"):
        est = estimate_lognormal_duration_params(_df(*fittable, *sparse), DATAPULL, min_fit_n=30)
    assert est["Judicial Review"]["agreement"].get("proxied") is True  # proxied, NOT dropped


def test_simplify_case_types_groupings_and_va_label_are_one_mapping():
    raw = pd.Series([
        "Civil Cases", "Civil Appeals", "Divorce and Separation", "Succession (Probate & Administration - P&A)",
        "Commercial Cases", "Tax Appeals", "Criminal Cases", "Judicial Review", "Anti-Corruption",
    ])
    default = simplify_case_types(raw)
    assert list(default) == [
        "Civil group", "Civil group", "Family group", "Family group",
        "Commercial and tax group", "Commercial and tax group",
        "Criminal Cases", "Judicial Review", "Anti-Corruption",  # pass-through incl. unmapped
    ]
    # VA's reference label is the SAME grouping, only "Family group" relabeled - the whole
    # point of a single taxonomy: the two consumers can never drift apart.
    va = simplify_case_types(raw, family_group_label="AAAFamily group")
    assert list(va) == list(default.replace("Family group", "AAAFamily group"))
