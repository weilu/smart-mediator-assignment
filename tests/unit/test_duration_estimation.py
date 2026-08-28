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
    {"mediator_id": np.nan},                                              # issue 1: missing mediator id
    {"mediator_appointment_date": np.nan},                               # issue 2: appointment date missing
    {"conclusion_date": "2022-01-15"},                                    # issue 3: conclusion before appointment
    {"referral_date": "2020-06-01", "mediator_appointment_date": "2020-07-01",
     "conclusion_date": "2020-09-01"},                                    # issue 4: pandemic (appointment date)
    {"outcome_name": "Terminated"},                                       # caseoutcome == 4
    {"case_outcome_agreement": np.nan},                                   # no observed outcome
])
def test_each_exclusion_drops_the_case(override):
    # the four data-quality issues 02_Hazard_DMP.do drops, plus terminated / no-outcome.
    assert len(clean_hazard_sample(_df(_clean_case(**override)), DATAPULL)) == 0


def test_cutoff_drops_too_new_by_default():
    # "too new": a case appointed within the cutoff (default 180 days) of the pull is dropped -
    # it hasn't had time to conclude. Passing cutoff=0 disables the filter and keeps it.
    too_new = _clean_case(referral_date="2023-12-01", mediator_appointment_date="2023-12-15",
                          conclusion_date="2023-12-20")  # appt 17 days before the 2024-01-01 pull
    assert len(clean_hazard_sample(_df(too_new), DATAPULL)) == 0             # default 180: dropped
    assert len(clean_hazard_sample(_df(too_new), DATAPULL, cutoff=0)) == 1   # kept when disabled


def test_lognormal_mle_is_normal_fit_to_log_duration():
    logd = np.log(pd.Series([10.0, 20.0, 30.0, 40.0, 50.0]))
    fit = _lognormal_mle(logd)
    assert fit["Intercept"] == pytest.approx(logd.mean())
    assert fit["Sigma"] == pytest.approx(logd.std(ddof=0))   # ML (population) SD, matching streg
    assert fit["Number of observations"] == 5


def test_sparse_type_is_fit_on_own_data_not_proxied():
    # A rare type with only a handful of usable cases is fit on its own data (used as-is),
    # not proxied - the agreed policy proxies only types with no usable case at all.
    fittable = [_clean_case(case_type="Criminal Cases", case_outcome_agreement=(i % 2))
                for i in range(500)]
    sparse = [_clean_case(case_type="Judicial Review", case_outcome_agreement=(i % 2))
              for i in range(4)]  # 2 agreement + 2 no agreement
    est = estimate_lognormal_duration_params(_df(*fittable, *sparse), DATAPULL)
    assert not est["Judicial Review"]["agreement"].get("proxied")
    assert est["Judicial Review"]["agreement"]["Number of observations"] == 2
    assert not est["Criminal Cases"]["agreement"].get("proxied")


def test_zero_usable_type_is_proxied_not_dropped():
    # A type present in the raw pull but with ALL rows failing cleaning (here: terminated) must
    # still be modeled via the global-pool proxy, not silently dropped from the case-type universe.
    fittable = [_clean_case(case_type="Criminal Cases", case_outcome_agreement=(i % 2))
                for i in range(600)]
    zero_usable = [_clean_case(case_type="Judicial Review", outcome_name="Terminated")
                   for _ in range(5)]  # dropped by the hazard sample -> empty group
    est = estimate_lognormal_duration_params(_df(*fittable, *zero_usable), DATAPULL)
    assert "Judicial Review" in est
    assert est["Judicial Review"]["agreement"].get("proxied") is True


def test_oversized_proxied_type_warns():
    # A type with NO usable case (all fail cleaning) proxies from the global pool; if its arrival
    # share is large the guard warns - the pooled duration is a poor stand-in at that scale.
    fittable = [_clean_case(case_type="Criminal Cases", case_outcome_agreement=(i % 2))
                for i in range(80)]
    zero_usable = [_clean_case(case_type="Judicial Review", outcome_name="Terminated")
                   for _ in range(20)]  # 20% of arrivals, all dropped -> empty group -> proxy
    with pytest.warns(UserWarning, match="proxied case type"):
        est = estimate_lognormal_duration_params(_df(*fittable, *zero_usable), DATAPULL)
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
