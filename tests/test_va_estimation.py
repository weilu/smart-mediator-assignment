"""
Tests for VA estimation module.
"""

import pytest
from dataclasses import dataclass
from datetime import datetime, date
from typing import Optional, Union

from smart_mediator_assignment.algorithm.va_estimation import (
    VAEstimationConfig,
    MediatorVAEstimate,
    CasePrediction,
    VAEstimationResult,
    estimate_va,
    estimate_va_from_prepared,
)
from smart_mediator_assignment.algorithm import va_estimation
from smart_mediator_assignment.core.case import CaseProtocol, SimpleCase
import pandas as pd
import numpy as np


@dataclass
class MockCase:
    """Mock case implementing CaseProtocol (unified protocol)."""

    id: int
    mediator_id: Optional[int]
    case_outcome_agreement: Optional[int]
    mediator_appointment_date: Optional[Union[date, datetime]]
    referral_date: Union[date, datetime]
    conclusion_date: Optional[Union[date, datetime]]
    case_status: str
    case_type: str
    court_station: str
    court_type: str
    referral_mode: str
    p_value: Optional[float] = None


class TestCaseProtocol:
    """Test that MockCase conforms to the protocol."""

    def test_mock_case_conforms_to_protocol(self):
        case = MockCase(
            id=1,
            mediator_id=100,
            case_outcome_agreement=1,
            mediator_appointment_date=datetime(2023, 1, 15),
            referral_date=datetime(2023, 1, 1),
            conclusion_date=datetime(2023, 3, 15),
            case_status="CONCLUDED",
            case_type="Family group",
            court_station="MILIMANI",
            court_type="Magistrate Court",
            referral_mode="Referred by Court",
        )
        assert isinstance(case, CaseProtocol)


def _case(cid, appt, referral, outcome=1, mediator_id=1):
    return SimpleCase(id=cid, case_type="Family group", court_station="MILIMANI",
        referral_date=referral, p_value=0.5, mediator_id=mediator_id, case_outcome_agreement=outcome,
        mediator_appointment_date=appt, conclusion_date=appt, case_status="CONCLUDED",
        court_type="Magistrate", referral_mode="Referred by Court")


def test_window_filters_on_appointment_date_not_referral():
    inside = _case(1, appt=date(2022, 6, 1), referral=date(2019, 1, 1))
    outside = _case(2, appt=date(2018, 1, 1), referral=date(2022, 6, 1))
    # Add more cases for regression stability
    padding = [_case(cid, appt=date(2022, 6, 1), referral=date(2022, 1, 1), mediator_id=2+i)
               for i, cid in enumerate(range(3, 35))]
    cfg = VAEstimationConfig(reference_date=datetime(2023, 6, 1), days_since_appt_threshold=0)
    result = estimate_va([inside, outside] + padding, config=cfg, start_date="2022-01-01", end_date="2023-01-01")
    ids = {c.case_id for c in result.case_predictions}
    assert 1 in ids and 2 not in ids


def _mk(cid, med, outcome):
    return SimpleCase(id=cid, case_type="Family group", court_station="MILIMANI",
        referral_date=date(2022, 1, 1), p_value=0.5, mediator_id=med, case_outcome_agreement=outcome,
        mediator_appointment_date=date(2022, 1, 10), conclusion_date=date(2022, 2, 1),
        case_status="CONCLUDED", court_type="Magistrate", referral_mode="Referred by Court")


def test_p_pred_present_for_singleton_mediator_case():
    cases = [_mk(1, 1, 1), _mk(2, 1, 0), _mk(3, 2, 1), _mk(4, 2, 0), _mk(99, 3, 1)]
    cfg = VAEstimationConfig(reference_date=datetime(2023, 6, 1), days_since_appt_threshold=0, min_med_cases=2)
    result = estimate_va(cases, config=cfg, start_date="2021-01-01", end_date="2023-01-01")
    preds = {c.case_id: c.p_pred for c in result.case_predictions}
    assert 99 in preds and preds[99] == preds[99]  # present and not NaN


def _station_case(cid, station, mediator, outcome):
    # Fixing case_type/appt date/referral_mode/court_type/outcome to the fitted
    # model's reference categories isolates court_station's own contribution to
    # p_pred: any two such cases should differ only by their station coefficient.
    return SimpleCase(id=cid, case_type="Divorce and Separation", court_station=station,
        referral_date=date(2022, 1, 1), p_value=0.5, mediator_id=mediator, case_outcome_agreement=outcome,
        mediator_appointment_date=date(2022, 1, 10), conclusion_date=date(2022, 2, 1),
        case_status="CONCLUDED", court_type="Magistrate", referral_mode="Referred by Court")


def test_small_court_station_coefficient_is_applied_not_zeroed():
    """df_case must be relabeled 'zzzSmall' (backfilled from df) for stations collapsed
    below min_court_station_cases, or the zzzSmall param merge never matches and the
    station's coefficient is silently zeroed via skipna=True (VA_Antoine.py step 4.1)."""
    cases = []
    # 10 mediators x 2 MILIMANI cases each. Mediators 1 and 2 (which also sit at TINY,
    # below) get outcome=0 at MILIMANI so their within-mediator jump to TINY's outcome=1
    # is only explainable by court_station, not by their (absorbed) mediator fixed effect.
    # Other mediators just add baseline variation for regression stability.
    for m in range(1, 11):
        milimani_outcome = 0 if m in (1, 2) else (1 if m <= 5 else 0)
        cases.append(_station_case(100 + 2 * m, "MILIMANI", m, milimani_outcome))
        cases.append(_station_case(101 + 2 * m, "MILIMANI", m, milimani_outcome))

    # A station with only 2 concluded cases collapses to 'zzzSmall' (threshold below).
    cases.append(_station_case(9001, "TINY", 1, outcome=1))
    cases.append(_station_case(9002, "TINY", 2, outcome=1))

    cfg = VAEstimationConfig(reference_date=datetime(2023, 6, 1), days_since_appt_threshold=0,
                              min_med_cases=2, min_court_station_cases=2)
    result = estimate_va(cases, config=cfg, start_date="2021-01-01", end_date="2023-01-01")
    preds = {c.case_id: c.p_pred for c in result.case_predictions}

    # Mediator 1's MILIMANI case shares every covariate with its TINY case except
    # court_station (and the intercept term is keyed only on outcome=0/1, both mapped to
    # the same single fitted intercept, so it cancels regardless). MILIMANI
    # ('AAAMilimani') is the omitted reference category (coefficient 0 by construction),
    # so any difference here is exactly TINY's zzzSmall contribution.
    milimani_reference_p_pred = preds[100 + 2 * 1]
    tiny_p_pred = preds[9001]

    assert abs(tiny_p_pred - milimani_reference_p_pred) > 1e-6, (
        f"small-station coefficient not applied: tiny={tiny_p_pred} vs "
        f"milimani_reference={milimani_reference_p_pred}"
    )


class TestVAEstimationConfig:
    """Test VAEstimationConfig."""

    def test_default_config(self):
        config = VAEstimationConfig.default()
        assert config.min_med_cases == 2
        assert config.days_since_appt_threshold == 180
        assert isinstance(config.reference_date, datetime)

    def test_custom_config(self):
        config = VAEstimationConfig(
            reference_date=datetime(2025, 1, 1),
            min_med_cases=5,
            min_court_station_cases=50,
        )
        assert config.min_med_cases == 5
        assert config.min_court_station_cases == 50


class TestVAEstimationResult:
    """Test VAEstimationResult dataclass."""

    def test_get_va_dict(self):
        result = VAEstimationResult(
            mediator_vas=[
                MediatorVAEstimate(mediator_id=1, va=0.1, n_cases=10),
                MediatorVAEstimate(mediator_id=2, va=-0.05, n_cases=5),
            ],
            case_predictions=[],
            sigma=0.1,
        )

        va_dict = result.get_va_dict()
        assert va_dict == {1: 0.1, 2: -0.05}

    def test_get_p_pred_dict(self):
        result = VAEstimationResult(
            mediator_vas=[],
            case_predictions=[
                CasePrediction(case_id=100, mediator_id=1, p_pred=0.5, va=0.1, case_outcome_agreement=1),
                CasePrediction(case_id=101, mediator_id=2, p_pred=0.4, va=-0.05, case_outcome_agreement=0),
            ],
            sigma=0.1,
        )

        p_dict = result.get_p_pred_dict()
        assert p_dict == {100: 0.5, 101: 0.4}


def generate_synthetic_cases(
    n_cases: int = 500,
    n_mediators: int = 20,
    seed: int = 42
) -> list:
    """Generate synthetic cases for testing."""
    np.random.seed(seed)

    case_types = ['Divorce and Separation', 'Civil Cases', 'Commercial Cases']
    court_stations = ['MILIMANI', 'KAKAMEGA', 'MOMBASA', 'KISUMU']
    court_types = ['Magistrate Court', 'High Court']
    referral_modes = ['Referred by Court', 'Request by Parties', 'Screened from Registry']

    # Generate mediator VAs (ground truth)
    mediator_vas = {i: np.random.normal(0, 0.1) for i in range(1, n_mediators + 1)}

    cases = []
    base_date = datetime(2020, 1, 1)

    for i in range(n_cases):
        # Random dates
        days_offset = np.random.randint(0, 365 * 3)
        referral_date = datetime(2020, 1, 1) + pd.Timedelta(days=int(days_offset))
        appt_date = referral_date + pd.Timedelta(days=np.random.randint(1, 30))
        conclusion_date = appt_date + pd.Timedelta(days=np.random.randint(30, 180))

        mediator_id = np.random.randint(1, n_mediators + 1)
        case_type = np.random.choice(case_types)
        court_station = np.random.choice(court_stations)

        # Simulate outcome based on base rate + mediator VA
        base_prob = 0.4
        prob = base_prob + mediator_vas[mediator_id]
        outcome = 1 if np.random.random() < prob else 0

        cases.append(MockCase(
            id=i + 1,
            mediator_id=mediator_id,
            case_outcome_agreement=outcome,
            mediator_appointment_date=appt_date,
            referral_date=referral_date,
            conclusion_date=conclusion_date,
            case_status="CONCLUDED",
            case_type=case_type,
            court_station=court_station,
            court_type=np.random.choice(court_types),
            referral_mode=np.random.choice(referral_modes),
        ))

    return cases


@pytest.mark.slow
class TestEstimateVAIntegration:
    """Integration tests for estimate_va function."""

    @pytest.fixture(scope="class")
    def synthetic_cases(self):
        """Generate synthetic cases for testing."""
        return generate_synthetic_cases(n_cases=1000, n_mediators=30, seed=42)

    def test_estimate_va_returns_result(self, synthetic_cases):
        """Test that estimate_va returns a VAEstimationResult."""
        config = VAEstimationConfig(
            reference_date=datetime(2025, 1, 1),
            min_med_cases=2,
            days_since_appt_threshold=30,
        )

        result = estimate_va(
            cases=synthetic_cases,
            config=config,
            start_date="2020-01-01",
            end_date="2023-01-01",
        )

        assert isinstance(result, VAEstimationResult)
        assert len(result.mediator_vas) > 0
        assert len(result.case_predictions) > 0
        assert result.sigma > 0

    def test_va_values_reasonable(self, synthetic_cases):
        """Test that VA values are in reasonable range."""
        config = VAEstimationConfig(
            reference_date=datetime(2025, 1, 1),
            min_med_cases=2,
            days_since_appt_threshold=30,
        )

        result = estimate_va(
            cases=synthetic_cases,
            config=config,
            start_date="2020-01-01",
            end_date="2023-01-01",
        )

        vas = [m.va for m in result.mediator_vas]
        assert all(-1 < va < 1 for va in vas), f"VA values out of range: {vas}"

    def test_p_pred_values_reasonable(self, synthetic_cases):
        """Test that p_pred values are in reasonable range."""
        config = VAEstimationConfig(
            reference_date=datetime(2025, 1, 1),
            min_med_cases=2,
            days_since_appt_threshold=30,
        )

        result = estimate_va(
            cases=synthetic_cases,
            config=config,
            start_date="2020-01-01",
            end_date="2023-01-01",
        )

        p_preds = [c.p_pred for c in result.case_predictions]
        assert all(-0.5 < p < 1.5 for p in p_preds), f"p_pred values out of range"

    def test_sigma_positive(self, synthetic_cases):
        """Test that sigma is positive."""
        config = VAEstimationConfig(
            reference_date=datetime(2025, 1, 1),
            min_med_cases=2,
            days_since_appt_threshold=30,
        )

        result = estimate_va(
            cases=synthetic_cases,
            config=config,
            start_date="2020-01-01",
            end_date="2023-01-01",
        )

        assert result.sigma > 0

    def test_date_filtering(self, synthetic_cases):
        """Test that date filtering works correctly."""
        config = VAEstimationConfig(
            reference_date=datetime(2025, 1, 1),
            min_med_cases=2,
            days_since_appt_threshold=30,
        )

        result_wide = estimate_va(
            cases=synthetic_cases,
            config=config,
            start_date="2020-01-01",
            end_date="2023-01-01",
        )

        result_narrow = estimate_va(
            cases=synthetic_cases,
            config=config,
            start_date="2021-01-01",
            end_date="2022-01-01",
        )

        assert len(result_narrow.case_predictions) < len(result_wide.case_predictions)


def test_quasiyear_uses_true_month_end_not_day_28():
    # Reference at end of June (30 days). Appt 2023-06-29 is within the most recent year
    # window -> bucket 0. The old day-28 upper bound (2023-06-28) would drop it to NaN.
    # The older 2010 row is an anchor so the oldest-bucket collapse targets it, not row 0.
    df = pd.DataFrame({'med_appt_date': pd.to_datetime(['2023-06-29', '2010-01-01'])})
    out = va_estimation._assign_quasiyear(df, datetime(2023, 6, 30))
    assert out.loc[0, 'quasiyear'] == 0


def test_quasiyear_collapses_short_oldest_bucket():
    # Two appts ~1 month apart in the oldest reachable window -> the short oldest bucket
    # is merged into the previous one, so both share one quasiyear value.
    df = pd.DataFrame({'med_appt_date': pd.to_datetime(['1994-06-10', '1994-07-10'])})
    out = va_estimation._assign_quasiyear(df, datetime(2023, 6, 30))
    assert out['quasiyear'].nunique() == 1


def _df(ct):
    c = SimpleCase(id=1, case_type=ct, court_station="MILIMANI", referral_date=date(2022,1,1),
        p_value=0.5, mediator_id=1, case_outcome_agreement=1,
        mediator_appointment_date=date(2022,1,1), conclusion_date=date(2022,2,1),
        case_status="CONCLUDED", court_type="Magistrate", referral_mode="Referred by Court")
    return va_estimation._cases_to_dataframe([c])


def test_commercial_and_tax_group_mapping():
    for ct in ("Commercial Cases", "Tax Appeals"):
        out = va_estimation._simplify_case_types(_df(ct))
        assert out.loc[0, "casetype_simplified"] == "Commercial and tax group"


def _mk_prepared_case(cid, appt, outcome):
    return SimpleCase(id=cid, case_type="Family group", court_station="MILIMANI",
        referral_date=date(2021, 1, 1), p_value=0.5, mediator_id=1, case_outcome_agreement=outcome,
        mediator_appointment_date=appt, conclusion_date=appt, case_status="CONCLUDED",
        court_type="Magistrate", referral_mode="Referred by Court")


def test_estimate_va_returns_prepared_frame():
    cases = [_mk_prepared_case(i, date(2022, 1, 5 + i), i % 2) for i in range(6)]
    cfg = VAEstimationConfig(reference_date=datetime(2023, 6, 1), days_since_appt_threshold=0)
    result = estimate_va(cases, config=cfg, start_date="2021-01-01", end_date="2023-01-01")
    assert result.prepared is not None


def test_estimate_va_from_prepared_reuses_frame_on_subwindow():
    cases = [_mk_prepared_case(i, date(2022, 1, 5 + i), i % 2) for i in range(6)]
    cfg = VAEstimationConfig(reference_date=datetime(2023, 6, 1), days_since_appt_threshold=0)
    prepared = estimate_va(cases, config=cfg, start_date="2021-01-01", end_date="2023-01-01").prepared
    second = estimate_va_from_prepared(prepared, config=cfg,
                                       start_date="2022-01-01", end_date="2022-01-08")
    ids = {c.case_id for c in second.case_predictions}
    assert ids and ids.issubset({0, 1, 2})


def test_estimate_va_from_prepared_matches_fresh_on_same_window():
    # Reuse on the SAME window the prepared frame was built from must reproduce the
    # fresh VA exactly: the fitted `df` (and thus mediator_vas) is identical either way.
    cases = [_mk_prepared_case(i, date(2022, 1, 5 + i), i % 2) for i in range(6)]
    cfg = VAEstimationConfig(reference_date=datetime(2023, 6, 1), days_since_appt_threshold=0)
    fresh = estimate_va(cases, config=cfg, start_date="2021-01-01", end_date="2023-01-01")
    reused = estimate_va_from_prepared(fresh.prepared, config=cfg,
                                       start_date="2021-01-01", end_date="2023-01-01")
    fresh_vas = fresh.get_va_dict()
    reused_vas = reused.get_va_dict()
    # NaN != NaN under Python equality, and this fixture's single-mediator/6-case
    # window makes the shrinkage covariance degenerate (both paths produce NaN VA
    # here) -- so compare keys plus np.isclose(equal_nan=True) rather than a plain
    # dict `==`, per the brief's documented float-determinism fallback.
    assert reused_vas.keys() == fresh_vas.keys()
    for mediator_id, fresh_va in fresh_vas.items():
        assert np.isclose(reused_vas[mediator_id], fresh_va, atol=1e-9, equal_nan=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
