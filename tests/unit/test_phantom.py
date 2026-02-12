import pytest
from datetime import date, datetime

import numpy as np

from smart_mediator_assignment.algorithm.phantom import (
    generate_phantom_cases,
    estimate_case_arrivals,
)
from tests.fixtures import (
    SCENARIO1_AVG_CASE_RATE,
    SCENARIO1_AVG_P_VAL,
    SCENARIO1_MED_BY_CRT_CASE_TYPE,
)


class TestGeneratePhantomCases:
    """Tests for phantom case generation."""

    def test_generates_phantom_cases(self):
        """Test that phantom cases are generated."""
        phantom_cases, next_id = generate_phantom_cases(
            current_day=date(2023, 1, 1),
            time_horizon=10,
            avg_case_rate=SCENARIO1_AVG_CASE_RATE,
            avg_p_val_by_crt_case_type=SCENARIO1_AVG_P_VAL,
            med_by_court_case_type=SCENARIO1_MED_BY_CRT_CASE_TYPE,
            court_stations=["MILIMANI", "KAKAMEGA"],
            case_types=["Family group"],
            starting_id=-1,
            seed=42,
        )

        assert len(phantom_cases) > 0
        assert next_id < -1

    def test_phantom_ids_are_negative(self):
        """Test that all phantom case IDs are negative."""
        phantom_cases, _ = generate_phantom_cases(
            current_day=date(2023, 1, 1),
            time_horizon=10,
            avg_case_rate=SCENARIO1_AVG_CASE_RATE,
            avg_p_val_by_crt_case_type=SCENARIO1_AVG_P_VAL,
            med_by_court_case_type=SCENARIO1_MED_BY_CRT_CASE_TYPE,
            court_stations=["MILIMANI", "KAKAMEGA"],
            case_types=["Family group"],
            starting_id=-1,
            seed=42,
        )

        for case in phantom_cases:
            assert case.id < 0

    def test_phantom_dates_are_future(self):
        """Test that phantom cases have future dates."""
        current_day = date(2023, 1, 1)
        phantom_cases, _ = generate_phantom_cases(
            current_day=current_day,
            time_horizon=10,
            avg_case_rate=SCENARIO1_AVG_CASE_RATE,
            avg_p_val_by_crt_case_type=SCENARIO1_AVG_P_VAL,
            med_by_court_case_type=SCENARIO1_MED_BY_CRT_CASE_TYPE,
            court_stations=["MILIMANI", "KAKAMEGA"],
            case_types=["Family group"],
            starting_id=-1,
            seed=42,
        )

        for case in phantom_cases:
            assert case.referral_date >= current_day

    def test_reproducibility_with_seed(self):
        """Test that same seed produces same results."""
        args = dict(
            current_day=date(2023, 1, 1),
            time_horizon=10,
            avg_case_rate=SCENARIO1_AVG_CASE_RATE,
            avg_p_val_by_crt_case_type=SCENARIO1_AVG_P_VAL,
            med_by_court_case_type=SCENARIO1_MED_BY_CRT_CASE_TYPE,
            court_stations=["MILIMANI", "KAKAMEGA"],
            case_types=["Family group"],
            starting_id=-1,
            seed=42,
        )

        cases1, id1 = generate_phantom_cases(**args)
        cases2, id2 = generate_phantom_cases(**args)

        assert len(cases1) == len(cases2)
        assert id1 == id2
        for c1, c2 in zip(cases1, cases2):
            assert c1.id == c2.id
            assert c1.court_station == c2.court_station

    def test_different_seeds_produce_different_results(self):
        """Test that different seeds produce different results over many trials."""
        args = dict(
            current_day=date(2023, 1, 1),
            time_horizon=30,
            avg_case_rate=SCENARIO1_AVG_CASE_RATE,
            avg_p_val_by_crt_case_type=SCENARIO1_AVG_P_VAL,
            med_by_court_case_type=SCENARIO1_MED_BY_CRT_CASE_TYPE,
            court_stations=["MILIMANI", "KAKAMEGA"],
            case_types=["Family group"],
            starting_id=-1,
        )

        cases1, _ = generate_phantom_cases(**args, seed=42)
        cases2, _ = generate_phantom_cases(**args, seed=123)

        different_count = len(cases1) != len(cases2)
        if not different_count and len(cases1) > 0:
            different_dates = any(
                c1.referral_date != c2.referral_date or c1.court_station != c2.court_station
                for c1, c2 in zip(cases1, cases2)
            )
            different_count = different_dates

        assert different_count or len(cases1) == 0

    def test_zero_time_horizon(self):
        """Test that zero time horizon produces no cases."""
        phantom_cases, next_id = generate_phantom_cases(
            current_day=date(2023, 1, 1),
            time_horizon=0,
            avg_case_rate=SCENARIO1_AVG_CASE_RATE,
            avg_p_val_by_crt_case_type=SCENARIO1_AVG_P_VAL,
            med_by_court_case_type=SCENARIO1_MED_BY_CRT_CASE_TYPE,
            court_stations=["MILIMANI", "KAKAMEGA"],
            case_types=["Family group"],
            starting_id=-1,
            seed=42,
        )

        assert len(phantom_cases) == 0
        assert next_id == -1

    def test_uses_avg_p_val(self):
        """Test that p-values come from avg_p_val mapping."""
        phantom_cases, _ = generate_phantom_cases(
            current_day=date(2023, 1, 1),
            time_horizon=10,
            avg_case_rate=SCENARIO1_AVG_CASE_RATE,
            avg_p_val_by_crt_case_type=SCENARIO1_AVG_P_VAL,
            med_by_court_case_type=SCENARIO1_MED_BY_CRT_CASE_TYPE,
            court_stations=["MILIMANI", "KAKAMEGA"],
            case_types=["Family group"],
            starting_id=-1,
            seed=42,
        )

        for case in phantom_cases:
            assert case.p_value == pytest.approx(0.4, abs=0.01)

    def test_datetime_support(self):
        """Test that datetime objects work as well as date objects."""
        phantom_cases, _ = generate_phantom_cases(
            current_day=datetime(2023, 1, 1, 10, 30),
            time_horizon=10,
            avg_case_rate=SCENARIO1_AVG_CASE_RATE,
            avg_p_val_by_crt_case_type=SCENARIO1_AVG_P_VAL,
            med_by_court_case_type=SCENARIO1_MED_BY_CRT_CASE_TYPE,
            court_stations=["MILIMANI", "KAKAMEGA"],
            case_types=["Family group"],
            starting_id=-1,
            seed=42,
        )

        assert len(phantom_cases) > 0


class TestEstimateCaseArrivals:
    """Tests for case arrival estimation."""

    def test_estimates_arrivals(self):
        """Test that arrivals are estimated correctly."""
        estimates = estimate_case_arrivals(
            avg_case_rate=SCENARIO1_AVG_CASE_RATE,
            court_stations=["MILIMANI", "KAKAMEGA"],
            case_types=["Family group"],
            days=1,
        )

        assert ("Family group", "MILIMANI") in estimates
        assert ("Family group", "KAKAMEGA") in estimates
        assert estimates[("Family group", "MILIMANI")] == pytest.approx(0.055)
        assert estimates[("Family group", "KAKAMEGA")] == pytest.approx(0.05)

    def test_multi_day_estimation(self):
        """Test estimation over multiple days."""
        estimates = estimate_case_arrivals(
            avg_case_rate=SCENARIO1_AVG_CASE_RATE,
            court_stations=["MILIMANI", "KAKAMEGA"],
            case_types=["Family group"],
            days=10,
        )

        assert estimates[("Family group", "MILIMANI")] == pytest.approx(0.55)
        assert estimates[("Family group", "KAKAMEGA")] == pytest.approx(0.5)

    def test_missing_case_type(self):
        """Test that missing case types are skipped."""
        estimates = estimate_case_arrivals(
            avg_case_rate=SCENARIO1_AVG_CASE_RATE,
            court_stations=["MILIMANI"],
            case_types=["Unknown Type"],
            days=1,
        )

        assert len(estimates) == 0

    def test_missing_court_station(self):
        """Test that missing court stations are skipped."""
        estimates = estimate_case_arrivals(
            avg_case_rate=SCENARIO1_AVG_CASE_RATE,
            court_stations=["Unknown Station"],
            case_types=["Family group"],
            days=1,
        )

        assert len(estimates) == 0
