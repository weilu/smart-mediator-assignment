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
)
from smart_mediator_assignment.core.case import CaseProtocol
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
