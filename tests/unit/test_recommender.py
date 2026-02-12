import pytest
from datetime import date

from smart_mediator_assignment import (
    SimpleCase,
    BeliefState,
    MediatorBelief,
    AlgorithmConfig,
    get_recommendations,
    get_recommendations_batch,
)
from tests.fixtures import (
    SCENARIO1_VALID_MEDS,
    SCENARIO1_MED_BY_CRT_CASE_TYPE,
    SCENARIO1_MED_VA,
    SCENARIO1_AVG_CASE_RATE,
    SCENARIO1_AVG_P_VAL,
    SCENARIO2_VALID_MEDS,
    SCENARIO2_MED_BY_CRT_CASE_TYPE,
    SCENARIO2_MED_VA,
)


def make_belief_state(med_va, global_sigma=0.12):
    """Create a belief state from VA dictionary."""
    beliefs = {
        mid: MediatorBelief(mediator_id=mid, mu=va, sigma=global_sigma, case_history=([], []))
        for mid, va in med_va.items()
    }
    return BeliefState(beliefs=beliefs, global_sigma=global_sigma)


class TestGetRecommendations:
    """Tests for get_recommendations."""

    def test_returns_recommendation_result(self):
        """Test that result has correct structure."""
        case = SimpleCase(
            id=1,
            case_type="Family group",
            court_station="KAKAMEGA",
            referral_date=date(2023, 1, 1),
            p_value=0.5,
        )

        belief_state = make_belief_state(SCENARIO1_MED_VA)
        config = AlgorithmConfig(capacity=3, lambda_penalty=1.0, time_horizon=10)

        result = get_recommendations(
            case=case,
            eligible_mediator_ids=SCENARIO1_VALID_MEDS,
            mediator_case_loads={1: 0, 2: 0, 3: 0},
            belief_state=belief_state,
            med_by_court_case_type=SCENARIO1_MED_BY_CRT_CASE_TYPE,
            config=config,
            current_day=date(2023, 1, 1),
        )

        assert result.case_id == 1
        assert len(result.recommendations) > 0

    def test_top_mediator_is_best_va(self):
        """Test that top mediator has highest VA when loads are equal."""
        case = SimpleCase(
            id=1,
            case_type="Family group",
            court_station="KAKAMEGA",
            referral_date=date(2023, 1, 1),
            p_value=0.5,
        )

        belief_state = make_belief_state(SCENARIO1_MED_VA)
        config = AlgorithmConfig(capacity=3, lambda_penalty=1.0, time_horizon=10)

        result = get_recommendations(
            case=case,
            eligible_mediator_ids=SCENARIO1_VALID_MEDS,
            mediator_case_loads={1: 0, 2: 0, 3: 0},
            belief_state=belief_state,
            med_by_court_case_type=SCENARIO1_MED_BY_CRT_CASE_TYPE,
            config=config,
            current_day=date(2023, 1, 1),
        )

        top_mediator = result.get_top_mediator()
        assert top_mediator == 1

    def test_respects_eligibility(self):
        """Test that only eligible mediators are recommended."""
        case = SimpleCase(
            id=1,
            case_type="Family group",
            court_station="MILIMANI",
            referral_date=date(2023, 1, 1),
            p_value=0.5,
        )

        belief_state = make_belief_state(SCENARIO1_MED_VA)
        config = AlgorithmConfig(capacity=3, lambda_penalty=1.0, time_horizon=10)

        result = get_recommendations(
            case=case,
            eligible_mediator_ids=SCENARIO1_VALID_MEDS,
            mediator_case_loads={1: 0, 2: 0, 3: 0},
            belief_state=belief_state,
            med_by_court_case_type=SCENARIO1_MED_BY_CRT_CASE_TYPE,
            config=config,
            current_day=date(2023, 1, 1),
        )

        assert result.get_top_mediator() == 1
        assert len(result.recommendations) == 1

    def test_considers_case_loads(self):
        """Test that high case loads influence assignment."""
        case = SimpleCase(
            id=1,
            case_type="Family group",
            court_station="KAKAMEGA",
            referral_date=date(2023, 1, 1),
            p_value=0.5,
        )

        belief_state = make_belief_state(SCENARIO1_MED_VA)
        config = AlgorithmConfig(capacity=3, lambda_penalty=1.0, time_horizon=10)

        result = get_recommendations(
            case=case,
            eligible_mediator_ids=SCENARIO1_VALID_MEDS,
            mediator_case_loads={1: 3, 2: 0, 3: 0},
            belief_state=belief_state,
            med_by_court_case_type=SCENARIO1_MED_BY_CRT_CASE_TYPE,
            config=config,
            current_day=date(2023, 1, 1),
        )

        top_mediator = result.get_top_mediator()
        assert top_mediator in [2, 3]

    def test_ground_truth_strategy(self):
        """Test that ground truth strategy uses provided VAs."""
        case = SimpleCase(
            id=1,
            case_type="Family group",
            court_station="KAKAMEGA",
            referral_date=date(2023, 1, 1),
            p_value=0.5,
        )

        belief_state = make_belief_state({1: -0.1, 2: 0.2, 3: 0.0})
        ground_truth = {1: 0.3, 2: -0.1, 3: 0.0}

        config = AlgorithmConfig(capacity=3, lambda_penalty=1.0, time_horizon=10)

        result = get_recommendations(
            case=case,
            eligible_mediator_ids=SCENARIO1_VALID_MEDS,
            mediator_case_loads={1: 0, 2: 0, 3: 0},
            belief_state=belief_state,
            med_by_court_case_type=SCENARIO1_MED_BY_CRT_CASE_TYPE,
            config=config,
            current_day=date(2023, 1, 1),
            strategy="ground",
            ground_truth_vas=ground_truth,
        )

        assert result.get_top_mediator() == 1

    def test_no_eligible_mediators_returns_empty(self):
        """Test that no eligible mediators returns empty recommendations."""
        case = SimpleCase(
            id=1,
            case_type="Commercial Cases",
            court_station="MILIMANI",
            referral_date=date(2023, 1, 1),
            p_value=0.5,
        )

        belief_state = make_belief_state(SCENARIO1_MED_VA)
        config = AlgorithmConfig(capacity=3, lambda_penalty=1.0, time_horizon=10)

        result = get_recommendations(
            case=case,
            eligible_mediator_ids=SCENARIO1_VALID_MEDS,
            mediator_case_loads={1: 0, 2: 0, 3: 0},
            belief_state=belief_state,
            med_by_court_case_type=SCENARIO1_MED_BY_CRT_CASE_TYPE,
            config=config,
            current_day=date(2023, 1, 1),
        )

        assert len(result.recommendations) == 0
        assert result.get_top_mediator() is None


class TestGetRecommendationsBatch:
    """Tests for get_recommendations_batch."""

    def test_batch_returns_results_for_all_cases(self):
        """Test that batch returns results for all cases."""
        cases = [
            SimpleCase(
                id=i,
                case_type="Family group",
                court_station="KAKAMEGA",
                referral_date=date(2023, 1, 1),
                p_value=0.5,
            )
            for i in range(1, 6)
        ]

        belief_state = make_belief_state(SCENARIO1_MED_VA)
        config = AlgorithmConfig(capacity=3, lambda_penalty=1.0, time_horizon=10)

        results = get_recommendations_batch(
            cases=cases,
            eligible_mediator_ids=SCENARIO1_VALID_MEDS,
            mediator_case_loads={1: 0, 2: 0, 3: 0},
            belief_state=belief_state,
            med_by_court_case_type=SCENARIO1_MED_BY_CRT_CASE_TYPE,
            config=config,
            current_day=date(2023, 1, 1),
            generate_phantoms=False,
        )

        assert len(results) == 5
        for case_id in range(1, 6):
            assert case_id in results
            assert results[case_id].case_id == case_id

    def test_batch_with_phantom_generation(self):
        """Test batch with automatic phantom case generation."""
        cases = [
            SimpleCase(
                id=1,
                case_type="Family group",
                court_station="KAKAMEGA",
                referral_date=date(2023, 1, 1),
                p_value=0.5,
            )
        ]

        belief_state = make_belief_state(SCENARIO1_MED_VA)
        config = AlgorithmConfig(capacity=3, lambda_penalty=1.0, time_horizon=10)

        results = get_recommendations_batch(
            cases=cases,
            eligible_mediator_ids=SCENARIO1_VALID_MEDS,
            mediator_case_loads={1: 0, 2: 0, 3: 0},
            belief_state=belief_state,
            med_by_court_case_type=SCENARIO1_MED_BY_CRT_CASE_TYPE,
            config=config,
            avg_case_rate=SCENARIO1_AVG_CASE_RATE,
            avg_p_val_by_crt_case_type=SCENARIO1_AVG_P_VAL,
            court_stations=["MILIMANI", "KAKAMEGA"],
            case_types=["Family group"],
            current_day=date(2023, 1, 1),
            generate_phantoms=True,
            seed=42,
        )

        assert 1 in results

    def test_empty_cases_returns_empty_dict(self):
        """Test that empty cases list returns empty dict."""
        belief_state = make_belief_state(SCENARIO1_MED_VA)
        config = AlgorithmConfig(capacity=3, lambda_penalty=1.0, time_horizon=10)

        results = get_recommendations_batch(
            cases=[],
            eligible_mediator_ids=SCENARIO1_VALID_MEDS,
            mediator_case_loads={1: 0, 2: 0, 3: 0},
            belief_state=belief_state,
            med_by_court_case_type=SCENARIO1_MED_BY_CRT_CASE_TYPE,
            config=config,
            current_day=date(2023, 1, 1),
        )

        assert results == {}


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_scenario1_full_workflow(self):
        """Test complete workflow with Scenario 1."""
        case = SimpleCase(
            id=1,
            case_type="Family group",
            court_station="KAKAMEGA",
            referral_date=date(2023, 1, 1),
            p_value=0.5,
        )

        belief_state = BeliefState.from_init_va(
            {mid: {"mu": 0.0, "sd": 0.12} for mid in SCENARIO1_VALID_MEDS},
            global_sigma=0.12,
        )

        config = AlgorithmConfig(capacity=3, lambda_penalty=1.0, time_horizon=10)

        result = get_recommendations(
            case=case,
            eligible_mediator_ids=SCENARIO1_VALID_MEDS,
            mediator_case_loads={1: 0, 2: 0, 3: 0},
            belief_state=belief_state,
            med_by_court_case_type=SCENARIO1_MED_BY_CRT_CASE_TYPE,
            config=config,
            strategy="ground",
            ground_truth_vas=SCENARIO1_MED_VA,
            current_day=date(2023, 1, 1),
        )

        assert result.get_top_mediator() == 1

    def test_scenario2_cross_station(self):
        """Test Scenario 2 with cross-station mediator."""
        cases = [
            SimpleCase(
                id=1,
                case_type="Family group",
                court_station="MILIMANI",
                referral_date=date(2023, 1, 1),
                p_value=0.5,
            ),
            SimpleCase(
                id=2,
                case_type="Family group",
                court_station="KAKAMEGA",
                referral_date=date(2023, 1, 1),
                p_value=0.5,
            ),
        ]

        belief_state = BeliefState.from_init_va(
            {mid: {"mu": 0.0, "sd": 0.12} for mid in SCENARIO2_VALID_MEDS},
            global_sigma=0.12,
        )

        config = AlgorithmConfig(capacity=3, lambda_penalty=1.0, time_horizon=10)

        results = get_recommendations_batch(
            cases=cases,
            eligible_mediator_ids=SCENARIO2_VALID_MEDS,
            mediator_case_loads={1: 0, 2: 0, 3: 0},
            belief_state=belief_state,
            med_by_court_case_type=SCENARIO2_MED_BY_CRT_CASE_TYPE,
            config=config,
            strategy="ground",
            ground_truth_vas=SCENARIO2_MED_VA,
            current_day=date(2023, 1, 1),
            generate_phantoms=False,
        )

        assert 1 in results
        assert 2 in results
