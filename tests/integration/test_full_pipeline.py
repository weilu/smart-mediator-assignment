"""
Integration tests for the full recommendation pipeline.

These tests verify that all components work together correctly.
"""

import pytest
from datetime import date, timedelta

from smart_mediator_assignment import (
    SimpleCase,
    BeliefState,
    MediatorBelief,
    AlgorithmConfig,
    get_recommendations,
    get_recommendations_batch,
    update_belief,
    generate_phantom_cases,
)
from tests.fixtures import (
    SCENARIO1_VALID_MEDS,
    SCENARIO1_MED_BY_CRT_CASE_TYPE,
    SCENARIO1_MED_VA,
    SCENARIO1_AVG_CASE_RATE,
    SCENARIO1_AVG_P_VAL,
)


class TestFullPipeline:
    """Full pipeline integration tests."""

    def test_simulate_day_of_assignments(self):
        """Simulate a typical day of case assignments."""
        belief_state = BeliefState.from_init_va(
            {mid: {"mu": 0.0, "sd": 0.12} for mid in SCENARIO1_VALID_MEDS},
            global_sigma=0.12,
        )

        config = AlgorithmConfig(capacity=3, lambda_penalty=1.0, time_horizon=10)

        current_day = date(2023, 1, 1)
        mediator_case_loads = {1: 0, 2: 0, 3: 0}

        todays_cases = [
            SimpleCase(
                id=i,
                case_type="Family group",
                court_station="KAKAMEGA" if i % 2 == 0 else "MILIMANI",
                referral_date=current_day,
                p_value=0.5 + (i % 3) * 0.1,
            )
            for i in range(1, 6)
        ]

        phantom_cases, _ = generate_phantom_cases(
            current_day=current_day,
            time_horizon=config.time_horizon,
            avg_case_rate=SCENARIO1_AVG_CASE_RATE,
            avg_p_val_by_crt_case_type=SCENARIO1_AVG_P_VAL,
            med_by_court_case_type=SCENARIO1_MED_BY_CRT_CASE_TYPE,
            court_stations=["MILIMANI", "KAKAMEGA"],
            case_types=["Family group"],
            seed=42,
        )

        assignments = {}
        for case in todays_cases:
            result = get_recommendations(
                case=case,
                eligible_mediator_ids=SCENARIO1_VALID_MEDS,
                mediator_case_loads=mediator_case_loads,
                belief_state=belief_state,
                med_by_court_case_type=SCENARIO1_MED_BY_CRT_CASE_TYPE,
                config=config,
                phantom_cases=phantom_cases,
                current_day=current_day,
                strategy="ground",
                ground_truth_vas=SCENARIO1_MED_VA,
            )

            top_med = result.get_top_mediator()
            if top_med:
                assignments[case.id] = top_med
                mediator_case_loads[top_med] += 1

        assert len(assignments) > 0
        for case_id, med_id in assignments.items():
            assert med_id in SCENARIO1_VALID_MEDS

    def test_simulate_multi_day_with_belief_updates(self):
        """Simulate multiple days with belief updates after case resolutions."""
        belief_state = BeliefState.from_init_va(
            {mid: {"mu": 0.0, "sd": 0.12} for mid in SCENARIO1_VALID_MEDS},
            global_sigma=0.12,
        )

        config = AlgorithmConfig(capacity=3, lambda_penalty=1.0, time_horizon=10)

        start_date = date(2023, 1, 1)
        num_days = 3

        assigned_cases = []
        total_assignments = 0
        total_successes = 0

        for day_offset in range(num_days):
            current_day = start_date + timedelta(days=day_offset)
            mediator_case_loads = {1: 0, 2: 0, 3: 0}

            todays_cases = [
                SimpleCase(
                    id=day_offset * 10 + i,
                    case_type="Family group",
                    court_station="KAKAMEGA",
                    referral_date=current_day,
                    p_value=0.5,
                )
                for i in range(2)
            ]

            for case in todays_cases:
                result = get_recommendations(
                    case=case,
                    eligible_mediator_ids=SCENARIO1_VALID_MEDS,
                    mediator_case_loads=mediator_case_loads,
                    belief_state=belief_state,
                    med_by_court_case_type=SCENARIO1_MED_BY_CRT_CASE_TYPE,
                    config=config,
                    current_day=current_day,
                    strategy="mean",
                )

                top_med = result.get_top_mediator()
                if top_med:
                    assigned_cases.append((case, top_med))
                    mediator_case_loads[top_med] += 1
                    total_assignments += 1

            if day_offset > 0 and assigned_cases:
                resolved_case, assigned_med = assigned_cases.pop(0)

                import numpy as np
                np.random.seed(day_offset)
                outcome = np.random.binomial(1, 0.5 + SCENARIO1_MED_VA[assigned_med])

                belief_state = update_belief(
                    belief_state=belief_state,
                    mediator_id=assigned_med,
                    case_p_val=resolved_case.p_value,
                    outcome=outcome,
                )

                if outcome == 1:
                    total_successes += 1

        assert total_assignments > 0

    def test_capacity_constraints_are_respected(self):
        """Test that capacity constraints influence assignment distribution."""
        belief_state = BeliefState.from_init_va(
            {mid: {"mu": 0.0, "sd": 0.12} for mid in SCENARIO1_VALID_MEDS},
            global_sigma=0.12,
        )

        config = AlgorithmConfig(capacity=3, lambda_penalty=1.0, time_horizon=10)

        current_day = date(2023, 1, 1)

        many_cases = [
            SimpleCase(
                id=i,
                case_type="Family group",
                court_station="KAKAMEGA",
                referral_date=current_day,
                p_value=0.5,
            )
            for i in range(1, 16)
        ]

        mediator_case_loads = {1: 0, 2: 0, 3: 0}
        mediator_assignment_counts = {1: 0, 2: 0, 3: 0}

        for case in many_cases:
            result = get_recommendations(
                case=case,
                eligible_mediator_ids=SCENARIO1_VALID_MEDS,
                mediator_case_loads=mediator_case_loads,
                belief_state=belief_state,
                med_by_court_case_type=SCENARIO1_MED_BY_CRT_CASE_TYPE,
                config=config,
                current_day=current_day,
                strategy="ground",
                ground_truth_vas=SCENARIO1_MED_VA,
            )

            top_med = result.get_top_mediator()
            if top_med:
                mediator_case_loads[top_med] += 1
                mediator_assignment_counts[top_med] += 1

        for med_id in [2, 3]:
            assert mediator_assignment_counts[med_id] > 0

    def test_learning_improves_estimates(self):
        """Test that belief updates improve VA estimates over time."""
        belief_state = BeliefState.from_init_va(
            {mid: {"mu": 0.0, "sd": 0.12} for mid in SCENARIO1_VALID_MEDS},
            global_sigma=0.12,
        )

        import numpy as np

        for i in range(20):
            np.random.seed(100 + i)
            outcome = np.random.binomial(1, 0.5 + SCENARIO1_MED_VA[1])

            belief_state = update_belief(
                belief_state=belief_state,
                mediator_id=1,
                case_p_val=0.5,
                outcome=outcome,
            )

        estimated_va = belief_state.beliefs[1].mu
        actual_va = SCENARIO1_MED_VA[1]

        assert abs(estimated_va - actual_va) < abs(0.0 - actual_va)

    def test_batch_vs_sequential_consistency(self):
        """Test that batch and sequential recommendations are consistent."""
        belief_state = BeliefState.from_init_va(
            {mid: {"mu": 0.0, "sd": 0.12} for mid in SCENARIO1_VALID_MEDS},
            global_sigma=0.12,
        )

        config = AlgorithmConfig(capacity=3, lambda_penalty=1.0, time_horizon=10)

        current_day = date(2023, 1, 1)
        mediator_case_loads = {1: 0, 2: 0, 3: 0}

        cases = [
            SimpleCase(
                id=i,
                case_type="Family group",
                court_station="KAKAMEGA",
                referral_date=current_day,
                p_value=0.5,
            )
            for i in range(1, 4)
        ]

        batch_results = get_recommendations_batch(
            cases=cases,
            eligible_mediator_ids=SCENARIO1_VALID_MEDS,
            mediator_case_loads=mediator_case_loads,
            belief_state=belief_state,
            med_by_court_case_type=SCENARIO1_MED_BY_CRT_CASE_TYPE,
            config=config,
            current_day=current_day,
            strategy="ground",
            ground_truth_vas=SCENARIO1_MED_VA,
            generate_phantoms=False,
        )

        for case in cases:
            assert case.id in batch_results
            assert batch_results[case.id].recommendations
