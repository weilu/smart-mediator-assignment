import pytest
from datetime import date, datetime

from smart_mediator_assignment.core import SimpleCase
from smart_mediator_assignment.solver import LPSolver
from smart_mediator_assignment.config import AlgorithmConfig
from tests.fixtures import (
    SCENARIO1_VALID_MEDS,
    SCENARIO1_MED_BY_CRT_CASE_TYPE,
    SCENARIO1_MED_VA,
    SCENARIO2_VALID_MEDS,
    SCENARIO2_MED_BY_CRT_CASE_TYPE,
    SCENARIO2_MED_VA,
)


class TestLPSolver:
    """Tests for the LP solver."""

    def test_basic_assignment_single_case(self):
        """Test that a single case gets assigned to the only available mediator."""
        case = SimpleCase(
            id=1,
            case_type_id="Family group",
            court_station_id="MILIMANI",
            referral_date=date(2023, 1, 1),
            p_value=0.5,
        )

        solver = LPSolver(
            valid_mediators=SCENARIO1_VALID_MEDS,
            mediator_case_loads={1: 0, 2: 0, 3: 0},
            capacity=3,
            mediator_vas=SCENARIO1_MED_VA,
            med_by_court_case_type=SCENARIO1_MED_BY_CRT_CASE_TYPE,
            lambda_penalty=1.0,
            time_horizon=10,
            use_gurobi=False,
        )

        result = solver.solve([case], current_day=date(2023, 1, 1))

        assert 1 in result
        assert len(result[1]) == 1
        assigned_mediator = result[1][0][0]
        assert assigned_mediator == 1

    def test_assignment_multiple_mediators_available(self):
        """Test assignment when multiple mediators are available."""
        case = SimpleCase(
            id=1,
            case_type_id="Family group",
            court_station_id="KAKAMEGA",
            referral_date=date(2023, 1, 1),
            p_value=0.5,
        )

        solver = LPSolver(
            valid_mediators=SCENARIO1_VALID_MEDS,
            mediator_case_loads={1: 0, 2: 0, 3: 0},
            capacity=3,
            mediator_vas=SCENARIO1_MED_VA,
            med_by_court_case_type=SCENARIO1_MED_BY_CRT_CASE_TYPE,
            lambda_penalty=1.0,
            time_horizon=10,
            use_gurobi=False,
        )

        result = solver.solve([case], current_day=date(2023, 1, 1))

        assert 1 in result
        assigned_mediator = result[1][0][0]
        assert assigned_mediator == 1

    def test_assignment_respects_capacity(self):
        """Test that assignments respect capacity constraints."""
        cases = [
            SimpleCase(
                id=i,
                case_type_id="Family group",
                court_station_id="KAKAMEGA",
                referral_date=date(2023, 1, 1),
                p_value=0.5,
            )
            for i in range(1, 10)
        ]

        solver = LPSolver(
            valid_mediators=SCENARIO1_VALID_MEDS,
            mediator_case_loads={1: 0, 2: 0, 3: 0},
            capacity=3,
            mediator_vas=SCENARIO1_MED_VA,
            med_by_court_case_type=SCENARIO1_MED_BY_CRT_CASE_TYPE,
            lambda_penalty=1.0,
            time_horizon=10,
            use_gurobi=False,
        )

        result = solver.solve(cases, current_day=date(2023, 1, 1))

        assert len(result) == 9
        for case_id in result:
            assert len(result[case_id]) >= 1
            assert result[case_id][0][1] > 0

    def test_assignment_with_existing_loads(self):
        """Test assignment when mediators have existing case loads."""
        case = SimpleCase(
            id=1,
            case_type_id="Family group",
            court_station_id="KAKAMEGA",
            referral_date=date(2023, 1, 1),
            p_value=0.5,
        )

        solver = LPSolver(
            valid_mediators=SCENARIO1_VALID_MEDS,
            mediator_case_loads={1: 3, 2: 0, 3: 0},
            capacity=3,
            mediator_vas=SCENARIO1_MED_VA,
            med_by_court_case_type=SCENARIO1_MED_BY_CRT_CASE_TYPE,
            lambda_penalty=1.0,
            time_horizon=10,
            use_gurobi=False,
        )

        result = solver.solve([case], current_day=date(2023, 1, 1))

        assert 1 in result
        assigned_mediator = result[1][0][0]
        assert assigned_mediator in [2, 3]

    def test_no_eligible_mediators(self):
        """Test case with no eligible mediators returns empty assignment."""
        case = SimpleCase(
            id=1,
            case_type_id="Commercial Cases",
            court_station_id="MILIMANI",
            referral_date=date(2023, 1, 1),
            p_value=0.5,
        )

        solver = LPSolver(
            valid_mediators=SCENARIO1_VALID_MEDS,
            mediator_case_loads={1: 0, 2: 0, 3: 0},
            capacity=3,
            mediator_vas=SCENARIO1_MED_VA,
            med_by_court_case_type=SCENARIO1_MED_BY_CRT_CASE_TYPE,
            lambda_penalty=1.0,
            time_horizon=10,
            use_gurobi=False,
        )

        result = solver.solve([case], current_day=date(2023, 1, 1))

        assert len(result) == 0

    def test_scenario2_cross_station_mediator(self):
        """Test Scenario 2 where mediator 1 spans both stations."""
        cases = [
            SimpleCase(
                id=1,
                case_type_id="Family group",
                court_station_id="MILIMANI",
                referral_date=date(2023, 1, 1),
                p_value=0.5,
            ),
            SimpleCase(
                id=2,
                case_type_id="Family group",
                court_station_id="KAKAMEGA",
                referral_date=date(2023, 1, 1),
                p_value=0.5,
            ),
        ]

        solver = LPSolver(
            valid_mediators=SCENARIO2_VALID_MEDS,
            mediator_case_loads={1: 0, 2: 0, 3: 0},
            capacity=3,
            mediator_vas=SCENARIO2_MED_VA,
            med_by_court_case_type=SCENARIO2_MED_BY_CRT_CASE_TYPE,
            lambda_penalty=1.0,
            time_horizon=10,
            use_gurobi=False,
        )

        result = solver.solve(cases, current_day=date(2023, 1, 1))

        assert 1 in result
        assert 2 in result

    def test_phantom_cases_affect_assignment(self):
        """Test that phantom cases influence the assignment distribution."""
        real_case = SimpleCase(
            id=1,
            case_type_id="Family group",
            court_station_id="KAKAMEGA",
            referral_date=date(2023, 1, 1),
            p_value=0.5,
        )

        phantom_cases = [
            SimpleCase(
                id=-i,
                case_type_id="Family group",
                court_station_id="KAKAMEGA",
                referral_date=date(2023, 1, 2),
                p_value=0.5,
            )
            for i in range(1, 10)
        ]

        solver = LPSolver(
            valid_mediators=SCENARIO1_VALID_MEDS,
            mediator_case_loads={1: 0, 2: 0, 3: 0},
            capacity=3,
            mediator_vas=SCENARIO1_MED_VA,
            med_by_court_case_type=SCENARIO1_MED_BY_CRT_CASE_TYPE,
            lambda_penalty=1.0,
            time_horizon=10,
            use_gurobi=False,
        )

        result = solver.solve([real_case], phantom_cases=phantom_cases, current_day=date(2023, 1, 1))

        assert 1 in result

    def test_optimal_value_returned(self):
        """Test that optimal value is accessible after solving."""
        case = SimpleCase(
            id=1,
            case_type_id="Family group",
            court_station_id="KAKAMEGA",
            referral_date=date(2023, 1, 1),
            p_value=0.5,
        )

        solver = LPSolver(
            valid_mediators=SCENARIO1_VALID_MEDS,
            mediator_case_loads={1: 0, 2: 0, 3: 0},
            capacity=3,
            mediator_vas=SCENARIO1_MED_VA,
            med_by_court_case_type=SCENARIO1_MED_BY_CRT_CASE_TYPE,
            lambda_penalty=1.0,
            time_horizon=10,
            use_gurobi=False,
        )

        solver.solve([case], current_day=date(2023, 1, 1))
        optimal_value = solver.get_optimal_value()

        assert optimal_value is not None
        assert optimal_value > 0

    def test_datetime_support(self):
        """Test that datetime objects work as well as date objects."""
        case = SimpleCase(
            id=1,
            case_type_id="Family group",
            court_station_id="MILIMANI",
            referral_date=datetime(2023, 1, 1, 10, 30),
            p_value=0.5,
        )

        solver = LPSolver(
            valid_mediators=SCENARIO1_VALID_MEDS,
            mediator_case_loads={1: 0, 2: 0, 3: 0},
            capacity=3,
            mediator_vas=SCENARIO1_MED_VA,
            med_by_court_case_type=SCENARIO1_MED_BY_CRT_CASE_TYPE,
            lambda_penalty=1.0,
            time_horizon=10,
            use_gurobi=False,
        )

        result = solver.solve([case], current_day=datetime(2023, 1, 1, 0, 0))

        assert 1 in result


class TestAlgorithmConfig:
    """Tests for AlgorithmConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = AlgorithmConfig()

        assert config.capacity == 3
        assert config.lambda_penalty == 1.0
        assert config.time_horizon == 10
        assert config.use_gurobi is False

    def test_custom_config(self):
        """Test custom configuration values."""
        config = AlgorithmConfig(
            capacity=5,
            lambda_penalty=2.0,
            time_horizon=20,
            use_gurobi=True,
        )

        assert config.capacity == 5
        assert config.lambda_penalty == 2.0
        assert config.time_horizon == 20
        assert config.use_gurobi is True

    def test_invalid_capacity(self):
        """Test that invalid capacity raises error."""
        with pytest.raises(ValueError):
            AlgorithmConfig(capacity=0)

    def test_invalid_lambda(self):
        """Test that negative lambda raises error."""
        with pytest.raises(ValueError):
            AlgorithmConfig(lambda_penalty=-1.0)

    def test_invalid_time_horizon(self):
        """Test that invalid time horizon raises error."""
        with pytest.raises(ValueError):
            AlgorithmConfig(time_horizon=0)
