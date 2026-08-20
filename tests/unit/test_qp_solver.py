import importlib.util
import pytest
from datetime import date
from smart_mediator_assignment import SimpleCase, AlgorithmConfig

osqp_missing = importlib.util.find_spec("osqp") is None


def test_config_defaults_to_lp():
    assert AlgorithmConfig().use_qp is False


@pytest.mark.skipif(osqp_missing, reason="osqp extra not installed")
def test_qp_solver_assigns_single_case():
    from smart_mediator_assignment import QPSolver
    solver = QPSolver(capacity=3, lambda_penalty=1.0, time_horizon=10,
                      valid_mediators=[1, 2],
                      mediator_case_loads={1: 0, 2: 1},
                      mediator_vas={1: 0.1, 2: -0.05},
                      med_by_court_case_type={"MILIMANI": {"Family group": [1, 2]}})
    case = SimpleCase(id=1, case_type="Family group", court_station="MILIMANI",
                      referral_date=date(2023, 1, 1), p_value=0.5)
    dist = solver.solve([case], phantom_cases=[], current_day=date(2023, 1, 1))
    assert 1 in dist
    total = sum(p for _, p in dist[1])
    assert abs(total - 1.0) < 1e-6
    assert dist[1][0][0] == 1  # higher-VA, lower-load mediator ranked first


@pytest.mark.skipif(not osqp_missing, reason="osqp installed")
def test_qp_solver_import_error_without_osqp():
    from smart_mediator_assignment import QPSolver
    with pytest.raises(ImportError, match="osqp"):
        QPSolver(capacity=3, lambda_penalty=1.0, time_horizon=10, valid_mediators=[1],
                 mediator_case_loads={1: 0}, mediator_vas={1: 0.0},
                 med_by_court_case_type={"MILIMANI": {"Family group": [1]}})
