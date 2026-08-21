import importlib.util
import pytest
from datetime import date
from smart_mediator_assignment import SimpleCase, AlgorithmConfig

osqp_missing = importlib.util.find_spec("osqp") is None
gurobi_missing = importlib.util.find_spec("gurobipy") is None


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


@pytest.mark.skipif(osqp_missing, reason="osqp extra not installed")
def test_qp_congested_activates_slack():
    # capacity=1 with 3 same-day cases eligible to both mediators forces the
    # capacity constraint to bind (slack xi capped at load+1=1), so mediator 1
    # cannot absorb all cases even at full slack -> the load must spread.
    from smart_mediator_assignment import QPSolver
    solver = QPSolver(capacity=1, lambda_penalty=1.0, time_horizon=10,
                      valid_mediators=[1, 2],
                      mediator_case_loads={1: 0, 2: 0},
                      mediator_vas={1: 0.2, 2: 0.1},
                      med_by_court_case_type={"MILIMANI": {"Family group": [1, 2]}})
    cases = [
        SimpleCase(id=i, case_type="Family group", court_station="MILIMANI",
                   referral_date=date(2023, 1, 1), p_value=0.5)
        for i in (1, 2, 3)
    ]
    dist = solver.solve(cases, phantom_cases=[], current_day=date(2023, 1, 1))

    for case_id in (1, 2, 3):
        assert case_id in dist
        assert abs(sum(p for _, p in dist[case_id]) - 1.0) < 1e-6

    med1_mass = sum(p for cid in (1, 2, 3) for m, p in dist[cid] if m == 1)
    assert med1_mass < 3.0 - 1e-3  # capacity limit prevents piling all on mediator 1
    med2_mass = sum(p for cid in (1, 2, 3) for m, p in dist[cid] if m == 2)
    assert med2_mass > 1e-3  # mediator 2 absorbs the overflow


@pytest.mark.skipif(osqp_missing, reason="osqp extra not installed")
def test_qp_phantom_case_leq_one():
    # A real case fully consumes the single mediator's capacity; the phantom then
    # competes only against quadratic slack penalty, so it is partially assigned
    # (< 1) and must NOT be force-normalized to 1.
    from smart_mediator_assignment import QPSolver
    solver = QPSolver(capacity=1, lambda_penalty=1.0, time_horizon=10,
                      valid_mediators=[1],
                      mediator_case_loads={1: 0},
                      mediator_vas={1: 0.1},
                      med_by_court_case_type={"MILIMANI": {"Family group": [1]}})
    real = SimpleCase(id=1, case_type="Family group", court_station="MILIMANI",
                      referral_date=date(2023, 1, 1), p_value=0.5)
    phantom = SimpleCase(id=-1, case_type="Family group", court_station="MILIMANI",
                         referral_date=date(2023, 1, 1), p_value=0.5)
    dist = solver.solve([real], phantom_cases=[phantom],
                        current_day=date(2023, 1, 1))

    assert -1 in dist
    phantom_sum = sum(p for _, p in dist[-1])
    assert phantom_sum <= 1.0 + 1e-6
    assert phantom_sum < 1.0 - 1e-3  # not force-normalized like real cases


@pytest.mark.skipif(osqp_missing, reason="osqp extra not installed")
def test_qp_vs_lp_agree_uncongested():
    from smart_mediator_assignment import QPSolver, LPSolver
    kwargs = dict(capacity=3, lambda_penalty=1.0, time_horizon=10,
                  valid_mediators=[1, 2],
                  mediator_case_loads={1: 0, 2: 1},
                  mediator_vas={1: 0.1, 2: -0.05},
                  med_by_court_case_type={"MILIMANI": {"Family group": [1, 2]}})
    case = SimpleCase(id=1, case_type="Family group", court_station="MILIMANI",
                      referral_date=date(2023, 1, 1), p_value=0.5)

    qp_dist = QPSolver(**kwargs).solve([case], phantom_cases=[],
                                       current_day=date(2023, 1, 1))
    lp_dist = LPSolver(**kwargs).solve([case], phantom_cases=[],
                                       current_day=date(2023, 1, 1))

    assert qp_dist[1][0][0] == lp_dist[1][0][0] == 1


@pytest.mark.skipif(not osqp_missing, reason="osqp installed")
def test_qp_solver_import_error_without_osqp():
    from smart_mediator_assignment import QPSolver
    with pytest.raises(ImportError, match="osqp"):
        QPSolver(capacity=3, lambda_penalty=1.0, time_horizon=10, valid_mediators=[1],
                 mediator_case_loads={1: 0}, mediator_vas={1: 0.0},
                 med_by_court_case_type={"MILIMANI": {"Family group": [1]}})


@pytest.mark.skipif(not gurobi_missing, reason="gurobipy installed")
def test_qp_solver_gurobi_import_error_without_gurobipy():
    from smart_mediator_assignment import QPSolver
    with pytest.raises(ImportError, match="gurobi"):
        QPSolver(capacity=3, lambda_penalty=1.0, time_horizon=10, valid_mediators=[1],
                 mediator_case_loads={1: 0}, mediator_vas={1: 0.0},
                 med_by_court_case_type={"MILIMANI": {"Family group": [1]}},
                 use_gurobi=True)


@pytest.mark.skipif(gurobi_missing, reason="gurobipy extra not installed")
@pytest.mark.skipif(osqp_missing, reason="osqp extra not installed")
def test_qp_gurobi_matches_osqp():
    # Convex QP: Gurobi (paper's original solver) and OSQP must reach the same
    # optimum. Requires a Gurobi license, so this is skipped where none exists.
    from smart_mediator_assignment import QPSolver
    kwargs = dict(capacity=3, lambda_penalty=1.0, time_horizon=10,
                  valid_mediators=[1, 2],
                  mediator_case_loads={1: 0, 2: 1},
                  mediator_vas={1: 0.1, 2: -0.05},
                  med_by_court_case_type={"MILIMANI": {"Family group": [1, 2]}})
    case = SimpleCase(id=1, case_type="Family group", court_station="MILIMANI",
                      referral_date=date(2023, 1, 1), p_value=0.5)

    try:
        gp_dist = QPSolver(use_gurobi=True, **kwargs).solve(
            [case], phantom_cases=[], current_day=date(2023, 1, 1))
    except Exception as e:  # noqa: BLE001
        if "license" in str(e).lower() or "size-limited" in str(e).lower():
            pytest.skip(f"Gurobi license unavailable: {e}")
        raise
    osqp_dist = QPSolver(use_gurobi=False, **kwargs).solve(
        [case], phantom_cases=[], current_day=date(2023, 1, 1))

    assert gp_dist[1][0][0] == osqp_dist[1][0][0]  # same top mediator
    gp_probs = dict(gp_dist[1])
    osqp_probs = dict(osqp_dist[1])
    assert gp_probs.keys() == osqp_probs.keys()
    for med, p in gp_probs.items():
        assert abs(p - osqp_probs[med]) < 1e-4
