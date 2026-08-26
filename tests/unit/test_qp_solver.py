import importlib.util
import pytest
from datetime import date, timedelta
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
def test_qp_infeasible_status_raises():
    """A non-null iterate from an infeasible/unsolved solve must raise, not be returned -
    it does not satisfy the constraints (matches the reference SlackedQPwithLoadOSQP.py)."""
    import numpy as np
    from smart_mediator_assignment import QPSolver

    class _Info:
        status = "primal infeasible"

    class _Res:
        info = _Info()
        x = np.array([0.5, 0.5])

    class _Prob:
        def solve(self):
            return _Res()

    solver = QPSolver(capacity=1, lambda_penalty=1.0, time_horizon=10,
                      valid_mediators=[1], mediator_case_loads={1: 0},
                      mediator_vas={1: 0.1}, med_by_court_case_type={})
    solver._prob = _Prob()
    with pytest.raises(RuntimeError, match="primal infeasible"):
        solver._solve_model()


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


# Captured from the retired SlackedQPwithLoadOSQP.extract_mediator_shadow_prices on the
# tight-capacity scenario below (capacity=1, 2 mediators, 3 same-day cases) before deletion.
_SHADOW_GOLDEN = {1: 1.0499999999999998, 2: 0.95}


def _tight_cap_solver(use_gurobi=False):
    from smart_mediator_assignment import QPSolver
    return QPSolver(valid_mediators=[1, 2], mediator_case_loads={1: 0, 2: 0}, capacity=1,
                    mediator_vas={1: 0.2, 2: 0.1},
                    med_by_court_case_type={"MILIMANI": {"Family group": [1, 2]}},
                    lambda_penalty=1.0, time_horizon=10, use_gurobi=use_gurobi)


def _tight_cap_cases():
    return [SimpleCase(id=i + 1, case_type="Family group", court_station="MILIMANI",
                       referral_date=date(2023, 1, 15), p_value=0.5) for i in range(3)]


@pytest.mark.skipif(osqp_missing, reason="osqp extra not installed")
def test_shadow_prices_match_retired_osqp_golden():
    solver = _tight_cap_solver()
    solver.solve(_tight_cap_cases(), phantom_cases=[], current_day=date(2023, 1, 15))
    sp = solver.extract_mediator_shadow_prices()
    for u, g in _SHADOW_GOLDEN.items():
        assert abs(sp[u] - g) < 1e-3, (u, sp[u], g)


@pytest.mark.skipif(osqp_missing, reason="osqp extra not installed")
def test_shadow_prices_zero_when_uncongested():
    from smart_mediator_assignment import QPSolver
    solver = QPSolver(valid_mediators=[1, 2], mediator_case_loads={1: 0, 2: 0}, capacity=5,
                      mediator_vas={1: 0.2, 2: 0.1},
                      med_by_court_case_type={"MILIMANI": {"Family group": [1, 2]}},
                      lambda_penalty=1.0, time_horizon=10)
    solver.solve([SimpleCase(id=1, case_type="Family group", court_station="MILIMANI",
                             referral_date=date(2023, 1, 1), p_value=0.5)],
                 phantom_cases=[], current_day=date(2023, 1, 1))
    assert all(abs(v) < 1e-6 for v in solver.extract_mediator_shadow_prices().values())


def test_shadow_prices_zero_without_solve():
    # no solve -> no duals -> zeros (never AttributeErrors, unlike the old missing method)
    assert _tight_cap_solver().extract_mediator_shadow_prices() == {}


@pytest.mark.skipif(gurobi_missing, reason="gurobipy extra not installed")
@pytest.mark.skipif(osqp_missing, reason="osqp extra not installed")
def test_gurobi_shadow_prices_match_osqp():
    osqp_solver = _tight_cap_solver(use_gurobi=False)
    osqp_solver.solve(_tight_cap_cases(), phantom_cases=[], current_day=date(2023, 1, 15))
    osqp_sp = osqp_solver.extract_mediator_shadow_prices()

    gp_solver = _tight_cap_solver(use_gurobi=True)
    try:
        gp_solver.solve(_tight_cap_cases(), phantom_cases=[], current_day=date(2023, 1, 15))
    except Exception as e:  # noqa: BLE001
        if "license" in str(e).lower() or "size-limited" in str(e).lower():
            pytest.skip(f"Gurobi license unavailable: {e}")
        raise
    gp_sp = gp_solver.extract_mediator_shadow_prices()
    for u in osqp_sp:
        assert abs(gp_sp[u] - osqp_sp[u]) < 1e-3, (u, gp_sp[u], osqp_sp[u])


# Realistic golden captured from the retired SlackedQPwithLoadOSQP: 6 mediators with real VAs
# (from tests/fixtures/va_golden_df_mediator_20260513.csv in the research repo), 2 court stations
# x 2 case types, varied eligibility, 12 cases with staggered arrivals, loads near capacity.
# Exercises the (mediator, day) -> capacity-row mapping across many rows and case types.
_REAL_VAS = {1335: -0.239793, 855: -0.234265, 907: -0.013297,
             1070: -0.013273, 1206: 0.202901, 1626: 0.231258}
_REAL_LOADS = {1206: 2, 1626: 2, 907: 1, 1070: 0, 855: 1, 1335: 0}
_REAL_ELIG = {"MILIMANI": {"Family group": [1206, 1626, 907], "Civil group": [1206, 1070]},
              "KAKAMEGA": {"Family group": [1626, 855], "Civil group": [907, 855, 1335]}}
_REAL_CASE_SPECS = [
    ("MILIMANI", "Family group", 0.6, 0), ("MILIMANI", "Family group", 0.5, 1),
    ("MILIMANI", "Family group", 0.55, 2), ("MILIMANI", "Civil group", 0.4, 0),
    ("MILIMANI", "Civil group", 0.45, 3), ("KAKAMEGA", "Family group", 0.5, 0),
    ("KAKAMEGA", "Family group", 0.6, 1), ("KAKAMEGA", "Family group", 0.35, 2),
    ("KAKAMEGA", "Civil group", 0.5, 0), ("KAKAMEGA", "Civil group", 0.45, 1),
    ("KAKAMEGA", "Civil group", 0.4, 2), ("KAKAMEGA", "Civil group", 0.55, 3),
]
_REAL_SHADOW_GOLDEN = {855: 0.1763742, 907: 0.3973422, 1070: 0.0,
                       1206: 0.6135402, 1335: 0.1708462, 1626: 0.6418972}


def _realistic_solver(use_gurobi=False):
    from smart_mediator_assignment import QPSolver
    return QPSolver(valid_mediators=list(_REAL_VAS), mediator_case_loads=dict(_REAL_LOADS),
                    capacity=3, mediator_vas=dict(_REAL_VAS), med_by_court_case_type=_REAL_ELIG,
                    lambda_penalty=1.0, time_horizon=10, use_gurobi=use_gurobi)


def _realistic_cases():
    base = date(2023, 1, 15)
    return [SimpleCase(id=i + 1, case_type=t, court_station=s,
                       referral_date=base + timedelta(days=o), p_value=p)
            for i, (s, t, p, o) in enumerate(_REAL_CASE_SPECS)]


@pytest.mark.skipif(osqp_missing, reason="osqp extra not installed")
def test_shadow_prices_match_retired_osqp_realistic_golden():
    solver = _realistic_solver()
    solver.solve(_realistic_cases(), phantom_cases=[], current_day=date(2023, 1, 15))
    sp = solver.extract_mediator_shadow_prices()
    assert set(sp) == set(_REAL_SHADOW_GOLDEN)
    for u, g in _REAL_SHADOW_GOLDEN.items():
        assert abs(sp[u] - g) < 1e-6, (u, sp[u], g)
