import os
import re
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple, Union

from pulp import (
    LpProblem,
    LpMaximize,
    LpMinimize,
    LpVariable,
    LpAffineExpression,
    value,
    PULP_CBC_CMD,
)

from .base import BaseSolver, AssignmentDistribution
from ..core.types import MediatorId, CaseId, CaseLoads, MediatorVAs, MedByCrtCaseType
from ..core.case import CaseProtocol
from ..config import AlgorithmConfig


def _setup_gurobi_env(config: AlgorithmConfig) -> None:
    """Set up Gurobi environment variables from config or existing env."""
    if config.gurobi_access_id:
        os.environ["GRB_WLSACCESSID"] = config.gurobi_access_id
    if config.gurobi_secret:
        os.environ["GRB_WLSSECRET"] = config.gurobi_secret
    if config.gurobi_license_id:
        os.environ["GRB_LICENSEID"] = config.gurobi_license_id


class LPSolver(BaseSolver):
    """
    Linear Programming solver for mediator assignment.

    Uses PuLP to formulate and solve an LP that maximizes expected case
    resolution while respecting mediator capacity constraints.
    """

    def __init__(
        self,
        valid_mediators: List[MediatorId],
        mediator_case_loads: CaseLoads,
        capacity: int,
        mediator_vas: MediatorVAs,
        med_by_court_case_type: MedByCrtCaseType,
        lambda_penalty: float = 1.0,
        time_horizon: int = 10,
        use_gurobi: bool = False,
        config: Optional[AlgorithmConfig] = None,
    ):
        """
        Initialize the LP solver.

        Args:
            valid_mediators: List of active mediator IDs
            mediator_case_loads: Current case load per mediator
            capacity: Maximum cases per mediator without penalty
            mediator_vas: VA estimate per mediator
            med_by_court_case_type: Mapping of court_station -> case_type -> [mediator_ids]
            lambda_penalty: Penalty weight for exceeding capacity
            time_horizon: Number of days to look ahead
            use_gurobi: Whether to use Gurobi solver (requires license)
            config: Optional AlgorithmConfig for additional settings
        """
        self.valid_mediators = valid_mediators
        self.mediator_case_loads = mediator_case_loads
        self.capacity = capacity
        self.mediator_vas = mediator_vas
        self.med_by_court_case_type = med_by_court_case_type
        self.lambda_penalty = lambda_penalty
        self.time_horizon = time_horizon
        self.use_gurobi = use_gurobi
        self.config = config

        self._current_day: Optional[Union[date, datetime]] = None
        self._model: Optional[LpProblem] = None
        self._us: List[MediatorId] = []
        self._vs: List[CaseId] = []
        self._edges: List[Tuple[MediatorId, CaseId]] = []
        self._case_arrival_time_by_id: Dict[CaseId, Union[date, datetime, int]] = {}
        self._case_p_vals: Dict[CaseId, float] = {}

        if use_gurobi and config:
            _setup_gurobi_env(config)

    def _build_graph(
        self,
        unassigned_cases: List[CaseProtocol],
        phantom_cases: List[CaseProtocol],
    ) -> None:
        """Build the bipartite graph of mediators to cases."""
        self._us = list(self.valid_mediators)
        self._vs = []
        self._edges = []
        self._case_arrival_time_by_id = {}
        self._case_p_vals = {}

        all_cases = phantom_cases + unassigned_cases

        for case in all_cases:
            station_id = case.court_station
            type_id = case.case_type

            if station_id not in self.med_by_court_case_type:
                continue
            if type_id not in self.med_by_court_case_type[station_id]:
                continue

            relevant_meds = [
                m
                for m in self.med_by_court_case_type[station_id][type_id]
                if m in self._us
            ]

            if not relevant_meds:
                continue

            self._vs.append(case.id)
            self._case_arrival_time_by_id[case.id] = case.referral_date
            self._case_p_vals[case.id] = case.p_value

            for med_id in relevant_meds:
                self._edges.append((med_id, case.id))

    def _build_primal(self) -> None:
        """Build the primal LP formulation."""
        self._model = LpProblem("primalLPSlacked", LpMaximize)

        x = {e: LpVariable(f"x_{e}", lowBound=0, upBound=1) for e in self._edges}
        ep_slacks = {
            med_id: LpVariable(
                f"lmda_{med_id}",
                lowBound=0,
                upBound=self.mediator_case_loads[med_id] + 1,
            )
            for med_id in self._us
        }

        for v in self._vs:
            edges_v = [e for e in self._edges if e[1] == v]
            constraint_lst = [(x[e], 1) for e in edges_v]

            if v >= 0:
                self._model += (
                    LpAffineExpression(constraint_lst) == 1,
                    f"Case_assignment_constraint_{v}",
                )
            else:
                self._model += (
                    LpAffineExpression(constraint_lst) <= 1,
                    f"Phantom_case_constraint_{v}",
                )

        for d in range(self.time_horizon):
            for u in self._us:
                edges_u = [e for e in self._edges if e[0] == u]
                constraint_lst = []

                for e in edges_u:
                    case_arrival = self._case_arrival_time_by_id[e[1]]

                    if self._current_day is None:
                        coef = int(d <= case_arrival)
                    else:
                        if isinstance(case_arrival, (date, datetime)) and isinstance(
                            self._current_day, (date, datetime)
                        ):
                            days_diff = (case_arrival - self._current_day).days
                        else:
                            days_diff = case_arrival - self._current_day
                        coef = int(d <= days_diff)

                    constraint_lst.append((x[e], coef))

                self._model += (
                    self.mediator_case_loads[u]
                    + LpAffineExpression(constraint_lst)
                    - ep_slacks[u]
                    <= self.capacity,
                    f"CapacityConstraint_{u}_{d}",
                )

        obj_list_x = []
        for e in self._edges:
            success_prob = float(self._case_p_vals[e[1]] + self.mediator_vas[e[0]])
            obj_list_x.append((x[e], success_prob))

        obj_list_slack = []
        for u in self._us:
            penalty = float(self.lambda_penalty) * float(
                max(self.mediator_case_loads[u], 0)
            )
            obj_list_slack.append((ep_slacks[u], penalty))

        self._model += LpAffineExpression(obj_list_x) - LpAffineExpression(
            obj_list_slack
        )

    def _build_dual(self) -> None:
        """Build the dual LP formulation."""
        self._model = LpProblem("dualLPSlacked", LpMinimize)

        self._vs = list(set(e[1] for e in self._edges))

        alpha = {}
        for v in self._vs:
            if v > 0:
                alpha[v] = LpVariable(f"alpha_{v}")
            else:
                alpha[v] = LpVariable(f"alpha_{v}", lowBound=0)

        beta = {
            (u, t): LpVariable(f"beta_{(u, t)}", lowBound=0)
            for u in self._us
            for t in range(self.time_horizon)
        }
        gamma = {u: LpVariable(f"gamma_{u}", lowBound=0) for u in self._us}

        for e in self._edges:
            u, v = e[0], e[1]
            constraint_lst = []

            for t in range(self.time_horizon):
                case_arrival = self._case_arrival_time_by_id[v]
                if self._current_day is None:
                    coef = int(t <= case_arrival)
                else:
                    if isinstance(case_arrival, (date, datetime)):
                        days_diff = (case_arrival - self._current_day).days
                    else:
                        days_diff = case_arrival - self._current_day
                    coef = int(t <= days_diff)

                constraint_lst.append((beta[(u, t)], coef))

            success_prob = float(self._case_p_vals[v]) + float(self.mediator_vas[u])
            self._model += alpha[v] + LpAffineExpression(constraint_lst) >= success_prob

        for u in self._us:
            constraint_lst = [(beta[(u, t)], 1) for t in range(self.time_horizon)]
            self._model += (
                LpAffineExpression(constraint_lst) - gamma[u]
                <= float(self.lambda_penalty) * float(self.mediator_case_loads[u])
            )

        obj_lst_alpha = [(alpha[v], 1) for v in self._vs]
        obj_lst_beta = []
        obj_lst_gamma = []

        for u in self._us:
            for t in range(self.time_horizon):
                obj_lst_beta.append(
                    (beta[(u, t)], self.capacity - self.mediator_case_loads[u])
                )
            obj_lst_gamma.append((gamma[u], self.mediator_case_loads[u] + 1))

        self._model += (
            LpAffineExpression(obj_lst_alpha)
            + LpAffineExpression(obj_lst_beta)
            + LpAffineExpression(obj_lst_gamma)
        )

    def _solve_model(self, verbose: bool = False) -> None:
        """Solve the LP model."""
        if self._model is None:
            raise RuntimeError("Model not built. Call solve() first.")

        if self.use_gurobi:
            try:
                from pulp import GUROBI

                self._model.solve(GUROBI(msg=verbose))
            except ImportError:
                raise ImportError(
                    "Gurobi not available. Install gurobipy or set use_gurobi=False"
                )
        else:
            self._model.solve(PULP_CBC_CMD(msg=verbose))

    def _extract_assignments(self) -> AssignmentDistribution:
        """Extract assignment distribution from solved model."""
        if self._model is None:
            raise RuntimeError("Model not solved")

        assignments: AssignmentDistribution = {}

        for var in self._model.variables():
            if var.varValue is None or var.varValue == 0:
                continue

            match = re.match(r"x_\(([^,]+),_([^)\s]+)\)", var.name)
            if not match:
                continue

            med_id = int(float(match.group(1)))
            case_id_str = match.group(2)

            if case_id_str.startswith("_"):
                case_id = -int(float(case_id_str[1:]))
            else:
                case_id = int(float(case_id_str))

            if case_id not in assignments:
                assignments[case_id] = []

            assignments[case_id].append((med_id, var.varValue))

        for case_id in assignments:
            assignments[case_id] = sorted(
                assignments[case_id], key=lambda x: x[1], reverse=True
            )

        return assignments

    def solve(
        self,
        cases: List[CaseProtocol],
        phantom_cases: Optional[List[CaseProtocol]] = None,
        current_day: Optional[Union[date, datetime]] = None,
    ) -> AssignmentDistribution:
        """
        Solve the assignment problem.

        Args:
            cases: List of cases to assign
            phantom_cases: Optional list of phantom (future) cases
            current_day: Current date for time-horizon calculations

        Returns:
            Dictionary mapping case_id -> [(mediator_id, probability), ...]
        """
        self._current_day = current_day
        phantom_cases = phantom_cases or []

        self._build_graph(cases, phantom_cases)
        self._build_primal()
        self._solve_model()

        return self._extract_assignments()

    def get_optimal_value(self) -> Optional[float]:
        """Get the optimal objective value after solving."""
        if self._model is None:
            return None
        return value(self._model.objective)

    def extract_shadow_prices(
        self,
    ) -> Tuple[Dict[CaseId, float], Dict[MediatorId, float], Dict[MediatorId, float]]:
        """
        Extract shadow prices from dual solution.

        Returns:
            Tuple of (alpha, beta, gamma) dictionaries
        """
        if self._model is None:
            raise RuntimeError("Model not solved")

        alpha = {}
        beta = {}
        gamma = {}

        for var in self._model.variables():
            name = var.name
            val = var.varValue

            if name.startswith("alpha_"):
                idx = int(name.split("_")[-1])
                alpha[idx] = val
            elif name.startswith("beta_("):
                inside = name[len("beta_(") : -1]
                med_str, time_str = inside.split(",_")
                med = int(float(med_str))

                if med not in beta:
                    beta[med] = {}
                beta[med][int(time_str)] = val
            elif name.startswith("gamma_"):
                med_str = name.split("_")[-1]
                med = int(float(med_str))
                gamma[med] = val

        aggregated_beta = {}
        for med in beta:
            aggregated_beta[med] = sum(beta[med].values())

        return alpha, aggregated_beta, gamma
