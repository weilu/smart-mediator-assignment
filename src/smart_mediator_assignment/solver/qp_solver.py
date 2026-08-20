import warnings
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.sparse as sp

from .base import BaseSolver, AssignmentDistribution
from ..core.types import MediatorId, CaseId, CaseLoads, MediatorVAs, MedByCrtCaseType
from ..core.case import CaseProtocol
from ..config import AlgorithmConfig


class QPSolver(BaseSolver):
    """
    Quadratic Programming solver for mediator assignment, backed by OSQP.

    Ported from cadaster-algo's ``slackedQP``. Solves the same assignment
    problem as ``LPSolver`` but penalizes capacity slack quadratically
    (``lambda * xi^2``) instead of linearly, which OSQP handles directly:

        max  sum_e x_e * (p_v + va_u)  -  lambda * sum_u xi_u^2
        s.t. per-case assignment, slacked per-day capacity, and box bounds.

    OSQP minimizes ``0.5 z' P z + q' z`` subject to ``l <= A z <= u``, so the
    maximization objective is negated during construction.

    Drop-in replacement for ``LPSolver``: identical constructor signature and
    ``solve()`` output contract. ``use_gurobi`` is accepted for parity but
    ignored (OSQP is the backend).
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
        try:
            import osqp
        except ImportError as e:
            raise ImportError(
                "QPSolver requires the 'osqp' extra. Install with: pip install "
                "'smart-mediator-assignment[qp]'"
            ) from e
        self._osqp = osqp

        self.valid_mediators = valid_mediators
        self.mediator_case_loads = mediator_case_loads
        self.capacity = capacity
        self.mediator_vas = mediator_vas
        self.med_by_court_case_type = med_by_court_case_type
        self.lambda_penalty = float(lambda_penalty)
        self.time_horizon = int(time_horizon)
        self.use_gurobi = use_gurobi  # ignored; kept for LPSolver signature parity
        self.config = config

        self._current_day: Optional[Union[date, datetime]] = None
        self._us: List[MediatorId] = []
        self._vs: List[CaseId] = []
        self._edges: List[Tuple[MediatorId, CaseId]] = []
        self._case_arrival_time_by_id: Dict[CaseId, Union[date, datetime, int]] = {}
        self._case_p_vals: Dict[CaseId, float] = {}

        self._x_index: Dict[Tuple[MediatorId, CaseId], int] = {}
        self._xi_index: Dict[MediatorId, int] = {}
        self._n_x = 0
        self._n_xi = 0
        self._n_var = 0

        self._prob = None
        self._res = None

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

        for case in phantom_cases + unassigned_cases:
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

        self._edges = list(set(self._edges))

    def _active_indicator(self, case_id: CaseId, d: int) -> int:
        """Whether case ``case_id`` still occupies capacity on horizon day ``d``."""
        case_arrival = self._case_arrival_time_by_id[case_id]
        if self._current_day is None:
            return int(d <= case_arrival)
        if isinstance(case_arrival, (date, datetime)) and isinstance(
            self._current_day, (date, datetime)
        ):
            days_diff = (case_arrival - self._current_day).days
        else:
            days_diff = case_arrival - self._current_day
        return int(d <= days_diff)

    def _build_indices(self) -> None:
        self._x_index = {e: idx for idx, e in enumerate(self._edges)}
        self._n_x = len(self._edges)

        self._xi_index = {u: self._n_x + offset for offset, u in enumerate(self._us)}
        self._n_xi = len(self._us)

        self._n_var = self._n_x + self._n_xi

    def _build_primal(self) -> None:
        """Assemble the OSQP problem (P, q, A, l, u) and set up the solver."""
        self._build_indices()

        # Quadratic term: lambda * xi^2. OSQP uses 0.5 z'Pz, so 2*lambda on the diagonal.
        P_diag = np.zeros(self._n_var, dtype=float)
        if self.lambda_penalty != 0.0:
            for u in self._us:
                P_diag[self._xi_index[u]] = 2.0 * self.lambda_penalty
        P = sp.diags(P_diag, format="csc")

        q = np.zeros(self._n_var, dtype=float)
        for (u, v), idx in self._x_index.items():
            success_prob = float(self._case_p_vals[v]) + float(self.mediator_vas[u])
            q[idx] = -success_prob  # negate: original objective is a maximization

        rows: List[int] = []
        cols: List[int] = []
        data: List[float] = []
        lower: List[float] = []
        upper: List[float] = []
        row_id = 0

        # (C1) per-case assignment: real cases (id >= 0) must sum to 1;
        # phantom cases (id < 0) may sum to at most 1.
        edges_by_v: Dict[CaseId, List[Tuple[MediatorId, CaseId]]] = {}
        for e in self._edges:
            edges_by_v.setdefault(e[1], []).append(e)

        for v in dict.fromkeys(self._vs):
            E_v = edges_by_v.get(v, [])
            if not E_v:
                continue

            for e in E_v:
                rows.append(row_id)
                cols.append(self._x_index[e])
                data.append(1.0)

            if v >= 0:
                lower.append(1.0)
                upper.append(1.0)
            else:
                lower.append(-np.inf)
                upper.append(1.0)
            row_id += 1

        # (C2) slacked per-day capacity: sum of active x - xi <= capacity - load.
        edges_by_u: Dict[MediatorId, List[Tuple[MediatorId, CaseId]]] = {}
        for e in self._edges:
            edges_by_u.setdefault(e[0], []).append(e)

        for d in range(self.time_horizon):
            for med in self._us:
                for (_, v) in edges_by_u.get(med, []):
                    if self._active_indicator(v, d):
                        rows.append(row_id)
                        cols.append(self._x_index[(med, v)])
                        data.append(1.0)

                rows.append(row_id)
                cols.append(self._xi_index[med])
                data.append(-1.0)

                rhs = float(self.capacity) - float(self.mediator_case_loads[med])
                lower.append(-np.inf)
                upper.append(rhs)
                row_id += 1

        # Box bounds 0 <= x <= 1 as explicit rows.
        for e, idx in self._x_index.items():
            rows.append(row_id)
            cols.append(idx)
            data.append(1.0)
            lower.append(0.0)
            upper.append(np.inf)
            row_id += 1

            rows.append(row_id)
            cols.append(idx)
            data.append(1.0)
            lower.append(-np.inf)
            upper.append(1.0)
            row_id += 1

        # Box bounds 0 <= xi <= load + 1.
        for med in self._us:
            idx = self._xi_index[med]
            xi_ub = float(self.mediator_case_loads[med] + 1)

            rows.append(row_id)
            cols.append(idx)
            data.append(1.0)
            lower.append(0.0)
            upper.append(np.inf)
            row_id += 1

            rows.append(row_id)
            cols.append(idx)
            data.append(1.0)
            lower.append(-np.inf)
            upper.append(xi_ub)
            row_id += 1

        A = sp.csc_matrix((data, (rows, cols)), shape=(row_id, self._n_var))
        l = np.array(lower, dtype=float)
        u = np.array(upper, dtype=float)

        self._prob = self._osqp.OSQP()
        self._prob.setup(
            P=P,
            q=q,
            A=A,
            l=l,
            u=u,
            verbose=False,
            polish=True,
            eps_abs=1e-6,
            eps_rel=1e-6,
            max_iter=100000,
            scaled_termination=True,
            warm_start=True,
            adaptive_rho=True,
        )

    def _solve_model(self) -> np.ndarray:
        """Run OSQP and return the primal result vector."""
        if self._prob is None:
            raise RuntimeError("Model not built. Call solve() first.")

        self._res = self._prob.solve()
        status = str(self._res.info.status).lower()

        if status == "solved":
            pass
        elif self._res.x is not None:
            # Accept a non-optimal primal iterate (inaccurate / max-iter reached),
            # but surface it: per-case normalization keeps sums at 1, yet the
            # underlying assignment may be suboptimal. Matches the reference intent.
            warnings.warn(
                f"OSQP did not solve to optimality (status: {self._res.info.status}); "
                "returning the available primal iterate.",
                stacklevel=2,
            )
        else:
            raise RuntimeError(
                f"OSQP solve failed with status: {self._res.info.status}"
            )

        return self._res.x

    def _extract_assignments(
        self, z: np.ndarray, tol: float = 1e-9
    ) -> AssignmentDistribution:
        """Read edge assignment values from the OSQP result vector."""
        assignments: AssignmentDistribution = {}

        for (u, v), idx in self._x_index.items():
            val = float(z[idx])
            if val <= tol:
                continue
            assignments.setdefault(v, []).append((u, val))

        for v, pairs in assignments.items():
            # Real cases (id >= 0) are constrained to sum to 1; normalize so the
            # contract holds exactly regardless of solver precision. Phantom cases
            # (id < 0) are only bounded <= 1, so leave their sums as solved.
            if v >= 0:
                total = sum(p for _, p in pairs)
                if total <= 0:
                    assignments[v] = []  # degenerate: no mass, avoid divide-by-zero
                    continue
                pairs = [(u, p / total) for u, p in pairs]
            assignments[v] = sorted(pairs, key=lambda pair: pair[1], reverse=True)

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
            sorted by probability descending
        """
        self._current_day = current_day
        phantom_cases = phantom_cases or []

        self._build_graph(cases, phantom_cases)
        self._build_primal()
        z = self._solve_model()

        return self._extract_assignments(z)
