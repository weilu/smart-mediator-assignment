"""
Main entry point for mediator assignment recommendations.

This module provides the high-level API for getting mediator recommendations
for cases, combining the LP solver, belief state, and phantom case generation.
"""

from dataclasses import dataclass
from datetime import date, datetime
from typing import Dict, List, Optional, Union

from ..core.case import CaseProtocol, SimpleCase
from ..core.belief import BeliefState
from ..core.types import (
    MediatorId,
    CaseId,
    CaseLoads,
    MediatorVAs,
    MedByCrtCaseType,
    AvgCaseRate,
    AvgPValByCrtCaseType,
    CourtStationId,
    CaseTypeId,
)
from ..solver.lp_solver import LPSolver
from ..algorithm.phantom import generate_phantom_cases
from ..algorithm.strategies import get_strategy
from ..config import AlgorithmConfig


@dataclass
class MediatorRecommendation:
    """A single mediator recommendation with assignment probability."""
    mediator_id: MediatorId
    assignment_probability: float
    va_estimate: float


@dataclass
class RecommendationResult:
    """Result of a recommendation request for a single case."""
    case_id: CaseId
    recommendations: List[MediatorRecommendation]

    def get_top_mediator(self) -> Optional[MediatorId]:
        """Get the top recommended mediator ID."""
        if not self.recommendations:
            return None
        return self.recommendations[0].mediator_id


def get_recommendations(
    case: CaseProtocol,
    eligible_mediator_ids: List[MediatorId],
    mediator_case_loads: CaseLoads,
    belief_state: BeliefState,
    med_by_court_case_type: MedByCrtCaseType,
    config: AlgorithmConfig,
    phantom_cases: Optional[List[CaseProtocol]] = None,
    current_day: Optional[Union[date, datetime]] = None,
    strategy: str = "mean",
    ground_truth_vas: Optional[MediatorVAs] = None,
) -> RecommendationResult:
    """
    Get mediator recommendations for a single case.

    This is the main entry point for the assignment algorithm. It runs the LP
    solver to determine optimal mediator assignments based on the current
    belief state and optionally phantom (future) cases.

    Args:
        case: The case to assign
        eligible_mediator_ids: List of mediator IDs eligible for assignment
        mediator_case_loads: Current case load per mediator
        belief_state: Current belief state with VA estimates
        med_by_court_case_type: Mediator eligibility mapping
        config: Algorithm configuration
        phantom_cases: Optional pre-generated phantom cases
        current_day: Current date (defaults to case referral date)
        strategy: VA estimation strategy ('mean', 'sample', 'ground',
                  'mu-futAVG', 'mu-futSAMPLE', 'mu-futGRND')
        ground_truth_vas: Ground truth VAs (required for 'ground' strategy)

    Returns:
        RecommendationResult containing ranked mediator recommendations
    """
    if current_day is None:
        current_day = case.referral_date

    va_strategy = get_strategy(strategy)
    va_estimates = va_strategy.get_va_estimates(belief_state, ground_truth_vas)

    filtered_vas = {
        mid: va_estimates[mid]
        for mid in eligible_mediator_ids
        if mid in va_estimates
    }

    solver = LPSolver(
        valid_mediators=eligible_mediator_ids,
        mediator_case_loads=mediator_case_loads,
        capacity=config.capacity,
        mediator_vas=filtered_vas,
        med_by_court_case_type=med_by_court_case_type,
        lambda_penalty=config.lambda_penalty,
        time_horizon=config.time_horizon,
        use_gurobi=config.use_gurobi,
        config=config,
    )

    assignments = solver.solve(
        cases=[case],
        phantom_cases=phantom_cases,
        current_day=current_day,
    )

    recommendations = []
    if case.id in assignments:
        for med_id, prob in assignments[case.id]:
            recommendations.append(
                MediatorRecommendation(
                    mediator_id=med_id,
                    assignment_probability=prob,
                    va_estimate=filtered_vas.get(med_id, 0.0),
                )
            )

    return RecommendationResult(case_id=case.id, recommendations=recommendations)


def get_recommendations_batch(
    cases: List[CaseProtocol],
    eligible_mediator_ids: List[MediatorId],
    mediator_case_loads: CaseLoads,
    belief_state: BeliefState,
    med_by_court_case_type: MedByCrtCaseType,
    config: AlgorithmConfig,
    avg_case_rate: Optional[AvgCaseRate] = None,
    avg_p_val_by_crt_case_type: Optional[AvgPValByCrtCaseType] = None,
    court_stations: Optional[List[CourtStationId]] = None,
    case_types: Optional[List[CaseTypeId]] = None,
    current_day: Optional[Union[date, datetime]] = None,
    strategy: str = "mean",
    ground_truth_vas: Optional[MediatorVAs] = None,
    generate_phantoms: bool = True,
    seed: Optional[int] = None,
) -> Dict[CaseId, RecommendationResult]:
    """
    Get mediator recommendations for multiple cases at once.

    This method is more efficient than calling get_recommendations() for each
    case individually, as it solves a single LP for all cases.

    Args:
        cases: List of cases to assign
        eligible_mediator_ids: List of eligible mediator IDs
        mediator_case_loads: Current case load per mediator
        belief_state: Current belief state
        med_by_court_case_type: Mediator eligibility mapping
        config: Algorithm configuration
        avg_case_rate: Average case arrival rates (for phantom generation)
        avg_p_val_by_crt_case_type: Average p-values (for phantom generation)
        court_stations: Court stations to consider (for phantom generation)
        case_types: Case types to consider (for phantom generation)
        current_day: Current date
        strategy: VA estimation strategy
        ground_truth_vas: Ground truth VAs (for 'ground' strategy)
        generate_phantoms: Whether to generate phantom cases
        seed: Random seed for phantom generation

    Returns:
        Dictionary mapping case_id -> RecommendationResult
    """
    if not cases:
        return {}

    if current_day is None:
        current_day = cases[0].referral_date

    va_strategy = get_strategy(strategy)
    va_estimates = va_strategy.get_va_estimates(belief_state, ground_truth_vas)

    filtered_vas = {
        mid: va_estimates[mid]
        for mid in eligible_mediator_ids
        if mid in va_estimates
    }

    phantom_cases = []
    if generate_phantoms and avg_case_rate and court_stations and case_types:
        phantom_cases, _ = generate_phantom_cases(
            current_day=current_day,
            time_horizon=config.time_horizon,
            avg_case_rate=avg_case_rate,
            avg_p_val_by_crt_case_type=avg_p_val_by_crt_case_type or {},
            med_by_court_case_type=med_by_court_case_type,
            court_stations=court_stations,
            case_types=case_types,
            starting_id=-1,
            seed=seed,
        )

    solver = LPSolver(
        valid_mediators=eligible_mediator_ids,
        mediator_case_loads=mediator_case_loads,
        capacity=config.capacity,
        mediator_vas=filtered_vas,
        med_by_court_case_type=med_by_court_case_type,
        lambda_penalty=config.lambda_penalty,
        time_horizon=config.time_horizon,
        use_gurobi=config.use_gurobi,
        config=config,
    )

    assignments = solver.solve(
        cases=cases,
        phantom_cases=phantom_cases,
        current_day=current_day,
    )

    results = {}
    for case in cases:
        recommendations = []
        if case.id in assignments:
            for med_id, prob in assignments[case.id]:
                recommendations.append(
                    MediatorRecommendation(
                        mediator_id=med_id,
                        assignment_probability=prob,
                        va_estimate=filtered_vas.get(med_id, 0.0),
                    )
                )

        results[case.id] = RecommendationResult(
            case_id=case.id, recommendations=recommendations
        )

    return results
