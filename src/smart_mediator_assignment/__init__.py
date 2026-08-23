"""
Smart Mediator Assignment Package.

This package provides an LP-based algorithm for optimizing mediator
assignments in court mediation systems.
"""

from .config import AlgorithmConfig
from .core import (
    CaseProtocol,
    SimpleCase,
    MediatorProtocol,
    SimpleMediator,
    MediatorBelief,
    BeliefState,
)
from .solver import LPSolver, QPSolver, AssignmentDistribution
from .algorithm import (
    compute_posterior,
    update_belief,
    update_belief_batch,
    generate_phantom_cases,
    get_strategy,
    VAEstimationConfig,
    MediatorVAEstimate,
    CasePrediction,
    VAEstimationResult,
    estimate_va,
    estimate_va_from_prepared,
    estimate_lognormal_duration_params,
    clean_hazard_sample,
    read_xlsx_duration_params,
    simplify_case_types,
    LEGACY_2_SIMPLIFIED_CTYPE,
    CASE_TYPE_NAMES,
)
from .assignment import (
    MediatorRecommendation,
    RecommendationResult,
    get_recommendations,
    get_recommendations_batch,
)

__version__ = "0.3.0"

__all__ = [
    # Config
    "AlgorithmConfig",
    # Core types
    "CaseProtocol",
    "SimpleCase",
    "MediatorProtocol",
    "SimpleMediator",
    "MediatorBelief",
    "BeliefState",
    # Solver
    "LPSolver",
    "QPSolver",
    "AssignmentDistribution",
    # Algorithm - Bayesian
    "compute_posterior",
    "update_belief",
    "update_belief_batch",
    "generate_phantom_cases",
    "get_strategy",
    # Algorithm - VA Estimation
    "VAEstimationConfig",
    "MediatorVAEstimate",
    "CasePrediction",
    "VAEstimationResult",
    "estimate_va",
    "estimate_va_from_prepared",
    # Case-type taxonomy (shared by VA + duration)
    "LEGACY_2_SIMPLIFIED_CTYPE",
    "CASE_TYPE_NAMES",
    "simplify_case_types",
    # Duration estimation
    "estimate_lognormal_duration_params",
    "clean_hazard_sample",
    "read_xlsx_duration_params",
    # Assignment
    "MediatorRecommendation",
    "RecommendationResult",
    "get_recommendations",
    "get_recommendations_batch",
]
