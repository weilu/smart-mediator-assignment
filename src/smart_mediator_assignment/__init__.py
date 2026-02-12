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
from .solver import LPSolver, AssignmentDistribution
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
)
from .assignment import (
    MediatorRecommendation,
    RecommendationResult,
    get_recommendations,
    get_recommendations_batch,
)

__version__ = "0.1.0"

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
    # Assignment
    "MediatorRecommendation",
    "RecommendationResult",
    "get_recommendations",
    "get_recommendations_batch",
]
