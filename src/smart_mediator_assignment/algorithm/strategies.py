"""
Belief sampling strategies for mediator VA estimation.

These strategies determine how VA estimates are obtained from the belief state
for use in the assignment algorithm.
"""

from typing import Dict, Optional

import numpy as np

from ..core.belief import BeliefState
from ..core.types import MediatorId, MediatorVAs


class VAStrategy:
    """Base class for VA estimation strategies."""

    def get_va_estimates(
        self,
        belief_state: BeliefState,
        ground_truth_vas: Optional[MediatorVAs] = None,
    ) -> MediatorVAs:
        raise NotImplementedError


class MeanStrategy(VAStrategy):
    """Use the posterior mean as the VA estimate."""

    def get_va_estimates(
        self,
        belief_state: BeliefState,
        ground_truth_vas: Optional[MediatorVAs] = None,
    ) -> MediatorVAs:
        return belief_state.get_va_estimates(strategy="mean")


class SampleStrategy(VAStrategy):
    """Sample from the posterior distribution for VA estimates."""

    def get_va_estimates(
        self,
        belief_state: BeliefState,
        ground_truth_vas: Optional[MediatorVAs] = None,
    ) -> MediatorVAs:
        return belief_state.get_va_estimates(strategy="sample")


class GroundTruthStrategy(VAStrategy):
    """Use ground truth VA values (requires ground truth to be provided)."""

    def get_va_estimates(
        self,
        belief_state: BeliefState,
        ground_truth_vas: Optional[MediatorVAs] = None,
    ) -> MediatorVAs:
        if ground_truth_vas is None:
            raise ValueError("Ground truth VAs required for ground truth strategy")
        return ground_truth_vas


def get_strategy(strategy_name: str) -> VAStrategy:
    """
    Get a VA strategy instance by name.

    Args:
        strategy_name: One of 'mean', 'sample', 'ground', 'mu-futAVG',
                       'mu-futSAMPLE', 'mu-futGRND'

    Returns:
        VAStrategy instance
    """
    strategy_map = {
        "mean": MeanStrategy,
        "sample": SampleStrategy,
        "ground": GroundTruthStrategy,
        "mu-futAVG": MeanStrategy,
        "mu-futSAMPLE": SampleStrategy,
        "mu-futGRND": GroundTruthStrategy,
    }

    if strategy_name not in strategy_map:
        raise ValueError(
            f"Unknown strategy: {strategy_name}. "
            f"Available: {list(strategy_map.keys())}"
        )

    return strategy_map[strategy_name]()
