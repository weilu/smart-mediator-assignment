from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import copy

from .types import MediatorId, CaseHistory


@dataclass
class MediatorBelief:
    mediator_id: MediatorId
    mu: float
    sigma: float
    case_history: CaseHistory = field(default_factory=lambda: ([], []))

    def copy(self) -> "MediatorBelief":
        return MediatorBelief(
            mediator_id=self.mediator_id,
            mu=self.mu,
            sigma=self.sigma,
            case_history=(
                list(self.case_history[0]),
                list(self.case_history[1]),
            ),
        )


@dataclass
class BeliefState:
    beliefs: Dict[MediatorId, MediatorBelief]
    global_sigma: float

    def copy(self) -> "BeliefState":
        return BeliefState(
            beliefs={mid: b.copy() for mid, b in self.beliefs.items()},
            global_sigma=self.global_sigma,
        )

    def get_va_estimates(self, strategy: str = "mean") -> Dict[MediatorId, float]:
        """
        Get VA estimates for all mediators.

        Args:
            strategy: 'mean' uses posterior mean, 'sample' samples from posterior,
                      'ground' would require ground truth (must be provided externally)

        Returns:
            Dictionary mapping mediator_id -> VA estimate
        """
        import numpy as np

        estimates = {}
        for mid, belief in self.beliefs.items():
            if strategy == "mean":
                estimates[mid] = belief.mu
            elif strategy == "sample":
                estimates[mid] = np.random.normal(belief.mu, belief.sigma)
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
        return estimates

    def get_belief(self, mediator_id: MediatorId) -> MediatorBelief:
        return self.beliefs[mediator_id]

    @classmethod
    def from_init_va(
        cls,
        med_init_va: Dict[MediatorId, Dict[str, float]],
        global_sigma: float,
    ) -> "BeliefState":
        """
        Create BeliefState from med_init_va format used in cadaster-algo.

        Args:
            med_init_va: Dict of {mediator_id: {'mu': float, 'sd': float}}
            global_sigma: Global prior sigma value

        Returns:
            BeliefState instance
        """
        beliefs = {}
        for mid, va_dict in med_init_va.items():
            beliefs[mid] = MediatorBelief(
                mediator_id=mid,
                mu=va_dict["mu"],
                sigma=va_dict["sd"],
                case_history=([], []),
            )
        return cls(beliefs=beliefs, global_sigma=global_sigma)
