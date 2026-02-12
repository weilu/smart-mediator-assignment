from abc import ABC, abstractmethod
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple, Union

from ..core.types import MediatorId, CaseId, CaseLoads, MediatorVAs, MedByCrtCaseType
from ..core.case import CaseProtocol


AssignmentDistribution = Dict[CaseId, List[Tuple[MediatorId, float]]]


class BaseSolver(ABC):
    """Abstract base class for mediator assignment solvers."""

    @abstractmethod
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
        pass
