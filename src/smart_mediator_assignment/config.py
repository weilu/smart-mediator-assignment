from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AlgorithmConfig:
    """Configuration for the mediator assignment algorithm."""

    capacity: int = 3
    lambda_penalty: float = 1.0
    time_horizon: int = 10
    use_gurobi: bool = False

    gurobi_access_id: Optional[str] = field(default=None, repr=False)
    gurobi_secret: Optional[str] = field(default=None, repr=False)
    gurobi_license_id: Optional[str] = field(default=None, repr=False)

    def __post_init__(self):
        if self.capacity < 1:
            raise ValueError("capacity must be >= 1")
        if self.lambda_penalty < 0:
            raise ValueError("lambda_penalty must be >= 0")
        if self.time_horizon < 1:
            raise ValueError("time_horizon must be >= 1")
