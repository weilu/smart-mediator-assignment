from dataclasses import dataclass
from typing import Protocol


class MediatorProtocol(Protocol):
    @property
    def id(self) -> int: ...

    @property
    def is_active(self) -> bool: ...


@dataclass
class SimpleMediator:
    """Concrete implementation for use in simulations and testing."""
    id: int
    is_active: bool = True
