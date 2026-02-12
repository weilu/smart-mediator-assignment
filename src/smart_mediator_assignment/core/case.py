from dataclasses import dataclass
from datetime import date, datetime
from typing import Protocol, Union

class CaseProtocol(Protocol):
    @property
    def id(self) -> int: ...

    @property
    def case_type_id(self) -> str: ...

    @property
    def court_station_id(self) -> str: ...

    @property
    def referral_date(self) -> Union[date, datetime]: ...

    @property
    def p_value(self) -> float: ...


@dataclass
class SimpleCase:
    """Concrete implementation for use in simulations and testing."""
    id: int
    case_type_id: str
    court_station_id: str
    referral_date: Union[date, datetime]
    p_value: float

    @classmethod
    def from_court_case(cls, court_case) -> "SimpleCase":
        """Create SimpleCase from cadaster-algo Court_Case object."""
        return cls(
            id=court_case.id,
            case_type_id=court_case.type_id,
            court_station_id=court_case.station_id,
            referral_date=court_case.incoming_day,
            p_value=court_case.p_val,
        )
