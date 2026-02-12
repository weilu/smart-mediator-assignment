from dataclasses import dataclass
from datetime import date, datetime
from typing import Protocol, Union, Optional, runtime_checkable


@runtime_checkable
class CaseProtocol(Protocol):
    """
    Protocol for cases used throughout the smart-mediator-assignment package.

    This unified protocol supports both:
    1. LP-based assignment (needs case_type, court_station, referral_date, p_value)
    2. VA estimation (needs outcome, dates, status, etc.)

    Compatible with Django Case model in cadaster-kenya-mediation.
    For Django models, ForeignKey fields should return the related object's name.
    """

    @property
    def id(self) -> int:
        """Unique case identifier."""
        ...

    @property
    def case_type(self) -> str:
        """Case type name (e.g., 'Divorce and Separation', 'Civil Cases')."""
        ...

    @property
    def court_station(self) -> str:
        """Court station name (e.g., 'MILIMANI', 'KAKAMEGA')."""
        ...

    @property
    def referral_date(self) -> Union[date, datetime]:
        """Date case was referred for mediation."""
        ...

    @property
    def p_value(self) -> Optional[float]:
        """Predicted probability of agreement (computed by VA estimation). None if not yet computed."""
        ...

    @property
    def mediator_id(self) -> Optional[int]:
        """ID of assigned mediator (None if unassigned)."""
        ...

    @property
    def case_outcome_agreement(self) -> Optional[int]:
        """Binary outcome: 1 for agreement, 0 for no agreement, None if pending."""
        ...

    @property
    def mediator_appointment_date(self) -> Optional[Union[date, datetime]]:
        """Date mediator was appointed to the case."""
        ...

    @property
    def conclusion_date(self) -> Optional[Union[date, datetime]]:
        """Date case was concluded (None if pending)."""
        ...

    @property
    def case_status(self) -> str:
        """Case status: 'PENDING', 'CONCLUDED', etc."""
        ...

    @property
    def court_type(self) -> str:
        """Court type: 'High Court', 'Court of Appeal', 'Magistrate Court', etc."""
        ...

    @property
    def referral_mode(self) -> str:
        """How case was referred: 'Referred by Court', 'Request by Parties', etc."""
        ...


@dataclass
class SimpleCase:
    """
    Concrete implementation for use in simulations and testing.

    For LP assignment, only id, case_type, court_station, referral_date, and p_value are required.
    For VA estimation, additional fields are needed.
    """

    id: int
    case_type: str
    court_station: str
    referral_date: Union[date, datetime]
    p_value: Optional[float] = None

    # VA estimation fields (optional for assignment-only use)
    mediator_id: Optional[int] = None
    case_outcome_agreement: Optional[int] = None
    mediator_appointment_date: Optional[Union[date, datetime]] = None
    conclusion_date: Optional[Union[date, datetime]] = None
    case_status: str = "PENDING"
    court_type: str = ""
    referral_mode: str = ""

    # Aliases for backward compatibility with cadaster-algo
    @property
    def case_type_id(self) -> str:
        """Alias for case_type (backward compatibility)."""
        return self.case_type

    @property
    def court_station_id(self) -> str:
        """Alias for court_station (backward compatibility)."""
        return self.court_station

    @classmethod
    def from_court_case(cls, court_case) -> "SimpleCase":
        """Create SimpleCase from cadaster-algo Court_Case object."""
        return cls(
            id=court_case.id,
            case_type=court_case.type_id,
            court_station=court_case.station_id,
            referral_date=court_case.incoming_day,
            p_value=court_case.p_val,
        )
