"""
Phantom case generation for look-ahead optimization.

This module generates simulated future cases based on Poisson arrival rates
to help the LP solver make forward-looking assignment decisions.
"""

import random
from datetime import date, datetime, timedelta
from typing import Dict, List, Tuple, Union

import numpy as np

from ..core.case import SimpleCase
from ..core.types import (
    AvgCaseRate,
    AvgPValByCrtCaseType,
    MedByCrtCaseType,
    CourtStationId,
    CaseTypeId,
)


def generate_phantom_cases(
    current_day: Union[date, datetime],
    time_horizon: int,
    avg_case_rate: AvgCaseRate,
    avg_p_val_by_crt_case_type: AvgPValByCrtCaseType,
    med_by_court_case_type: MedByCrtCaseType,
    court_stations: List[CourtStationId],
    case_types: List[CaseTypeId],
    starting_id: int = -1,
    default_p_val: float = 0.5,
    seed: int = None,
) -> Tuple[List[SimpleCase], int]:
    """
    Generate phantom (simulated future) cases using Poisson sampling.

    Args:
        current_day: Current date
        time_horizon: Number of days to look ahead
        avg_case_rate: Average daily case arrival rates by case_type and court_station
        avg_p_val_by_crt_case_type: Average p-values by (case_type, court_station)
        med_by_court_case_type: Mapping of eligible mediators
        court_stations: List of court station IDs to consider
        case_types: List of case type IDs to consider
        starting_id: Starting ID for phantom cases (should be negative)
        default_p_val: Default p-value when not found in mapping
        seed: Random seed for reproducibility

    Returns:
        Tuple of (list of phantom cases, next available phantom ID)
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    phantom_id = starting_id
    phantom_cases_with_order = []

    for fut_day in range(time_horizon):
        arrival_date = _add_days(current_day, fut_day)

        pairs = [
            (court_station, case_type)
            for court_station in court_stations
            for case_type in case_types
        ]

        for court_station, case_type in pairs:
            if court_station not in med_by_court_case_type:
                continue
            if case_type not in med_by_court_case_type[court_station]:
                continue

            if case_type not in avg_case_rate:
                continue
            if court_station not in avg_case_rate[case_type]:
                continue

            lambda_rate = avg_case_rate[case_type][court_station]
            num_cases = np.random.poisson(lambda_rate)

            for _ in range(num_cases):
                p_val_key = (case_type, court_station)
                if p_val_key in avg_p_val_by_crt_case_type:
                    p_val = avg_p_val_by_crt_case_type[p_val_key]
                else:
                    p_val = default_p_val

                order_key = random.uniform(0, 1)

                phantom_case = SimpleCase(
                    id=phantom_id,
                    case_type=case_type,
                    court_station=court_station,
                    referral_date=arrival_date,
                    p_value=p_val - 0.1,
                )

                phantom_cases_with_order.append((order_key, phantom_case))
                phantom_id -= 1

    phantom_cases_with_order.sort(key=lambda x: x[0])
    phantom_cases = [case for _, case in phantom_cases_with_order]

    return phantom_cases, phantom_id


def _add_days(
    base_date: Union[date, datetime], days: int
) -> Union[date, datetime]:
    """Add days to a date or datetime object."""
    if isinstance(base_date, datetime):
        return base_date + timedelta(days=days)
    return base_date + timedelta(days=days)


def estimate_case_arrivals(
    avg_case_rate: AvgCaseRate,
    court_stations: List[CourtStationId],
    case_types: List[CaseTypeId],
    days: int = 1,
) -> Dict[Tuple[CaseTypeId, CourtStationId], float]:
    """
    Estimate expected case arrivals over a period.

    Args:
        avg_case_rate: Average daily case rates
        court_stations: Court stations to consider
        case_types: Case types to consider
        days: Number of days to estimate over

    Returns:
        Dictionary mapping (case_type, court_station) -> expected arrivals
    """
    estimates = {}

    for case_type in case_types:
        if case_type not in avg_case_rate:
            continue

        for court_station in court_stations:
            if court_station not in avg_case_rate[case_type]:
                continue

            lambda_rate = avg_case_rate[case_type][court_station]
            estimates[(case_type, court_station)] = lambda_rate * days

    return estimates
