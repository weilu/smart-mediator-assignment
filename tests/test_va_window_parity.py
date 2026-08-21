"""Guards the conditional window filter that keeps the library bit-parity with
cadaster-algo's VA_Antoine.py: the start/end window (and quasiyear re-anchoring) is
applied ONLY when end_date < reference_date. When end_date == reference_date (the
datapull), all cases are used and waiting-for-appointment (NaT med_appt_date) rows are
retained for prediction — dropping them silently diverges from the reference."""
import math
from datetime import datetime, date

from smart_mediator_assignment import estimate_va, VAEstimationConfig
from smart_mediator_assignment.core.case import SimpleCase


def _fitted():
    return [
        SimpleCase(
            id=i, case_type="Family group", court_station="MILIMANI",
            referral_date=date(2022, 1, 1), p_value=0.5, mediator_id=(i % 2) + 1,
            case_outcome_agreement=i % 2, mediator_appointment_date=date(2022, 1, 10),
            conclusion_date=date(2022, 2, 1), case_status="CONCLUDED",
            court_type="Magistrate", referral_mode="Referred by Court",
        )
        for i in range(6)
    ]


_WAITING = SimpleCase(
    id=999, case_type="Family group", court_station="MILIMANI",
    referral_date=date(2023, 5, 1), p_value=0.5, mediator_id=None,
    case_outcome_agreement=None, mediator_appointment_date=None,
    conclusion_date=None, case_status="PENDING",
    court_type="Magistrate", referral_mode="Referred by Court",
)


def test_end_at_reference_retains_waiting_for_appointment_case():
    # end_date == reference_date -> no window filter -> NaT-appt case kept (would be
    # dropped by an unconditional med_appt.between() filter, which returns False for NaT).
    cfg = VAEstimationConfig(reference_date=datetime(2023, 6, 1), days_since_appt_threshold=0)
    result = estimate_va(_fitted() + [_WAITING], config=cfg,
                         start_date="2016-04-06", end_date="2023-06-01")
    preds = {c.case_id: c.p_pred for c in result.case_predictions}
    assert 999 in preds
    assert math.isfinite(preds[999])


def test_end_before_reference_applies_window_filter():
    # end_date < reference_date -> window filter active -> an out-of-window appt is excluded.
    cfg = VAEstimationConfig(reference_date=datetime(2024, 6, 1), days_since_appt_threshold=0)
    out_of_window = SimpleCase(
        id=42, case_type="Family group", court_station="MILIMANI",
        referral_date=date(2019, 1, 1), p_value=0.5, mediator_id=1,
        case_outcome_agreement=1, mediator_appointment_date=date(2018, 1, 1),
        conclusion_date=date(2018, 2, 1), case_status="CONCLUDED",
        court_type="Magistrate", referral_mode="Referred by Court",
    )
    result = estimate_va(_fitted() + [out_of_window], config=cfg,
                         start_date="2021-01-01", end_date="2023-01-01")
    ids = {c.case_id for c in result.case_predictions}
    assert 42 not in ids
