"""Case-duration lognormal parameters, reproduced from the Stata pipeline in Python.

The duration parameters (Intercept, Sigma per case type x outcome) were historically
produced by two Stata scripts run on a WB-side data pull:
  - cleaning.do        -> builds the analysis sample (exclusion / issue6 / issue7 flags)
  - 02_Hazard_AS.do    -> streg, distribution(lognormal) -> Casetypes_parameters_lognormal.xlsx

This module reproduces both, faithfully, on a deidentified case pull, so the params
re-compute automatically on every fresh pull and cover every case type the RCT handles.
Stata fit only 8 of 10 types; the two too-sparse-to-fit types (Constitution and Human Rights,
Judicial Review) get a pooled proxy duration so their cases are still simulated - the SMaRT RCT
is system-wide across all case types (PAP_Nov2024), so none are dropped.

The intercept-only lognormal AFT with no censoring (the analysis sample keeps only concluded,
non-terminated cases with an observed outcome) has a closed-form MLE: it is just a Normal
fit to log(duration). No survival optimizer is needed.

These durations are a simulation input (they model how long a case occupies a mediator over
the horizon); the production assignment algorithm does not use them.
"""
import logging
import warnings

import numpy as np
import pandas as pd

from .case_types import simplify_case_types

_log = logging.getLogger(__name__)

# --- cleaning.do locals (verbatim) -------------------------------------------
CUTOFF = 300                                   # days: "too new" (issue 6) threshold
PANDEMIC_START = pd.Timestamp("2020-03-15")
PANDEMIC_END = pd.Timestamp("2021-06-30")
POST_PANDEMIC = pd.Timedelta(days=60)          # keep excluding this long after pandemic_end

# Below this many usable cases per outcome, a case type cannot be reliably fit on its own data
# and is given a proxy duration (the pooled all-cases distribution) instead. Its cases are
# still modeled: the SMaRT RCT is system-wide across all case types (PAP_Nov2024), so every
# arriving type must occupy mediator capacity in the sim - none are dropped.
MIN_FIT_N = 30

# A proxied type's duration is only approximate, which matters only at volume. Warn if a
# proxied type's arrival share exceeds this - a prompt to obtain real params for it.
MAX_PROXY_SHARE = 0.01

# Correct identity of the 16 columns in the legacy Casetypes_parameters_lognormal.xlsx,
# established empirically by matching N-size + intercepts back to reproduced categories.
# The xlsx etable order is [casetype_simplified code {1,4,5,7,8,11,13,14}] x [agreement, no agreement].
XLSX_COLUMN_IDENTITY = [
    ("Children Custody and Maintenance", "agreement"),      # code 1
    ("Children Custody and Maintenance", "no agreement"),
    ("Commercial and tax group", "agreement"),              # code 4
    ("Commercial and tax group", "no agreement"),
    ("Criminal Cases", "agreement"),                        # code 5
    ("Criminal Cases", "no agreement"),
    ("Employment and Labour Relations Cases (ELRC)", "agreement"),   # code 7
    ("Employment and Labour Relations Cases (ELRC)", "no agreement"),
    ("Environment and Land Cases (ELC)", "agreement"),      # code 8
    ("Environment and Land Cases (ELC)", "no agreement"),
    ("Matrimonial Property Cases", "agreement"),            # code 11
    ("Matrimonial Property Cases", "no agreement"),
    ("Civil group", "agreement"),                           # code 13
    ("Civil group", "no agreement"),
    ("Family group", "agreement"),                          # code 14
    ("Family group", "no agreement"),
]


def _lognormal_mle(log_durations: pd.Series) -> dict:
    """Closed-form MLE of an intercept-only lognormal AFT with no censoring.

    Equivalent to Stata `streg, distribution(lognormal)` on an all-events sample:
    Intercept = mean(ln t), sigma = population SD(ln t) (streg uses the ML /N estimator).
    """
    intercept = float(log_durations.mean())
    sigma = float(log_durations.std(ddof=0))
    return {
        "Intercept": intercept,
        "Insigma": float(np.log(sigma)) if sigma > 0 else float("nan"),
        "Sigma": sigma,
        "Number of observations": int(len(log_durations)),
    }


def clean_hazard_sample(df: pd.DataFrame, datapull: pd.Timestamp) -> pd.DataFrame:
    """Reproduce cleaning.do + 02_Hazard_AS.do sample selection.

    Returns concluded, non-terminated cases with an observed agreement outcome and no
    data-quality/too-new/pandemic issue, with added `case_days_med`, `casetype_simplified`,
    and `log_case_days_med` columns.
    """
    df = df.copy()
    ref = pd.to_datetime(df["referral_date"], errors="coerce")
    appt = pd.to_datetime(df["mediator_appointment_date"], errors="coerce")
    concl = pd.to_datetime(df["conclusion_date"], errors="coerce")

    # case_days_med: days under mediation; datapull fallback for non-concluded (cleaning.do:144-145)
    cdm = (concl - appt).dt.days
    cdm = cdm.where(df["case_status"] != "PENDING", (datapull - appt).dt.days)
    df["case_days_med"] = cdm

    feasible_gap = (datapull - appt).dt.days      # days elapsed since appointment
    gap = (appt - ref).dt.days                     # appointment lag

    # issue == missing  <=>  none of cleaning.do's conditions 1..7 apply. The hazard script
    # drops exclusion(1-5) + issue6 + issue7, i.e. every flagged case, so the sample is
    # exactly the unflagged cases.
    no_issue = (
        df["mediator_id"].notna()                  # not issue 1
        & (gap >= 0)                               # not issue 2
        & appt.notna()                             # not issue 3
        & (cdm >= 0)                               # not issue 4
        & ((feasible_gap - cdm) >= 0)              # not issue 5
        & (feasible_gap >= CUTOFF)                 # not issue 6 (too new)
        & ~((ref > PANDEMIC_START) & (ref < PANDEMIC_END + POST_PANDEMIC))  # not issue 7 (pandemic)
    )

    df["casetype_simplified"] = simplify_case_types(df["case_type"])

    sample = df[
        no_issue
        & (df["case_status"] == "CONCLUDED")
        & (df["outcome_name"] != "Terminated")     # caseoutcome == 4
        & (df["case_days_med"] > 0)                # stset drops t <= 0
        & (df["case_outcome_agreement"].isin([0, 1]))
    ].copy()
    sample["log_case_days_med"] = np.log(sample["case_days_med"])
    return sample


def estimate_lognormal_duration_params(
    df: pd.DataFrame, datapull: pd.Timestamp, min_fit_n: int = MIN_FIT_N
) -> dict:
    """Fit lognormal duration params per (case type, outcome) from a raw case pull.

    Returns {case_type_name: {"agreement": {...}, "no agreement": {...}}} for EVERY case type
    present in the pull - the SMaRT RCT is system-wide, so all arriving types are modeled. A
    type with < `min_fit_n` usable cases in an outcome (Constitution and Human Rights, Judicial
    Review) is given a proxy for that outcome - the pooled all-cases distribution, marked
    "proxied": True - rather than dropped, so its cases still occupy mediator capacity.

    NOTE: the proxy is a modeling choice the paper sim never made (it scoped to 8 fittable
    types); the research team should sanity-check it. A guard warns if a proxied type's
    arrival share is large enough that the approximation could matter.
    """
    sample = clean_hazard_sample(df, datapull)
    outcomes = [("agreement", 1), ("no agreement", 0)]

    # Pooled fallback per outcome, for types too sparse to fit on their own data.
    pooled = {
        name: _lognormal_mle(sample.loc[sample["case_outcome_agreement"] == flag,
                                        "log_case_days_med"])
        for name, flag in outcomes
    }

    params = {}
    for ct, g in sample.groupby("casetype_simplified"):
        entry = {}
        for name, flag in outcomes:
            logd = g.loc[g["case_outcome_agreement"] == flag, "log_case_days_med"]
            if len(logd) >= min_fit_n:
                entry[name] = _lognormal_mle(logd)
            else:
                proxy = dict(pooled[name])
                proxy["proxied"] = True
                proxy["proxy_n"] = int(len(logd))  # this type's own (insufficient) count
                entry[name] = proxy
        params[ct] = entry

    _guard_proxied_types(df, params)
    return params


def _guard_proxied_types(df: pd.DataFrame, params: dict) -> None:
    """Log which case types are proxied and warn if a proxied type is too large to approximate."""
    proxied = {ct for ct, entry in params.items()
               if any(o.get("proxied") for o in entry.values())}
    if not proxied:
        return

    simplified = simplify_case_types(df["case_type"])
    shares = simplified.value_counts(normalize=True)
    proxied_shares = {ct: float(shares.get(ct, 0.0)) for ct in proxied}

    _log.info("duration params proxied (pooled) for sparse types: %s",
              {ct: f"{s:.3%}" for ct, s in proxied_shares.items()})
    oversized = {ct: s for ct, s in proxied_shares.items() if s > MAX_PROXY_SHARE}
    if oversized:
        warnings.warn(
            "proxied case type(s) exceed the safe share "
            f"(> {MAX_PROXY_SHARE:.0%}): "
            f"{ {ct: f'{s:.2%}' for ct, s in oversized.items()} }; their durations are "
            "approximated by the pooled distribution - obtain real params for them"
        )


def read_xlsx_duration_params(path) -> dict:
    """Read the legacy Casetypes_parameters_lognormal.xlsx with the CORRECT column identity.

    Reference reader for parity checks against the Stata output (not wired into the sim - the
    sim always reproduces from the pull). Returns only the 8 types Stata actually fit;
    Constitution and Human Rights + Judicial Review are absent (never fit).
    """
    x = pd.read_excel(path, header=None)
    # rows: 1=Intercept, 3=lnsigma, 5=sigma, 7=N ; cols 1..16 are the estimates.
    params = {}
    for col, (name, outcome) in enumerate(XLSX_COLUMN_IDENTITY, start=1):
        params.setdefault(name, {})[outcome] = {
            "Intercept": float(x.iloc[1, col]),
            "Insigma": float(x.iloc[3, col]),
            "Sigma": float(x.iloc[5, col]),
            "Number of observations": int(x.iloc[7, col]),
        }
    return params
