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

# --- 01_cleaning.do locals ---------------------------------------------------
# Pandemic window (excluded), keyed on mediator_appointment_date (01_cleaning.do issue 4).
PANDEMIC_START = pd.Timestamp("2020-03-15")
PANDEMIC_END = pd.Timestamp("2021-06-30")
POST_PANDEMIC = pd.Timedelta(days=60)          # keep excluding this long after pandemic_end

# "Too new" cutoff: minimum days between appointment and datapull. The current Stata pipeline
# (02_Hazard_DMP.do) omits this filter; whether to apply it is a research decision deferred to
# the economist, so it's a parameter defaulting to 0 (no exclusion). Older pulls used 180 / 300.
DEFAULT_CUTOFF = 0

MIN_FIT_N = 30          # min usable cases per outcome to fit a type; fewer -> pooled proxy
MAX_PROXY_SHARE = 0.01  # warn if a proxied type exceeds this share of arrivals

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


def clean_hazard_sample(
    df: pd.DataFrame, datapull: pd.Timestamp, cutoff: int = DEFAULT_CUTOFF
) -> pd.DataFrame:
    """Reproduce the 01_cleaning.do + 02_Hazard_DMP.do sample selection.

    The hazard script drops four data-quality issues - missing mediator id, missing appointment
    date, conclusion-before-appointment, and pandemic appointment - plus terminated cases, and
    fits on concluded cases with an observed agreement outcome. Returns that sample with added
    `case_days_med`, `casetype_simplified`, and `log_case_days_med` columns.

    `cutoff` is the optional "too new" threshold (min days between appointment and datapull); the
    current Stata pipeline omits it, so it defaults to 0 (no exclusion) - see DEFAULT_CUTOFF.
    """
    df = df.copy()
    appt = pd.to_datetime(df["mediator_appointment_date"], errors="coerce")
    concl = pd.to_datetime(df["conclusion_date"], errors="coerce")

    # case_days_med: days under mediation; datapull fallback for non-concluded (01_cleaning.do)
    cdm = (concl - appt).dt.days
    cdm = cdm.where(df["case_status"] != "PENDING", (datapull - appt).dt.days)
    df["case_days_med"] = cdm

    feasible_gap = (datapull - appt).dt.days      # days elapsed since appointment

    no_issue = (
        df["mediator_id"].notna()                  # issue 1: missing mediator id
        & appt.notna()                             # issue 2: appointment date missing
        & (cdm >= 0)                               # issue 3: conclusion before appointment
        & ~((appt > PANDEMIC_START) & (appt < PANDEMIC_END + POST_PANDEMIC))  # issue 4: pandemic (appt date)
        & (feasible_gap >= cutoff)                 # "too new": omitted upstream, no-op at cutoff=0
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
    df: pd.DataFrame, datapull: pd.Timestamp, min_fit_n: int = MIN_FIT_N,
    cutoff: int = DEFAULT_CUTOFF,
) -> dict:
    """Fit lognormal duration params per (case type, outcome) from a raw case pull.

    Returns {case_type_name: {"agreement": {...}, "no agreement": {...}}} for every case type
    in the pull. A type with < `min_fit_n` usable cases in an outcome falls back to the pooled
    proxy (marked "proxied": True) rather than being dropped. The proxy is a modeling choice
    the paper sim never faced - the research team should sanity-check it; a guard warns if a
    proxied type's arrival share is large enough to matter. `cutoff` is the optional "too new"
    threshold (default 0 = no exclusion; see clean_hazard_sample).
    """
    sample = clean_hazard_sample(df, datapull, cutoff)
    outcomes = [("agreement", 1), ("no agreement", 0)]

    # Pooled fallback per outcome, for types too sparse to fit on their own data.
    pooled = {
        name: _lognormal_mle(sample.loc[sample["case_outcome_agreement"] == flag,
                                        "log_case_days_med"])
        for name, flag in outcomes
    }

    # Model every simplified type present in the pull, not just those with usable rows: a
    # type whose rows all fail cleaning yields an empty group and falls through to the pooled
    # proxy, rather than being silently dropped from the sim's case-type universe.
    empty = sample.iloc[0:0]
    grouped = {ct: sub for ct, sub in sample.groupby("casetype_simplified")}
    params = {}
    for ct in simplify_case_types(df["case_type"]).dropna().unique():
        g = grouped.get(ct, empty)
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


