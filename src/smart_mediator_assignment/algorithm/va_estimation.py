"""
Mediator Value Added (VA) estimation using absorbing regression with shrinkage.

This module provides batch estimation of mediator VA from historical case data.
The approach uses an absorbing least squares regression to control for case
characteristics, then applies shrinkage to produce stable VA estimates.
"""

from dataclasses import dataclass
from datetime import datetime, date
from typing import Union, Optional, List, Dict
import calendar
import pandas as pd
import numpy as np

from ..core.types import MediatorId
from ..core.case import CaseProtocol
from .case_types import simplify_case_types


@dataclass
class VAEstimationConfig:
    """Configuration for VA estimation."""

    reference_date: datetime
    min_med_cases: int = 2
    min_court_station_cases: int = 35
    min_case_type_cases: int = 2
    days_since_appt_threshold: int = 180
    pending_outcome_threshold_days: int = 90
    pandemic_start: datetime = datetime(2020, 3, 15)
    pandemic_end: datetime = datetime(2021, 8, 29)

    @classmethod
    def default(cls) -> "VAEstimationConfig":
        """Create config with default reference date of today."""
        return cls(reference_date=datetime.now())


@dataclass
class MediatorVAEstimate:
    """VA estimate for a single mediator."""

    mediator_id: MediatorId
    va: float
    n_cases: int


@dataclass
class CasePrediction:
    """Predicted probability and VA for a single case."""

    case_id: int
    mediator_id: int
    p_pred: float
    va: float
    case_outcome_agreement: Optional[int]


@dataclass
class VAEstimationResult:
    """Results from VA estimation."""

    mediator_vas: List[MediatorVAEstimate]
    case_predictions: List[CasePrediction]
    sigma: float
    # Clean, collapsed, pre-fit frame -- reusable as the input to
    # estimate_va_from_prepared for incremental refresh without re-cleaning.
    prepared: Optional[pd.DataFrame] = None

    def get_va_dict(self) -> Dict[MediatorId, float]:
        """Return VA estimates as dictionary."""
        return {m.mediator_id: m.va for m in self.mediator_vas}

    def get_p_pred_dict(self) -> Dict[int, float]:
        """Return case p_pred values as dictionary."""
        return {c.case_id: c.p_pred for c in self.case_predictions}


def _cases_to_dataframe(cases: List[CaseProtocol]) -> pd.DataFrame:
    """Convert list of cases to DataFrame for processing."""
    records = []
    for case in cases:
        records.append({
            'id': case.id,
            'mediator_id': case.mediator_id,
            'case_outcome_agreement': case.case_outcome_agreement,
            'mediator_appointment_date': case.mediator_appointment_date,
            'referral_date': case.referral_date,
            'conclusion_date': case.conclusion_date,
            'case_status': case.case_status,
            'case_type': case.case_type,
            'court_station': case.court_station,
            'court_type': case.court_type,
            'referral_mode': case.referral_mode,
        })
    return pd.DataFrame(records)


def _calculate_half_means(group: pd.DataFrame, mediator_id: int) -> pd.DataFrame:
    """Calculate half-split means for shrinkage estimation."""
    n = len(group)
    first_half = group.iloc[:n // 2]
    second_half = group.iloc[n // 2:]
    first_half_mean = first_half['residuals'].mean()
    second_half_mean = second_half['residuals'].mean()
    group = group.copy()
    group['half_mean'] = [first_half_mean] * len(first_half) + [second_half_mean] * len(second_half)
    group['mediator_id'] = mediator_id
    return group


def _simplify_case_types(df: pd.DataFrame) -> pd.DataFrame:
    """Simplify case types into broader groupings for regression.

    Delegates to the shared taxonomy (algorithm.case_types). Uses the 'AAAFamily group'
    label so the family grouping sorts first and becomes the omitted reference category in
    the fixed-effects regression.

    Args:
        df: DataFrame with 'case_type' column

    Returns:
        DataFrame with added 'casetype_simplified' column
    """
    df = df.copy()
    df['casetype_simplified'] = simplify_case_types(
        df['case_type'], family_group_label='AAAFamily group'
    )
    return df


def _assign_quasiyear(df: pd.DataFrame, reference_date) -> pd.DataFrame:
    """Assign quasiyear and appt_month columns to cases based on appointment date.

    Buckets appointments into roughly annual buckets using true calendar month-ends
    (via calendar.monthrange) as boundaries. Collapses the oldest bucket into the
    previous one when it spans < 365 days.

    Args:
        df: DataFrame with 'med_appt_date' column
        reference_date: Reference date for bucketing

    Returns:
        DataFrame with added 'appt_month' and 'quasiyear' columns
    """
    df = df.copy()
    df['appt_month'] = df['med_appt_date'].dt.month
    df['quasiyear'] = np.nan
    ref_month, ref_year = reference_date.month, reference_date.year
    for t in range(31):
        yu = ref_year - t
        ub = datetime(yu, ref_month, calendar.monthrange(yu, ref_month)[1])
        yl = ref_year - t - 1
        lb = datetime(yl, ref_month, calendar.monthrange(yl, ref_month)[1])
        df.loc[(df['med_appt_date'] <= ub) & (df['med_appt_date'] > lb), 'quasiyear'] = t
    # collapse the oldest bucket into the previous one when it spans < 1 year
    oldest_qy = df['quasiyear'].max()
    if pd.notna(oldest_qy):
        max_qy = df.loc[df['quasiyear'] == oldest_qy, 'med_appt_date'].max()
        min_qy = df.loc[df['quasiyear'] == oldest_qy, 'med_appt_date'].min()
        if pd.notna(max_qy) and (max_qy - min_qy).days < 365:
            df.loc[df['quasiyear'] == oldest_qy, 'quasiyear'] -= 1
    return df


def estimate_va(
    cases: List[CaseProtocol],
    config: Optional[VAEstimationConfig] = None,
    start_date: Optional[Union[str, datetime]] = None,
    end_date: Optional[Union[str, datetime]] = None,
) -> VAEstimationResult:
    """
    Estimate mediator Value Added (VA) from historical case data.

    Uses absorbing least squares regression with shrinkage to produce
    stable VA estimates that account for case characteristics.

    Args:
        cases: List of cases conforming to CaseProtocol
        config: Configuration for estimation (uses defaults if None)
        start_date: Filter cases with mediator_appointment_date >= start_date
        end_date: Filter cases with mediator_appointment_date < end_date

    Returns:
        VAEstimationResult containing mediator VAs, case predictions, sigma,
        and the reusable `prepared` frame (see estimate_va_from_prepared).
    """
    if config is None:
        config = VAEstimationConfig.default()

    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, '%Y-%m-%d')

    # Match VA_Antoine.py:89-96: only apply the start/end window (and re-anchor the
    # quasiyear bucketing on end_date) when end_date precedes the datapull/reference
    # date. When end_date == reference_date, all cases are used and waiting-for-
    # appointment (NaT med_appt_date) cases are retained for prediction.
    apply_window = (
        start_date is not None and end_date is not None
        and end_date < config.reference_date
    )
    anchor_date = end_date if apply_window else config.reference_date

    df = _cases_to_dataframe(cases)

    # Data preparation
    df['med_appt_date'] = pd.to_datetime(df['mediator_appointment_date'])
    df['days_since_appt'] = (config.reference_date - df['med_appt_date']).dt.days
    df['referral_date'] = pd.to_datetime(df['referral_date'])
    df['concl_date'] = pd.to_datetime(df['conclusion_date'])

    # Calculate days of mediation
    df.loc[df['case_status'] != 'PENDING', 'case_days_med'] = \
        (df['concl_date'] - df['med_appt_date']).dt.days
    df.loc[df['case_days_med'].isnull(), 'case_days_med'] = \
        (config.reference_date - df['med_appt_date']).dt.days

    # Simplify case types (Family group as reference)
    df = _simplify_case_types(df)

    # Court indicators
    df['highcourt'] = (df['court_type'] == 'High Court').astype(int)
    df['courtofappeal'] = (df['court_type'] == 'Court of Appeal').astype(int)

    # Milimani as reference court station
    df.loc[df['court_station'] == 'MILIMANI', 'court_station'] = 'AAAMilimani'

    # Generate quasi-year and month indicators
    df = _assign_quasiyear(df, anchor_date)

    # Case-level frame: captured before the "data issue" drops below, so cases
    # excluded from the fit (missing mediator, pandemic period, singleton
    # mediator/court-station/case-type groups) can still get a p_pred.
    df_case = df.copy()

    # Drop invalid cases
    df = df.dropna(subset=['mediator_id'])
    df = df.dropna(subset=['med_appt_date'])
    # df = df.dropna(subset=['referral_date'])
    df = df[df['case_days_med'] >= 0]

    # Exclude pandemic period
    df = df.loc[~df['med_appt_date'].between(
        config.pandemic_start, config.pandemic_end, inclusive="both"
    )]

    # Window filter (VA_Antoine.py:93) -- only when end_date precedes the reference date;
    # applied to df_case too so the fitted and prediction frames share one population.
    # When end_date == reference_date the reference uses all cases (NaT-appt rows kept).
    if apply_window:
        df = df.loc[df['med_appt_date'].between(start_date, end_date, inclusive="left")]
        df_case = df_case.loc[df_case['med_appt_date'].between(start_date, end_date, inclusive="left")]

    # Group small mediators
    concltotal = df[df['case_status'] == 'CONCLUDED'].groupby('mediator_id').size()
    df = df.merge(concltotal.rename('concltotal_med'), on='mediator_id', how='left')
    df.loc[
        (df['concltotal_med'] < config.min_med_cases) | pd.isna(df['concltotal_med']),
        'mediator_id'
    ] = -999

    # Group small court stations
    concltotal = df[df['case_status'] == 'CONCLUDED'].groupby('court_station').size()
    df = df.merge(concltotal.rename('concltotal_cs'), on='court_station', how='left')
    df.loc[
        (df['concltotal_cs'] <= config.min_court_station_cases) | pd.isna(df['concltotal_cs']),
        'court_station'
    ] = 'zzzSmall'

    # Group small case types
    concltotal = df[df['case_status'] == 'CONCLUDED'].groupby('casetype_simplified').size()
    df = df.merge(concltotal.rename('concltotal_ct'), on='casetype_simplified', how='left')
    df.loc[df['concltotal_ct'] < config.min_case_type_cases, 'casetype_simplified'] = 'zzzSmall'

    # Clean, collapsed, pre-fit frame -- captured before df_case's label backfill
    # below and before _fit_and_score mutates df, so it can be reused verbatim by
    # estimate_va_from_prepared for incremental refresh.
    prepared_frame = df.copy()

    # Backfill the small-group collapse (mediator_id/court_station/casetype_simplified ->
    # -999/zzzSmall) onto df_case for rows that survived into the fitted df. Without this,
    # no df_case row is ever labeled 'zzzSmall', so the zzzSmall param merge below never
    # matches and small-station/small-casetype cases get their coefficient silently
    # zeroed via skipna=True (VA_Antoine.py:229-235).
    df_case = df_case.set_index('id')
    df_case.update(df.set_index('id')[['mediator_id', 'court_station', 'casetype_simplified']])
    df_case = df_case.reset_index()

    # Waiting-for-appointment cases (no med_appt_date -> NaN quasiyear/appt_month from
    # _assign_quasiyear) still need prediction covariates: bucket them into the most
    # recent quasiyear/month so they get a p_pred instead of a silently-zeroed one.
    qy_mask = df_case['quasiyear'].isna() & ~df_case['med_appt_date'].between(
        config.pandemic_start, config.pandemic_end, inclusive="both")
    df_case.loc[qy_mask, 'quasiyear'] = df['quasiyear'].max()
    df_case.loc[df_case['appt_month'].isna(), 'appt_month'] = anchor_date.month

    result = _fit_and_score(df, df_case, config)
    result.prepared = prepared_frame
    return result


def estimate_va_from_prepared(
    prepared: pd.DataFrame,
    config: Optional[VAEstimationConfig] = None,
    *,
    start_date: Union[str, datetime],
    end_date: Union[str, datetime],
) -> VAEstimationResult:
    """
    Re-estimate mediator VA from a previously prepared (cleaned, collapsed) frame.

    Skips the cleaning pipeline entirely -- `prepared` must be a frame produced
    by `estimate_va` (i.e. `result.prepared`). Filters by both `med_appt_date`
    and `concl_date` within `[start_date, end_date)`, since a growing window's
    conclusion dates lag appointment dates.

    Args:
        prepared: Clean, collapsed frame from a prior `estimate_va` call
        config: Configuration for estimation (uses defaults if None)
        start_date: Filter cases with med_appt_date/concl_date >= start_date
        end_date: Filter cases with med_appt_date/concl_date < end_date

    Returns:
        VAEstimationResult containing mediator VAs, case predictions, sigma,
        and `prepared` set to the input frame (unchanged, for further reuse).
    """
    if config is None:
        config = VAEstimationConfig.default()

    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, '%Y-%m-%d')

    df = prepared.loc[
        prepared['med_appt_date'].between(start_date, end_date, inclusive="left")
        & prepared['concl_date'].between(start_date, end_date, inclusive="left")
    ].copy()
    # Labels already collapsed in `prepared` -- unlike the fresh path, no 4.1 backfill here.
    df_case = df.copy()

    result = _fit_and_score(df, df_case, config)
    result.prepared = prepared
    return result


def _fit_and_score(
    df: pd.DataFrame,
    df_case: pd.DataFrame,
    config: VAEstimationConfig,
) -> VAEstimationResult:
    """
    Shared core: regression, prediction, and shrinkage over an already-cleaned frame.

    Shared by estimate_va (fresh path) and estimate_va_from_prepared (reuse path).
    Does NOT set `.prepared` -- each entry point sets it after calling this.
    """
    import statsmodels.api as sm
    from linearmodels.iv import absorbing

    # Estimation dataset (cases appointed >= threshold days ago)
    df_estim = df[df['days_since_appt'] >= config.days_since_appt_threshold]
    df_estim = df_estim[[
        'case_outcome_agreement', 'appt_month', 'quasiyear',
        'casetype_simplified', 'highcourt', 'courtofappeal',
        'court_station', 'referral_mode', 'mediator_id', 'med_appt_date'
    ]].dropna()

    df_estim[['mediator_id', 'appt_month', 'quasiyear']] = df_estim[[
        'mediator_id', 'appt_month', 'quasiyear'
    ]].astype('category')

    # Run absorbing regression
    depend = df_estim['case_outcome_agreement']
    independ = sm.tools.tools.add_constant(df_estim[[
        'appt_month', 'quasiyear', 'casetype_simplified', 'highcourt',
        'courtofappeal', 'court_station', 'referral_mode'
    ]])
    catgrcl = df_estim[['mediator_id']]

    model = absorbing.AbsorbingLS(depend, independ, absorb=catgrcl, drop_absorbed=True)
    model_res = model.fit(cov_type='robust')

    # Extract parameters
    params = model_res.params.to_frame().reset_index()
    params = params.rename(columns={"parameter": "value"})
    split_cols = params['index'].str.split('.', expand=True)
    params['var'] = split_cols[0].astype('str')
    params['var_cat'] = split_cols[1] if 1 in split_cols.columns else None
    params['empty'] = split_cols[2] if 2 in split_cols.columns else None

    # Build parameter dictionaries
    dependent_vars = [
        'appt_month', 'quasiyear', 'casetype_simplified', 'court_station',
        'referral_mode', 'highcourt', 'courtofappeal', 'const'
    ]
    params_dict = {}
    for x in dependent_vars:
        params_dict[f"params_{x}"] = params[params['var'] == x].copy()
        params_dict[f"params_{x}"] = params_dict[f"params_{x}"].rename(
            columns={"var_cat": x, "value": f"{x}_val"}
        )
        params_dict[f"params_{x}"] = params_dict[f"params_{x}"][[x, f"{x}_val"]]
        if x in ('highcourt', 'courtofappeal'):
            params_dict[f"params_{x}"][[x]] = 1
            params_dict[f"params_{x}"].loc[-1] = [0, 0]
        if x in ('appt_month', 'quasiyear', 'highcourt', 'courtofappeal'):
            params_dict[f"params_{x}"][[x]] = params_dict[f"params_{x}"][[x]].astype('float64')

    # Add omitted categories
    params_dict['params_appt_month'].loc[-1] = [1, 0]
    params_dict['params_quasiyear'].loc[-1] = [0, 0]
    params_dict['params_casetype_simplified'].loc[-1] = ['AAAFamily group', 0]
    params_dict['params_court_station'].loc[-1] = ['AAAMilimani', 0]
    params_dict['params_referral_mode'].loc[-1] = ['Referred by Court', 0]
    params_dict['params_const'] = params_dict['params_const'].rename(
        columns={"const": 'case_outcome_agreement'}
    )
    params_dict['params_const']['case_outcome_agreement'] = 1
    params_dict['params_const'].loc[-1] = [0, params_dict['params_const']['const_val'].mean()]

    # Handle pending cases (applied to both frames; the fresh-pending drop is df-only
    # since df_case must still carry every windowed case through to prediction)
    df.loc[
        (df['days_since_appt'] > config.pending_outcome_threshold_days) &
        (df['case_status'] == 'PENDING'),
        'case_outcome_agreement'
    ] = 0
    df_case.loc[
        (df_case['days_since_appt'] > config.pending_outcome_threshold_days) &
        (df_case['case_status'] == 'PENDING'),
        'case_outcome_agreement'
    ] = 0
    df = df[~(
        (df['days_since_appt'] <= config.pending_outcome_threshold_days) &
        (df['case_status'] == 'PENDING')
    )]

    # Merge parameters to get predictions over ALL cases (df_case), not just the
    # fitted sample, so cases dropped above still get a p_pred.
    df_case = df_case.merge(params_dict['params_appt_month'], on='appt_month', how='left')
    df_case = df_case.merge(params_dict['params_quasiyear'], on='quasiyear', how='left')
    df_case = df_case.merge(params_dict['params_casetype_simplified'], on='casetype_simplified', how='left')
    df_case = df_case.merge(params_dict['params_court_station'], on='court_station', how='left')
    df_case = df_case.merge(params_dict['params_referral_mode'], on='referral_mode', how='left')
    df_case = df_case.merge(params_dict['params_highcourt'], on='highcourt', how='left')
    df_case = df_case.merge(params_dict['params_courtofappeal'], on='courtofappeal', how='left')
    # params_const is keyed on case_outcome_agreement {0,1}; pending cases (null outcome) do
    # not match, so const_val stays NaN and the skipna sum below drops the intercept for them.
    # Faithful to VA_Antoine.py - changing it would diverge from the parity baseline.
    df_case = df_case.merge(params_dict['params_const'], on='case_outcome_agreement', how='left')

    # Calculate predictions and residuals
    df_case['p_pred'] = df_case[[
        'appt_month_val', 'quasiyear_val', 'casetype_simplified_val',
        'court_station_val', 'referral_mode_val', 'highcourt_val',
        'courtofappeal_val', 'const_val'
    ]].sum(axis=1, skipna=True)

    # court_station values grouped into 'zzzSmall' in df but absent from df_case
    # (captured before that grouping) won't match a fitted category -> backfill.
    small_mask = df_case['court_station'] == 'zzzSmall'
    if small_mask.any():
        cs_small_value = df_case.loc[small_mask, 'court_station_val'].iloc[0]
        df_case['court_station_val'] = df_case['court_station_val'].fillna(cs_small_value)

    df_case['residuals'] = df_case['case_outcome_agreement'] - df_case['p_pred']

    # Map predictions/residuals back onto the fitted df for shrinkage/VA below
    df['p_pred'] = df['id'].map(df_case.set_index('id')['p_pred'])
    df['residuals'] = df['id'].map(df_case.set_index('id')['residuals'])

    # Calculate VA with shrinkage
    df['total_med_cases'] = df.groupby('mediator_id')['residuals'].transform('count')
    df = df.sort_values(['mediator_id', 'med_appt_date', 'id'])

    df['temp1'] = df.groupby('mediator_id')['residuals'].transform(
        lambda x: x.iloc[:len(x) // 2].mean()
    )
    df['average1'] = df.groupby('mediator_id')['temp1'].transform('mean')
    df['temp2'] = df.groupby('mediator_id')['residuals'].transform(
        lambda x: x.iloc[len(x) // 2:].mean()
    )
    df['average2'] = df.groupby('mediator_id')['temp2'].transform('mean')

    # Covariance for shrinkage
    average1_2 = df.groupby('mediator_id')[['average1', 'average2']].first().dropna()
    cov_avgs = np.cov(average1_2['average1'], average1_2['average2'])[0, 1]
    sigma = np.sqrt(cov_avgs)

    # Half-case deviation
    df = df.sort_values(by=['mediator_id', 'med_appt_date', 'id']).reset_index(drop=True)
    df = pd.concat([
        _calculate_half_means(group, mediator_id)
        for mediator_id, group in df.groupby('mediator_id', sort=False)
    ]).reset_index(drop=True)
    df['halfcases_dev'] = df['residuals'] - df['half_mean']

    var_halfcases_dev = df['halfcases_dev'].std() ** 2
    var_residuals = df['residuals'].std() ** 2
    final_rst = var_residuals - cov_avgs - var_halfcases_dev

    # Shrinkage calculation
    df['h'] = 1 / (final_rst + (var_halfcases_dev / (0.5 * df['total_med_cases'])))
    df['h2'] = 1 / (2 * df['h'])
    df['shrinkage'] = cov_avgs / (cov_avgs + df['h2'])
    df['shrinkage_med'] = df.groupby('mediator_id')['shrinkage'].transform('mean')
    df['naive_med'] = df.groupby('mediator_id')['residuals'].transform('mean')
    df['va'] = df['shrinkage_med'] * df['naive_med']

    # Build results
    df_mediator = df.groupby('mediator_id').agg({
        'va': 'first',
        'total_med_cases': 'first'
    }).reset_index()

    mediator_vas = [
        MediatorVAEstimate(
            mediator_id=int(row['mediator_id']),
            va=float(row['va']),
            n_cases=int(row['total_med_cases'])
        )
        for _, row in df_mediator.iterrows()
        if row['mediator_id'] != -999
    ]

    # VA is only defined for mediators in the fitted df; cases present only in
    # df_case (dropped above for data issues) get NaN here.
    df_case['va'] = df_case['id'].map(df.set_index('id')['va'])

    case_predictions = [
        CasePrediction(
            case_id=int(row['id']),
            mediator_id=int(row['mediator_id']) if pd.notna(row['mediator_id']) else -999,
            p_pred=float(row['p_pred']),
            va=float(row['va']),
            case_outcome_agreement=int(row['case_outcome_agreement']) if pd.notna(row['case_outcome_agreement']) else None
        )
        for _, row in df_case[['id', 'mediator_id', 'p_pred', 'va', 'case_outcome_agreement']].iterrows()
    ]

    return VAEstimationResult(
        mediator_vas=mediator_vas,
        case_predictions=case_predictions,
        sigma=float(sigma)
    )
