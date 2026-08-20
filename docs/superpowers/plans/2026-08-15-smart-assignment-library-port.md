# PR A — Smart-Assignment Library Port + QP Path Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `smart-mediator-assignment` the canonical implementation: port `cadaster-algo` `main`'s VA + phantom changes (commit `89d0039`), add an OSQP-based QP solver as an *optional* install target, and end installable/version-bumped so the companion `cadaster-algo` PR (PR B) can depend on it.

**Architecture:** Update `algorithm/va_estimation.py` in place (window filter, quasiyear, case-type taxonomy, df/df_case prediction coverage, parameterized reference date + reusable prepared frame); isolate phantom RNG; add `solver/qp_solver.py` (`QPSolver(BaseSolver)`) ported from `cadaster-algo/SlackedQPwithLoadOSQP.py`, adapted to library types and selected via `AlgorithmConfig.use_qp`; declare `osqp` as an optional-dependency extra so the default install stays lean. The LP solver, CRPS, and Bayesian belief code are not touched.

**Tech Stack:** Python ≥3.10, pandas, numpy, scipy, statsmodels, linearmodels, pulp, (optional) osqp, pytest. Package manager: `uv`.

## Global Constraints

- Reference for VA: `/Users/wlu4/workspace/kenya/kenya-mediation/cadaster-algo/VA_Antoine.py::calculate_VA_Antoine` (@ `main`, `89d0039`).
- Reference for QP: `/Users/wlu4/workspace/kenya/kenya-mediation/cadaster-algo/SlackedQPwithLoadOSQP.py::slackedQP` (OSQP-based; imports `osqp`, `scipy.sparse`).
- Do NOT modify `solver/lp_solver.py`, `crps.py`, `core/belief.py`, `algorithm/bayesian.py` (unchanged on `main`).
- **No backward-compatibility obligation.** Nothing consumes the library's public API in production yet (the Django app doesn't import it; `cadaster-algo`'s only importers are the dormant adapter + 2 tests, all rewritten in PR B; plus the library's own tests). This pre-integration window is the right time to shape the API cleanly — break signatures freely and update the in-repo callers/tests in the same PR. The safety rail is **numeric parity** (the golden VA fixture in PR B), NOT signature stability.
- `osqp` MUST be an optional extra (like the existing `gurobi` extra) — NOT a default/runtime dependency. `import smart_mediator_assignment` must succeed with osqp absent; only constructing `QPSolver` may require it.
- Repo owner's Python rules: no "what" comments; preserve comments when moving code; imports at file top; update tests on signature/behavior change; add tests for new behavior.
- Run tests with **`uv run python -m pytest`** from `smart-mediator-assignment/` (the bare `pytest` console script mis-resolves to a stale global path in this env; `python -m pytest` uses the project venv correctly). Substitute this wherever a step shows `uv run pytest`.
- Reference categories (sort-prefix hacks that force omitted categories in the absorbing regression) stay: case-type `AAAFamily group`, court-station `AAAMilimani`, referral-mode `Referred by Court`.
- **Naming decision:** do NOT add a `legacy_case_type` field. In the library `case_type` is already the granular type and simplification is internal to `estimate_va`; cadaster-algo's `legacy_case_type` name (implies deprecated; actually the original granular type) is not carried into the canonical repo. PR B's adapter maps cadaster `legacy_case_type` → `SimpleCase.case_type`.

---

### Task 1: Isolate phantom-generation RNG

`generate_phantom_cases` mutates process-global `np.random`/`random` state. Switch to local RNG instances (matches `main`'s determinism refactor).

**Files:**
- Modify: `src/smart_mediator_assignment/algorithm/phantom.py`
- Test: `tests/unit/test_phantom.py` (add)

**Interfaces:**
- Produces: `generate_phantom_cases(..., seed)` — same seed yields identical output without disturbing global `np.random`/`random`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_phantom.py  (add)
import numpy as np, random
from datetime import date
from smart_mediator_assignment import generate_phantom_cases

_ARGS = dict(
    current_day=date(2023, 1, 1), time_horizon=5,
    avg_case_rate={"Family group": {"MILIMANI": 2.0}},
    avg_p_val_by_crt_case_type={("Family group", "MILIMANI"): 0.5},
    med_by_court_case_type={"MILIMANI": {"Family group": [1, 2]}},
    court_stations=["MILIMANI"], case_types=["Family group"],
)

def test_same_seed_is_reproducible():
    a, _ = generate_phantom_cases(**_ARGS, seed=7)
    b, _ = generate_phantom_cases(**_ARGS, seed=7)
    assert [(c.id, c.referral_date) for c in a] == [(c.id, c.referral_date) for c in b]

def test_does_not_disturb_global_rng():
    np.random.seed(123); random.seed(123)
    exp_np, exp_py = np.random.rand(), random.random()
    np.random.seed(123); random.seed(123)
    generate_phantom_cases(**_ARGS, seed=999)
    assert np.random.rand() == exp_np and random.random() == exp_py
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_phantom.py -k global_rng -v`
Expected: FAIL — global RNG was mutated.

- [ ] **Step 3: Implement local RNG**

In `phantom.py`, replace the seeding block and draw sites:

```python
# was: if seed is not None: np.random.seed(seed); random.seed(seed)
rng = np.random.default_rng(seed)
py_rng = random.Random(seed)
```
```python
num_cases = rng.poisson(lambda_rate)   # was np.random.poisson(lambda_rate)
```
```python
order_key = py_rng.uniform(0, 1)       # was random.uniform(0, 1)
```

Keep `import random` (needed for `random.Random`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_phantom.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/smart_mediator_assignment/algorithm/phantom.py tests/unit/test_phantom.py
git commit -m "fix(phantom): use isolated local RNG for reproducible phantom generation"
```

---

### Task 2: VA — restrict the estimation window by `mediator_appointment_date`

`main` filters cases into the regression by `med_appt_date.between(start, end, inclusive="left")`; the library filters by `referral_date`. Change which cases enter estimation.

**Files:**
- Modify: `src/smart_mediator_assignment/algorithm/va_estimation.py` (date-range filter block, ~lines 192–196; and the `dropna(['referral_date'])` at line 184)
- Test: `tests/test_va_estimation.py` (add)

**Interfaces:**
- Produces: a case is included when `start_date <= mediator_appointment_date < end_date`, independent of `referral_date`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_va_estimation.py  (add)
from datetime import datetime, date
from smart_mediator_assignment import estimate_va, VAEstimationConfig
from smart_mediator_assignment.core.case import SimpleCase

def _case(cid, appt, referral, outcome=1):
    return SimpleCase(id=cid, case_type="Family group", court_station="MILIMANI",
        referral_date=referral, p_value=0.5, mediator_id=1, case_outcome_agreement=outcome,
        mediator_appointment_date=appt, conclusion_date=appt, case_status="CONCLUDED",
        court_type="Magistrate", referral_mode="Referred by Court")

def test_window_filters_on_appointment_date_not_referral():
    inside = _case(1, appt=date(2022, 6, 1), referral=date(2019, 1, 1))
    outside = _case(2, appt=date(2018, 1, 1), referral=date(2022, 6, 1))
    cfg = VAEstimationConfig(reference_date=datetime(2023, 6, 1), days_since_appt_threshold=0)
    result = estimate_va([inside, outside], config=cfg, start_date="2022-01-01", end_date="2023-01-01")
    ids = {c.case_id for c in result.case_predictions}
    assert 1 in ids and 2 not in ids
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_va_estimation.py::test_window_filters_on_appointment_date_not_referral -v`
Expected: FAIL.

- [ ] **Step 3: Implement the filter change**

Replace the date-range filter:

```python
if start_date is not None and end_date is not None:
    df = df.loc[df['med_appt_date'].between(start_date, end_date, inclusive="left")]
elif start_date is not None:
    df = df.loc[df['med_appt_date'] >= start_date]
elif end_date is not None:
    df = df.loc[df['med_appt_date'] < end_date]
```

Comment out `df = df.dropna(subset=['referral_date'])` (main no longer drops on referral/conclusion date). Keep `mediator_id` and `med_appt_date` dropna.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_va_estimation.py -v`
Expected: PASS (new test + existing recovery test).

- [ ] **Step 5: Commit**

```bash
git add src/smart_mediator_assignment/algorithm/va_estimation.py tests/test_va_estimation.py
git commit -m "fix(va): filter estimation window by mediator_appointment_date (left-closed)"
```

---

### Task 3: VA — quasiyear buckets use true month length + oldest-bucket collapse

`main` uses `calendar.monthrange(year, month)[1]` for bucket bounds (vs hardcoded day-28) and collapses the oldest bucket into the previous one when it spans < 365 days.

**Files:**
- Modify: `src/smart_mediator_assignment/algorithm/va_estimation.py` (quasiyear loop, ~lines 172–179; add `import calendar`)
- Test: `tests/test_va_estimation.py` (add)

**Interfaces:**
- Produces: quasiyear boundaries anchored on the true last calendar day of `reference_date.month`; no `ValueError` for reference dates in 30-day months / Feb-29.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_va_estimation.py  (add)
from datetime import datetime, date
from smart_mediator_assignment import estimate_va, VAEstimationConfig
from smart_mediator_assignment.core.case import SimpleCase

def test_estimate_va_runs_with_reference_in_february():
    cases = [SimpleCase(id=i, case_type="Family group", court_station="MILIMANI",
        referral_date=date(2022, 1, 1), p_value=0.5, mediator_id=1, case_outcome_agreement=i % 2,
        mediator_appointment_date=date(2022, 2, 10), conclusion_date=date(2022, 3, 1),
        case_status="CONCLUDED", court_type="Magistrate", referral_mode="Referred by Court")
        for i in range(4)]
    cfg = VAEstimationConfig(reference_date=datetime(2024, 2, 29), days_since_appt_threshold=0)
    estimate_va(cases, config=cfg, start_date="2021-01-01", end_date="2023-01-01")  # must not raise
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_va_estimation.py::test_estimate_va_runs_with_reference_in_february -v`
Expected: FAIL (day-28 anchoring mis-buckets / raises).

- [ ] **Step 3: Implement the calendar-correct loop**

Add `import calendar` at top. Replace the quasiyear loop:

```python
df['appt_month'] = df['med_appt_date'].dt.month
df['quasiyear'] = np.nan
ref_month, ref_year = config.reference_date.month, config.reference_date.year
for t in range(31):
    yu = ref_year - t
    ub = datetime(yu, ref_month, calendar.monthrange(yu, ref_month)[1])
    yl = ref_year - t - 1
    lb = datetime(yl, ref_month, calendar.monthrange(yl, ref_month)[1])
    df.loc[(df['med_appt_date'] <= ub) & (df['med_appt_date'] > lb), 'quasiyear'] = t

oldest_qy = df['quasiyear'].max()
if pd.notna(oldest_qy):
    max_qy = df.loc[df['quasiyear'] == oldest_qy, 'med_appt_date'].max()
    min_qy = df.loc[df['quasiyear'] == oldest_qy, 'med_appt_date'].min()
    if pd.notna(max_qy) and (max_qy - min_qy).days < 365:
        df.loc[df['quasiyear'] == oldest_qy, 'quasiyear'] -= 1
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_va_estimation.py -k "february or recover" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/smart_mediator_assignment/algorithm/va_estimation.py tests/test_va_estimation.py
git commit -m "fix(va): quasiyear buckets use calendar month length + collapse short oldest bucket"
```

---

### Task 4: VA — add "Commercial and tax group" case-type mapping

`main` maps `Commercial Cases` and `Tax Appeals` into `Commercial and tax group`. Extract the simplification into a testable helper and add the rule.

**Files:**
- Modify: `src/smart_mediator_assignment/algorithm/va_estimation.py` (casetype block ~lines 154–163)
- Test: `tests/test_va_estimation.py` (add)

**Interfaces:**
- Produces: module-level `_simplify_case_types(df) -> df` with `casetype_simplified`; `Commercial Cases`/`Tax Appeals` → `Commercial and tax group`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_va_estimation.py  (add)
from datetime import date
from smart_mediator_assignment.algorithm import va_estimation
from smart_mediator_assignment.core.case import SimpleCase

def _df(ct):
    c = SimpleCase(id=1, case_type=ct, court_station="MILIMANI", referral_date=date(2022,1,1),
        p_value=0.5, mediator_id=1, case_outcome_agreement=1,
        mediator_appointment_date=date(2022,1,1), conclusion_date=date(2022,2,1),
        case_status="CONCLUDED", court_type="Magistrate", referral_mode="Referred by Court")
    return va_estimation._cases_to_dataframe([c])

def test_commercial_and_tax_group_mapping():
    for ct in ("Commercial Cases", "Tax Appeals"):
        out = va_estimation._simplify_case_types(_df(ct))
        assert out.loc[0, "casetype_simplified"] == "Commercial and tax group"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_va_estimation.py::test_commercial_and_tax_group_mapping -v`
Expected: FAIL — helper absent.

- [ ] **Step 3: Implement the helper**

```python
def _simplify_case_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['casetype_simplified'] = df['case_type']
    df.loc[df['case_type'].isin(['Civil Cases', 'Civil Appeals']), 'casetype_simplified'] = 'Civil group'
    df.loc[df['case_type'].isin([
        'Divorce and Separation', 'Family Appeals', 'Family Miscellaneous',
        'Succession (Probate & Administration - P&A)'
    ]), 'casetype_simplified'] = 'AAAFamily group'
    df.loc[df['case_type'].isin(['Commercial Cases', 'Tax Appeals']),
           'casetype_simplified'] = 'Commercial and tax group'
    return df
```

Replace the inline block in `estimate_va` with `df = _simplify_case_types(df)`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_va_estimation.py -k "commercial or recover" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/smart_mediator_assignment/algorithm/va_estimation.py tests/test_va_estimation.py
git commit -m "feat(va): map Commercial Cases and Tax Appeals to Commercial and tax group"
```

---

### Task 5: VA — predict `p_pred`/residuals over all cases via a `df_case` frame

`main` fits on the cleaned `df` but computes `p_pred`/residuals over a `df_case` frame that keeps data-issue rows, then maps back to `df` for shrinkage; it also fills `court_station_val` for unseen stations with the `zzzSmall` coefficient.

**Files:**
- Modify: `src/smart_mediator_assignment/algorithm/va_estimation.py` (capture `df_case` before drops; prediction/merge block ~lines 291–307; results assembly ~lines 363–372)
- Test: `tests/test_va_estimation.py` (add)

**Interfaces:**
- Produces: `case_predictions` includes a `p_pred` for every windowed case that has prediction covariates, including cases excluded from the fit (e.g. singleton mediator grouped to `-999`). `mediator_vas` still computed over the fitted `df` (excludes `-999`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_va_estimation.py  (add)
from datetime import datetime, date
from smart_mediator_assignment import estimate_va, VAEstimationConfig
from smart_mediator_assignment.core.case import SimpleCase

def _mk(cid, med, outcome):
    return SimpleCase(id=cid, case_type="Family group", court_station="MILIMANI",
        referral_date=date(2022,1,1), p_value=0.5, mediator_id=med, case_outcome_agreement=outcome,
        mediator_appointment_date=date(2022,1,10), conclusion_date=date(2022,2,1),
        case_status="CONCLUDED", court_type="Magistrate", referral_mode="Referred by Court")

def test_p_pred_present_for_singleton_mediator_case():
    cases = [_mk(1,1,1), _mk(2,1,0), _mk(3,2,1), _mk(4,2,0), _mk(99,3,1)]
    cfg = VAEstimationConfig(reference_date=datetime(2023,6,1), days_since_appt_threshold=0, min_med_cases=2)
    result = estimate_va(cases, config=cfg, start_date="2021-01-01", end_date="2023-01-01")
    preds = {c.case_id: c.p_pred for c in result.case_predictions}
    assert 99 in preds and preds[99] == preds[99]  # present and not NaN
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_va_estimation.py::test_p_pred_present_for_singleton_mediator_case -v`
Expected: FAIL (or passes trivially today — keep as regression guard; the change below aligns the prediction surface with `main`).

- [ ] **Step 3: Implement the df/df_case split**

Mirror `VA_Antoine.py:177,229–247,330–408`:
- Capture `df_case = df.copy()` immediately after the quasiyear block, before the `dropna(['mediator_id'])`/`case_days_med>=0`/pandemic/small-group drops.
- Ensure `df_case` carries `casetype_simplified` (Task 4 helper runs before capture).
- Compute `params_dict` as today from the fitted `df`, then merge param contributions onto `df_case` and:

```python
df_case['p_pred'] = df_case[['appt_month_val', 'quasiyear_val', 'casetype_simplified_val',
    'court_station_val', 'referral_mode_val', 'highcourt_val', 'courtofappeal_val',
    'const_val']].sum(axis=1, skipna=True)
small_mask = df_case['court_station'] == 'zzzSmall'
if small_mask.any():
    cs_small_value = df_case.loc[small_mask, 'court_station_val'].iloc[0]
    df_case['court_station_val'] = df_case['court_station_val'].fillna(cs_small_value)
df_case['residuals'] = df_case['case_outcome_agreement'] - df_case['p_pred']
df['p_pred'] = df['id'].map(df_case.set_index('id')['p_pred'])
df['residuals'] = df['id'].map(df_case.set_index('id')['residuals'])
```

- Apply the pending-outcome coercion (`days_since_appt > threshold & PENDING -> outcome 0`) to BOTH `df` and `df_case`; drop fresh-pending from `df` only.
- Build `case_predictions` from `df_case` (all cases); keep `mediator_vas` from `df` (excludes `-999`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_va_estimation.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/smart_mediator_assignment/algorithm/va_estimation.py tests/test_va_estimation.py
git commit -m "fix(va): predict p_pred/residuals over all cases via df_case, fill small-station coef"
```

---

### Task 6: VA — return a reusable prepared frame + split out `estimate_va_from_prepared`

The dry-run re-estimates VA on a growing window every 7 days. `main` supports this via a pre-cleaned frame (skips re-cleaning; filters by `med_appt_date` AND `concl_date` within the window). Since there's no backward-compat obligation, expose this as a **clean second entry point** rather than an optional `prepared=` param with a sentinel guard. **PR B's VA cutover depends on this** (the adapter shim reuses the prepared frame as its cache).

**Files:**
- Modify: `src/smart_mediator_assignment/algorithm/va_estimation.py` (`VAEstimationResult`; extract a shared core; add `estimate_va_from_prepared`), `src/smart_mediator_assignment/__init__.py` (export)
- Test: `tests/test_va_estimation.py` (add)

**Interfaces:**
- Produces:
  - `estimate_va(cases, config=None, start_date=None, end_date=None) -> VAEstimationResult` (signature unchanged; now always populates `result.prepared`).
  - `estimate_va_from_prepared(prepared, config=None, start_date=None, end_date=None) -> VAEstimationResult` (new; skips cleaning; filters the prepared frame by `med_appt_date` AND `concl_date` within `[start, end)`).
  - `VAEstimationResult.prepared: Optional[pd.DataFrame] = None` (the cleaned frame).
  - Both delegate to a private `_fit_and_score(df, df_case, config) -> VAEstimationResult` holding the shared regression → prediction → shrinkage core. No sentinel/`None`-guard on `estimate_va`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_va_estimation.py  (add)
from datetime import datetime, date
from smart_mediator_assignment import estimate_va, estimate_va_from_prepared, VAEstimationConfig
from smart_mediator_assignment.core.case import SimpleCase

def _mk(cid, appt, outcome):
    return SimpleCase(id=cid, case_type="Family group", court_station="MILIMANI",
        referral_date=date(2021,1,1), p_value=0.5, mediator_id=1, case_outcome_agreement=outcome,
        mediator_appointment_date=appt, conclusion_date=appt, case_status="CONCLUDED",
        court_type="Magistrate", referral_mode="Referred by Court")

def test_estimate_va_returns_prepared_frame():
    cases = [_mk(i, date(2022,1,5+i), i % 2) for i in range(6)]
    cfg = VAEstimationConfig(reference_date=datetime(2023,6,1), days_since_appt_threshold=0)
    result = estimate_va(cases, config=cfg, start_date="2021-01-01", end_date="2023-01-01")
    assert result.prepared is not None

def test_estimate_va_from_prepared_reuses_frame_on_subwindow():
    cases = [_mk(i, date(2022,1,5+i), i % 2) for i in range(6)]
    cfg = VAEstimationConfig(reference_date=datetime(2023,6,1), days_since_appt_threshold=0)
    prepared = estimate_va(cases, config=cfg, start_date="2021-01-01", end_date="2023-01-01").prepared
    second = estimate_va_from_prepared(prepared, config=cfg,
                                       start_date="2022-01-01", end_date="2022-01-08")
    ids = {c.case_id for c in second.case_predictions}
    assert ids and ids.issubset({0, 1, 2})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_va_estimation.py -k "prepared" -v`
Expected: FAIL — `estimate_va_from_prepared` not defined / `prepared` field missing.

- [ ] **Step 3: Implement the split**

- Add `prepared: Optional[pd.DataFrame] = None` to `VAEstimationResult`.
- Extract the regression → prediction → shrinkage core (everything after cleaning) into `_fit_and_score(df, df_case, config) -> VAEstimationResult`, and have it set `result.prepared = df` (the cleaned frame).
- `estimate_va(cases, config, start_date, end_date)`: run `_cases_to_dataframe` + the cleaning pipeline (Tasks 2–5) to produce `df` and `df_case`, then `return _fit_and_score(df, df_case, config)`.
- `estimate_va_from_prepared(prepared, config, start_date, end_date)`: coerce `start_date`/`end_date` (reuse the existing str→datetime helper), then:

```python
df = prepared.loc[
    prepared['med_appt_date'].between(start_date, end_date, inclusive="left")
    & prepared['concl_date'].between(start_date, end_date, inclusive="left")
].copy()
df_case = df.copy()
return _fit_and_score(df, df_case, config)
```

- In `__init__.py`, add `estimate_va_from_prepared` to imports and `__all__`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_va_estimation.py -v`
Expected: PASS.

- [ ] **Step 5: Update the README VA example**

In `README.md`, note that `estimate_va` returns `result.prepared`, and add a one-line example of `estimate_va_from_prepared(result.prepared, ...)` for incremental refresh.

- [ ] **Step 6: Commit**

```bash
git add src/smart_mediator_assignment/algorithm/va_estimation.py src/smart_mediator_assignment/__init__.py tests/test_va_estimation.py README.md
git commit -m "feat(va): return prepared frame + estimate_va_from_prepared for incremental refresh"
```

---

### Task 7: Add an OSQP-based QP solver as an optional target

Port `SlackedQPwithLoadOSQP.py::slackedQP` into the library as `QPSolver(BaseSolver)`, adapted to library types, selectable via config. Declare `osqp` as an optional extra — NOT installed by default.

**Files:**
- Create: `src/smart_mediator_assignment/solver/qp_solver.py`
- Modify: `src/smart_mediator_assignment/config.py` (add `use_qp`), `src/smart_mediator_assignment/assignment/recommender.py` (solver selection), `src/smart_mediator_assignment/__init__.py` (export `QPSolver`), `pyproject.toml` (optional `qp` extra)
- Test: `tests/unit/test_qp_solver.py` (create)

**Interfaces:**
- Consumes: `BaseSolver.solve(cases, phantom_cases, current_day) -> AssignmentDistribution` (from `solver/base.py`); `AlgorithmConfig` fields `capacity`, `lambda_penalty`, `time_horizon`.
- Produces: `QPSolver(capacity, lambda_penalty, time_horizon, ...)` mirroring `LPSolver`'s constructor, `.solve(...)` returning an `AssignmentDistribution` (`{case_id: [(mediator_id, prob), ...]}` sorted desc). `AlgorithmConfig.use_qp: bool = False`. `import smart_mediator_assignment` succeeds without `osqp`; constructing `QPSolver` without `osqp` raises `ImportError` with an actionable message.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_qp_solver.py
import importlib.util
import pytest
from datetime import date
from smart_mediator_assignment import SimpleCase, AlgorithmConfig

osqp_missing = importlib.util.find_spec("osqp") is None

def test_config_defaults_to_lp():
    assert AlgorithmConfig().use_qp is False

@pytest.mark.skipif(osqp_missing, reason="osqp extra not installed")
def test_qp_solver_assigns_single_case():
    from smart_mediator_assignment import QPSolver
    solver = QPSolver(capacity=3, lambda_penalty=1.0, time_horizon=10,
                      valid_mediators=[1, 2],
                      mediator_case_loads={1: 0, 2: 1},
                      med_vas={1: 0.1, 2: -0.05},
                      med_by_court_case_type={"MILIMANI": {"Family group": [1, 2]}})
    case = SimpleCase(id=1, case_type="Family group", court_station="MILIMANI",
                      referral_date=date(2023, 1, 1), p_value=0.5)
    dist = solver.solve([case], phantom_cases=[], current_day=date(2023, 1, 1))
    assert 1 in dist
    total = sum(p for _, p in dist[1])
    assert abs(total - 1.0) < 1e-6
    assert dist[1][0][0] == 1  # higher-VA, lower-load mediator ranked first

@pytest.mark.skipif(not osqp_missing, reason="osqp installed")
def test_qp_solver_import_error_without_osqp():
    from smart_mediator_assignment import QPSolver
    with pytest.raises(ImportError, match="osqp"):
        QPSolver(capacity=3, lambda_penalty=1.0, time_horizon=10, valid_mediators=[1],
                 mediator_case_loads={1: 0}, med_vas={1: 0.0},
                 med_by_court_case_type={"MILIMANI": {"Family group": [1]}})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_qp_solver.py -v`
Expected: FAIL — `QPSolver` not exported / `use_qp` missing.

- [ ] **Step 3: Add the optional `qp` extra**

In `pyproject.toml` under `[project.optional-dependencies]`, add alongside `gurobi`:

```toml
qp = ["osqp>=0.6.3"]
```

Do NOT add `osqp` to `[project.dependencies]`.

- [ ] **Step 4: Add `use_qp` to config**

In `config.py`, add `use_qp: bool = False` to `AlgorithmConfig` (no extra validation needed).

- [ ] **Step 5: Implement `QPSolver`**

Create `solver/qp_solver.py` with `class QPSolver(BaseSolver)`. Mirror `LPSolver.__init__`'s parameters (`capacity`, `lambda_penalty`, `time_horizon`, plus the per-instance `valid_mediators`, `mediator_case_loads`, `med_vas`, `med_by_court_case_type`, optional `phantom` args — match `LPSolver`'s exact constructor for drop-in parity). Lazy-import osqp at construction:

```python
def __init__(self, ...):
    try:
        import osqp  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "QPSolver requires the 'osqp' extra. Install with: pip install "
            "'smart-mediator-assignment[qp]'"
        ) from e
    ...
```

Port the OSQP problem construction from `cadaster-algo/SlackedQPwithLoadOSQP.py`: index building (`_build_indices` :134), big-graph edges (`build_big_graph` :96, `_active_indicator` :123), primal matrices (`buildPrimal` :148–325), `solvePrimal` (:411–493), and `extract_assignment_distribution` (:521–538). Adapt every `Court_Case` access to `CaseProtocol` fields (`case.id`, `case.case_type`, `case.court_station`, `case.referral_date`, `case.p_value`); replace the `_VarView`/`_ModelView` Gurobi-mimics with direct reads of the OSQP result vector; drop the CSV/logging params (`stats_csv_path`, `experiment_tag`, `enable_logging`) — not needed in the library. Return `{case_id: [(mediator_id, prob), ...]}` sorted by prob descending, matching `LPSolver.solve`'s output contract (`solver/base.py`).

- [ ] **Step 6: Wire solver selection + export**

In `recommender.py`, at both solver-construction sites (~lines 99 and 203), branch on config:

```python
if config.use_qp:
    from ..solver.qp_solver import QPSolver
    solver = QPSolver(capacity=config.capacity, lambda_penalty=config.lambda_penalty,
                      time_horizon=config.time_horizon, valid_mediators=eligible_mediator_ids,
                      mediator_case_loads=mediator_case_loads, med_vas=va_estimates,
                      med_by_court_case_type=med_by_court_case_type)
else:
    solver = LPSolver(...)  # unchanged
```

(Use the same argument values already passed to `LPSolver` at those sites.) In `__init__.py`, add `QPSolver` to imports and `__all__`.

- [ ] **Step 7: Run tests to verify they pass**

Run: `uv sync --extra qp && uv run pytest tests/unit/test_qp_solver.py -v`
Expected: PASS (the assign test runs; the import-error test is skipped when osqp is present).

- [ ] **Step 8: Confirm default install has no osqp dependency**

Run: `uv run python -c "import importlib.util as u; import smart_mediator_assignment; print('osqp_installed=', u.find_spec('osqp') is not None)"`
Expected: library imports cleanly regardless; document that `osqp` only arrives via `[qp]`.

- [ ] **Step 9: Commit**

```bash
git add src/smart_mediator_assignment/solver/qp_solver.py src/smart_mediator_assignment/config.py \
        src/smart_mediator_assignment/assignment/recommender.py src/smart_mediator_assignment/__init__.py \
        pyproject.toml tests/unit/test_qp_solver.py
git commit -m "feat(solver): add OSQP-based QPSolver behind optional [qp] extra and config.use_qp"
```

---

### Task 8: Full-suite verification, version bump, installability check

**Files:**
- Modify: `pyproject.toml` (version)

- [ ] **Step 1: Run the entire suite (default + qp extra)**

Run: `uv run pytest -q` then `uv sync --extra qp && uv run pytest -q`
Expected: PASS in both.

- [ ] **Step 2: Confirm the synthetic recovery test still recovers VA**

Run: `uv run pytest tests/test_va_estimation.py -k recover -v`
Expected: PASS (estimated VA correlates with synthesized ground truth).

- [ ] **Step 3: Confirm editable-installability (PR B depends on this)**

Run: `uv run pip install -e . && uv run python -c "import smart_mediator_assignment; print(smart_mediator_assignment.__name__)"`
Expected: prints `smart_mediator_assignment`.

- [ ] **Step 4: Bump version**

In `pyproject.toml`, bump `version = "0.1.0"` → `"0.2.0"` (VA behavior changed + QP added; consumers re-pin).

- [ ] **Step 5: Commit + tag**

```bash
git add pyproject.toml README.md
git commit -m "chore: bump to 0.2.0 (VA/phantom port from cadaster-algo main + optional QP solver)"
git tag v0.2.0
```

---

## Out of scope (handled in PR B / later)

- Re-pointing `cadaster-algo` at the library and deleting duplicates (`VA_Antoine.py`, `Court_case.py`, `SlakedLPwithLoad.py`, the `real_avg_var_sample` belief method) — see `cadaster-algo/docs/.../2026-08-15-cadaster-algo-retire-duplicates.md` (PR B).
- Phase 1 (prod dry-run) and Phase 2 (production integration) — separate plans (spec §9).

## Self-review notes
- Spec coverage: implements spec §4 Phase 0 PR-A (VA filter, quasiyear, Commercial/tax, df/df_case, prepared frame; phantom RNG; QP path as optional target). `legacy_case_type` intentionally dropped (naming decision in Global Constraints).
- No placeholders: each task has runnable tests and concrete edits. The QP matrix port references exact source line ranges in `SlackedQPwithLoadOSQP.py` and pins the input/output contract via `BaseSolver` + tests (a transcription-from-named-source, not a vague "implement QP").
- Type consistency: `estimate_va` gains only optional params; `QPSolver` mirrors `LPSolver`'s constructor and `BaseSolver.solve` contract; `use_qp` added to `AlgorithmConfig` and read in `recommender.py`; `_simplify_case_types` and `VAEstimationResult.prepared` referenced consistently.
