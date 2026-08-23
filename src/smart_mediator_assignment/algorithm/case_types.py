"""Canonical case-type taxonomy shared across the algorithm.

Both VA estimation and duration estimation collapse the raw court case types into the same
simplified groupings. Historically each kept its own copy of that mapping, which is exactly
how they can silently drift apart. This module is the single source of truth.

The only per-consumer difference is the label for the family grouping: VA uses
``AAAFamily group`` so it sorts first and becomes the omitted reference category in the
fixed-effects regression, while the simulation/duration code uses the plain ``Family group``.
That is expressed via the ``family_group_label`` parameter, not a second mapping.
"""
import pandas as pd

# Raw DB case_type -> simplified grouping (simulation-facing "Family group" label). Callers
# that need a lookup table (e.g. the RCT sim's per-case eligibility mapping) use this dict;
# callers transforming a column use simplify_case_types().
LEGACY_2_SIMPLIFIED_CTYPE = {
    "Matrimonial Property Cases": "Matrimonial Property Cases",
    "Criminal Cases": "Criminal Cases",
    "Commercial Cases": "Commercial and tax group",
    "Employment and Labour Relations Cases (ELRC)": "Employment and Labour Relations Cases (ELRC)",
    "Family Appeals": "Family group",
    "Civil Cases": "Civil group",
    "Children Custody and Maintenance": "Children Custody and Maintenance",
    "Civil Appeals": "Civil group",
    "Divorce and Separation": "Family group",
    "Succession (Probate & Administration - P&A)": "Family group",
    "Family Miscellaneous": "Family group",
    "Environment and Land Cases (ELC)": "Environment and Land Cases (ELC)",
    "Judicial Review": "Judicial Review",
    "Constitution and Human Rights": "Constitution and Human Rights",
    "Tax Appeals": "Commercial and tax group",
}

# The 10 simplified case types (simulation-facing names).
CASE_TYPE_NAMES = [
    "Family group",
    "Children Custody and Maintenance",
    "Civil group",
    "Commercial and tax group",
    "Constitution and Human Rights",
    "Criminal Cases",
    "Employment and Labour Relations Cases (ELRC)",
    "Environment and Land Cases (ELC)",
    "Judicial Review",
    "Matrimonial Property Cases",
]


def simplify_case_types(case_type: pd.Series, family_group_label: str = "Family group") -> pd.Series:
    """Map raw case types to their simplified grouping.

    Args:
        case_type: Series of raw case_type strings.
        family_group_label: label for the Divorce/Family/Succession grouping. Defaults to
            "Family group"; pass "AAAFamily group" for the VA regression reference category.

    Returns:
        Series (same index) of simplified case-type strings; unknown types pass through.
    """
    simplified = case_type.map(lambda c: LEGACY_2_SIMPLIFIED_CTYPE.get(c, c))
    if family_group_label != "Family group":
        simplified = simplified.replace("Family group", family_group_label)
    return simplified
