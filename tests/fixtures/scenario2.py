"""
Scenario 2: Test case with 3 mediators with different coverage.

Mediators 1 and 2 can handle cases in MILIMANI.
Mediators 1 and 3 can handle cases in KAKAMEGA.
"""

SCENARIO2_AVG_CASE_RATE = {
    "Family group": {
        "MILIMANI": 0.055,
        "KAKAMEGA": 0.05,
    }
}

SCENARIO2_AVG_P_VAL = {
    ("Family group", "MILIMANI"): 0.5,
    ("Family group", "KAKAMEGA"): 0.5,
}

SCENARIO2_VALID_MEDS = [1, 2, 3]

SCENARIO2_MED_BY_CRT_CASE_TYPE = {
    "MILIMANI": {
        "Family group": [1, 2],
    },
    "KAKAMEGA": {
        "Family group": [1, 3],
    },
}

SCENARIO2_CRT_CASE_TYPE_BY_MED = {
    1: [("Family group", "MILIMANI"), ("Family group", "KAKAMEGA")],
    2: [("Family group", "MILIMANI")],
    3: [("Family group", "KAKAMEGA")],
}

SCENARIO2_MED_VA = {
    1: 0.1,
    2: 0.05,
    3: -0.1,
}
