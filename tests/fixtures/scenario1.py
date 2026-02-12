"""
Scenario 1: Simple test case with 3 mediators.

Mediator 1 can handle cases in both MILIMANI and KAKAMEGA.
Mediators 2 and 3 can only handle cases in KAKAMEGA.
"""

SCENARIO1_AVG_CASE_RATE = {
    "Family group": {
        "MILIMANI": 0.055,
        "KAKAMEGA": 0.05,
    }
}

SCENARIO1_AVG_P_VAL = {
    ("Family group", "MILIMANI"): 0.5,
    ("Family group", "KAKAMEGA"): 0.5,
}

SCENARIO1_VALID_MEDS = [1, 2, 3]

SCENARIO1_MED_BY_CRT_CASE_TYPE = {
    "MILIMANI": {
        "Family group": [1],
    },
    "KAKAMEGA": {
        "Family group": [1, 2, 3],
    },
}

SCENARIO1_CRT_CASE_TYPE_BY_MED = {
    1: [("Family group", "MILIMANI"), ("Family group", "KAKAMEGA")],
    2: [("Family group", "KAKAMEGA")],
    3: [("Family group", "KAKAMEGA")],
}

SCENARIO1_MED_VA = {
    1: 0.1,
    2: 0.05,
    3: -0.1,
}
