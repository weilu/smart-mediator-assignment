# Smart Mediator Assignment

An LP-based algorithm for optimizing mediator assignments in court mediation systems.

## Installation

Using uv (recommended):
```bash
uv sync
source .venv/bin/activate
```

Optional variants:

```bash
uv sync --dev                  # development dependencies
uv sync --extra gurobi         # Gurobi support
uv sync --dev --extra gurobi   # dev + Gurobi
```

Using pip (alternative):
```bash
pip install -e .
pip install -e ".[dev]"  # for development
pip install -e ".[gurobi]"  # for Gurobi support
```

## Usage

### Basic Usage

```python
from datetime import date
from smart_mediator_assignment import (
    SimpleCase,
    BeliefState,
    AlgorithmConfig,
    get_recommendations,
)

# Create a case
case = SimpleCase(
    id=1,
    case_type_id="Family group",
    court_station_id="KAKAMEGA",
    referral_date=date(2023, 1, 1),
    p_value=0.5,
)

# Set up belief state
belief_state = BeliefState.from_init_va(
    {1: {"mu": 0.0, "sd": 0.12}, 2: {"mu": 0.05, "sd": 0.12}},
    global_sigma=0.12,
)

# Configure algorithm
config = AlgorithmConfig(capacity=3, lambda_penalty=1.0, time_horizon=10)

# Get recommendations
result = get_recommendations(
    case=case,
    eligible_mediator_ids=[1, 2],
    mediator_case_loads={1: 0, 2: 1},
    belief_state=belief_state,
    med_by_court_case_type={"KAKAMEGA": {"Family group": [1, 2]}},
    config=config,
    current_day=date(2023, 1, 1),
)

top_mediator = result.get_top_mediator()
```

### Belief Updates

```python
from smart_mediator_assignment import update_belief

# After case resolution
new_belief_state = update_belief(
    belief_state=belief_state,
    mediator_id=1,
    case_p_val=0.5,
    outcome=1,  # 1 for success, 0 for failure
)
```

### Batch Recommendations

```python
from smart_mediator_assignment import get_recommendations_batch

results = get_recommendations_batch(
    cases=cases_list,
    eligible_mediator_ids=[1, 2, 3],
    mediator_case_loads={1: 0, 2: 0, 3: 0},
    belief_state=belief_state,
    med_by_court_case_type=med_mapping,
    config=config,
    avg_case_rate=avg_case_rate,
    avg_p_val_by_crt_case_type=avg_p_val,
    court_stations=["MILIMANI", "KAKAMEGA"],
    case_types=["Family group"],
    generate_phantoms=True,
)
```

## Testing

```bash
pytest tests/
```

## License

MIT
