import pytest
import numpy as np

from smart_mediator_assignment.core.belief import MediatorBelief, BeliefState
from smart_mediator_assignment.algorithm.bayesian import (
    compute_posterior,
    update_belief,
    update_belief_batch,
)


class TestComputePosterior:
    """Tests for posterior computation."""

    def test_empty_history_returns_prior(self):
        """With no cases, posterior should be close to prior."""
        empty_history = ([], [])
        mu, sigma = compute_posterior(
            case_history=empty_history,
            prior_mu=0.0,
            prior_sigma=0.12,
        )

        assert abs(mu - 0.0) < 0.01
        assert abs(sigma - 0.12) < 0.01

    def test_successful_cases_increase_mu(self):
        """Cases with positive outcomes should increase the VA estimate."""
        history = ([], [0.5, 0.5, 0.5])
        mu, sigma = compute_posterior(
            case_history=history,
            prior_mu=0.0,
            prior_sigma=0.12,
        )

        assert mu > 0.0

    def test_failed_cases_decrease_mu(self):
        """Cases with negative outcomes should decrease the VA estimate."""
        history = ([0.5, 0.5, 0.5], [])
        mu, sigma = compute_posterior(
            case_history=history,
            prior_mu=0.0,
            prior_sigma=0.12,
        )

        assert mu < 0.0

    def test_more_evidence_reduces_sigma(self):
        """More case history should reduce uncertainty (sigma)."""
        single_case_history = ([], [0.5])
        mu1, sigma1 = compute_posterior(
            case_history=single_case_history,
            prior_mu=0.0,
            prior_sigma=0.12,
        )

        many_case_history = ([], [0.5] * 10)
        mu2, sigma2 = compute_posterior(
            case_history=many_case_history,
            prior_mu=0.0,
            prior_sigma=0.12,
        )

        assert sigma2 < sigma1

    def test_mixed_outcomes(self):
        """Mixed outcomes should result in moderate VA estimate."""
        history = ([0.5, 0.5], [0.5, 0.5])
        mu, sigma = compute_posterior(
            case_history=history,
            prior_mu=0.0,
            prior_sigma=0.12,
        )

        assert abs(mu) < 0.1

    def test_handles_extreme_p_values(self):
        """Should handle edge cases with extreme p-values."""
        history = ([], [0.1, 0.9])
        mu, sigma = compute_posterior(
            case_history=history,
            prior_mu=0.0,
            prior_sigma=0.12,
        )

        assert not np.isnan(mu)
        assert not np.isnan(sigma)
        assert sigma > 0


class TestUpdateBelief:
    """Tests for belief state updates."""

    def test_update_single_mediator(self):
        """Test updating belief for a single mediator."""
        initial_beliefs = {
            1: MediatorBelief(mediator_id=1, mu=0.0, sigma=0.12, case_history=([], [])),
            2: MediatorBelief(mediator_id=2, mu=0.0, sigma=0.12, case_history=([], [])),
        }
        belief_state = BeliefState(beliefs=initial_beliefs, global_sigma=0.12)

        new_state = update_belief(
            belief_state=belief_state,
            mediator_id=1,
            case_p_val=0.5,
            outcome=1,
        )

        assert new_state.beliefs[1].case_history[1] == [0.5]
        assert new_state.beliefs[2].case_history == ([], [])
        assert new_state.beliefs[1].mu > belief_state.beliefs[1].mu

    def test_update_preserves_other_mediators(self):
        """Updating one mediator should not affect others."""
        initial_beliefs = {
            1: MediatorBelief(mediator_id=1, mu=0.05, sigma=0.10, case_history=([0.3], [0.7])),
            2: MediatorBelief(mediator_id=2, mu=-0.02, sigma=0.11, case_history=([], [0.5])),
        }
        belief_state = BeliefState(beliefs=initial_beliefs, global_sigma=0.12)

        new_state = update_belief(
            belief_state=belief_state,
            mediator_id=1,
            case_p_val=0.6,
            outcome=0,
        )

        assert new_state.beliefs[2].mu == -0.02
        assert new_state.beliefs[2].sigma == 0.11
        assert new_state.beliefs[2].case_history == ([], [0.5])

    def test_immutability(self):
        """Update should not modify the original belief state."""
        initial_beliefs = {
            1: MediatorBelief(mediator_id=1, mu=0.0, sigma=0.12, case_history=([], [])),
        }
        belief_state = BeliefState(beliefs=initial_beliefs, global_sigma=0.12)
        original_mu = belief_state.beliefs[1].mu

        new_state = update_belief(
            belief_state=belief_state,
            mediator_id=1,
            case_p_val=0.5,
            outcome=1,
        )

        assert belief_state.beliefs[1].mu == original_mu
        assert belief_state.beliefs[1].case_history == ([], [])


class TestUpdateBeliefBatch:
    """Tests for batch belief updates."""

    def test_batch_update(self):
        """Test updating with multiple outcomes at once."""
        initial_beliefs = {
            1: MediatorBelief(mediator_id=1, mu=0.0, sigma=0.12, case_history=([], [])),
        }
        belief_state = BeliefState(beliefs=initial_beliefs, global_sigma=0.12)

        outcomes = [(0.5, 1), (0.6, 1), (0.4, 0)]
        new_state = update_belief_batch(
            belief_state=belief_state,
            mediator_id=1,
            case_outcomes=outcomes,
        )

        assert len(new_state.beliefs[1].case_history[0]) == 1
        assert len(new_state.beliefs[1].case_history[1]) == 2

    def test_batch_equals_sequential(self):
        """Batch update should produce same result as sequential updates."""
        initial_beliefs = {
            1: MediatorBelief(mediator_id=1, mu=0.0, sigma=0.12, case_history=([], [])),
        }
        belief_state = BeliefState(beliefs=initial_beliefs, global_sigma=0.12)

        outcomes = [(0.5, 1), (0.6, 0)]

        batch_state = update_belief_batch(
            belief_state=belief_state,
            mediator_id=1,
            case_outcomes=outcomes,
        )

        sequential_state = belief_state
        for p_val, outcome in outcomes:
            sequential_state = update_belief(
                belief_state=sequential_state,
                mediator_id=1,
                case_p_val=p_val,
                outcome=outcome,
            )

        assert abs(batch_state.beliefs[1].mu - sequential_state.beliefs[1].mu) < 0.001
        assert (
            abs(batch_state.beliefs[1].sigma - sequential_state.beliefs[1].sigma) < 0.001
        )


class TestBeliefState:
    """Tests for BeliefState class."""

    def test_from_init_va(self):
        """Test creating BeliefState from init_va format."""
        med_init_va = {
            1: {"mu": 0.05, "sd": 0.10},
            2: {"mu": -0.02, "sd": 0.12},
        }

        state = BeliefState.from_init_va(med_init_va, global_sigma=0.12)

        assert state.beliefs[1].mu == 0.05
        assert state.beliefs[1].sigma == 0.10
        assert state.beliefs[2].mu == -0.02
        assert state.beliefs[2].sigma == 0.12

    def test_get_va_estimates_mean(self):
        """Test getting VA estimates using mean strategy."""
        beliefs = {
            1: MediatorBelief(mediator_id=1, mu=0.1, sigma=0.12, case_history=([], [])),
            2: MediatorBelief(mediator_id=2, mu=-0.05, sigma=0.12, case_history=([], [])),
        }
        state = BeliefState(beliefs=beliefs, global_sigma=0.12)

        estimates = state.get_va_estimates(strategy="mean")

        assert estimates[1] == 0.1
        assert estimates[2] == -0.05

    def test_get_va_estimates_sample(self):
        """Test getting VA estimates using sample strategy."""
        np.random.seed(42)
        beliefs = {
            1: MediatorBelief(mediator_id=1, mu=0.1, sigma=0.12, case_history=([], [])),
        }
        state = BeliefState(beliefs=beliefs, global_sigma=0.12)

        samples = [state.get_va_estimates(strategy="sample")[1] for _ in range(100)]
        mean_sample = np.mean(samples)

        assert abs(mean_sample - 0.1) < 0.05

    def test_get_va_estimates_invalid_strategy(self):
        """Test that invalid strategy raises error."""
        beliefs = {
            1: MediatorBelief(mediator_id=1, mu=0.1, sigma=0.12, case_history=([], [])),
        }
        state = BeliefState(beliefs=beliefs, global_sigma=0.12)

        with pytest.raises(ValueError):
            state.get_va_estimates(strategy="invalid")

    def test_copy(self):
        """Test that copy creates independent state."""
        beliefs = {
            1: MediatorBelief(mediator_id=1, mu=0.1, sigma=0.12, case_history=([], [0.5])),
        }
        original = BeliefState(beliefs=beliefs, global_sigma=0.12)

        copied = original.copy()
        copied.beliefs[1].case_history[1].append(0.6)

        assert len(original.beliefs[1].case_history[1]) == 1
        assert len(copied.beliefs[1].case_history[1]) == 2
