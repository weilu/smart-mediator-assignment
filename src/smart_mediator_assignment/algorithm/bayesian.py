"""
Bayesian belief update for mediator value-added estimates.

This module implements the posterior computation and belief updates
used in the mediator assignment algorithm.
"""

import math
from typing import Tuple, List

import numpy as np
from scipy.integrate import quad

from ..core.belief import MediatorBelief, BeliefState
from ..core.types import MediatorId, CaseHistory


def _log_likelihood(
    mu: float,
    case_history: CaseHistory,
    prior_mu: float,
    prior_sigma: float,
) -> float:
    """
    Compute the (un-normalized) log posterior density at mu.

    Args:
        mu: The VA value to evaluate
        case_history: Tuple of (failed_p_vals, success_p_vals)
        prior_mu: Prior mean
        prior_sigma: Prior standard deviation

    Returns:
        Log of un-normalized posterior density
    """
    failed_p_vals, success_p_vals = case_history

    log_likelihood = np.longdouble(0)

    for p_val in success_p_vals:
        prob = np.clip(p_val + mu, 0.00001, 2)
        log_likelihood += np.log(prob)

    for p_val in failed_p_vals:
        prob = np.clip(1 - p_val - mu, 0.00001, 2)
        log_likelihood += np.log(prob)

    log_prior = -0.5 * ((mu - prior_mu) / prior_sigma) ** 2
    return log_likelihood + log_prior


def compute_posterior(
    case_history: CaseHistory,
    prior_mu: float,
    prior_sigma: float,
    prev_mu: float = None,
    prev_sigma: float = None,
) -> Tuple[float, float]:
    """
    Compute the posterior mean and standard deviation for a mediator's VA.

    Uses numerical integration to compute the posterior distribution given
    the case history and prior.

    Args:
        case_history: Tuple of (failed_p_vals, success_p_vals)
        prior_mu: Prior mean (usually 0 or from external VA estimate)
        prior_sigma: Prior/global standard deviation
        prev_mu: Previous posterior mean (for fallback)
        prev_sigma: Previous posterior sigma (for fallback)

    Returns:
        Tuple of (posterior_mean, posterior_sd)
    """
    min_sigma = 1e-6

    if prev_sigma is None:
        prev_sigma = prior_sigma
        prev_mu = prior_mu

    if prior_sigma is None or prior_sigma <= 0 or not np.isfinite(prior_sigma):
        prior_sigma = max(prev_sigma, min_sigma)

    integration_points = (
        prior_mu - 6 * prior_sigma,
        prior_mu - 5 * prior_sigma,
        prior_mu - 4 * prior_sigma,
        prior_mu - 3 * prior_sigma,
        prior_mu - 2 * prior_sigma,
        prior_mu - prior_sigma,
        prior_mu + prior_sigma,
        prior_mu + 2 * prior_sigma,
        prior_mu + 3 * prior_sigma,
        prior_mu + 4 * prior_sigma,
        prior_mu + 5 * prior_sigma,
    )

    scale = 1_000_000

    def pdf(x: float) -> float:
        return np.exp(_log_likelihood(x, case_history, prior_mu, prior_sigma)) * scale

    try:
        normalization, _ = quad(pdf, -1, 1, limit=100, points=integration_points)

        if normalization <= 0 or not np.isfinite(normalization):
            return prev_mu, max(prev_sigma, min_sigma)

        mean, _ = quad(
            lambda x: x * pdf(x), -1, 1, limit=100, points=integration_points
        )
        mean /= normalization

        mean_square, _ = quad(
            lambda x: x**2 * pdf(x), -1, 1, limit=100, points=integration_points
        )
        mean_square /= normalization

        variance = mean_square - mean**2
        if variance < 0:
            variance = 0.0

        sd = np.sqrt(variance)

        if math.isnan(mean) or math.isnan(sd):
            return prior_mu, prior_sigma

        return mean, sd

    except Exception:
        return prev_mu if prev_mu is not None else prior_mu, max(prev_sigma, min_sigma)


def update_belief(
    belief_state: BeliefState,
    mediator_id: MediatorId,
    case_p_val: float,
    outcome: int,
    prior_mu: float = 0.0,
) -> BeliefState:
    """
    Update the belief state after observing a case outcome.

    Args:
        belief_state: Current belief state
        mediator_id: ID of the mediator who handled the case
        case_p_val: P-value of the case
        outcome: 1 for success (agreement), 0 for failure
        prior_mu: Prior mean for the update

    Returns:
        New BeliefState with updated belief for the mediator
    """
    new_state = belief_state.copy()
    belief = new_state.beliefs[mediator_id]

    new_history = (
        list(belief.case_history[0]),
        list(belief.case_history[1]),
    )
    new_history[outcome].append(case_p_val)

    new_mu, new_sigma = compute_posterior(
        case_history=new_history,
        prior_mu=prior_mu,
        prior_sigma=belief_state.global_sigma,
        prev_mu=belief.mu,
        prev_sigma=belief.sigma,
    )

    new_state.beliefs[mediator_id] = MediatorBelief(
        mediator_id=mediator_id,
        mu=new_mu,
        sigma=new_sigma,
        case_history=tuple(new_history),
    )

    return new_state


def update_belief_batch(
    belief_state: BeliefState,
    mediator_id: MediatorId,
    case_outcomes: List[Tuple[float, int]],
    prior_mu: float = 0.0,
) -> BeliefState:
    """
    Update belief state with multiple case outcomes at once.

    Args:
        belief_state: Current belief state
        mediator_id: ID of the mediator
        case_outcomes: List of (p_val, outcome) tuples
        prior_mu: Prior mean for the update

    Returns:
        New BeliefState with updated belief
    """
    new_state = belief_state.copy()
    belief = new_state.beliefs[mediator_id]

    new_history = (
        list(belief.case_history[0]),
        list(belief.case_history[1]),
    )

    for p_val, outcome in case_outcomes:
        new_history[outcome].append(p_val)

    new_mu, new_sigma = compute_posterior(
        case_history=new_history,
        prior_mu=prior_mu,
        prior_sigma=belief_state.global_sigma,
        prev_mu=belief.mu,
        prev_sigma=belief.sigma,
    )

    new_state.beliefs[mediator_id] = MediatorBelief(
        mediator_id=mediator_id,
        mu=new_mu,
        sigma=new_sigma,
        case_history=tuple(new_history),
    )

    return new_state
