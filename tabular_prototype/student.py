"""Student policy hierarchy and trajectory collection."""

import numpy as np
from abc import ABC, abstractmethod
from typing import List, Optional
from dataclasses import dataclass

from .environment import GridEnv


class NPGPolicy(ABC):
    """Abstract base class for policies compatible with Natural Policy Gradient.

    Defines the minimal interface any policy parameterization must expose
    for NPG-based training loops (vanilla PG, NPG, PAV-RL, etc.).

    Concrete subclasses:
        TabularSoftmaxPolicy  – explicit theta[s, a] table for tabular MDPs
        (future) NeuralSoftmaxPolicy – neural-network parameterization
    """

    def __init__(self, n_actions: int):
        self.n_actions = n_actions

    @abstractmethod
    def get_probs(self, state) -> np.ndarray:
        """Return pi(.|state) as a probability vector of length n_actions."""
        ...

    def sample_action(self, state, rng: np.random.Generator) -> int:
        """Sample an action from pi(.|state)."""
        probs = self.get_probs(state)
        return rng.choice(self.n_actions, p=probs)

    def log_prob(self, state, action: int) -> float:
        """Compute log pi(action|state)."""
        probs = self.get_probs(state)
        return np.log(probs[action] + 1e-10)

    @abstractmethod
    def get_parameters(self) -> np.ndarray:
        """Return a flat copy of all trainable parameters."""
        ...

    @abstractmethod
    def set_parameters(self, params: np.ndarray) -> None:
        """Overwrite trainable parameters from a flat array."""
        ...

    @abstractmethod
    def compute_score(self, state, action: int) -> np.ndarray:
        """Compute the score function nabla_theta log pi(action|state) as a flat vector.

        This is the fundamental quantity for all policy-gradient methods.
        For NPG the natural gradient is F^{-1} * score, where F is the
        Fisher information matrix.
        """
        ...


class TabularSoftmaxPolicy(NPGPolicy):
    """Tabular softmax policy: pi(a|s) = softmax(theta[s, a]).

    Parameters are stored as a (n_states, n_actions) matrix theta.
    """

    def __init__(self, n_states: int, n_actions: int):
        super().__init__(n_actions)
        self.n_states = n_states
        self.theta = np.zeros((n_states, n_actions))

    def get_probs(self, state_idx: int) -> np.ndarray:
        logits = self.theta[state_idx]
        logits = logits - logits.max()
        exp_logits = np.exp(logits)
        return exp_logits / exp_logits.sum()

    def sample_action(self, state_idx: int, rng: np.random.Generator) -> int:
        probs = self.get_probs(state_idx)
        return rng.choice(self.n_actions, p=probs)

    def log_prob(self, state_idx: int, action: int) -> float:
        probs = self.get_probs(state_idx)
        return np.log(probs[action] + 1e-10)

    def get_parameters(self) -> np.ndarray:
        return self.theta.ravel().copy()

    def set_parameters(self, params: np.ndarray) -> None:
        self.theta = params.reshape(self.n_states, self.n_actions)

    def compute_score(self, state_idx: int, action: int) -> np.ndarray:
        """Score for tabular softmax: d log pi(a|s) / d theta[s', a'].

        Non-zero only for s' == state_idx:
            theta[s, a]  -> 1 - pi(a|s)    (the taken action)
            theta[s, a'] -> -pi(a'|s)      (all other actions)
        """
        score = np.zeros_like(self.theta)
        probs = self.get_probs(state_idx)
        score[state_idx] = -probs
        score[state_idx, action] += 1.0
        return score.ravel()


# Backward-compatible alias
TabularPolicy = TabularSoftmaxPolicy


@dataclass
class Transition:
    state_idx: int
    action: int
    reward: float
    next_state_idx: int
    log_prob: float
    timestep: int = 0
    done: bool = False


def collect_trajectory(
    env: GridEnv, policy: TabularSoftmaxPolicy, rng: np.random.Generator
) -> List[Transition]:
    """Collect a single trajectory with timestep tracking and early termination."""
    trajectory = []
    state = env.reset()

    for t in range(env.horizon):
        state_idx = env.state_to_idx(state)
        action = policy.sample_action(state_idx, rng)
        log_prob = policy.log_prob(state_idx, action)

        step_result = env.step(state, action)
        if len(step_result) == 3:
            next_state, reward, done = step_result
        else:
            next_state, reward = step_result
            done = False

        next_state_idx = env.state_to_idx(next_state)

        trajectory.append(Transition(
            state_idx=state_idx,
            action=action,
            reward=reward,
            next_state_idx=next_state_idx,
            log_prob=log_prob,
            timestep=t,
            done=done
        ))

        if done:
            break

        state = next_state

    return trajectory


def collect_trajectories(
    env: GridEnv, policy: TabularSoftmaxPolicy,
    n_trajectories: int, rng: np.random.Generator
) -> List[List[Transition]]:
    """Collect multiple trajectories."""
    return [collect_trajectory(env, policy, rng) for _ in range(n_trajectories)]
