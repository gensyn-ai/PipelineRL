"""Rollout mixing strategies for cross-actor GRPO training."""

import random
from abc import ABC, abstractmethod
from typing import List


class RolloutMixingStrategy(ABC):
    """Base class for cross-actor rollout mixing strategies."""
    
    @abstractmethod
    def select_rollouts(
        self,
        primary_rollouts: List[dict],
        other_rollouts: List[dict],
        n_primary: int,
        n_other: int,
    ) -> List[dict]:
        """
        Select and mix rollouts from primary and other actors.
        
        Args:
            primary_rollouts: Rollouts from primary actor for this group
            other_rollouts: Rollouts from other actors for this group (combined)
            n_primary: Number to select from primary actor
            n_other: Number to select from other actors
            
        Returns:
            Mixed list of exactly (n_primary + n_other) rollouts with reassigned rollout_index
            
        Raises:
            ValueError: If insufficient rollouts available
        """
        pass


class RandomMixingStrategy(RolloutMixingStrategy):
    """Randomly sample rollouts from primary and other actors."""
    
    def select_rollouts(
        self,
        primary_rollouts: List[dict],
        other_rollouts: List[dict],
        n_primary: int,
        n_other: int,
    ) -> List[dict]:
        if len(primary_rollouts) < n_primary:
            raise ValueError(
                f"Not enough primary rollouts: {len(primary_rollouts)} < {n_primary}"
            )
        if len(other_rollouts) < n_other:
            raise ValueError(
                f"Not enough other rollouts: {len(other_rollouts)} < {n_other}"
            )
        
        # Randomly sample from each source
        selected_primary = random.sample(primary_rollouts, n_primary)
        selected_other = random.sample(other_rollouts, n_other)
        
        # Combine and shuffle to avoid bias
        mixed = selected_primary + selected_other
        random.shuffle(mixed)
        
        # Reassign rollout_index to 0-(n_primary+n_other-1)
        for idx, rollout in enumerate(mixed):
            rollout['rollout_index'] = idx
        
        return mixed


# Registry for mixing strategies
MIXING_STRATEGIES = {
    "random": RandomMixingStrategy,
}


def get_mixing_strategy(strategy_name: str) -> RolloutMixingStrategy:
    """Get a mixing strategy instance by name."""
    if strategy_name not in MIXING_STRATEGIES:
        raise ValueError(
            f"Unknown mixing strategy: {strategy_name}. "
            f"Available strategies: {list(MIXING_STRATEGIES.keys())}"
        )
    return MIXING_STRATEGIES[strategy_name]()
