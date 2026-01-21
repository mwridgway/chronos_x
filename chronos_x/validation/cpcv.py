"""Combinatorial Purged Cross-Validation (CPCV).

Implements cross-validation methods from "Advances in Financial Machine Learning"
that properly handle overlapping labels and temporal dependencies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations
from typing import TYPE_CHECKING, Iterator

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class CPCVConfig:
    """Configuration for Combinatorial Purged Cross-Validation."""

    n_splits: int = 5  # Number of CV splits
    n_test_splits: int = 2  # Number of test groups per fold
    purge_gap: int = 10  # Number of samples to purge between train/test
    embargo_pct: float = 0.01  # Percentage of samples to embargo after test


class PurgedKFold:
    """K-Fold cross-validation with purging and embargo.

    Purging removes training samples that overlap with test samples.
    Embargo adds a gap after each test set to prevent information leakage.
    """

    def __init__(
        self,
        n_splits: int = 5,
        purge_gap: int = 10,
        embargo_pct: float = 0.01,
    ) -> None:
        self.n_splits = n_splits
        self.purge_gap = purge_gap
        self.embargo_pct = embargo_pct

    def split(
        self,
        X: NDArray,
        y: NDArray | None = None,
        groups: NDArray | None = None,
        t1: NDArray | None = None,
    ) -> Iterator[tuple[NDArray[np.int64], NDArray[np.int64]]]:
        """Generate train/test indices with purging and embargo.

        Args:
            X: Feature matrix
            y: Labels (optional)
            groups: Group labels (optional)
            t1: Array of prediction end times (for purging)

        Yields:
            Tuple of (train_indices, test_indices)
        """
        n_samples = len(X)
        indices = np.arange(n_samples)

        # Calculate embargo size
        embargo_size = int(n_samples * self.embargo_pct)

        # Generate fold boundaries
        fold_size = n_samples // self.n_splits
        fold_starts = [i * fold_size for i in range(self.n_splits)]
        fold_starts.append(n_samples)

        for fold_idx in range(self.n_splits):
            test_start = fold_starts[fold_idx]
            test_end = fold_starts[fold_idx + 1]
            test_indices = indices[test_start:test_end]

            # Initial train indices (all except test)
            train_mask = np.ones(n_samples, dtype=bool)
            train_mask[test_start:test_end] = False

            # Apply purging before test set
            purge_start = max(0, test_start - self.purge_gap)
            train_mask[purge_start:test_start] = False

            # Apply embargo after test set
            embargo_end = min(n_samples, test_end + embargo_size)
            train_mask[test_end:embargo_end] = False

            # Additional purging based on t1 (prediction end times)
            if t1 is not None:
                # Find training samples whose predictions overlap with test period
                test_start_time = t1[test_start] if test_start < len(t1) else t1[-1]
                for train_idx in np.where(train_mask)[0]:
                    if train_idx < len(t1) and t1[train_idx] >= test_start_time:
                        train_mask[train_idx] = False

            train_indices = indices[train_mask]

            yield train_indices, test_indices


class CombinatorialPurgedKFold:
    """Combinatorial Purged K-Fold Cross-Validation.

    Generates all combinations of test folds, providing more
    independent test sets for robust performance estimation.
    """

    def __init__(
        self,
        n_splits: int = 5,
        n_test_splits: int = 2,
        purge_gap: int = 10,
        embargo_pct: float = 0.01,
    ) -> None:
        self.n_splits = n_splits
        self.n_test_splits = n_test_splits
        self.purge_gap = purge_gap
        self.embargo_pct = embargo_pct

        # Calculate number of combinations
        self.n_folds = int(
            math.factorial(n_splits)
            / (
                math.factorial(n_test_splits)
                * math.factorial(n_splits - n_test_splits)
            )
        )

    def split(
        self,
        X: NDArray,
        y: NDArray | None = None,
        groups: NDArray | None = None,
        t1: NDArray | None = None,
    ) -> Iterator[tuple[NDArray[np.int64], NDArray[np.int64]]]:
        """Generate combinatorial train/test splits.

        Args:
            X: Feature matrix
            y: Labels (optional)
            groups: Group labels (optional)
            t1: Array of prediction end times (for purging)

        Yields:
            Tuple of (train_indices, test_indices)
        """
        n_samples = len(X)
        indices = np.arange(n_samples)

        # Calculate sizes
        embargo_size = int(n_samples * self.embargo_pct)
        fold_size = n_samples // self.n_splits

        # Create fold boundaries
        fold_bounds = [(i * fold_size, (i + 1) * fold_size) for i in range(self.n_splits)]
        fold_bounds[-1] = (fold_bounds[-1][0], n_samples)  # Adjust last fold

        # Generate all test combinations
        test_combinations = list(combinations(range(self.n_splits), self.n_test_splits))

        for test_fold_indices in test_combinations:
            # Build test set from selected folds
            test_mask = np.zeros(n_samples, dtype=bool)
            for fold_idx in test_fold_indices:
                start, end = fold_bounds[fold_idx]
                test_mask[start:end] = True

            test_indices = indices[test_mask]

            # Build train set with purging and embargo
            train_mask = np.ones(n_samples, dtype=bool)

            for fold_idx in test_fold_indices:
                start, end = fold_bounds[fold_idx]

                # Remove test samples
                train_mask[start:end] = False

                # Apply purging before test fold
                purge_start = max(0, start - self.purge_gap)
                train_mask[purge_start:start] = False

                # Apply embargo after test fold
                embargo_end = min(n_samples, end + embargo_size)
                train_mask[end:embargo_end] = False

            # Additional purging based on t1
            if t1 is not None:
                test_start_idx = test_indices[0]
                test_start_time = t1[test_start_idx] if test_start_idx < len(t1) else t1[-1]

                for train_idx in np.where(train_mask)[0]:
                    if train_idx < len(t1) and t1[train_idx] >= test_start_time:
                        train_mask[train_idx] = False

            train_indices = indices[train_mask]

            yield train_indices, test_indices

    def get_n_splits(self) -> int:
        """Return number of splits."""
        return self.n_folds


class WalkForwardCV:
    """Walk-forward cross-validation for time series.

    Expanding or sliding window approach that respects temporal ordering.
    """

    def __init__(
        self,
        n_splits: int = 5,
        train_size: int | float | None = None,
        test_size: int | float | None = None,
        gap: int = 0,
        expanding: bool = True,
    ) -> None:
        self.n_splits = n_splits
        self.train_size = train_size
        self.test_size = test_size
        self.gap = gap
        self.expanding = expanding

    def split(
        self,
        X: NDArray,
        y: NDArray | None = None,
        groups: NDArray | None = None,
    ) -> Iterator[tuple[NDArray[np.int64], NDArray[np.int64]]]:
        """Generate walk-forward train/test splits.

        Args:
            X: Feature matrix
            y: Labels (optional)
            groups: Group labels (optional)

        Yields:
            Tuple of (train_indices, test_indices)
        """
        n_samples = len(X)
        indices = np.arange(n_samples)

        # Calculate test size
        if self.test_size is None:
            test_size = n_samples // (self.n_splits + 1)
        elif isinstance(self.test_size, float):
            test_size = int(n_samples * self.test_size)
        else:
            test_size = self.test_size

        # Calculate train size
        if self.train_size is None:
            # Use all available data up to test set
            min_train_size = test_size
        elif isinstance(self.train_size, float):
            min_train_size = int(n_samples * self.train_size)
        else:
            min_train_size = self.train_size

        # Generate splits
        for split_idx in range(self.n_splits):
            test_end = n_samples - (self.n_splits - split_idx - 1) * test_size
            test_start = test_end - test_size

            if self.expanding:
                train_start = 0
            else:
                train_start = max(0, test_start - self.gap - min_train_size)

            train_end = test_start - self.gap

            if train_end <= train_start:
                continue

            train_indices = indices[train_start:train_end]
            test_indices = indices[test_start:test_end]

            yield train_indices, test_indices


def compute_sample_weight(
    t1: NDArray,
    concurrent_threshold: int = 1,
) -> NDArray[np.float64]:
    """Compute sample weights based on label concurrency.

    Samples with more overlapping predictions get lower weights
    to prevent overrepresentation.

    Args:
        t1: Array of prediction end times
        concurrent_threshold: Minimum concurrent count

    Returns:
        Array of sample weights
    """
    n_samples = len(t1)
    weights = np.ones(n_samples)

    # Count concurrent labels for each sample
    concurrent = np.ones(n_samples, dtype=int)

    for i in range(n_samples):
        for j in range(i):
            # Check if sample j's prediction overlaps with sample i
            if t1[j] > i:
                concurrent[i] += 1
                concurrent[j] += 1

    # Weight is inverse of concurrency
    weights = 1.0 / np.maximum(concurrent, concurrent_threshold)

    # Normalize
    weights = weights / weights.sum() * n_samples

    return weights


def generate_cpcv_paths(
    n_splits: int,
    n_test_splits: int,
) -> list[list[int]]:
    """Generate all possible backtest paths through CPCV folds.

    Each path represents a complete walk through all data,
    with each fold appearing in exactly one test set.

    Args:
        n_splits: Number of CV splits
        n_test_splits: Number of test groups per combination

    Returns:
        List of paths, where each path is a list of combination indices
    """
    # Generate all test combinations
    all_combinations = list(combinations(range(n_splits), n_test_splits))

    # Find paths that cover all folds exactly once
    def find_paths(
        used_folds: set,
        current_path: list,
        remaining_combinations: list,
    ) -> list[list[int]]:
        if len(used_folds) == n_splits:
            return [current_path]

        paths = []
        for i, combo in enumerate(remaining_combinations):
            if not any(f in used_folds for f in combo):
                new_used = used_folds | set(combo)
                new_path = current_path + [all_combinations.index(combo)]
                new_remaining = remaining_combinations[i + 1 :]
                paths.extend(find_paths(new_used, new_path, new_remaining))

        return paths

    return find_paths(set(), [], all_combinations)


class TimeSeriesSplit:
    """Time series split with gap and minimum training size."""

    def __init__(
        self,
        n_splits: int = 5,
        gap: int = 0,
        max_train_size: int | None = None,
        test_size: int | None = None,
    ) -> None:
        self.n_splits = n_splits
        self.gap = gap
        self.max_train_size = max_train_size
        self.test_size = test_size

    def split(
        self,
        X: NDArray,
        y: NDArray | None = None,
        groups: NDArray | None = None,
    ) -> Iterator[tuple[NDArray[np.int64], NDArray[np.int64]]]:
        """Generate time series train/test splits."""
        n_samples = len(X)
        indices = np.arange(n_samples)

        test_size = self.test_size or (n_samples // (self.n_splits + 1))

        for i in range(self.n_splits):
            test_end = n_samples - (self.n_splits - i - 1) * test_size
            test_start = test_end - test_size
            train_end = test_start - self.gap

            if self.max_train_size:
                train_start = max(0, train_end - self.max_train_size)
            else:
                train_start = 0

            if train_end <= train_start:
                continue

            yield indices[train_start:train_end], indices[test_start:test_end]

    def get_n_splits(self) -> int:
        """Return number of splits."""
        return self.n_splits
