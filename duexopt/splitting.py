# duexopt/splitting.py

import numpy as np
from typing import Optional, List, Tuple

def _generate_kfolds_indices(
    n_samples: int,
    n_splits: int,
    shuffle: bool = True,
    random_state: Optional[int] = None
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Generates K-Fold training and validation indices without sklearn dependency.

    Args:
        n_samples: Total number of samples.
        n_splits: Number of folds (k). Must be at least 2.
        shuffle: Whether to shuffle the data before splitting indices.
        random_state: Seed for the random number generator for shuffling.

    Returns:
        A list of tuples, where each tuple contains (train_indices, val_indices)
        for one fold. Indices are integer arrays.

    Raises:
        ValueError: If n_splits is invalid or less than 2, or > n_samples.
    """
    if not isinstance(n_splits, int) or n_splits < 2:
        raise ValueError(f"n_splits must be an integer >= 2, got {n_splits}.")
    if n_splits > n_samples:
        raise ValueError(f"Cannot have n_splits={n_splits} > n_samples={n_samples}.")

    indices = np.arange(n_samples)

    if shuffle:
        # Use NumPy's recommended way for seeding
        rng = np.random.default_rng(random_state)
        rng.shuffle(indices)

    # Determine the sizes of each fold, distributing remainder
    fold_sizes = np.full(n_splits, n_samples // n_splits, dtype=int)
    remainder = n_samples % n_splits
    fold_sizes[:remainder] += 1
    if sum(fold_sizes) != n_samples: # Sanity check
         raise RuntimeError("Internal error: Fold sizes do not sum to n_samples.")

    cv_indices = []
    current_idx_pos = 0
    for fold_size in fold_sizes:
        start, stop = current_idx_pos, current_idx_pos + fold_size
        # Get validation indices for the current fold
        val_indices = indices[start:stop]

        # Get training indices (all indices *not* in the current validation set)
        # Create a boolean mask for efficiency
        train_mask = np.ones(n_samples, dtype=bool)
        train_mask[start:stop] = False
        train_indices = indices[train_mask]

        cv_indices.append((train_indices, val_indices))
        current_idx_pos = stop

    return cv_indices