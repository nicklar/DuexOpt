# duexopt/evaluator.py

import numpy as np
import numbers # For type checking
from typing import Dict, Any, Callable, Optional, List, Tuple

# Define type hint for the user's evaluation function (copied for clarity)
EvalFuncType = Callable[[Dict[str, Any], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]], float]

# --- Top-Level Function that Runs CV (Pickleable) ---
# duexopt/evaluator.py (or wherever _run_cv_for_single_param_set lives)

def _run_cv_for_single_param_set(
    params_dict: Dict[str, Any],
    eval_func: EvalFuncType,
    X: np.ndarray, # Can now assume X is not None
    y: Optional[np.ndarray],
    cv: int, # Will always be >= 2 here
    fold_indices: List[Tuple[np.ndarray, np.ndarray]], # Can assume this is not None
    direction: str,
    verbose: int
) -> float:
    """Calculates the CV score for a single parameter set."""
    if verbose > 1: print(f"  [Worker Eval] Params: {params_dict} ({cv}-Fold CV)")
    objective_score: float
    fold_scores = []

    # --- REMOVED 'if cv <= 1:' block ---

    # Perform K-Fold CV (always runs now)
    for i, (train_idx, val_idx) in enumerate(fold_indices):
        if verbose > 2: print(f"    [Worker CV Fold {i+1}/{cv}] Starting evaluation...")
        try:
            X_train, X_val = X[train_idx], X[val_idx]
            y_train = y[train_idx] if y is not None else None
            y_val = y[val_idx] if y is not None else None

            # Check if validation set is empty (can happen with LeaveOneOut conceptually)
            # Though our KFold implementation shouldn't produce empty val sets for k < N
            if X_val.shape[0] == 0:
                 print(f"Warning: Fold {i+1} has empty validation set. Skipping.")
                 # Or append inf? Let's append inf to penalize unusual splits.
                 fold_scores.append(float('inf'))
                 continue

            score = eval_func(params_dict, X_train, y_train, X_val, y_val) # eval_func still needs X_val

            if not isinstance(score, numbers.Real): raise TypeError(f"eval_func ret type {type(score)} fold {i+1}")
            if not np.isfinite(score):
                 print(f"Warning: eval_func returned non-finite score ({score}) for fold {i+1}, params {params_dict}. Treating as failure (inf).")
                 fold_scores.append(float('inf'))
            else:
                 fold_scores.append(float(score))
            if verbose > 2: print(f"    [Worker CV Fold {i+1}/{cv}] Score: {score:.5g}")
        except Exception as e: print(f"Warning: eval_func fail fold {i+1}, params {params_dict}: {e}"); fold_scores.append(float('inf'))

    valid_scores = [s for s in fold_scores if np.isfinite(s)]
    if not valid_scores:
        objective_score = float('inf')
        if verbose > 1: print(f"    (All CV folds failed or returned non-finite scores)")
    else:
        objective_score = float(np.mean(valid_scores))
        if verbose > 1: print(f"    (Mean CV score: {objective_score:.5g})")

    final_score_for_optimizer = objective_score if direction == 'minimize' else (-objective_score if np.isfinite(objective_score) else objective_score)
    if verbose > 1: print(f"  [Worker Result] Params: {params_dict} -> Internal Score: {final_score_for_optimizer:.5g}")
    return final_score_for_optimizer