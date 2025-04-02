# duexopt/core.py

import numpy as np
import time
import datetime
import numbers # For type checking
from typing import Dict, Any, Callable, Optional, Union, List, Tuple

# Import necessary components from the library
from .result import OptimizationResult
from .space import _parse_space # Used for validation early
from .strategies import _run_grid, _run_random, _run_shgo
from .splitting import _generate_kfolds_indices
# Import the CV runner function (assuming it's moved to evaluator.py)
from .evaluator import _run_cv_for_single_param_set
from .exceptions import ConfigurationError, MethodError, SpaceConfigurationError

# Define type hint for the user's evaluation function
EvalFuncType = Callable[[Dict[str, Any], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]], float]

# --- Main Optimize Function ---
def optimize(
    eval_func: EvalFuncType,
    space_config: Dict[str, tuple],
    method: str,
    X: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
    cv: int = 5, # Default to 5 folds, explicitly require >= 2
    cv_shuffle: bool = True,
    method_options: Optional[Dict[str, Any]] = None,
    n_jobs: int = 1,
    random_state: Optional[int] = None,
    direction: str = 'minimize',
    verbose: int = 0
) -> OptimizationResult:
    """
    Performs hyperparameter optimization using specified method and cross-validation.

    Args:
        eval_func: The function to evaluate a parameter set.
                   Signature: `eval_func(params, X_train, y_train, X_val, y_val) -> score`.
                   It receives sliced data for train/validation according to cv strategy.
        space_config: Dictionary defining the search space. Keys determine parameter order.
                      Example: {'p1': ('real', 0.1, 1.0, 'log'), 'p2': ('integer', 1, 10),
                                'p3': ('categorical', ['a', 'b'])}
                      Format: {name: (type, arg1, arg2, ...)} where type is
                      'real', 'integer', or 'categorical'. Requires Python 3.7+.
        method: The optimization method ('grid', 'random', 'shgo').
        X: Feature data (NumPy array). Required if cv >= 2.
        y: Target data (NumPy array). Can be None (e.g., unsupervised).
           Required by internal KFold splitting if cv >= 2, even if not used by eval_func.
        cv: Number of cross-validation folds. Must be >= 2.
        cv_shuffle: Whether to shuffle data indices before splitting in K-Fold CV.
        method_options: Dictionary of options specific to the chosen method.
                        e.g., {'n_trials': 100} for 'random',
                              {'grid_points': {'p1': 10, 'p2': 5}} for 'grid',
                              {'maxiter': 100, 'sampling_method':'sobol'} for 'shgo'.
        n_jobs: Number of parallel jobs to run (-1 uses all cores).
                Used for grid/random search parallel evaluation via concurrent.futures,
                or passed as 'workers' option to SHGO.
        random_state: Seed for reproducibility (CV shuffle, random sampling).
        direction: 'minimize' (default) or 'maximize'. Whether the score returned
                   by eval_func should be minimized or maximized.
        verbose: Verbosity level (0 = silent, 1 = basic info, 2 = detailed progress).

    Returns:
        An OptimizationResult object containing the best parameters, best score,
        history (for grid/random), and metadata.

    Raises:
        ConfigurationError: If input arguments are invalid (e.g., cv < 2, missing X).
        MethodError: If an unsupported method is specified.
        SpaceConfigurationError: If space_config format is invalid.
        TypeError: If inputs have incorrect types.
    """
    start_time_total = time.time()

    # --- Input Validation ---
    if not callable(eval_func):
        raise TypeError("eval_func must be a callable function.")
    if not isinstance(space_config, dict) or not space_config:
        raise ConfigurationError("space_config must be a non-empty dictionary.")
    valid_methods = ['grid', 'random', 'shgo']
    if method not in valid_methods:
        raise MethodError(f"Invalid method '{method}'. Choose from {valid_methods}.")
    if not isinstance(cv, int):
        raise ConfigurationError("cv must be an integer.")
    if cv < 2:
        raise ConfigurationError(f"cv must be >= 2 for cross-validation, got {cv}.")
    if direction not in ['minimize', 'maximize']:
        raise ConfigurationError("direction must be 'minimize' or 'maximize'.")
    if n_jobs is None: n_jobs = 1 # Handle None case
    if not isinstance(n_jobs, int): raise ConfigurationError("n_jobs must be an integer.")
    if not isinstance(verbose, int) or verbose < 0: raise ConfigurationError("verbose must be a non-negative integer.")

    # Validate space config structure early and get parameter order
    parsed_space: Dict[str, Dict[str, Any]] = {}
    param_order: List[str] = [] # Store order for SHGO mapping
    try:
        # Keep parsed_space and param_order for potential use later if needed
        parsed_space, param_order = _parse_space(space_config)
    except Exception as e:
        raise ConfigurationError(f"Invalid space_config: {e}") from e

    if verbose > 0:
        print(f"[DuexOpt] Starting optimization. Method: {method}, CV folds: {cv}, Parallel jobs: {n_jobs}")
        now = datetime.datetime.now(); print(f"[DuexOpt] Current time: {now.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"[DuexOpt] Optimizing {len(param_order)} parameters: {param_order}")

    # --- CV Setup (Assuming cv >= 2 now) ---
    fold_indices: List[Tuple[np.ndarray, np.ndarray]] = []
    n_samples = 0
    if X is None:
        raise ConfigurationError("X must be provided for cross-validation (cv >= 2).")
    if not isinstance(X, np.ndarray):
            raise TypeError("X must be a NumPy array for CV.")
    n_samples = X.shape[0]
    if y is not None:
        if not isinstance(y, np.ndarray):
                raise TypeError("y must be a NumPy array if provided.")
        if y.shape[0] != n_samples:
                raise ConfigurationError(f"X ({n_samples}) and y ({y.shape[0]}) must have the same number of samples.")

    try:
        # Generate the fold indices that will be passed in the context
        fold_indices = _generate_kfolds_indices(n_samples, cv, shuffle=cv_shuffle, random_state=random_state)
        if verbose > 0:
            print(f"[DuexOpt] Generated {len(fold_indices)} folds using internal KFold (shuffle={cv_shuffle}).")
    except Exception as e:
        raise ConfigurationError(f"Failed to generate CV folds: {e}") from e


    # --- Prepare Context for Objective Evaluation ---
    # This dictionary holds everything needed by _run_cv_for_single_param_set
    # It will be passed to the strategies
    objective_context = {
        'eval_func': eval_func,
        'X': X,
        'y': y,
        'cv': cv, # Although cv>=2, pass it in case needed internally
        'fold_indices': fold_indices,
        'direction': direction,
        'verbose': verbose # Pass verbosity level for potential use in worker
    }

    # --- Dispatch to Selected Strategy ---
    best_params: Optional[Dict[str, Any]] = None
    best_value_internal: float = float('inf') # Value being optimized (always minimized internally)
    history: List[Tuple[Dict[str, Any], float]] = []
    strategy_metadata: Dict[str, Any] = {} # Collect strategy-specific results

    try:
        if method == 'grid':
            best_params, best_value_internal, history = _run_grid(
                objective_context, space_config, method_options, n_jobs, verbose
            )
        elif method == 'random':
            best_params, best_value_internal, history = _run_random(
                objective_context, space_config, method_options, n_jobs, random_state, verbose
            )
        elif method == 'shgo':
            # Note: _run_shgo now returns metadata as well
            best_params, best_value_internal, history_shgo, shgo_meta = _run_shgo(
                objective_context, space_config, method_options, n_jobs, random_state, verbose
            )
            history = history_shgo # Assign (likely empty) history
            strategy_metadata.update(shgo_meta) # Add SHGO specific results
        else:
            # This path should ideally not be reached due to earlier validation
            raise MethodError(f"Internal Error: Method '{method}' dispatch failed.")

    except Exception as e:
         print(f"ERROR: Optimization strategy '{method}' failed unexpectedly: {e}")
         import traceback; traceback.print_exc() # Print full traceback for debugging
         # Return a result indicating failure
         final_metadata = { 'method': method, 'cv': cv, 'n_jobs': n_jobs, 'error': str(e) }
         return OptimizationResult(None, float('inf'), [], final_metadata)


    # --- Final Result Formatting ---
    # Convert optimized value back to original scale if maximization was used
    best_value_final = best_value_internal if direction == 'minimize' else \
                       (-best_value_internal if np.isfinite(best_value_internal) else best_value_internal)

    # Adjust history scores back to original scale as well
    final_history = []
    for params, internal_score in history:
        # Check if score is finite before potentially negating infinity
        original_score = internal_score if direction == 'minimize' else \
                         (-internal_score if np.isfinite(internal_score) else internal_score)
        final_history.append((params, original_score))

    total_elapsed_time = time.time() - start_time_total
    final_metadata = {
        'method': method,
        'cv': cv,
        'n_jobs': n_jobs,
        'total_time_s': total_elapsed_time,
        'direction': direction,
        # Maybe add number of func evals if strategies return it?
        **strategy_metadata # Add strategy-specific data (like shgo_result_object)
    }
    # Avoid adding potentially huge space_config to metadata unless requested via verbose?
    # if verbose > 0: final_metadata['space_config'] = space_config

    if verbose > 0:
         print(f"[DuexOpt] Optimization finished. Total time: {total_elapsed_time:.2f}s")

    # Ensure best_params is None if best_value isn't finite, unless strategy failed completely
    if not np.isfinite(best_value_final) and best_params is not None:
        if verbose > 0: print("[DuexOpt] Warning: No finite best value found, setting best_params to None.")
        best_params = None


    return OptimizationResult(
        best_params=best_params, # Will be None if no finite score found
        best_value=best_value_final,
        history=final_history,
        metadata=final_metadata
    )