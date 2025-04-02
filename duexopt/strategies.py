# duexopt/strategies.py

import numpy as np
import time
import os
import scipy.optimize
from concurrent.futures import ProcessPoolExecutor
import functools # <--- IMPORT functools
import numbers # <-- Ensure this is imported
from typing import Dict, Any, Callable, Optional, List, Tuple

# Import necessary components from the library
from .result import OptimizationResult
from .space import _parse_space, _space_to_bounds, _sample_random_point, _generate_grid_points
from .exceptions import ConfigurationError
# Import the top-level CV runner function (assuming it's in core.py)
from .evaluator import _run_cv_for_single_param_set

# Define type hint for the context dictionary
ObjectiveContextType = Dict[str, Any]

# --- Worker for ProcessPoolExecutor ---
# This receives the *context dictionary* via functools.partial
def _parallel_worker(params_dict: Dict[str, Any], context: ObjectiveContextType) -> Tuple[Dict[str, Any], float]:
    """Worker function that calls _run_cv_for_single_param_set with context. Runs in worker process."""
    # This function should be pickleable as it's top-level
    try:
        # Call the top-level function that handles CV etc.
        score = _run_cv_for_single_param_set(
            params_dict=params_dict,
            # Unpack context correctly
            eval_func=context['eval_func'],
            X=context['X'],
            y=context['y'],
            cv=context['cv'],
            fold_indices=context['fold_indices'],
            direction=context['direction'],
            verbose=context['verbose'] # Pass verbose level
        )
        # Ensure score is finite float before returning
        if not isinstance(score, numbers.Real) or not np.isfinite(score):
            score = float('inf')

        return params_dict, float(score)
    except Exception as e:
        print(f"Error in worker evaluating {params_dict}: {e}")
        # import traceback; traceback.print_exc() # Uncomment for deep debugging in worker
        return params_dict, float('inf')


# --- Strategy Functions ---

def _run_grid(
    objective_context: ObjectiveContextType,
    space_config: Dict[str, tuple],
    options: Optional[Dict[str, Any]],
    n_jobs: int,
    verbose: int
) -> Tuple[Optional[Dict[str, Any]], float, List[Tuple[Dict[str, Any], float]]]:
    """Performs Grid Search, potentially in parallel."""
    start_time = time.time()
    options = options or {}
    grid_points_config = options.get('grid_points')
    if grid_points_config is None: raise ConfigurationError("Grid needs 'grid_points' in options.")

    parsed_space, param_order = _parse_space(space_config)
    try:
        grid_points = list(_generate_grid_points(parsed_space, grid_points_config, param_order))
    except Exception as e: raise ConfigurationError(f"Error generating grid points: {e}") from e
    total_points = len(grid_points)
    if total_points == 0: print("Warning: Grid search generated zero points."); return None, float('inf'), []
    if verbose > 0: print(f"[DuexOpt] Starting Grid Search: {total_points} points total.")

    results_list: List[Tuple[Dict[str, Any], float]] = []

    if n_jobs == 1:
        # --- Sequential Execution ---
        if verbose > 1: print("[DuexOpt] Running Grid Search sequentially.")
        for i, params in enumerate(grid_points):
            # Call the CV runner directly, unpacking context
            try:
                # Pass params_dict as first arg, then unpack context dict for others
                score = _run_cv_for_single_param_set(params, **objective_context)
                if not isinstance(score, numbers.Real) or not np.isfinite(score): score = float('inf')
                results_list.append((params, score))
            except Exception as e:
                print(f"Error evaluating {params} sequentially: {e}")
                results_list.append((params, float('inf')))
            # Progress update (removed internal best score tracking here)
            if verbose > 1 and (i + 1) % (max(1, total_points // 10)) == 0:
                 print(f"  [Grid Progress] Evaluated {i+1}/{total_points}.")
    else:
        # --- Parallel Execution ---
        actual_jobs = os.cpu_count() if n_jobs == -1 else n_jobs
        if verbose > 0: print(f"[DuexOpt] Running Grid Search with {actual_jobs} parallel workers.")
        # Prepare the worker function using partial (binds context)
        worker_func = functools.partial(_parallel_worker, context=objective_context)
        try:
            with ProcessPoolExecutor(max_workers=actual_jobs) as executor:
                # Map the partial worker function over the grid points
                results_list = list(executor.map(worker_func, grid_points))
        except Exception as e:
            print(f"Error during parallel grid execution: {e}. Results may be incomplete.")
            # results_list might be partially populated or empty

    # --- Process results (common path) ---
    best_score_internal = float('inf')
    best_params = None
    valid_results = [(p, s) for p, s in results_list if np.isfinite(s)]

    if valid_results:
        # Find the minimum score among valid results
        best_params, best_score_internal = min(valid_results, key=lambda item: item[1])
    elif results_list: # No finite scores found, return first point evaluated
        print("Warning: Grid search did not find a finite best score.")
        best_params, best_score_internal = results_list[0] # Score will be inf
    else: # No results at all
         print("Error: Grid search yielded no results.")
         # best_params remains None, best_score_internal remains inf

    elapsed_time = time.time() - start_time
    if verbose > 0: print(f"[DuexOpt] Grid Search finished in {elapsed_time:.2f}s. Best internal score: {best_score_internal:.5g}")
    # History contains internal scores (adjusted for direction)
    return best_params, best_score_internal, results_list


def _run_random(
    objective_context: ObjectiveContextType,
    space_config: Dict[str, tuple],
    options: Optional[Dict[str, Any]],
    n_jobs: int,
    random_state: Optional[int],
    verbose: int
) -> Tuple[Optional[Dict[str, Any]], float, List[Tuple[Dict[str, Any], float]]]:
    """Performs Random Search, potentially in parallel."""
    start_time = time.time()
    options = options or {}; n_trials = options.get('n_trials', 50)
    if not isinstance(n_trials, int) or n_trials <= 0: raise ConfigurationError("'n_trials' positive int needed.")

    parsed_space, param_order = _parse_space(space_config)
    rng = np.random.default_rng(random_state)
    param_sets = [_sample_random_point(parsed_space, rng) for _ in range(n_trials)]
    if verbose > 0: print(f"[DuexOpt] Starting Random Search: {n_trials} trials.")

    results_list: List[Tuple[Dict[str, Any], float]] = []

    # Prepare the worker function using partial (binds context)
    worker_func = functools.partial(_parallel_worker, context=objective_context)

    if n_jobs == 1:
         # --- Sequential Execution ---
        if verbose > 1: print("[DuexOpt] Running Random Search sequentially.")
        for i, params in enumerate(param_sets):
            # Call the partial function directly
             _params, score = worker_func(params)
             results_list.append((_params, score))
             # Progress update (removed internal best score tracking here)
             if verbose > 1 and (i + 1) % (max(1, n_trials // 10)) == 0:
                  print(f"  [Random Progress] Evaluated {i+1}/{n_trials}.")
    else:
        # --- Parallel Execution ---
        actual_jobs = os.cpu_count() if n_jobs == -1 else n_jobs
        if verbose > 0: print(f"[DuexOpt] Running Random Search with {actual_jobs} parallel workers.")
        try:
            with ProcessPoolExecutor(max_workers=actual_jobs) as executor:
                results_list = list(executor.map(worker_func, param_sets))
        except Exception as e:
            print(f"Error during parallel random execution: {e}. Results may be incomplete.")

    # --- Process results (common path) ---
    best_score_internal = float('inf')
    best_params = None
    valid_results = [(p, s) for p, s in results_list if np.isfinite(s)]

    if valid_results:
        best_params, best_score_internal = min(valid_results, key=lambda item: item[1])
    elif results_list:
        print("Warning: Random search did not find a finite best score.")
        best_params, best_score_internal = results_list[0]
    else:
         print("Error: Random search yielded no results.")

    elapsed_time = time.time() - start_time
    if verbose > 0: print(f"[DuexOpt] Random Search finished in {elapsed_time:.2f}s. Best internal score: {best_score_internal:.5g}")
    return best_params, best_score_internal, results_list


def _run_shgo(
    objective_context: ObjectiveContextType,
    space_config: Dict[str, tuple],
    options: Optional[Dict[str, Any]],
    n_jobs: int,
    random_state: Optional[int], # Used by CV splitting via context
    verbose: int
) -> Tuple[Optional[Dict[str, Any]], float, List[Tuple[Dict[str, Any], float]], Dict[str, Any]]: # Return metadata too
    """Runs optimization using scipy.optimize.shgo."""
    start_time = time.time()
    options = options or {}

    parsed_space, param_order = _parse_space(space_config)
    try:
        bounds = _space_to_bounds(parsed_space, param_order)
        # Check bounds immediately
        checked_bounds = []
        for low, high in bounds:
            if not np.isfinite(low) or not np.isfinite(high):
                raise ConfigurationError(f"SHGO requires finite bounds, found: ({low}, {high}).")
            checked_bounds.append((low, high))
    except Exception as e:
        raise ConfigurationError(f"Error processing bounds for SHGO: {e}") from e

    # --- Define Wrapper for SHGO ---
    # Use functools.partial to create a callable with context bound
    # This callable still needs to accept the array 'x_arr' from SHGO
    # The target _run_cv_for_single_param_set expects params_dict first.

    # Define the core function SHGO calls (must take array)
    def shgo_func_target(x_arr: np.ndarray, p_order: List[str], context: ObjectiveContextType) -> float:
         params_dict = {name: val for name, val in zip(p_order, x_arr)}
         # Call the main CV runner function
         score = _run_cv_for_single_param_set(params_dict=params_dict, **context)
         return score

    # Create the final function object for SHGO using partial
    shgo_objective_for_scipy = functools.partial(
        shgo_func_target,
        p_order=param_order,
        context=objective_context
    )
    # --- End Wrapper Definition ---


    shgo_options = options.copy()
    if n_jobs != 1:
        actual_jobs = os.cpu_count() if n_jobs == -1 else n_jobs
        shgo_options['workers'] = actual_jobs
        if verbose > 0: print(f"[DuexOpt] Running SHGO with {actual_jobs} parallel workers (via SHGO internal).")
    elif verbose > 0: print(f"[DuexOpt] Starting SHGO (sequential).")


    best_params_dict: Optional[Dict[str, Any]] = None
    best_score_internal = float('inf')
    history: List[Tuple[Dict[str, Any], float]] = [] # Keep history empty for SHGO V1
    shgo_result_object = None
    metadata_shgo = {}

    try:
        shgo_result = scipy.optimize.shgo(
            func=shgo_objective_for_scipy, # Pass the partial function
            bounds=checked_bounds,
            options=shgo_options
        )
        shgo_result_object = shgo_result
        metadata_shgo['shgo_result_object'] = shgo_result_object

        # Check if shgo found a minimum (shgo_result.x might be None)
        if getattr(shgo_result, 'x', None) is not None and np.isfinite(shgo_result.fun):
            # SHGO only guarantees success=True if constraints met, check finite fun
            best_params_arr = shgo_result.x
            best_params_dict = {name: val for name, val in zip(param_order, best_params_arr)}
            best_score_internal = float(shgo_result.fun)
            # Note: shgo_result.fun should already account for internal minimization objective
        else:
             # If x is None or fun is not finite, treat as failure
             print(f"Warning: SHGO optimization did not converge to a finite minimum.")
             if hasattr(shgo_result,'message'): print(f"  Message: {shgo_result.message}")
             best_score_internal = float('inf')

    except Exception as e:
         print(f"Error during scipy.optimize.shgo execution: {e}")
         import traceback; traceback.print_exc() # Show traceback for SHGO errors
         best_score_internal = float('inf') # Ensure inf on error

    elapsed_time = time.time() - start_time
    msg = getattr(shgo_result_object, 'message', 'N/A (Error occurred)')
    succ = getattr(shgo_result_object, 'success', False) # Default to False on error
    if verbose > 0: print(f"[DuexOpt] SHGO finished in {elapsed_time:.2f}s. Success: {succ}. Message: {msg}. Best internal score: {best_score_internal:.5g}")

    return best_params_dict, best_score_internal, history, metadata_shgo