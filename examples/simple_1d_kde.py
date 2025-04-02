# examples/simple_1d_kde.py

import numpy as np
from scipy.stats import gaussian_kde
import time
import sys
import os
import numbers # For type checking in eval_func
from typing import Dict, Any, Callable, Optional, Union, List, Tuple
# Assuming 'duexopt' is installed or the project root is in PYTHONPATH
try:
    import duexopt
except ImportError:
    # Simplified path adjustment assuming script run from project root
    sys.path.insert(0, os.path.abspath('.'))
    try:
         import duexopt
    except ImportError:
        print("ERROR: Cannot find duexopt. Make sure you are in the project root directory")
        print("       AND have installed the package using 'uv pip install -e .'")
        sys.exit(1)

# --- 1. Define the Evaluation Function (for one fold - 1D KDE) ---
def evaluate_1d_kde_fold(params: dict, X_train: np.ndarray, y_train: Optional[np.ndarray], X_val: np.ndarray, y_val: Optional[np.ndarray]) -> float:
    """
    Fits a 1D Gaussian KDE on X_train and evaluates log-likelihood on X_val.
    y_train and y_val are ignored (unsupervised).
    Assumes X_val is always provided (since cv >= 2).
    """
    # --- REMOVED check for X_val is None ---

    # Ensure input data is treated as 1D
    if X_train.ndim > 1 and X_train.shape[1] != 1: raise ValueError("Expected 1D X_train data (N,) or (N,1)")
    if X_val.ndim > 1 and X_val.shape[1] != 1: raise ValueError("Expected 1D X_val data (N,) or (N,1)")
    X_train = X_train.squeeze()
    X_val = X_val.squeeze()
    if X_train.ndim == 0 or X_val.ndim == 0 or X_train.shape[0] < 2:
         return -np.inf # Need multiple points for KDE

    try:
        log_bw = params['log_bw']
    except KeyError as e:
        raise duexopt.ConfigurationError(f"Missing parameter in params dict: {e}")

    bw = np.exp(log_bw)

    if not isinstance(bw, numbers.Real) or not np.isfinite(bw) or bw < 1e-9:
         return -np.inf

    try:
        kde = gaussian_kde(X_train, bw_method=bw)
        log_likelihood = kde.logpdf(X_val)
        finite_ll_mask = np.isfinite(log_likelihood)
        if not np.any(finite_ll_mask): return -np.inf
        score = float(np.sum(log_likelihood[finite_ll_mask]))
        if not np.isfinite(score): return -np.inf
        return score
    except (np.linalg.LinAlgError, ValueError) as e:
        # print(f"Warning: LinAlgError/ValueError in KDE for params {params}: {e}. Penalizing.") # Optional debug
        return -np.inf
    except Exception as e:
        print(f"Warning: Unexpected error in KDE evaluation for params {params}: {e}. Penalizing.")
        return -np.inf

# --- Run Optimization Script ---
def run_simple_optimization():
    # --- 2. Generate Sample 1D Data ---
    np.random.seed(456)
    X_full = np.random.randn(200).reshape(-1, 1) # Ensure 2D shape (N, 1) initially
    y_full = None # Unsupervised

    print(f"Generated data shape: {X_full.shape}")

    # --- 3. Define Search Space (1 Parameter) ---
    search_space = { 'log_bw': ('real', -5.0, 1.0, 'linear') }

    # --- 4. Run Optimization (Grid Search Only, Sequential) ---
    print("\n--- Running Grid Search (1D Example, Sequential) ---")
    grid_opts = { 'grid_points': { 'log_bw': 25 } }
    total_grid_points = 25
    print(f"(Grid involves {total_grid_points} points)")
    start_grid = time.time()

    try:
        results_grid = duexopt.optimize(
            eval_func=evaluate_1d_kde_fold,
            X=X_full,
            y=y_full,
            space_config=search_space,
            method='grid',
            method_options=grid_opts,
            cv=5, # Use 5-fold CV
            cv_shuffle=True,
            n_jobs=1, # Run sequentially
            random_state=44,
            direction='maximize',
            verbose=1 # Change verbosity level as needed
        )
    except Exception as e:
        print("\nERROR: duexopt.optimize call failed!")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    end_grid = time.time()
    print("\nGrid Search Results:")
    print(results_grid)
    if results_grid.best_params:
        log_bw = results_grid.best_params['log_bw']
        bw = np.exp(log_bw)
        print(f"  -> Best Params: log_bw={log_bw:.3f}")
        print(f"  -> Calculated BW: bw={bw:.3f}")
    print(f"  -> Time: {end_grid - start_grid:.2f}s")

    # --- REMOVED Second Run with cv=1 ---

# --- Main execution guard ---
if __name__ == "__main__":
    run_simple_optimization()