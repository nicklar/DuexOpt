# examples/basic_usage.py

import numpy as np
from scipy.stats import gaussian_kde
import time
import sys
import os

# Assuming 'duexopt' is installed or the project root is in PYTHONPATH
try:
    import duexopt
except ImportError:
    print("DuexOpt not found. Make sure it's installed (`pip install .`) or adjust PYTHONPATH.")
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    try:
        import duexopt
    except ImportError:
        print("Failed to import duexopt even after path adjustment.")
        sys.exit(1)

# --- 1. Define the Evaluation Function (for one fold) ---
def evaluate_kde_fold(params: dict, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> float:
    # ... implementation ...
    # Add necessary imports like 'import numbers' INSIDE if needed by this func,
    # OR ensure they are imported globally in this script.
    # ... make sure it returns a float score ...
    # Check previous response for a more robust version handling LinAlgError etc.
    pass # Replace with your actual function code

# --- Run Optimization Script ---
def run_optimizations():
    # --- 2. Generate Sample 2D Data ---
    # ... data generation ...
    X_full = ...
    y_full = None

    # --- 3. Define Search Space ---
    # ... search_space definition ...
    search_space = {
        'log_bw_x': ('real', -5.0, 2.0, 'linear'),
        'log_bw_y': ('real', -5.0, 2.0, 'linear'),
        'rho':      ('real', -0.95, 0.95, 'linear')
    }

    # --- 4. Run Optimizations (Random, SHGO, Grid) ---
    print("\n--- Running Random Search ---")
    # ... call duexopt.optimize for random ...
    results_random = duexopt.optimize(
        eval_func=evaluate_kde_fold, X=X_full, y=y_full,
        space_config=search_space, method='random',
        method_options={'n_trials': 60}, cv=5, cv_shuffle=True,
        n_jobs=-1, random_state=42, direction='maximize', verbose=1
    )
    print(results_random)
    # ... (print details) ...


    print("\n--- Running SHGO ---")
    # ... call duexopt.optimize for shgo ...
    results_shgo = duexopt.optimize(
        eval_func=evaluate_kde_fold, X=X_full, y=y_full,
        space_config=search_space, method='shgo',
        method_options={'sampling_method': 'sobol'}, cv=5, cv_shuffle=True,
        n_jobs=1, random_state=43, direction='maximize', verbose=1
    )
    print(results_shgo)
    # ... (print details) ...

    print("\n--- Running Grid Search (Coarse Grid) ---")
     # ... call duexopt.optimize for grid ...
    grid_opts = {'grid_points': {'log_bw_x': 5, 'log_bw_y': 5, 'rho': 6}}
    results_grid = duexopt.optimize(
        eval_func=evaluate_kde_fold, X=X_full, y=y_full,
        space_config=search_space, method='grid',
        method_options=grid_opts, cv=5, cv_shuffle=True,
        n_jobs=-1, random_state=44, direction='maximize', verbose=1
    )
    print(results_grid)
    # ... (print details) ...


# --- Main execution guard for multiprocessing safety ---
# V V V V V V V V V V V V V V V V V V V V V V V V V V V V
if __name__ == "__main__":
    # On Windows, you might also need freeze_support:
    # from multiprocessing import freeze_support
    # freeze_support()

    run_optimizations() # Call the function that runs the optimize calls
# ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^