# duexopt/result.py

from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np

class OptimizationResult:
    """
    Stores the results of an optimization run.

    Attributes:
        best_params (Dict[str, Any]): Dictionary of the best parameter set found.
                                      Returns an empty dict if no valid result found.
        best_value (float): The objective function value corresponding to best_params
                           (always in the original scale, e.g., higher is better if
                           direction='maximize'). Returns inf or -inf if no valid result.
        history (List[Tuple[Dict[str, Any], float]]): A list storing tuples of
            (parameter_dict, objective_value_original_scale) for evaluated points.
            May not be fully populated for all methods (e.g., SHGO).
        metadata (Dict[str, Any]): Dictionary containing additional information,
            such as the raw result object from backend optimizers (like SHGO),
            total time, number of evaluations, or error messages.
    """
    def __init__(self,
                 best_params: Optional[Dict[str, Any]],
                 best_value: float,
                 history: Optional[List[Tuple[Dict[str, Any], float]]] = None,
                 metadata: Optional[Dict[str, Any]] = None):

        # Handle cases where optimization might completely fail
        self.best_params = best_params if best_params is not None else {}
        self.best_value = float(best_value) # Ensure it's a standard float

        if not isinstance(self.best_params, dict):
            # This check is mostly for internal consistency
            raise TypeError("Internal error: best_params must be a dictionary or None.")

        # Add a warning if the best value is non-finite, as params might be meaningless
        if not np.isfinite(self.best_value):
             print(f"Warning: Optimization resulted in non-finite best_value ({self.best_value}). "
                   f"best_params may correspond to the initial point or be unreliable.")

        self.history = history if history is not None else []
        self.metadata = metadata if metadata is not None else {}

    def __repr__(self):
        try:
            # Format finite numbers, handle inf/-inf explicitly
            if np.isinf(self.best_value) and self.best_value > 0:
                val_str = "+inf"
            elif np.isinf(self.best_value) and self.best_value < 0:
                 val_str = "-inf"
            elif np.isnan(self.best_value):
                 val_str = "nan"
            else:
                 val_str = f"{self.best_value:.5g}"
        except (TypeError, ValueError):
            val_str = str(self.best_value) # Fallback

        # Shorten params dict string if too long for repr
        max_repr_len = 60
        params_str = str(self.best_params)
        if len(params_str) > max_repr_len:
             params_str = params_str[:max_repr_len-3] + "..."

        return (f"OptimizationResult(best_value={val_str}, "
                f"best_params={params_str})")