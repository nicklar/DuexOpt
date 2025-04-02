# duexopt/space.py

import numpy as np
import itertools
import numbers # To check for numeric types robustly
from typing import Dict, Any, Tuple, List, Union, Generator, Optional

from .exceptions import SpaceConfigurationError, ConfigurationError

# Type alias for parsed parameter details
ParsedParam = Dict[str, Any]

def _parse_space(space_config: Dict[str, tuple]) -> Tuple[Dict[str, ParsedParam], List[str]]:
    """
    Parses and validates the space configuration dictionary.

    Returns a tuple containing:
        - parsed_space: Dict mapping param name to its detailed config dict.
        - param_order: List of parameter names in insertion order.
    """
    if not isinstance(space_config, dict):
        raise SpaceConfigurationError("space_config must be a dictionary.")
    if not space_config:
        raise SpaceConfigurationError("space_config cannot be empty.")

    parsed_space = {}
    param_order = list(space_config.keys()) # Relies on Python 3.7+ insertion order

    for name in param_order:
        config = space_config[name]
        if not isinstance(name, str) or not name:
            raise SpaceConfigurationError(f"Invalid parameter name: {name}")
        if not isinstance(config, tuple) or not config:
            raise SpaceConfigurationError(f"Invalid config for parameter '{name}': {config}. Must be a non-empty tuple.")

        param_type = config[0]
        details: ParsedParam = {'name': name, 'type': param_type}

        if param_type == 'real':
            if not (3 <= len(config) <= 4):
                raise SpaceConfigurationError(f"Invalid config for real parameter '{name}': {config}. Expected ('real', low, high, [scale='linear'|'log']).")
            low, high = config[1], config[2]
            if not isinstance(low, numbers.Real) or not isinstance(high, numbers.Real):
                raise SpaceConfigurationError(f"Bounds for real parameter '{name}' must be numeric (int or float).")
            if low >= high:
                raise SpaceConfigurationError(f"low bound ({low}) must be less than high bound ({high}) for real parameter '{name}'.")
            details['low'] = float(low)
            details['high'] = float(high)
            scale = 'linear'
            if len(config) == 4:
                 if not isinstance(config[3], str):
                     raise SpaceConfigurationError(f"Scale for real parameter '{name}' must be a string ('linear' or 'log').")
                 scale = config[3].lower()
            if scale not in ['linear', 'log']:
                raise SpaceConfigurationError(f"Invalid scale '{scale}' for real parameter '{name}'. Must be 'linear' or 'log'.")
            if scale == 'log' and details['low'] <= 0:
                 raise SpaceConfigurationError(f"low bound must be > 0 for log scale on parameter '{name}'.")
            details['scale'] = scale

        elif param_type == 'integer':
            if len(config) != 3:
                raise SpaceConfigurationError(f"Invalid config for integer parameter '{name}': {config}. Expected ('integer', low, high).")
            low, high = config[1], config[2]
            # Check if they are integer types or whole number floats
            if not (isinstance(low, numbers.Integral) or (isinstance(low, float) and low.is_integer())):
                 raise SpaceConfigurationError(f"low bound for integer parameter '{name}' must be an integer.")
            if not (isinstance(high, numbers.Integral) or (isinstance(high, float) and high.is_integer())):
                 raise SpaceConfigurationError(f"high bound for integer parameter '{name}' must be an integer.")
            low, high = int(low), int(high) # Cast to int
            if low > high:
                raise SpaceConfigurationError(f"low bound ({low}) cannot be greater than high bound ({high}) for integer parameter '{name}'.")
            details['low'] = low
            details['high'] = high # high is inclusive for SciPy bounds, adjusted for sampling

        elif param_type == 'categorical':
            if len(config) != 2 or not isinstance(config[1], (list, tuple)) or not config[1]:
                raise SpaceConfigurationError(f"Invalid config for categorical parameter '{name}': {config}. Expected ('categorical', [choice1, ...]). Choices cannot be empty.")
            choices = list(config[1])
            if len(set(choices)) != len(choices):
                 # Warning instead of error for duplicates
                 print(f"Warning: Duplicate choices found for categorical parameter '{name}'.")
            details['choices'] = choices
        else:
            raise SpaceConfigurationError(f"Unknown parameter type '{param_type}' for parameter '{name}'. Must be 'real', 'integer', or 'categorical'.")

        parsed_space[name] = details

    return parsed_space, param_order

def _space_to_bounds(parsed_space: Dict[str, ParsedParam], param_order: List[str]) -> List[Tuple[float, float]]:
    """Converts parsed space to ordered bounds list for SciPy optimizers."""
    bounds = []
    for name in param_order:
        details = parsed_space[name]
        param_type = details['type']
        if param_type == 'real':
            bounds.append((details['low'], details['high']))
        elif param_type == 'integer':
             # SciPy bounds are usually floats, represent integer range
             # SHGO works with float bounds. Rounding might happen if func needs strict int.
             bounds.append((float(details['low']), float(details['high'])))
        elif param_type == 'categorical':
             # SHGO/DE typically don't handle categorical directly via bounds
             raise SpaceConfigurationError(f"Cannot convert categorical parameter '{name}' to bounds for method 'shgo' or 'de'. Use 'random' or 'grid' search.")
    return bounds

def _sample_random_point(parsed_space: Dict[str, ParsedParam], rng: np.random.Generator) -> Dict[str, Any]:
    """Samples a single random point from the parsed space."""
    params = {}
    for name, details in parsed_space.items():
        param_type = details['type']
        if param_type == 'real':
            low, high, scale = details['low'], details['high'], details['scale']
            if scale == 'log':
                log_low, log_high = np.log(low), np.log(high)
                params[name] = np.exp(rng.uniform(log_low, log_high))
            else: # linear
                params[name] = rng.uniform(low, high)
        elif param_type == 'integer':
            # rng.integers is exclusive of high endpoint, so add 1
            low, high = details['low'], details['high']
            params[name] = rng.integers(low, high + 1, endpoint=False) # endpoint=False is default anyway
        elif param_type == 'categorical':
            params[name] = rng.choice(details['choices'])
    return params

def _generate_grid_points(parsed_space: Dict[str, ParsedParam],
                           grid_options: Dict[str, int],
                           param_order: List[str]
                           ) -> Generator[Dict[str, Any], None, None]:
    """Generates grid points based on parsed space and options."""
    param_grids = []

    for name in param_order:
        details = parsed_space[name]
        param_type = details['type']
        num_points = grid_options.get(name)

        # Validation for num_points
        if num_points is None:
            raise ConfigurationError(f"Number of grid points not specified for parameter '{name}' in method_options['grid_points'].")
        if not isinstance(num_points, int) or num_points < 1:
             raise ConfigurationError(f"Number of grid points for '{name}' must be a positive integer, got {num_points}.")

        if param_type == 'real':
            low, high, scale = details['low'], details['high'], details['scale']
            if num_points == 1:
                 points = np.array([(low + high) / 2.0]) # Midpoint for single point
            elif scale == 'log':
                points = np.logspace(np.log10(low), np.log10(high), num_points)
            else: # linear
                points = np.linspace(low, high, num_points)
            param_grids.append(points)

        elif param_type == 'integer':
            low, high = details['low'], details['high']
            if num_points == 1:
                 points = np.array([low + (high-low)//2]) # Midpoint rounded down
            elif high == low: # Handle case where range is a single integer
                 points = np.array([low])
            else:
                 # Generate integer points using linspace, rounding, and unique
                 # Clip to ensure bounds are included if num_points > 1
                 points = np.linspace(low, high, min(num_points, high - low + 1)) # Don't generate more points than integers available
                 points = np.round(points).astype(int)
                 points = np.unique(points) # Ensure unique integer points
            param_grids.append(points)

        elif param_type == 'categorical':
             choices = details['choices']
             if num_points > len(choices):
                 print(f"Warning: Requested {num_points} grid points for categorical '{name}', but only {len(choices)} unique choices exist. Using all {len(choices)} choices.")
                 points = np.array(list(set(choices))) # Use unique choices
             elif num_points == 1:
                 points = np.array([choices[0]]) # Just pick first? Or middle? First is simplest.
             else:
                 # If fewer points requested than available, how to choose?
                 # For grid search, it's most common to test ALL categories.
                 # Let's default to using all unique choices if num_points > 1
                 points = np.array(list(set(choices)))
                 if len(points) != num_points:
                     print(f"Warning: Using all {len(points)} unique choices for categorical '{name}' in grid search, ignoring requested num_points={num_points}.")
             param_grids.append(points)

    # Generate combinations using Cartesian product
    total_expected_points = np.prod([len(grid) for grid in param_grids])
    if total_expected_points > 1_000_000: # Warn if grid is very large
         print(f"Warning: Grid search involves {total_expected_points:.2g} points, which may consume significant time and memory.")

    for combo_values in itertools.product(*param_grids):
        yield {name: val for name, val in zip(param_order, combo_values)}