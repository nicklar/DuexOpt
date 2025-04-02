# duexopt/__init__.py

__version__ = "0.1.0"

# Check Python version at import time to ensure dict order preservation
import sys
if sys.version_info < (3, 7):
    print(
        f"Warning: DuexOpt relies on dictionary insertion order for parameter consistency, "
        f"which is guaranteed from Python 3.7 onwards. Your version: {sys.version}. "
        f"Behavior might be unpredictable.",
        file=sys.stderr
    )

# Import main components for user accessibility
# >>> MAKE SURE THESE LINES ARE PRESENT <<<
from .result import OptimizationResult
from .core import optimize # <--- This line imports the function
from .exceptions import ConfigurationError, SpaceConfigurationError, MethodError, OptionalDependencyNotFoundError

# Define what gets imported with 'from duexopt import *'
__all__ = [
    "optimize", # <--- And this makes 'optimize' available
    "OptimizationResult",
    "ConfigurationError",
    "SpaceConfigurationError",
    "MethodError",
    "OptionalDependencyNotFoundError",
]