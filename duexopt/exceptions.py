# duexopt/exceptions.py

class ConfigurationError(ValueError):
    """Base exception for configuration errors in DuexOpt."""
    pass

class SpaceConfigurationError(ConfigurationError):
    """Exception raised for errors in the space_config format."""
    pass

class MethodError(ConfigurationError):
    """Exception raised for invalid method specification."""
    pass

class OptionalDependencyNotFoundError(ImportError):
    """Exception raised when an optional dependency is needed but not installed."""
    pass