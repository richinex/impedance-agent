# src/core/exceptions.py
class BaseError(Exception):
    """Base exception for all custom exceptions"""

    pass


class DataLoadError(BaseError):
    """Exception raised for errors during data loading"""

    pass


class ConfigError(BaseError):
    """Exception raised for configuration errors"""

    pass


class ValidationError(BaseError):
    """Exception raised for validation errors"""

    pass


class FittingError(BaseError):
    """Exception raised for errors during fitting"""

    pass


class ExportError(BaseError):
    """Exception raised for errors during result export"""

    pass
