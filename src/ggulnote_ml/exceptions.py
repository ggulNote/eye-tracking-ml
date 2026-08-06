class PipelineError(RuntimeError):
    """Base error for an actionable pipeline failure."""


class ConfigurationError(PipelineError):
    """Raised when a configuration file is incomplete or inconsistent."""


class ContractError(PipelineError):
    """Raised when data does not satisfy a declared input/output contract."""


class OptionalDependencyError(PipelineError):
    """Raised when an optional integration dependency is unavailable."""

