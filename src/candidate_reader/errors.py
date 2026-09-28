class CandidateReaderError(Exception):
    """Base class for expected application failures."""


class ConfigurationError(CandidateReaderError):
    """Configuration is missing or unsafe."""


class ExtractionError(CandidateReaderError):
    """A source document could not be extracted."""


class StructuredOutputError(CandidateReaderError):
    """Model output could not be validated."""


class SourceReferenceError(CandidateReaderError):
    """Structured data cites source content that does not exist."""


class StorageError(CandidateReaderError):
    """Structured data could not be loaded or written."""


class CostBudgetExceededError(CandidateReaderError):
    """A model call would exceed the configured cost ceiling."""
