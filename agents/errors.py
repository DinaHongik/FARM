"""
Custom Exceptions for FARM Agentic System
=========================================

Defines exception classes for error handling throughout the
resolution pipeline.
"""


class FARMAgentError(Exception):
    """Base exception for all FARM agent errors."""

    def __init__(self, message: str, details: dict = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class RAGError(FARMAgentError):
    """Error loading candidates from RAG system."""
    pass


class ParsingError(FARMAgentError):
    """Error parsing LLM output as JSON."""

    def __init__(self, message: str, raw_output: str = None, details: dict = None):
        super().__init__(message, details)
        self.raw_output = raw_output


class ResolutionError(FARMAgentError):
    """Error during agent resolution."""
    pass


class AllPairsFailedError(FARMAgentError):
    """All trigger-action pairs have been rejected."""

    def __init__(self, rejection_history: list):
        message = f"All {len(rejection_history)} pairs were rejected"
        super().__init__(message)
        self.rejection_history = rejection_history


class VerifierError(FARMAgentError):
    """Error during contract verification."""
    pass


class LLMError(FARMAgentError):
    """Error calling the LLM."""

    def __init__(self, message: str, model: str = None, details: dict = None):
        super().__init__(message, details)
        self.model = model


class ConfigurationError(FARMAgentError):
    """Error in system configuration."""
    pass
