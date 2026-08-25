class LocatorError(Exception):
    """Base exception for all dialogue-locator errors."""
    pass


class AcquisitionError(LocatorError):
    """Raised when media acquisition fails."""
    pass


class ProbeError(LocatorError):
    """Raised when media probing fails."""
    pass


class TranscriptionError(LocatorError):
    """Raised when transcription fails."""
    pass


class AlignmentError(LocatorError):
    """Raised when audio-text alignment fails."""
    pass


class FrameExtractionError(LocatorError):
    """Raised when frame extraction fails."""
    pass
