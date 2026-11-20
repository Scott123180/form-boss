"""Errors the analysis raises for problems with the input rather than bugs. The CLI reports these
as a one-line message; anything else is a bug and keeps its traceback."""


class FormBossError(Exception):
    """Base class: the analysis cannot go ahead with this input."""


class VideoError(FormBossError):
    """The video cannot be opened or read."""


class TrackingError(FormBossError):
    """No lifter could be tracked reliably in the video."""


class DependencyError(FormBossError):
    """An external tool the analysis needs (ffmpeg) is missing."""


class ConfigError(FormBossError):
    """An option the user gave does not apply (unknown variation, strictness or lift)."""
