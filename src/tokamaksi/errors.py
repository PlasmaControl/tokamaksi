"""The errors and warnings tokamaksi raises."""

from __future__ import annotations


class TokamaksiError(Exception):
    """Base of every error tokamaksi raises on purpose; the CLI prints it in one line."""


class AccessError(TokamaksiError):
    """The Hub refused or hid the dataset: no token, no read access, or a wrong name."""


class OfflineError(TokamaksiError):
    """A release is not on disk and the network is switched off or unreachable."""


class ReleaseError(TokamaksiError):
    """A release directory is incomplete or a file does not match its checksum."""


class SchemaError(TokamaksiError):
    """A release is newer than this tokamaksi, or lacks columns it needs."""


class MissingPartError(TokamaksiError):
    """A part this release does not ship (a grid, a shot list, a signal group)."""


class NoDataError(TokamaksiError):
    """A signal or a 1 ms label set is not on disk for this shot, or the diagnostic or
    model did not run on it."""


class UnknownEventError(TokamaksiError, ValueError):
    """An event id the release's taxonomy does not know."""


class TermsWarning(UserWarning):
    """A reminder about what the rows you asked for are (and are not)."""
