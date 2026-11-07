"""Where tokamaksi looks, from the environment. Nothing here touches the network."""

from __future__ import annotations

import os
import re
from pathlib import Path

#: The release a bare `open_release()` opens: the one this version was cut against.
DEFAULT_RELEASE = "v1"
#: The private dataset on the Hugging Face Hub.
DEFAULT_REPO = "nc1/tokamak-si"

_TRUE = {"1", "true", "yes", "on"}
_TAG = re.compile(r"^v\d+(\.\d+)*$")


def is_tag(value: str) -> bool:
    """Whether `value` is a release tag (`v1`, `v2.1`) and not a path."""
    return bool(_TAG.match(value))


def _flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUE


def home() -> Path:
    """The folder releases are kept in: `$TOKAMAKSI_HOME`, else `~/.cache/tokamaksi`.

    A release lives at `<home>/releases/<tag>/`. Hugging Face's own cache location
    (`HF_HOME`) is never read or changed here.
    """
    chosen = os.environ.get("TOKAMAKSI_HOME", "").strip()
    if chosen:
        return Path(chosen).expanduser()
    base = os.environ.get("XDG_CACHE_HOME", "").strip()
    root = Path(base).expanduser() if base else Path.home() / ".cache"
    return root / "tokamaksi"


def repo_id() -> str:
    """The Hub dataset: `$TOKAMAKSI_REPO`, else `DEFAULT_REPO`."""
    return os.environ.get("TOKAMAKSI_REPO", "").strip() or DEFAULT_REPO


def default_release() -> str:
    """The release a call without one opens: `$TOKAMAKSI_RELEASE` (a tag or a directory)."""
    return os.environ.get("TOKAMAKSI_RELEASE", "").strip() or DEFAULT_RELEASE


def offline() -> bool:
    """True when `TOKAMAKSI_OFFLINE` or `HF_HUB_OFFLINE` asks for no network."""
    return _flag("TOKAMAKSI_OFFLINE") or _flag("HF_HUB_OFFLINE")


def corpus() -> Path | None:
    """The folder of per-shot signal files (`<shot>_processed.h5`): `$TOKAMAKSI_CORPUS`.

    None when unset; tokamaksi ships no default path to any cluster.
    """
    chosen = os.environ.get("TOKAMAKSI_CORPUS", "").strip()
    return Path(chosen).expanduser() if chosen else None


def fdp_labels() -> Path | None:
    """The folder of the all-shot 1 ms model labels (one `<event>/` folder each):
    `$TOKAMAKSI_FDP_LABELS`.

    None when unset; tokamaksi ships no default path to any cluster.
    """
    chosen = os.environ.get("TOKAMAKSI_FDP_LABELS", "").strip()
    return Path(chosen).expanduser() if chosen else None
