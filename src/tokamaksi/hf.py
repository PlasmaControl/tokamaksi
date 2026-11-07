"""Getting a release from the Hub: the only module that touches the network."""

from __future__ import annotations

import datetime as dt
import importlib
import json
import os
import shutil
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

from . import config
from .errors import AccessError, OfflineError, ReleaseError
from .schema import PULL_PATTERNS

#: `(repo, tag, destination, patterns) -> commit` : fetch the matching files of the
#: dataset at `tag` into `destination` and return the commit the tag pointed at.
#: Tests pass their own; the default talks to the Hub.
Backend = Callable[[str, str, Path, "tuple[str, ...]"], "str | None"]


def releases_dir(home: str | os.PathLike | None = None) -> Path:
    """`<TOKAMAKSI_HOME>/releases`, where pulled releases live."""
    return (Path(home).expanduser() if home else config.home()) / "releases"


def release_dir(tag: str, home: str | os.PathLike | None = None) -> Path:
    """Where release `tag` is (or would be) on disk."""
    return releases_dir(home) / tag


def _access_error(repo: str, tag: str, why: str) -> AccessError:
    return AccessError(
        f"cannot read {repo} at {tag} ({why}). The dataset is private: (1) ask the "
        f"maintainers for read access, (2) run `hf auth login` or set HF_TOKEN to a "
        f"token that has it, (3) run `tokamaksi pull {tag}` again."
    )


def _network_errors() -> tuple[type[BaseException], ...]:
    """The connection errors of whichever HTTP client this huggingface_hub uses
    (`requests` before 1.0, `httpx` since); either may be absent."""
    kinds = []
    for module, name in (("requests", "RequestException"), ("httpx", "HTTPError")):
        try:
            kinds.append(getattr(importlib.import_module(module), name))
        except ImportError:
            pass
    return tuple(kinds)


def hub_backend(
    repo: str, tag: str, dest: Path, patterns: tuple[str, ...]
) -> str | None:
    """Download `patterns` of dataset `repo` at `tag` into `dest` with huggingface_hub.

    The commit is pinned first, so the files come from one state. Files land as
    plain files in `dest` (no Hub-cache links), and HF_HOME is left alone. A token
    is found by huggingface_hub itself and is never read or shown here.
    """
    from huggingface_hub import HfApi, snapshot_download
    from huggingface_hub.errors import (
        GatedRepoError,
        HfHubHTTPError,
        LocalEntryNotFoundError,
        OfflineModeIsEnabled,
        RepositoryNotFoundError,
        RevisionNotFoundError,
    )

    try:
        commit = HfApi().dataset_info(repo, revision=tag).sha
        snapshot_download(
            repo,
            repo_type="dataset",
            revision=commit,
            local_dir=str(dest),
            allow_patterns=list(patterns),
        )
    except RevisionNotFoundError as error:
        raise ReleaseError(f"{repo} has no release tagged {tag}") from error
    except (RepositoryNotFoundError, GatedRepoError) as error:
        raise _access_error(repo, tag, type(error).__name__) from error
    except (LocalEntryNotFoundError, OfflineModeIsEnabled) as error:
        # The Hub could not be reached and nothing local answers (huggingface_hub
        # says so with these, not with a connection error). Before the HTTP clause:
        # before hub 1.0, LocalEntryNotFoundError is an HfHubHTTPError too.
        raise OfflineError(
            f"cannot reach the Hub to read {repo} at {tag} ({type(error).__name__}). "
            f"Check the network, or set TOKAMAKSI_OFFLINE=1 to use a release already "
            f"on disk."
        ) from error
    except HfHubHTTPError as error:
        status = getattr(getattr(error, "response", None), "status_code", None)
        if status in (401, 403, 404):
            raise _access_error(repo, tag, f"HTTP {status}") from error
        if status == 429 or (status or 0) >= 500:
            raise OfflineError(
                f"the Hub answered HTTP {status} for {repo} at {tag}; try again later, "
                f"or set TOKAMAKSI_OFFLINE=1 to use a release already on disk."
            ) from error
        raise
    except _network_errors() as error:
        raise OfflineError(
            f"cannot reach the Hub to read {repo} at {tag} ({type(error).__name__}: "
            f"{error}). Check the network, or set TOKAMAKSI_OFFLINE=1 to use a release "
            f"already on disk."
        ) from error
    return commit


def _conflict(target: Path, tag: str) -> ReleaseError:
    return ReleaseError(
        f"{target} exists but is not a release (it has no release.json). Remove it, or "
        f"run `tokamaksi pull {tag} --force` to replace it."
    )


def _is_empty_dir(path: Path) -> bool:
    return path.is_dir() and not any(path.iterdir())


def pull(
    tag: str,
    *,
    repo: str | None = None,
    force: bool = False,
    home: str | os.PathLike | None = None,
    backend: Backend | None = None,
) -> Path:
    """Fetch release `tag` into `<home>/releases/<tag>` and return that folder.

    The files go into a temporary folder beside it, are checked against the sha256
    and sizes `release.json` lists, and are then moved into place with one
    `os.replace`, so a failed or interrupted pull leaves nothing half-written. A
    tag that is already on disk is left as is unless `force`, which swaps the old
    folder aside first and puts it back if the new one cannot be installed (a
    complete release another process installs meanwhile is kept instead). A
    folder in the way that holds no `release.json` is not a release: an empty one is
    replaced, any other raises `ReleaseError` unless `force`. Everything but the
    `tables/` mirror is fetched (about 30 MB for v0.2.1).
    """
    from . import __version__
    from .release import _OPEN, Release

    if not config.is_tag(tag):
        raise ReleaseError(f"{tag!r} is not a release tag like 'v1'")
    target = release_dir(tag, home)
    if (target / "release.json").is_file() and not force:
        return target
    if target.exists() and not force and not _is_empty_dir(target):
        raise _conflict(target, tag)  # before a download that could not be used
    if config.offline():
        raise OfflineError(
            f"cannot pull {tag}: the network is off (TOKAMAKSI_OFFLINE or HF_HUB_OFFLINE)"
        )
    repo = repo or config.repo_id()
    parent = target.parent
    parent.mkdir(parents=True, exist_ok=True)
    partial = Path(tempfile.mkdtemp(prefix=f".partial-{tag}-", dir=parent))
    try:
        print(f"pulling {repo} at {tag} into {target} ...", file=sys.stderr)
        commit = (backend or hub_backend)(repo, tag, partial, tuple(PULL_PATTERNS))
        problems = Release(partial).check(deep=True)
        if problems:
            raise ReleaseError(
                f"the pulled copy of {tag} is not what release.json lists: "
                + "; ".join(problems[:5])
            )
        shutil.rmtree(partial / ".cache", ignore_errors=True)
        (partial / ".tokamaksi.json").write_text(
            json.dumps(
                {
                    "repo": repo,
                    "tag": tag,
                    "commit": commit,
                    "tokamaksi_version": __version__,
                    "pulled_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
                },
                indent=1,
            )
            + "\n",
            encoding="utf-8",
        )
        _set_modes(partial)
        if not _install(partial, target, tag, force=force):
            return target  # another process pulled the same tag first: keep theirs
        _OPEN.pop(target.resolve(), None)
        return target
    finally:
        shutil.rmtree(partial, ignore_errors=True)


def _install(partial: Path, target: Path, tag: str, *, force: bool) -> bool:
    """Move the verified `partial` to `target`; False when a complete release was
    put there meanwhile and is kept."""
    aside = None
    try:
        if force and target.exists():
            spot = Path(tempfile.mkdtemp(prefix=f".old-{tag}-", dir=target.parent))
            spot.rmdir()
            os.replace(target, spot)
            aside = spot  # set only once the old copy is really there
        elif _is_empty_dir(target):
            target.rmdir()
        os.replace(partial, target)
    except OSError as error:
        moved = aside is not None  # the old copy is out of the way
        taken = target.exists()  # judged before the old copy goes back
        if moved and not taken:
            try:
                os.replace(aside, target)  # put the old copy back
                aside = None
            except OSError:
                pass
        if taken and (target / "release.json").is_file() and (moved or not force):
            # Another process installed a complete release (between our two renames
            # when forcing): keep theirs, the old copy is not needed any more. A
            # forced pull that could not move the old copy aside still sees that
            # copy here, and is not this case.
            if aside is not None:
                shutil.rmtree(aside, ignore_errors=True)
            return False
        if taken and not force:
            raise _conflict(target, tag) from error
        kept = f"; the previous copy is in {aside}" if aside is not None else ""
        raise ReleaseError(
            f"could not install {tag} into {target}: {error}{kept}"
        ) from error
    if aside is not None:
        shutil.rmtree(aside, ignore_errors=True)
    return True


def _set_modes(folder: Path) -> None:
    """A pulled release is write-once and read-many: 0755 folders, 0644 files."""
    for path in [folder, *folder.rglob("*")]:
        path.chmod(0o755 if path.is_dir() else 0o644)
