"""Getting a release from the Hub: the only module that touches the network."""

from __future__ import annotations

import datetime as dt
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
    except HfHubHTTPError as error:
        status = getattr(getattr(error, "response", None), "status_code", None)
        if status in (401, 403, 404):
            raise _access_error(repo, tag, f"HTTP {status}") from error
        raise
    return commit


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
    tag that is already on disk is left as is unless `force`. Everything but the
    `tables/` mirror is fetched (about 20 MB for v1).
    """
    from .release import _OPEN, Release

    if not config.is_tag(tag):
        raise ReleaseError(f"{tag!r} is not a release tag like 'v1'")
    target = release_dir(tag, home)
    if (target / "release.json").is_file() and not force:
        return target
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
                    "pulled_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
                },
                indent=1,
            )
            + "\n",
            encoding="utf-8",
        )
        _set_modes(partial)
        if target.exists():
            if not force:
                # Another process pulled the same tag meanwhile: keep theirs.
                return target
            shutil.rmtree(target)
        os.replace(partial, target)
        _OPEN.pop(target.resolve(), None)
        return target
    finally:
        shutil.rmtree(partial, ignore_errors=True)


def _set_modes(folder: Path) -> None:
    """A pulled release is write-once and read-many: 0755 folders, 0644 files."""
    for path in [folder, *folder.rglob("*")]:
        path.chmod(0o755 if path.is_dir() else 0o644)
