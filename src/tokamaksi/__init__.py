"""Read the Tokamak-SI DIII-D event labels and draw them on a shot.

`import tokamaksi` is light: matplotlib, h5py and huggingface_hub load only when a
plot, a signal read or a download needs them.
"""

__version__ = "0.1.0"
#: The release schema this version reads (see docs/schema.md).
RELEASE_SCHEMA = 1

from . import config, hf, shotlists, signals  # noqa: E402
from .errors import (  # noqa: E402
    AccessError,
    MissingPartError,
    NoDataError,
    OfflineError,
    ReleaseError,
    SchemaError,
    TermsWarning,
    TokamaksiError,
    UnknownEventError,
)
from .grid import Grid, ShotGrid, load_grid  # noqa: E402
from .labels import at, individuals, load_labels, spans, unspecified  # noqa: E402
from .release import Release, open_release  # noqa: E402
from .taxonomy import Taxonomy  # noqa: E402

__all__ = [
    "AccessError",
    "Grid",
    "MissingPartError",
    "NoDataError",
    "OfflineError",
    "RELEASE_SCHEMA",
    "Release",
    "ReleaseError",
    "SchemaError",
    "ShotGrid",
    "Taxonomy",
    "TermsWarning",
    "TokamaksiError",
    "UnknownEventError",
    "at",
    "config",
    "hf",
    "individuals",
    "load_grid",
    "load_labels",
    "open_release",
    "shotlists",
    "signals",
    "spans",
    "unspecified",
]


def __getattr__(name):
    # `tokamaksi.plot` imports matplotlib, so it loads on first use.
    if name == "plot":
        import importlib

        return importlib.import_module(".plot", __name__)
    raise AttributeError(f"module 'tokamaksi' has no attribute {name!r}")
