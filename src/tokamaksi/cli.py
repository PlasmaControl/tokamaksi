"""The `tokamaksi` command: pull a release, look at it, plot a shot."""

from __future__ import annotations

import argparse
import sys

from . import __version__, config
from .errors import TokamaksiError
from .schema import TIER_ORDER

_EVALUATION_ONLY = (
    "The gold shots (split 'test') are for evaluation: do not train or tune on them."
)


def _split(text: str | None) -> list[str] | None:
    return [part.strip() for part in text.split(",") if part.strip()] if text else None


def _split_signals(text: str | None) -> list[str] | None:
    """`a:0,1,b:2` is two specs: a bare number continues the channel list before it."""
    specs: list[str] = []
    for part in _split(text) or []:
        if part.isdigit() and specs and ":" in specs[-1]:
            specs[-1] += "," + part
        else:
            specs.append(part)
    return specs or None


def _pull(args) -> int:
    from . import hf

    path = hf.pull(args.tag, repo=args.repo, force=args.force)
    print(path)
    print(_EVALUATION_ONLY)
    return 0


def _info(args) -> int:
    from .release import open_release

    rel = open_release(args.release, offline=True if args.offline else None)
    meta = rel.meta
    print(f"release   {rel.tag}   schema {rel.schema}")
    print(f"path      {rel.path}")
    print(f"repo      {meta.get('repo', '-')}   commit {rel.commit or '-'}")
    print(f"made_at   {meta.get('made_at', '-')}")
    print("labels    rows / shots")
    for key, row in meta.get("labels", {}).items():
        print(f"  {key:<18}{row.get('rows', '-'):>9}{row.get('shots', '-'):>8}")
    for name, row in meta.get("grids", {}).items():
        print(
            f"grid      {name}: {row.get('shots', '-')} shots, {row.get('runs', '-')} runs"
        )
    print(f"1 ms      {_fdp_events(rel)}")
    if args.check:
        problems = rel.check(deep=True)
        for problem in problems:
            print(f"PROBLEM   {problem}")
        print(
            "check     " + ("clean" if not problems else f"{len(problems)} problem(s)")
        )
        return 0 if not problems else 2
    return 0


def _fdp_events(rel) -> str:
    """The events with all-shot 1 ms model labels, or why none are found."""
    from . import fdp

    try:
        found = fdp.events(release=rel)
    except TokamaksiError:
        return "none found (set TOKAMAKSI_FDP_LABELS)"
    return (", ".join(found) or "none") + " (model output, tokamaksi.fdp)"


def _plot(args) -> int:
    from . import plot
    from .errors import MissingPartError
    from .release import open_release

    figure = plot.shot(
        args.shot,
        _split(args.events),
        signals=_split_signals(args.signals),
        tier=_split(args.tier) or "reviewed",
        t_range=tuple(args.t_range) if args.t_range else None,
        release=args.release,
        corpus=args.corpus,
        model=args.model,
    )
    figure.savefig(args.out, dpi=args.dpi)
    print(args.out)
    try:
        gold = open_release(args.release).shots("gold")
    except MissingPartError:
        return 0
    if args.shot in set(gold["shot"]):
        print(_EVALUATION_ONLY)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tokamaksi",
        description="Read the Tokamak-SI DIII-D event labels and plot them on a shot.",
    )
    parser.add_argument(
        "--version", action="version", version=f"tokamaksi {__version__}"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    pull = sub.add_parser(
        "pull", help="download a release from the Hub into TOKAMAKSI_HOME"
    )
    pull.add_argument("tag", nargs="?", default=config.default_release())
    pull.add_argument(
        "--repo",
        default=None,
        help="Hub dataset (default: TOKAMAKSI_REPO or nc1/tokamak-si)",
    )
    pull.add_argument(
        "--force", action="store_true", help="fetch again even if the tag is on disk"
    )
    pull.set_defaults(run=_pull)

    info = sub.add_parser("info", help="what a release holds")
    info.add_argument("--release", default=None, help="a tag or a release directory")
    info.add_argument("--offline", action="store_true", help="never touch the network")
    info.add_argument(
        "--check", action="store_true", help="verify every file against release.json"
    )
    info.set_defaults(run=_info)

    plot = sub.add_parser(
        "plot", help="draw one shot's labels (and signals) to an image"
    )
    plot.add_argument("shot", type=int)
    plot.add_argument(
        "--events", help="comma-separated event ids (default: every event with rows)"
    )
    plot.add_argument(
        "--signals",
        help="comma-separated signal specs, e.g. neutron_rate:0,bolo:0,1 is two (needs TOKAMAKSI_CORPUS)",
    )
    plot.add_argument(
        "--tier",
        default="reviewed",
        help=f"comma-separated, from {', '.join(TIER_ORDER)}",
    )
    plot.add_argument("--release", default=None, help="a tag or a release directory")
    plot.add_argument(
        "--corpus", default=None, help="folder of <shot>_processed.h5 files"
    )
    plot.add_argument(
        "--model",
        action="store_true",
        help="add the 1 ms model labels of each event (needs TOKAMAKSI_FDP_LABELS)",
    )
    plot.add_argument("--t-range", nargs=2, type=float, metavar=("START_MS", "STOP_MS"))
    plot.add_argument("--dpi", type=int, default=150)
    plot.add_argument(
        "--out", required=True, help="image file to write (.png, .pdf, ...)"
    )
    plot.set_defaults(run=_plot)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.run(args)
    except (TokamaksiError, ValueError, ImportError, OSError) as error:
        print(f"tokamaksi: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
