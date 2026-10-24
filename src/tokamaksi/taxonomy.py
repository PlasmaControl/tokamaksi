"""The release's `taxonomy.yaml`: events, categories, nesting and folding."""

from __future__ import annotations

import difflib
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from .errors import SchemaError, UnknownEventError
from .schema import GRID_MS, UNASSESSED

if TYPE_CHECKING:
    from .release import Release


class Taxonomy:
    """Events, their categories, parents and folding, read from a release.

    An event id is a name such as `edge_localized_mode`, or `<event>.<key>` for a
    subcategory (`mhd_n1.m2`), which lives inside its parent's saved spans.
    """

    def __init__(self, data: dict):
        events = data.get("events")
        if not isinstance(events, dict) or not events:
            raise SchemaError("taxonomy.yaml has no `events` table")
        grid = data.get("grid") or {}
        if (
            int(grid.get("grid_ms", GRID_MS)) != GRID_MS
            or int(grid.get("unassessed", UNASSESSED)) != UNASSESSED
        ):
            raise SchemaError(
                "taxonomy.yaml declares a different grid rule than this tokamaksi reads"
            )
        self._events = {str(name): dict(value or {}) for name, value in events.items()}
        self._grid = grid
        self.events: tuple[str, ...] = tuple(self._events)

    @classmethod
    def load(cls, release: str | Path | Release | None = None) -> Taxonomy:
        """The taxonomy of `release` (default: the default release)."""
        from .release import open_release

        if isinstance(release, (str, Path)) or release is None:
            release = open_release(release)
        return release.taxonomy()

    @classmethod
    def from_file(cls, path: str | Path) -> Taxonomy:
        with open(path, encoding="utf-8") as handle:
            return cls(yaml.safe_load(handle))

    def _entry(self, event: str) -> dict:
        if event not in self._events:
            near = difflib.get_close_matches(event, self.events, n=4, cutoff=0.5)
            hint = f"; did you mean {', '.join(near)}?" if near else ""
            raise UnknownEventError(f"unknown event {event!r}{hint}")
        return self._events[event]

    def __contains__(self, event: object) -> bool:
        return event in self._events

    def display(self, event: str) -> str:
        """The name to show a person."""
        return str(self._entry(event).get("display", event))

    def categories(self, event: str) -> dict[int, str]:
        """Category id -> name; `0` is `absent` where the event has an absent class."""
        return {
            int(k): str(v) for k, v in self._entry(event).get("categories", {}).items()
        }

    def category_id(self, event: str, name: str) -> int:
        """The id of the category called `name`."""
        for cid, label in self.categories(event).items():
            if label == name:
                return cid
        raise KeyError(f"{event} has no category {name!r}: {self.categories(event)}")

    def parent(self, event: str) -> str | None:
        """The event a subcategory nests inside, else None."""
        return self._entry(event).get("parent")

    def children(self, event: str) -> list[str]:
        """The subcategories of `event`, in taxonomy order."""
        self._entry(event)
        return [
            name for name in self.events if self._events[name].get("parent") == event
        ]

    def group(self, event: str) -> str | None:
        """The family an event is shown under (`MHD mode`), if any."""
        return self._entry(event).get("group")

    def mode(self, event: str) -> tuple[int, int] | None:
        """`(m, n)` of a mode subcategory (`mhd_n1.m2` is `(2, 1)`), else None."""
        entry = self._entry(event)
        if "m" in entry and "n" in entry:
            return int(entry["m"]), int(entry["n"])
        return None

    def folded_into(self, event: str) -> dict | None:
        """Where the review page shows `event` instead: `{"event": ..., "categories":
        {own id: id there}}`. `load_labels` on a folded event returns what its own
        tables say; the page and the grid show it as a category of the other event.
        """
        folded = self._entry(event).get("folded_into")
        if not folded:
            return None
        return {
            "event": folded["event"],
            "categories": {
                int(k): int(v) for k, v in folded.get("categories", {}).items()
            },
        }

    def on_review_page(self, event: str) -> bool:
        """Whether the review page lists this event on its own."""
        return bool(self._entry(event).get("on_review_page", False))

    def zero(self, event: str) -> str:
        """What category 0 means for this event on the grid."""
        zeros = self._grid.get("zero") or {}
        return str(zeros.get(event, "absent: assessed and not seen"))

    @property
    def grid_rule(self) -> str:
        """The full 10 ms grid rule, verbatim from the release."""
        return " ".join(str(self._grid.get("rule", "")).split())
