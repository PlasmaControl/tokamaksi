"""Opens a real release: skipped unless TOKAMAKSI_TEST_RELEASE names one. Never in CI."""

import os

import pytest

import tokamaksi as tk
from tokamaksi.schema import TIER_ORDER

pytestmark = pytest.mark.real_release
PATH = os.environ.get("TOKAMAKSI_TEST_RELEASE")


@pytest.mark.skipif(not PATH, reason="TOKAMAKSI_TEST_RELEASE is not set")
def test_a_real_release_opens_and_loads():
    rel = tk.open_release(PATH)
    assert rel.check(deep=True) == []
    on_disk = {p.name for p in (rel.path / "labels").iterdir() if p.is_dir()}
    assert on_disk and on_disk <= set(TIER_ORDER), on_disk - set(TIER_ORDER)
    assert set(rel.datasets()["tier"]) <= set(TIER_ORDER)
    for tier in ("reviewed", "original", "derived", "source"):
        df = tk.load_labels("edge_localized_mode", tier=tier, release=rel)
        assert set(df.kind) <= {"span", "individual", "unspecified"}
        assert str(df.iscrowd.dtype) == "boolean"
    gold = tk.shotlists.gold(rel)
    shot = tk.load_grid("gold", release=rel).decode(int(gold.shot.iloc[0]))
    assert shot.frames.dtype.name == "int8" and shot.frames.shape[1] > 0


@pytest.mark.skipif(not PATH, reason="TOKAMAKSI_TEST_RELEASE is not set")
def test_every_category_of_the_real_taxonomy_has_a_fixed_plot_colour():
    # A category added in a later release must get a colour in plot.CATEGORY_COLOURS
    # (or be one the plot leaves hatched), not a fallback that depends on the figure.
    pytest.importorskip("matplotlib")
    from tokamaksi import plot

    taxonomy = tk.open_release(PATH).taxonomy()
    missing = {
        (event, name)
        for event in taxonomy.events
        for name in taxonomy.categories(event).values()
        if not plot._not_assessed(name) and name not in plot.CATEGORY_COLOURS
    }
    assert not missing, sorted(missing)
