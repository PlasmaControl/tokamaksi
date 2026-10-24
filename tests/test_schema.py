import pytest

from tokamaksi import schema
from tokamaksi.errors import SchemaError


def test_pinned_constants():
    # Copied from FusionAIHub at commit 52c2a2bd; a change there is a change here.
    assert schema.GRID_MS == 10
    assert schema.UNASSESSED == -1
    assert schema.TIER_ORDER == ("reviewed", "original", "derived", "source")
    assert schema.SPLITS == ("train", "holdout", "test")


def test_release_tiers_are_known(rel):
    on_disk = {p.name for p in (rel.path / "labels").iterdir()}
    assert on_disk <= set(schema.TIER_ORDER)


def test_require_columns_names_the_missing():
    schema.require_columns(["a", "b"], ["a"], "t")
    with pytest.raises(SchemaError, match="b"):
        schema.require_columns(["a"], ["a", "b"], "t")
