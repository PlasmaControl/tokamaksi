import pytest

import tokamaksi as tk


def test_events_categories_and_nesting(rel):
    tax = rel.taxonomy()
    assert "mhd_n1.m2" in tax and tax.display("edge_localized_mode") == "ELM"
    assert (
        tax.categories("confinement")[6] == "imode"
        and len(tax.categories("confinement")) == 7
    )
    assert tax.category_id("confinement", "qh") == 3
    assert tax.parent("mhd_n1.m2") == "mhd_n1" and tax.parent("mhd_n1") is None
    assert tax.children("mhd_n1") == ["mhd_n1.m2"]
    assert tax.mode("mhd_n1.m2") == (2, 1) and tax.mode("mhd_n1") is None
    assert tax.group("mhd_n1.m2") == "MHD mode"


def test_folding_and_zero(rel):
    tax = rel.taxonomy()
    assert tax.folded_into("sawtooth_oscillation") == {
        "event": "mhd_n1",
        "categories": {1: 1, 2: 2},
    }
    assert tax.folded_into("mhd_n1") is None
    assert not tax.on_review_page("sawtooth_oscillation") and tax.on_review_page(
        "confinement"
    )
    assert tax.zero("confinement").startswith("the owner saved")
    assert "absent" in tax.zero("edge_localized_mode")
    assert tax.grid_rule


def test_unknown_event_suggests(rel):
    with pytest.raises(tk.UnknownEventError, match="confinement"):
        rel.taxonomy().display("confinment")


def test_a_different_grid_rule_is_refused():
    with pytest.raises(tk.SchemaError):
        tk.Taxonomy({"events": {"a": {}}, "grid": {"grid_ms": 50, "unassessed": -1}})
