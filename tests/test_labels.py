import warnings

import pytest

import tokamaksi as tk


def test_kind_follows_iscrowd(rel):
    elm = tk.load_labels("edge_localized_mode", shots=900001, release=rel)
    by = elm.groupby("kind").size().to_dict()
    assert by == {"span": 1, "individual": 3, "unspecified": 1}
    assert len(tk.spans(elm)) == 1 and len(tk.individuals(elm)) == 3
    absent = elm[elm.category_name == "absent"]
    assert set(absent.kind) == {"unspecified"} and absent.iscrowd.isna().all()
    assert str(elm.iscrowd.dtype) == "boolean"


def test_filters_and_sorting(rel):
    allrows = tk.load_labels(release=rel)
    assert list(allrows.shot) == sorted(allrows.shot)
    assert set(tk.load_labels(tier=["reviewed", "original"], release=rel).tier) == {
        "reviewed",
        "original",
    }
    one = tk.load_labels("confinement", tier="original", shots=900002, release=rel)
    assert list(one.shot) == [900002]
    assert list(tk.load_labels(tier="original", split="holdout", release=rel).shot) == [
        900005
    ]
    named = tk.load_labels(
        tier="original", dataset="confinement/format/example_lab_table", release=rel
    )
    assert len(named) == 3
    assert list(tk.load_labels("mhd_n1.m2", release=rel).event) == ["mhd_n1.m2"]
    assert set(tk.load_labels("mhd_n1", release=rel).event) == {
        "mhd_n1"
    }  # a parent has no children


def test_tier_never_falls_back(rel):
    # shot 900003 has reviewed rows (sawtooth) but no original ones.
    assert tk.load_labels(shots=900003, tier="original", release=rel).empty
    empty = tk.load_labels(shots=1, release=rel)
    assert empty.empty and "kind" in empty.columns


def test_unknown_event_lists_neighbours(rel):
    with pytest.raises(tk.UnknownEventError, match="edge_localized_mode"):
        tk.load_labels("edge_localised_mode", release=rel)
    with pytest.raises(ValueError, match="tier"):
        tk.load_labels(tier="gold", release=rel)


def test_dtypes_survive(rel):
    df = tk.load_labels("edge_localized_mode", release=rel)
    assert str(df.shot.dtype) == "int32" and str(df.category.dtype) == "int16"
    assert str(df.t_start_ms.dtype) == "float64"
    assert df.attrs["tokamaksi"]["release"] == "v1"


def test_at_covers_half_open_and_instants(rel):
    elm = tk.load_labels("edge_localized_mode", release=rel)
    assert set(tk.at(elm, 900001, 650).category_name) == {"present"}
    assert (
        len(tk.at(elm, 900001, 650)) == 1
    )  # the span only; 600-610 does not cover 650
    assert len(tk.at(elm, 900001, 605)) == 2
    assert len(tk.at(elm, 900001, 500)) == 1  # start is inside, 500 ends the absent row
    assert (
        len(tk.at(elm, 900001, 800)) == 2
    )  # the zero-length individual at 800, plus the span
    assert tk.at(elm, 900001, 5000).empty


def test_terms_warning_once_per_tier_and_for_test(rel):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        tk.load_labels(
            "edge_localized_mode", shots=900001, release=rel
        )  # reviewed train: silent
    with pytest.warns(tk.TermsWarning, match="source"):
        tk.load_labels(tier="source", release=rel)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        tk.load_labels(tier="source", release=rel)  # already shown
    with pytest.warns(tk.TermsWarning, match="evaluate"):
        tk.load_labels(split="test", release=rel)
    with pytest.warns(tk.TermsWarning, match="third-party"):
        tk.load_labels(tier="original", release=rel)
