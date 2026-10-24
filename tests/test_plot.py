import numpy as np
import pandas as pd
import pytest

import tokamaksi as tk

pytest.importorskip("matplotlib")
from matplotlib.colors import to_hex, to_rgb  # noqa: E402

from tokamaksi import plot  # noqa: E402


def _label_axes(fig):
    return fig.axes[-1]


def _hatched(ax):
    return [p for p in ax.patches if p.get_hatch()]


def test_rows_hatch_and_legend(rel, tmp_path):
    fig = plot.shot(900001, ["edge_localized_mode", "confinement"], release=rel)
    ax = _label_axes(fig)
    labels = [t.get_text() for t in ax.get_yticklabels()]
    assert labels == ["ELM [rev]", "confinement [rev]"]  # rows in the order asked
    assert len(_hatched(ax)) == 2  # unassessed background under every row
    legend = [t.get_text() for t in fig.legends[0].get_texts()]
    assert legend[0] == "unassessed" and {
        "span",
        "individual",
        "present",
        "absent",
    } <= set(legend)
    assert "uncertain" not in legend  # only what is drawn
    out = tmp_path / "shot.png"
    fig.savefig(out)
    assert out.stat().st_size > 2000


def test_unassessed_is_never_the_colour_of_zero(rel):
    ax = _label_axes(plot.shot(900001, ["edge_localized_mode"], release=rel))
    background = _hatched(ax)[0]
    assert background.get_gid() == "unassessed"
    assert background.get_facecolor()[:3] == (1.0, 1.0, 1.0)
    assert background.get_edgecolor()[:3] != (1.0, 1.0, 1.0)
    zero = [
        p
        for p in ax.patches
        if to_hex(p.get_facecolor()).lower() == plot.CATEGORY_COLOURS["absent"].lower()
    ]
    assert zero and all(not p.get_hatch() for p in zero)
    # The 0-500 ms absent row is drawn on top of the hatch, not instead of it.
    assert min(p.get_zorder() for p in zero) > background.get_zorder()
    assert (
        to_hex(background.get_edgecolor()).lower()
        != plot.CATEGORY_COLOURS["absent"].lower()
    )


def test_spans_and_individuals_differ_by_shape(rel):
    ax = _label_axes(plot.shot(900001, ["edge_localized_mode"], release=rel))
    by = {}
    for p in ax.patches:
        by.setdefault(p.get_gid(), []).append(p)
    assert len(by["span"]) == 1 and len(by["individual"]) == 3
    span, individual = by["span"][0], by["individual"][0]
    assert (
        not span.get_hatch() and span.get_alpha() is None
    )  # opaque: no hatch behind it
    assert to_hex(span.get_edgecolor()).lower() != plot.INK.lower()
    assert to_hex(individual.get_edgecolor()).lower() == plot.INK.lower()
    assert all(p.get_width() > 0 for p in by["individual"])  # the zero-length one shows
    assert len(ax.collections[0].get_offsets()) == 3  # one triangle per individual


def test_a_shot_without_labels_says_so(rel):
    fig = plot.shot(900003, ["edge_localized_mode"], release=rel)
    ax = _label_axes(fig)
    assert [t.get_text() for t in ax.texts] == ["no reviewed labels"]
    assert len(_hatched(ax)) == 1
    nothing = plot.shot(424242, release=rel)
    assert [t.get_text() for t in _label_axes(nothing).texts] == ["no reviewed labels"]


def test_no_tier_fallback(rel):
    # 900002 has original rows but no reviewed confinement: the reviewed row stays empty.
    ax = _label_axes(plot.shot(900002, ["confinement"], release=rel))
    assert [t.get_text() for t in ax.texts] == ["no reviewed labels"]
    both = _label_axes(
        plot.shot(900002, ["confinement"], tier=["reviewed", "original"], release=rel)
    )
    assert [t.get_text() for t in both.get_yticklabels()] == [
        "confinement [rev]",
        "confinement [orig]",
    ]


def test_subcategories_sit_under_their_parent(rel):
    ax = _label_axes(plot.shot(900002, release=rel))
    assert [t.get_text() for t in ax.get_yticklabels()] == [
        "mhd_n1 [rev]",
        "  mhd_n1.m2 [rev]",
    ]


def test_signals_stack_over_the_labels(rel, corpus):
    fig = plot.shot(
        900001,
        ["edge_localized_mode"],
        signals=["bolo:0,1", "neutron_rate:0"],
        release=rel,
        corpus=corpus,
        t_range=(0, 2000),
    )
    assert len(fig.axes) == 3  # two signal panels and the label rows
    top, mid, bottom = fig.axes
    assert top.get_ylabel() == "bolo[0,1]" and mid.get_ylabel() == "neutron_rate[0]"
    assert len(top.lines) == 2 and bottom.get_xlim() == (0.0, 2000.0) == top.get_xlim()
    with pytest.raises(tk.NoDataError):
        plot.shot(900001, signals=["pinj:0"], release=rel, corpus=corpus)


def test_timeline_alone(rel):
    labels = tk.load_labels("edge_localized_mode", shots=900001, release=rel)
    ax = plot.timeline(labels, taxonomy=rel.taxonomy())
    assert [t.get_text() for t in ax.get_yticklabels()] == ["ELM [rev]"]
    with pytest.raises(ValueError, match="one shot"):
        plot.timeline(tk.load_labels("edge_localized_mode", release=rel))
    assert np.allclose(ax.get_xlim(), (-30.0, 1530.0))


@pytest.mark.parametrize(
    "name, expected",
    [
        ("unassessed", True),
        ("Unassessed", True),
        ("unmarked (not assessed)", True),
        ("not_observable", True),
        ("not observable", True),
        ("Not  Observable", True),
        ("absent", False),
        ("uncertain", False),
        ("uncertain (segment below the confidence floor)", False),
        ("low (L)", False),
        ("present", False),
    ],
)
def test_which_category_names_mean_not_assessed(name, expected):
    assert plot._not_assessed(name) is expected


def test_not_assessed_rows_leave_the_hatch_showing(rel):
    # 900003 holds, in rule_a's table, present 100-300, unassessed 300-900 and
    # not_observable 900-1200; rule_b's table adds an individual at 150-160.
    fig = plot.shot(900003, ["resistive_wall_mode"], tier="derived", release=rel)
    ax = _label_axes(fig)
    boxes = [p for p in ax.patches if not p.get_hatch()]
    assert sorted((p.get_x(), p.get_width()) for p in boxes) == [(100, 200), (150, 10)]
    assert len(_hatched(ax)) == 2  # the hatch of each row is the only thing 300-1200
    legend = [t.get_text() for t in fig.legends[0].get_texts()]
    assert legend == ["unassessed or not observable", "present", "span", "individual"]


def test_a_tier_with_several_datasets_gets_a_row_each(rel):
    ax = _label_axes(
        plot.shot(900003, ["resistive_wall_mode"], tier="derived", release=rel)
    )
    assert [t.get_text() for t in ax.get_yticklabels()] == [
        "resistive_wall_mode [der] rule_a v2",
        "resistive_wall_mode [der] rule_b v1",
    ]
    # One dataset (or a dataset filter): no suffix.
    one = _label_axes(
        plot.shot(
            900003,
            ["resistive_wall_mode"],
            tier="derived",
            dataset="resistive_wall_mode/extend_rule_b/onsets",
            release=rel,
        )
    )
    assert [t.get_text() for t in one.get_yticklabels()] == [
        "resistive_wall_mode [der]"
    ]


def test_dataset_tags_fall_back_until_they_differ():
    def frame(*rows):
        return pd.DataFrame(rows, columns=["dataset", "method", "version"])

    by_method = plot._dataset_tags(
        frame(("a/x/one", "m1", "v1"), ("a/x/two", "m2", ""))
    )
    assert by_method == {"a/x/one": "m1 v1", "a/x/two": "m2"}
    same_method = plot._dataset_tags(frame(("a/x/one", "m", ""), ("a/x/two", "m", "")))
    assert same_method == {"a/x/one": "one", "a/x/two": "two"}
    same_name = plot._dataset_tags(
        frame(("a/one/data", "m", ""), ("a/two/data", "m", ""))
    )
    assert same_name == {"a/one/data": "a/one/data", "a/two/data": "a/two/data"}


def test_a_present_span_is_not_the_colour_of_absent():
    span = np.array(plot._tint(plot.CATEGORY_COLOURS["present"]))
    absent = np.array(to_rgb(plot.CATEGORY_COLOURS["absent"]))
    assert np.linalg.norm(span - absent) > 0.4


def _boxes(ax):
    """The drawn boxes as {(row from the bottom, start ms): colour}."""
    return {
        (round(p.get_y() - 0.1), p.get_x()): to_hex(p.get_facecolor()).lower()
        for p in ax.patches
        if not p.get_hatch()
    }


def test_a_producers_suffix_does_not_change_a_category_colour(rel):
    # 900004: the source table says qh / wpqh, the derived one `qh (QH)` / `wpqh (WPQH)`.
    fig = plot.shot(900004, ["confinement"], tier=["source", "derived"], release=rel)
    ax = _label_axes(fig)
    assert [t.get_text() for t in ax.get_yticklabels()] == [
        "confinement [src]",
        "confinement [der]",
    ]
    boxes = _boxes(ax)  # the source row is row 1 from the bottom, the derived row 0
    assert boxes[(1, 900.0)] == boxes[(0, 900.0)] == plot.CATEGORY_COLOURS["qh"].lower()
    assert (
        boxes[(1, 1200.0)]
        == boxes[(0, 1200.0)]
        == plot.CATEGORY_COLOURS["wpqh"].lower()
    )
    assert boxes[(0, 0.0)] == plot.CATEGORY_COLOURS["high"].lower()  # `high (H)`
    assert boxes[(0, 500.0)] == plot.CATEGORY_COLOURS["low"].lower()  # `low (L)`
    legend = fig.legends[0]
    names = [t.get_text() for t in legend.get_texts()]
    # `qh (QH)` is an abbreviation and adds nothing; the reason after `uncertain` stays.
    assert names == [
        "unassessed",
        "qh",
        "wpqh",
        "high",
        "low",
        "uncertain (segment below the confidence floor)",
    ]
    swatches = [to_hex(p.get_facecolor()).lower() for p in legend.get_patches()[1:]]
    assert len(set(swatches)) == len(swatches)  # no two entries share a colour


@pytest.mark.parametrize(
    "event, category, name, expected",
    [
        ("confinement", 1, "high (H)", "high"),
        ("confinement", 1, "high", "high"),
        ("confinement", 4, "wpqh (WPQH)", "wpqh"),
        (
            "confinement",
            5,
            "uncertain (segment below the confidence floor)",
            "uncertain",
        ),
        ("confinement", 2, "something else", "low"),  # the id decides, not the name
        ("mystery_event", 1, "high (H)", "high"),  # not in the taxonomy: strip
        ("confinement", 9, "new class (v2)", "new class"),  # unknown id: strip
        ("confinement", 9, "plain", "plain"),
    ],
)
def test_colours_go_by_the_taxonomy_name(rel, event, category, name, expected):
    assert plot._colour_name(rel.taxonomy(), event, category, name) == expected


@pytest.mark.parametrize(
    "name, expected",
    [("high (H)", "high"), ("marfe (candidate)", "marfe"), ("a (b) (c)", "a (b)")],
)
def test_without_a_taxonomy_a_trailing_parenthesis_is_dropped(name, expected):
    assert plot._colour_name(None, "confinement", 1, name) == expected


def test_an_unknown_name_never_takes_a_colour_in_use(rel):
    fixed = {c.lower() for c in plot.CATEGORY_COLOURS.values()}
    fallback = [c.lower() for c in plot.FALLBACK]
    assert len(set(fallback)) == len(fallback) and not set(fallback) & fixed
    colours = plot._Colours()
    unknown = [colours(f"mystery {i}") for i in range(len(fallback) + 4)]
    assert len({c.lower() for c in unknown}) == len(unknown)  # past the list too
    assert not {c.lower() for c in unknown} & fixed
    assert colours("mystery 0") == unknown[0]  # the same name keeps its colour
    # In a figure: two names the taxonomy cannot map beside known ones.
    labels = tk.load_labels("confinement", tier="derived", shots=900004, release=rel)
    names = {"wpqh (WPQH)": "zzz (new)", "qh (QH)": "yyy"}
    labels["category_name"] = labels["category_name"].replace(names)
    labels["category"] = 99  # no id the taxonomy knows
    ax = plot.timeline(labels)
    swatches = [
        to_hex(p.get_facecolor()).lower() for p in ax.get_legend().get_patches()
    ]
    assert len(set(swatches)) == len(swatches)


def test_default_rows_follow_the_tier_order_asked(rel):
    # Alphabetical order of tiers (original, reviewed, source) was the old default.
    ax = _label_axes(
        plot.shot(900001, tier=["source", "reviewed", "original"], release=rel)
    )
    assert [t.get_text() for t in ax.get_yticklabels()] == [
        "ELM [src]",
        "ELM [rev]",
        "confinement [rev]",
        "confinement [orig]",
    ]
    labels = tk.load_labels(
        None, tier=["reviewed", "original", "source"], shots=900001, release=rel
    )
    plain = plot.timeline(labels, taxonomy=rel.taxonomy())  # no order asked: TIER_ORDER
    assert [t.get_text() for t in plain.get_yticklabels()] == [
        "ELM [rev]",
        "ELM [src]",
        "confinement [rev]",
        "confinement [orig]",
    ]


def test_a_tuple_of_two_specs_is_two_specs():
    assert plot._spec_list(("filterscopes:1", "neutron_rate:3")) == [
        "filterscopes:1",
        "neutron_rate:3",
    ]
    for pair in (("bolo", 1), ("bolo", [0, 1]), ("bolo", (0, 1)), ("bolo", None)):
        assert plot._spec_list(pair) == [pair]
    assert plot._spec_list([("bolo", 1), "neutron_rate:0"]) == [
        ("bolo", 1),
        "neutron_rate:0",
    ]
    assert plot._spec_list("bolo:0,1") == ["bolo:0,1"]
    assert plot._spec_list(None) == []
    assert plot._spec_list(("bolo", ("neutron_rate", 0))) == [
        "bolo",
        ("neutron_rate", 0),
    ]


def test_signals_given_as_a_tuple_of_strings_stack(rel, corpus):
    fig = plot.shot(
        900001,
        ["edge_localized_mode"],
        signals=("bolo:0", "neutron_rate:1"),
        release=rel,
        corpus=corpus,
        t_range=(0, 2000),
    )
    assert [ax.get_ylabel() for ax in fig.axes[:2]] == ["bolo[0]", "neutron_rate[1]"]
    pair = plot.shot(
        900001,
        signals=("bolo", [0, 2]),
        release=rel,
        corpus=corpus,
        t_range=(0, 2000),
    )
    assert pair.axes[0].get_ylabel() == "bolo[0,2]" and len(pair.axes) == 2


def _frame(event, *rows, tier="derived", layer="extend"):
    """Labels of one shot as `load_labels` gives them: `(category, name, start, stop)`."""
    return pd.DataFrame(
        [
            {
                "shot": 900100,
                "event": event,
                "category": category,
                "category_name": name,
                "t_start_ms": float(start),
                "t_end_ms": float(stop),
                "kind": "unspecified",
                "tier": tier,
                "layer": layer,
                "dataset": f"{event}/{layer}/made_up",
                "method": "made_up",
                "version": "",
            }
            for category, name, start, stop in rows
        ]
    )


def _legend(obj):
    """The legend entries of a figure (`plot.shot`) or of an Axes (`plot.timeline`)."""
    legend = obj.legends[0] if hasattr(obj, "legends") else obj.get_legend()
    return [t.get_text() for t in legend.get_texts()]


def test_category_zero_of_confinement_outside_the_owners_save_is_not_assessed(rel):
    # 900004's source draft says category 0 `absent` on 0-900 ms, but the release
    # reads that as "nothing marked" (taxonomy zero: the owner saved no regime here).
    fig = plot.shot(900004, ["confinement"], tier=["source", "derived"], release=rel)
    ax = _label_axes(fig)
    boxes = _boxes(ax)  # the source row is row 1 from the bottom
    assert (1, 0.0) not in boxes and (1, 900.0) in boxes
    assert plot.CATEGORY_COLOURS["absent"].lower() not in boxes.values()
    assert len(_hatched(ax)) == 2  # one hatch per row, nothing painted over 0-900
    assert "absent" not in _legend(fig)


def test_zero_in_the_owners_save_is_painted_with_the_taxonomys_words(rel):
    fig = plot.shot(900004, ["confinement"], tier="reviewed", release=rel)
    ax = _label_axes(fig)
    assert _boxes(ax)[(0, 1500.0)] == plot.CATEGORY_COLOURS["absent"].lower()
    words = rel.taxonomy().zero("confinement")
    assert words == "the owner saved no regime here"
    assert _legend(fig) == ["unassessed", words, "high"]  # not `absent`
    # The same rows read from any other table are unassessed again.
    taxonomy = rel.taxonomy()
    labels = tk.load_labels("confinement", tier="reviewed", shots=900004, release=rel)
    other = labels.assign(tier="original", layer="format")
    assert (0, 1500.0) not in _boxes(plot.timeline(other, taxonomy=taxonomy))
    assert (0, 1700.0) in _boxes(plot.timeline(other, taxonomy=taxonomy))


def test_zero_that_means_absent_is_painted_in_every_tier(rel):
    # An ELM 0 is "assessed, not seen" wherever it comes from; the taxonomy says so.
    labels = tk.load_labels("edge_localized_mode", shots=900001, release=rel)
    assert rel.taxonomy().zero("edge_localized_mode").startswith("absent")
    for tier, layer in (("source", "source"), ("derived", "extend")):
        ax = plot.timeline(
            labels.assign(tier=tier, layer=layer), taxonomy=rel.taxonomy()
        )
        assert _boxes(ax)[(0, 0.0)] == plot.CATEGORY_COLOURS["absent"].lower()
        assert "absent" in _legend(ax)
    # Without a taxonomy there is nothing to say a 0 means less: it is painted.
    assert (0, 0.0) in _boxes(plot.timeline(labels.assign(tier="source")))


@pytest.mark.parametrize(
    "key, name, expected",
    [
        ("high", "high", "high"),
        ("high", "high (H)", "high"),
        ("low", "low (L)", "low"),
        ("qh", "qh (QH)", "qh"),
        ("wpqh", "wpqh (WPQH)", "wpqh"),
        ("high", "High", "high"),
        ("marfe", "marfe (candidate)", "marfe (candidate)"),
        (
            "uncertain",
            "uncertain (segment below the confidence floor)",
            "uncertain (segment below the confidence floor)",
        ),
        ("low", "something else", "something else"),
    ],
)
def test_the_legend_keeps_a_producers_qualifier(key, name, expected):
    assert plot._legend_name(key, name) == expected


def test_a_candidate_keeps_its_word_in_the_legend_with_the_colour_of_its_class():
    labels = _frame(
        "detachment",
        (3, "marfe (candidate)", 0, 100),
        (3, "marfe", 100, 200),
        (2, "detached", 200, 300),
    )
    ax = plot.timeline(labels)
    assert _legend(ax) == ["unassessed", "marfe (candidate)", "marfe", "detached"]
    boxes = _boxes(ax)
    assert (
        boxes[(0, 0.0)] == boxes[(0, 100.0)] == plot.CATEGORY_COLOURS["marfe"].lower()
    )


def test_names_that_mean_not_assessed_are_found_by_the_taxonomy_too(rel):
    # The raw name carries a suffix or a space, but the id is the taxonomy's
    # not_observable: still hatch, still out of the legend.
    labels = _frame(
        "edge_localized_mode",
        (3, "not_observable (ECE cutoff)", 0, 100),
        (3, "not observable", 100, 200),
        (1, "present", 200, 300),
    )
    ax = plot.timeline(labels, taxonomy=rel.taxonomy())
    assert sorted(_boxes(ax)) == [(0, 200.0)]
    assert _legend(ax) == ["unassessed or not observable", "present"]
    # With no taxonomy the trailing parenthesis is dropped, which is enough here.
    bare = plot.timeline(labels.iloc[[0]])
    assert _boxes(bare) == {} and _legend(bare) == ["unassessed or not observable"]


def test_q_min_hybrid_and_elevated_have_fixed_colours_in_a_ramp():
    ramp = [plot.CATEGORY_COLOURS[n] for n in ("low", "hybrid", "elevated", "high")]
    assert len({c.lower() for c in ramp}) == 4
    assert all(c.lower() not in {f.lower() for f in plot.FALLBACK} for c in ramp)

    # Colour-blind check: CIELAB distance after simulating protan, deutan and tritan
    # vision (Machado et al. 2009, severity 1) between every pair of a q_min figure.
    simulate = {
        "protan": [[0.152286, 1.052583, -0.204868], [0.114503, 0.786281, 0.099216], [-0.003882, -0.048116, 1.051998]],
        "deutan": [[0.367322, 0.860646, -0.227968], [0.280085, 0.672501, 0.047413], [-0.011820, 0.042940, 0.968881]],
        "tritan": [[1.255528, -0.076749, -0.178779], [-0.078411, 0.930809, 0.147602], [0.004733, 0.691367, 0.303900]],
    }  # fmt: skip
    xyz = np.array(
        [[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]]
    )

    def lab(colour, matrix):
        rgb = np.array(to_rgb(colour))
        linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
        linear = np.clip(np.array(matrix) @ linear, 0, 1)
        f = (xyz @ linear) / np.array([0.95047, 1.0, 1.08883])
        f = np.where(f > 0.008856, np.cbrt(f), 7.787 * f + 16 / 116)
        return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])

    shown = ramp + [plot.CATEGORY_COLOURS[n] for n in ("uncertain", "absent")]
    for kind, matrix in simulate.items():
        for i, one in enumerate(shown):
            for other in shown[:i]:
                gap = np.linalg.norm(lab(one, matrix) - lab(other, matrix))
                assert gap > 15, (kind, one, other, gap)


def test_a_figure_with_q_min_classes_uses_the_ramp():
    labels = _frame(
        "minimum_safety_factor",
        (1, "low", 0, 100),
        (2, "hybrid", 100, 200),
        (3, "elevated", 200, 300),
        (4, "high", 300, 400),
    )
    ax = plot.timeline(labels)
    assert _legend(ax) == ["unassessed", "low", "hybrid", "elevated", "high"]
    assert [to_hex(p.get_facecolor()).lower() for p in ax.get_legend().get_patches()][
        1:
    ] == [
        plot.CATEGORY_COLOURS[n].lower() for n in ("low", "hybrid", "elevated", "high")
    ]


def test_fallback_colours_go_by_the_categorys_position_not_the_draw_order():
    from tokamaksi.taxonomy import Taxonomy

    taxonomy = Taxonomy(
        {
            "events": {
                "newcomer": {
                    "categories": {0: "absent", 1: "zz1", 2: "zz2", 3: "zz3", 4: "zz4"}
                }
            }
        }
    )

    def colours(*rows):
        ax = plot.timeline(_frame("newcomer", *rows), taxonomy=taxonomy)
        patches = ax.get_legend().get_patches()
        return dict(
            zip(_legend(ax), (to_hex(p.get_facecolor()).lower() for p in patches))
        )

    first = colours((1, "zz1", 0, 10), (2, "zz2", 10, 20), (4, "zz4", 20, 30))
    again = colours((4, "zz4", 0, 10), (3, "zz3", 10, 20), (2, "zz2", 20, 30))
    assert first["zz2"] == again["zz2"] and first["zz4"] == again["zz4"]
    assert len({first["zz1"], first["zz2"], first["zz4"], again["zz3"]}) == 4
    assert first["zz1"] == plot.FALLBACK[1].lower()  # its position in the taxonomy
    # The same class on its own is the same colour again.
    assert colours((2, "zz2", 0, 10))["zz2"] == first["zz2"]


def test_an_unknown_name_of_another_event_never_shares_a_colour_in_a_figure():
    from tokamaksi.taxonomy import Taxonomy

    taxonomy = Taxonomy(
        {
            "events": {
                "first": {"categories": {0: "absent", 1: "yy1"}},
                "second": {"categories": {0: "absent", 1: "ww1"}},
            }
        }
    )
    labels = pd.concat(
        [_frame("first", (1, "yy1", 0, 10)), _frame("second", (1, "ww1", 0, 10))]
    )
    patches = plot.timeline(labels, taxonomy=taxonomy).get_legend().get_patches()
    colours = [to_hex(p.get_facecolor()).lower() for p in patches]
    assert len(set(colours)) == len(colours)  # same position, two names, two colours


def test_a_bad_spec_in_a_list_is_named_before_anything_is_read(rel, corpus):
    with pytest.raises(ValueError, match="bad signal spec 3"):
        plot.shot(900001, signals=["bolo:0", 3], release=rel, corpus=corpus)
    with pytest.raises(ValueError, match="bad signal spec"):
        plot.shot(900001, signals=["bolo", 3], release=rel, corpus=corpus)
