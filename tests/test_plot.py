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
