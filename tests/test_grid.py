import numpy as np
import pytest
from release_factory import GOLD, SILVER

import tokamaksi as tk


def test_decode_matches_a_hand_built_grid(rel):
    shot = rel.grid("silver").decode(SILVER[0])
    assert shot.events == ("edge_localized_mode", "confinement")
    assert shot.t0_ms == 0 and shot.frames.dtype == np.int8
    expect = np.full((2, 230), -1, dtype=np.int8)
    expect[0, 0:50], expect[0, 50:150], expect[0, 200:230] = (
        0,
        1,
        0,
    )  # 150..199 unassessed
    expect[1, 20:100] = 2  # confinement starts at frame 20
    assert np.array_equal(shot.frames, expect)
    assert (
        shot.categories("edge_localized_mode")[150:200] == tk.schema.UNASSESSED
    ).all()
    assert shot.time_ms[0] == 0 and shot.time_ms[3] == 30


def test_unassessed_is_not_zero(rel):
    shot = rel.grid("gold").decode(GOLD[0])
    gap = shot.assessed("edge_localized_mode")[150:200]
    assert not gap.any()
    assert shot.assessed("edge_localized_mode")[0]  # category 0, assessed
    assert shot.at(1700) == {"edge_localized_mode": -1, "confinement": -1}  # the gap
    assert shot.at(2000) == {"edge_localized_mode": 0, "confinement": -1}
    assert shot.at(600) == {"edge_localized_mode": 1, "confinement": 2}
    assert shot.at(10_000) == {"edge_localized_mode": -1, "confinement": -1}


def test_events_subset_lines_up_with_the_full_decode(rel):
    grid = rel.grid("silver")
    full = grid.decode(SILVER[1])
    part = grid.decode(SILVER[1], events=["confinement"])
    assert part.events == ("confinement",) and part.t0_ms == full.t0_ms
    assert np.array_equal(part.frames[0], full.categories("confinement"))
    empty = grid.decode(SILVER[1], events=["detachment"])
    assert (empty.frames == -1).all() and empty.frames.shape == (
        1,
        full.frames.shape[1],
    )


def test_missing_grid_and_shot_are_named(rel):
    with pytest.raises(tk.MissingPartError, match="gold, silver"):
        rel.grid("holdout")
    with pytest.raises(tk.MissingPartError, match="not in the gold grid"):
        rel.grid("gold").decode(1)
    assert rel.grid("gold").shots == GOLD
