"""The all-shot 1 ms model labels (`tokamaksi.fdp`), on a synthetic root.

Nothing here is real: the shots are 9000xx and every number is made up. The files
are written into the test's temporary folder and never committed.
"""

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import tokamaksi as tk
from tokamaksi import fdp
from tokamaksi.schema import pulled

STATUSES = {"labeled": "run", "no_fs": "no filterscope record"}


def _database(folder: Path, shots: list[dict], spans: list[dict], meta: dict) -> Path:
    """One database: the four tables and a README, as the producers write them."""
    folder.mkdir(parents=True)
    pd.DataFrame(shots).astype(
        {"shot": "int32", "window_start_ms": "Int32", "window_end_ms": "Int32"}
    ).to_parquet(folder / "shots.parquet")
    frame = pd.DataFrame(spans, columns=list(spans[0]) if spans else None)
    frame.astype(
        {"shot": "int32", "t_start_ms": "int32", "t_end_ms": "int32"}
    ).to_parquet(folder / "spans.parquet")
    labeled = {row["shot"] for row in shots if row["status"] == "labeled"}
    pd.DataFrame(
        {
            "shot": range(900000, 900004),
            "valid": [int(s in labeled) for s in range(900000, 900004)],
        }
    ).to_csv(folder / "valid.csv", index=False)
    (folder / "meta.json").write_text(json.dumps(meta))
    (folder / "README.md").write_text("synthetic\n")
    return folder


def make_root(root: Path) -> Path:
    """A binary event in a named subfolder (the cluster layout), a categorical one
    straight under its event folder (the release layout) and a folder that is not a
    database (the curated tables sit beside the databases on the cluster)."""
    _database(
        root / "edge_localized_mode" / "elm_synthetic_1ms",
        [
            {
                "shot": 900001,
                "status": "labeled",
                "window_start_ms": 100,
                "window_end_ms": 200,
                "notch": True,
            },
            {
                "shot": 900002,
                "status": "no_fs",
                "window_start_ms": None,
                "window_end_ms": None,
                "notch": False,
            },
        ],
        [
            {"shot": 900001, "t_start_ms": 120, "t_end_ms": 130},
            {"shot": 900001, "t_start_ms": 150, "t_end_ms": 160},
        ],
        {
            "event": "edge_localized_mode",
            "statuses": STATUSES,
            "caveats": ["synthetic"],
        },
    )
    _database(
        root / "confinement",
        [
            {
                "shot": 900001,
                "status": "labeled",
                "window_start_ms": 0,
                "window_end_ms": 50,
            }
        ],
        [
            {"shot": 900001, "t_start_ms": 10, "t_end_ms": 20, "category": 1},
            {"shot": 900001, "t_start_ms": 30, "t_end_ms": 40, "category": 5},
        ],
        {"event": "confinement", "categories": {"1": "H", "5": "uncertain"}},
    )
    (root / "events" / "edge_localized_mode").mkdir(parents=True)
    return root


@pytest.fixture
def root(tmp_path) -> Path:
    return make_root(tmp_path / "labels")


def test_where_the_labels_are_found(root, monkeypatch, tmp_path, synthetic):
    assert fdp.events(root) == ["confinement", "edge_localized_mode"]
    assert fdp.open("confinement", root).path == root / "confinement"
    with pytest.raises(tk.MissingPartError, match="TOKAMAKSI_FDP_LABELS"):
        fdp.open("confinement")
    monkeypatch.setenv("TOKAMAKSI_FDP_LABELS", str(root))
    elm = fdp.open("edge_localized_mode")
    assert elm.path.name == "elm_synthetic_1ms" and elm.meta["caveats"] == ["synthetic"]
    monkeypatch.setenv("TOKAMAKSI_FDP_LABELS", str(tmp_path / "unmounted"))
    with pytest.raises(tk.MissingPartError, match="not a folder"):
        fdp.events()
    # Neither: a release on disk that ships fdp_1ms/.
    monkeypatch.delenv("TOKAMAKSI_FDP_LABELS")
    release = shutil.copytree(synthetic, tmp_path / "v1")
    shutil.copytree(root, release / "fdp_1ms")
    assert (
        fdp.open("confinement", release=release).path
        == release / "fdp_1ms" / "confinement"
    )
    assert pulled("fdp_1ms/confinement/spans.parquet") and not pulled("tables/x.csv")


def test_decode_window_absent_present_and_outside(root):
    elm = fdp.open("edge_localized_mode", root)
    assert list(elm.labeled) == [900001] and elm.categories == fdp.BINARY
    t_ms, codes = elm.decode(900001)
    assert t_ms[0] == 100 and t_ms[-1] == 199 and codes.dtype == np.int8
    assert codes.sum() == 20 and codes[20:30].all() and not codes[:20].any()
    t_ms, codes = elm.decode(900001, t_range=(90, 210))
    assert t_ms[0] == 90 and len(codes) == 120
    assert (codes[:10] == -1).all() and (codes[-10:] == -1).all() and codes[10] == 0
    assert elm.at(900001, 125.5) == 1 and elm.at(900001, 130) == 0
    assert elm.at(900001, 99) == -1 and elm.at(900001, 200) == -1
    with pytest.raises(tk.NoDataError, match="no_fs \\(no filterscope record\\)"):
        elm.decode(900002)
    with pytest.raises(tk.NoDataError, match="not a plasma candidate"):
        elm.decode(900003)

    conf = fdp.open("confinement", root)
    t_ms, codes = conf.decode(900001)
    assert codes[15] == 1 and codes[35] == 5 and codes[25] == 0 and len(codes) == 50
    assert conf.categories == {1: "H", 5: "uncertain"}
    assert conf.valid["valid"].tolist() == [0, 1, 0, 0]


def test_a_missing_or_doubled_database_is_an_error(root):
    with pytest.raises(tk.UnknownEventError, match="confinement, edge_localized_mode"):
        fdp.open("alfven_eigenmode", root)
    (root / "alfven_eigenmode").mkdir()
    with pytest.raises(tk.NoDataError, match="not published"):
        fdp.open("alfven_eigenmode", root)
    shutil.copytree(
        root / "edge_localized_mode" / "elm_synthetic_1ms",
        root / "edge_localized_mode" / "elm_other_1ms",
    )
    with pytest.raises(tk.NoDataError, match="2 databases"):
        fdp.open("edge_localized_mode", root)


def test_plot_adds_a_model_row(root, rel, monkeypatch):
    pytest.importorskip("matplotlib")
    monkeypatch.setenv("TOKAMAKSI_FDP_LABELS", str(root))
    fig = tk.plot.shot(
        900001, ["edge_localized_mode", "confinement"], release=rel, model=True
    )
    rows = [t.get_text() for t in fig.axes[-1].get_yticklabels()]
    assert rows == [
        "ELM [rev]",
        "ELM [1 ms model, notched]",
        "confinement [rev]",
        "confinement [1 ms model]",
    ]
