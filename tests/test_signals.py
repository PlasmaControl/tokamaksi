import numpy as np
import pytest

import tokamaksi as tk
from tokamaksi import signals


def test_parse_specs():
    assert signals.parse("bolo") == ("bolo", None)
    assert signals.parse("bolo:2") == ("bolo", (2,))
    assert signals.parse("bolo:0, 2,5") == ("bolo", (0, 2, 5))
    assert signals.parse(("bolo", 3)) == ("bolo", (3,))
    assert signals.parse(("bolo", [0, 1])) == ("bolo", (0, 1))
    with pytest.raises(ValueError, match="bad signal spec"):
        signals.parse("bolo:x")


def test_read_gives_ms_and_the_asked_channels(corpus):
    trace = signals.read(900001, "bolo:1,2", corpus=corpus)
    assert (
        trace.group == "bolo"
        and trace.channels == (1, 2)
        and trace.label == "bolo[1,2]"
    )
    assert trace.y.shape == (2, 15000) and trace.y.dtype == np.float32
    assert (
        trace.t_ms[0] == 0 and abs(trace.t_ms[-1] - 2999.8) < 0.01
    )  # seconds in, ms out
    assert not trace.decimated


def test_t_range_is_a_slice_not_a_filter(corpus):
    trace = signals.read(900001, "neutron_rate:0", t_range=(100, 200), corpus=corpus)
    # The file stores float32 seconds, so the edges are only good to about a microsecond.
    assert 100 <= trace.t_ms[0] < 100.001 and 198.9 < trace.t_ms[-1] <= 200
    assert trace.y.shape[0] == 1 and 99 <= trace.y.shape[1] <= 101
    with pytest.raises(tk.NoDataError, match="no samples"):
        signals.read(900001, "neutron_rate:0", t_range=(9000, 9500), corpus=corpus)


def test_envelope_keeps_the_extremes(corpus):
    full = signals.read(900001, "ece:1", corpus=corpus)
    thin = signals.read(900001, "ece:1", corpus=corpus, max_points=100)
    assert thin.decimated and thin.t_ms.shape == (200,) == thin.y.shape[1:]
    assert thin.y.max() == full.y.max() and thin.y.min() == full.y.min()
    assert np.all(np.diff(thin.t_ms) >= 0)


def test_what_is_there_and_what_is_not(corpus, monkeypatch):
    assert signals.available(900001, corpus) == {
        "neutron_rate": (2, 3000),
        "bolo": (3, 15000),
        "ece": (2, 150000),
        "filterscopes": (4, 6000),
    }
    with pytest.raises(tk.NoDataError, match="sentinel"):
        signals.read(900001, "pinj:0", corpus=corpus)
    with pytest.raises(tk.NoDataError, match="no group 'sxr'"):
        signals.read(900001, "sxr:0", corpus=corpus)
    with pytest.raises(tk.NoDataError, match="900002"):
        signals.read(900002, "bolo:0", corpus=corpus)
    with pytest.raises(ValueError, match="3 channels"):
        signals.read(900001, "bolo:7", corpus=corpus)
    monkeypatch.setattr(signals, "MAX_BARE_CHANNELS", 2)
    with pytest.raises(ValueError, match="name some"):
        signals.read(900001, "bolo", corpus=corpus)


def test_no_corpus_configured_says_how(monkeypatch, corpus):
    with pytest.raises(tk.MissingPartError, match="TOKAMAKSI_CORPUS"):
        signals.read(900001, "bolo:0")
    monkeypatch.setenv("TOKAMAKSI_CORPUS", str(corpus))
    assert signals.read(900001, "bolo:0").y.shape[0] == 1


def test_live_fetching_is_a_clear_stub(corpus):
    with pytest.raises(
        (tk.MissingPartError, NotImplementedError), match="fdp|milestone 2"
    ):
        signals.read(900001, "bolo:0", corpus=corpus, source="fdp")
    with pytest.raises(ValueError, match="source"):
        signals.read(900001, "bolo:0", corpus=corpus, source="tape")


def test_channels_that_recorded_nothing_are_not_drawn_as_data(corpus):
    bare = signals.read(900001, "filterscopes", corpus=corpus)
    assert (
        bare.channels == (2, 3) and bare.y.shape[0] == 2 and not np.isnan(bare.y).any()
    )
    with pytest.raises(tk.NoDataError, match=r"channel\(s\) \[0\] hold only NaN"):
        signals.read(900001, "filterscopes:0,2", corpus=corpus)
    thin = signals.read(900001, "filterscopes:2", corpus=corpus, max_points=50)
    assert thin.decimated and not np.isnan(thin.y).any()


def test_the_envelope_skips_nan_gaps():
    t = np.arange(1000, dtype="float64")
    y = np.vstack([np.where(t < 500, np.nan, np.sin(t))]).astype("float32")
    _, thin = signals.envelope(t, y, 10)
    assert np.isnan(thin[0, :10]).all() and not np.isnan(thin[0, 10:]).any()


def test_the_many_channels_hint_names_live_channels(corpus, monkeypatch):
    # filterscopes channels 0 and 1 recorded nothing: the example must not name them.
    monkeypatch.setattr(signals, "MAX_BARE_CHANNELS", 2)
    with pytest.raises(ValueError, match=r"filterscopes:2,3") as caught:
        signals.read(900001, "filterscopes", corpus=corpus)
    assert "filterscopes:0" not in str(caught.value)
    # An all-NaN stretch finds nothing to name and says so.
    assert "none showed data" in signals._name_some("g", 40, [])
    assert "..." in signals._name_some("g", 40, list(range(20)))
