import pytest

from tokamaksi import cli, hf


def run(capsys, *argv):
    code = cli.main(list(argv))
    out = capsys.readouterr()
    return code, out.out, out.err


def test_info_and_check(capsys, synthetic):
    code, out, _ = run(capsys, "info", "--release", str(synthetic), "--check")
    assert code == 0 and "release   v1   schema 1" in out
    assert "reviewed/train" in out and "check     clean" in out


def test_plot_writes_a_png(capsys, synthetic, tmp_path):
    png = tmp_path / "shot.png"
    code, out, _ = run(
        capsys,
        "plot",
        "900006",
        "--release",
        str(synthetic),
        "--events",
        "edge_localized_mode",
        "--out",
        str(png),
    )
    assert code == 0 and png.stat().st_size > 2000
    assert "do not train or tune" in out  # a gold shot


def test_plot_with_signals(capsys, synthetic, corpus, tmp_path):
    png = tmp_path / "shot.png"
    code, _, _ = run(
        capsys,
        "plot",
        "900001",
        "--release",
        str(synthetic),
        "--corpus",
        str(corpus),
        "--signals",
        "bolo:0,neutron_rate:0",
        "--t-range",
        "0",
        "2000",
        "--out",
        str(png),
    )
    assert code == 0 and png.stat().st_size > 2000


def test_signal_specs_keep_their_channel_lists():
    assert cli._split_signals("neutron_rate:0,bolo:0,1,2,filterscopes") == [
        "neutron_rate:0",
        "bolo:0,1,2",
        "filterscopes",
    ]
    assert cli._split_signals(None) is None


def test_pull_with_a_fake_backend(capsys, monkeypatch, synthetic):
    import shutil

    monkeypatch.delenv("TOKAMAKSI_OFFLINE")
    monkeypatch.setattr(
        hf,
        "hub_backend",
        lambda repo, tag, dest, pats: (
            shutil.copytree(synthetic, dest, dirs_exist_ok=True) and "abc"
        ),
    )
    code, out, _ = run(capsys, "pull", "v1")
    assert code == 0 and str(hf.release_dir("v1")) in out and "evaluation" in out
    assert (hf.release_dir("v1") / "release.json").is_file()


@pytest.mark.parametrize(
    "argv, text",
    [
        (["info", "--release", "v9"], "tokamaksi pull v9"),  # offline in the tests
        (["plot", "1", "--events", "nope", "--out", "x.png"], "unknown event"),
        (["plot", "1", "--tier", "gold", "--out", "x.png"], "unknown tier"),
    ],
)
def test_user_errors_exit_two_without_a_traceback(
    capsys, synthetic, monkeypatch, argv, text
):
    monkeypatch.setenv("TOKAMAKSI_RELEASE", str(synthetic))
    code, out, err = run(capsys, *argv)
    assert code == 2 and text in err and "Traceback" not in err and out == ""
