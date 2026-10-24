import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "tokamaksi"


def test_import_is_light_and_stands_alone():
    code = (
        "import sys, tokamaksi\n"
        "heavy = ['matplotlib', 'huggingface_hub', 'h5py', 'torch', 'toksearch']\n"
        "own = ['labeler', 'shot_design', 'tokamak_foundation_model', 'faith']\n"
        "print([m for m in heavy + own if m in sys.modules])\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "[]"


def test_source_imports_nothing_from_fusionaihub():
    banned = ("labeler", "shot_design", "tokamak_foundation_model", "faith")
    for path in SRC.glob("*.py"):
        for line in path.read_text(encoding="utf-8").splitlines():
            words = line.split()
            if words[:1] in (["import"], ["from"]) and len(words) > 1:
                assert words[1].split(".")[0] not in banned, f"{path.name}: {line}"
