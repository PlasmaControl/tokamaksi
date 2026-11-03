"""The README's quickstart names things that exist."""

import re
from pathlib import Path

import tokamaksi
from tokamaksi import cli

README = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
CODE = "\n".join(re.findall(r"^    (.+)$", README, flags=re.M))


def test_python_names_exist():
    chains = re.findall(r"(?<![\w/])(?:tk|tokamaksi)((?:\.[A-Za-z_]\w*)+)", CODE)
    assert chains
    for chain in chains:
        obj = tokamaksi
        for name in chain.strip(".").split("."):
            assert hasattr(obj, name), f"README uses tk{chain}, which does not exist"
            obj = getattr(obj, name)


def test_cli_subcommands_and_flags_exist():
    parser = cli.build_parser()
    subs = parser._subparsers._group_actions[0].choices
    used = re.findall(r"^\s*tokamaksi (\w+)(.*)$", CODE, flags=re.M)
    assert {name for name, _ in used} <= set(subs)
    for name, rest in used:
        known = {opt for action in subs[name]._actions for opt in action.option_strings}
        for flag in re.findall(r"--[\w-]+", rest):
            assert flag in known, f"`tokamaksi {name}` has no {flag}"
