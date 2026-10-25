"""Every `shapesmith` command the README names exists with the options it is given (readme_check.py)."""
from pathlib import Path

from shapesmith.cli import app
from shapesmith.readme_check import cli_options, mentions, problems

README = Path(__file__).resolve().parents[1] / "README.md"


def test_the_readme_names_only_commands_and_options_of_the_cli():
    text = README.read_text()
    assert len(mentions(text, set(cli_options(app)))) > 30  # the steps block and the inline mentions
    assert problems(text, app) == []


def test_unknown_commands_and_options_are_found():
    text = "\n".join([
        "```bash",
        "shapesmith datacards -c run.yaml --variant s0 --no-systematics && shapesmith fit -c run.yaml --unblinded   # --not-an-option",
        "shapesmith limits -c run.yaml",
        "```",
        "Run `fit --background-only --masks`, then `combine -M AsymptoticLimits --toysFrequentist` (the combine tool).",
    ])
    assert problems(text, app) == [
        "`shapesmith fit` has no option --unblinded, in: shapesmith datacards -c run.yaml --variant s0 --no-systematics && shapesmith fit -c run.yaml --unblinded   # --not-an-option",
        "unknown command `shapesmith limits` in: shapesmith limits -c run.yaml",
        "`shapesmith fit` has no option --masks, in: fit --background-only --masks",
    ]
