"""The combination of the datacards of several runs (v7 spec 6.9: one run per era of Run 2): combineCards.py of the
card of one final state of every run, then the limit chain (limits.py) or the background-only fit (background_fit.py)
on the combined card, with their summaries.

The bins of every card keep their names (combineCards' label "."): they carry the era already
(datacards.bin_name), so that the cards of different eras never share a bin; two runs of one era are an error. The
shape files are referenced with absolute paths, so the combination lives in its own directory. nuisances.md lists
every nuisance with the runs it appears in and flags a name that ends in the token of another run's era (e.g.
`_2017` in the 2018 card), the check of spec 9.9 for nuisances doubled by mistake; a name ending in a prefix of the
run's own era (`_2016` in a 2016preVFP card) is the shared one of a year.
"""
from __future__ import annotations

import json
import logging
import shlex
from dataclasses import dataclass
from pathlib import Path

from shapesmith import cmssw
from shapesmith.background_fit import OUTPUT as FIT_OUTPUT
from shapesmith.background_fit import fit_commands, summarise_bins, summary_lines
from shapesmith.config import RunConfig
from shapesmith.datacards import bin_name
from shapesmith.limits import collect, limit_commands, write_summary
from shapesmith.model import Analysis

logger = logging.getLogger(__name__)
CARD = "combined.txt"
VALIDATION = "validation.json"


@dataclass(frozen=True)
class Run:
    """A run of the combination: its configuration, its analysis and the card of the chosen final state."""

    config: RunConfig
    analysis: Analysis
    card: Path

    @property
    def era(self) -> str:
        return self.analysis.era


def run_card(config: RunConfig, final_state: str) -> Path:
    """The datacard of a final state of a run: <output_dir>/datacards/<final state>/combined.txt."""
    return Path(config.output_dir) / "datacards" / final_state / CARD


def check_runs(runs: list[Run]) -> None:
    """Every card exists and every era appears once."""
    missing = [str(run.card) for run in runs if not run.card.exists()]
    if missing:
        raise FileNotFoundError("datacards missing (run `shapesmith datacards` for every era first):\n" + "\n".join(missing))
    eras = [run.era for run in runs]
    repeated = sorted({era for era in eras if eras.count(era) > 1})
    if repeated:
        raise ValueError(f"the eras {', '.join(repeated)} appear in more than one run; their bins would clash")


def combine_cards_command(cards: list[Path], output: str = CARD) -> str:
    """combineCards.py with the label "." (the bin names stay) and absolute card paths (absolute shape files)."""
    return "combineCards.py " + " ".join(shlex.quote(f".={Path(card).resolve()}") for card in cards) + f" > {output}"


def validate_command(card: str = CARD) -> str:
    """CombineHarvester's ValidateDatacards.py on the combined card, its findings in validation.json."""
    return f"ValidateDatacards.py {card} --jsonFile {VALIDATION} --printLevel 1"


def card_nuisances(card: Path) -> list[str]:
    """The nuisance names of a text datacard (lnN, shape and the like; rateParams and autoMCStats excluded)."""
    kinds = {"lnN", "shape", "shapeN", "shape?", "lnU", "gmN", "param"}
    names = []
    for line in Path(card).read_text().splitlines():
        tokens = line.split()
        if len(tokens) > 1 and tokens[1] in kinds and not line.startswith(("bin ", "process ", "rate ", "observation ")):
            names.append(tokens[0])
    return names


def foreign_token(name: str, era: str, eras: list[str]) -> str | None:
    """The era of another run that a nuisance name of the run of `era` ends in (None if none): `_<other>` where the
    other era is neither a prefix of the run's own era (a year shared by two halves) nor an extension of it."""
    for other in sorted(eras, key=len, reverse=True):
        if other != era and name.endswith(f"_{other}") and not era.startswith(other) and not other.startswith(era):
            return other
    return None


def nuisance_table(runs: list[Run]) -> tuple[dict[str, list[str]], list[str]]:
    """Nuisance -> the eras whose cards have it, and the flagged names (`name (era card): ends in another era`)."""
    eras = [run.era for run in runs]
    table: dict[str, list[str]] = {}
    flagged = []
    for run in runs:
        for name in card_nuisances(run.card):
            table.setdefault(name, []).append(run.era)
            other = foreign_token(name, run.era, eras)
            if other is not None:
                flagged.append(f"{name} ({run.era} card): ends in the era {other} of another run")
    return table, flagged


def write_nuisance_report(runs: list[Run], output: Path) -> Path:
    table, flagged = nuisance_table(runs)
    eras = [run.era for run in runs]
    lines = [f"# Nuisances of the combination of {', '.join(eras)}", ""]
    lines += ["## Flagged", ""] + ([f"- {item}" for item in flagged] or ["none"]) + [""]
    lines += ["| nuisance | " + " | ".join(eras) + " |", "|---|" + "---|" * len(eras)]
    lines += [f"| {name} | " + " | ".join("x" if era in present else "" for era in eras) + " |" for name, present in sorted(table.items())]
    path = Path(output) / "nuisances.md"
    path.write_text("\n".join(lines) + "\n")
    if flagged:
        logger.warning(f"{len(flagged)} nuisances end in the era of another run, see {path}:\n" + "\n".join(flagged))
    logger.info(f"{len(table)} nuisances in the combination: {path}")
    return path


def bin_map(runs: list[Run]) -> dict[str, tuple[str, str]]:
    """Combined bin -> (channel label `<channel> <era>`, category) of every run."""
    return {
        bin_name(run.analysis, channel, index): (f"{channel} {run.era}", category.name)
        for run in runs for channel in run.analysis.channels for index, category in enumerate(run.analysis.channel(channel).categories)
    }


def run_combination(runs: list[Run], output: Path, background_only: bool = False, skip_combine: bool = False, validate: bool = False) -> dict:
    """Combine the cards of the runs into <output>/combined.txt and run the limits or the background-only fit there;
    the summary (limits.json/md/pdf or background_fit.json/md) next to it."""
    check_runs(runs)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    write_nuisance_report(runs, output)
    combine = runs[0].config.combine
    if not skip_combine:
        logger.info(f"combining {len(runs)} cards ({', '.join(run.era for run in runs)}) into {output / CARD}")
        commands = [combine_cards_command([run.card for run in runs])] + ([f"{validate_command()} > validate.log 2>&1"] if validate else [])
        commands += fit_commands() if background_only else limit_commands()
        cmssw.run(commands, combine, output)
    if background_only:
        result = {"combination": summarise_bins(json.loads((output / f"{FIT_OUTPUT}.json").read_text()), bin_map(runs))}
        (output / "background_fit.json").write_text(json.dumps(result, indent=2))
        (output / "background_fit.md").write_text("\n".join(summary_lines("combination", result["combination"])))
        return result
    result = {"combination": collect(output)}
    write_summary(result, output)
    return result
