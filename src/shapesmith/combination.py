"""The combination of the datacards of several runs (v7 spec 6.9: one run per era of Run 2): combineCards.py of the
card of one final state of every run, then the limit chain (limits.py) or the background-only fit (background_fit.py)
on the combined card, with their summaries.

The bins of every card keep their names (combineCards' label "."): they carry the era already
(datacards.bin_name), so that the cards of different eras never share a bin; two runs of one era are an error. The
shape files are referenced with absolute paths, so the combination lives in its own directory. nuisances.md lists
every nuisance with the runs it appears in and flags a name that ends in the token of another run's era (e.g.
`_2017` in the 2018 card), the check of spec 9.9 for nuisances doubled by mistake; a name ending in a prefix of the
run's own era (`_2016` in a 2016preVFP card) is the shared one of a year.

The cards of a variant (`datacards --variant`) are <output_dir>/datacards/<variant>/<final state>/. The blinding.json
of the cards merge into <output>/blinding.json (masking.merge): the background-only fit masks their blinded channels
unless `unblind`. The limits are prefit Asimov limits (no masks), on r and on the signal cross section of the
analysis (limits.py; theory nuisances of every card frozen).
"""
from __future__ import annotations

import json
import logging
import shlex
from dataclasses import dataclass
from pathlib import Path

from shapesmith import cmssw, masking
from shapesmith.background_fit import OUTPUT as FIT_OUTPUT
from shapesmith.background_fit import fit_commands, plot_fit, summarise_bins, summary_lines
from shapesmith.cards import card_nuisances
from shapesmith.config import RunConfig
from shapesmith.datacards import bin_categories
from shapesmith.limits import collect, limit_commands, write_summary, xs_frozen
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


def datacard_dir(config: RunConfig, variant: str | None = None) -> Path:
    """The datacards of a run: <output_dir>/datacards/, of a variant <output_dir>/datacards/<variant>/."""
    return Path(config.output_dir) / "datacards" / (variant or "")


def run_card(config: RunConfig, final_state: str, variant: str | None = None) -> Path:
    """The datacard of a final state of a run: <output_dir>/datacards[/<variant>]/<final state>/combined.txt."""
    return datacard_dir(config, variant) / final_state / CARD


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
    """Combined bin -> (channel label `<channel> <era>`, category) of every run, the parts of a blinded category as
    `<category> lo` and `<category> hi` (datacards.bin_categories)."""
    return {card_bin: (f"{channel} {run.era}", category) for run in runs for card_bin, (channel, category) in bin_categories(run.analysis).items()}


def combined_blinding(runs: list[Run], output: Path) -> masking.Blinding:
    """<output>/blinding.json from those of the runs' cards, each checked against its card."""
    for run in runs:
        masking.check(masking.read(run.card.parent), run.card)
    return masking.write_merged([run.card.parent for run in runs], output)


def run_combination(runs: list[Run], output: Path, background_only: bool = False, skip_combine: bool = False, validate: bool = False, unblind: bool = False) -> dict:
    """Combine the cards of the runs into <output>/combined.txt and run the limits or the background-only fit there
    (the blinded channels of all cards masked unless `unblind`); the summary (limits.json/md/pdf or
    background_fit.json/md and the fit plots) next to it."""
    check_runs(runs)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    write_nuisance_report(runs, output)
    blinding = combined_blinding(runs, output)
    if unblind and background_only:
        masking.warn_unblind(output, blinding)
    masks = [] if unblind else blinding.assignments(1)
    combine = runs[0].config.combine
    cross_section = runs[0].analysis.signal_cross_section
    frozen = xs_frozen(cross_section, combine, [name for run in runs for name in card_nuisances(run.card)])
    if not skip_combine:
        logger.info(f"combining {len(runs)} cards ({', '.join(run.era for run in runs)}) into {output / CARD}")
        commands = [combine_cards_command([run.card for run in runs])] + ([f"{validate_command()} > validate.log 2>&1"] if validate else [])
        commands += fit_commands(combine=combine, masks=masks) if background_only else limit_commands(combine=combine, frozen=frozen)
        cmssw.run(commands, combine, output)
    if background_only:
        values = json.loads((output / f"{FIT_OUTPUT}.json").read_text())
        result = {"combination": summarise_bins(values, bin_map(runs), [] if unblind else blinding.masked)}
        result["combination"].update({"masked": [] if unblind else list(blinding.masked), **({"unblinded": True} if unblind else {})})
        (output / "background_fit.json").write_text(json.dumps(result, indent=2))
        (output / "background_fit.md").write_text("\n".join(summary_lines("combination", result["combination"])))
        plot_fit(values, result["combination"], runs[0].analysis, output, [] if unblind else blinding.masked, "combination")
        return result
    result = {"combination": collect(output, cross_section=cross_section, frozen=frozen)}
    write_summary(result, output)
    return result
