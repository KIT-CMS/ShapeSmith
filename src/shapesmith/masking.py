"""The blinding of a datacard (spec N14, 6.4) as the fits see it: the channel masks of <card dir>/blinding.json.

`datacards` writes blinding.json next to the combined.txt of a final state:

    {"edge_rule": "S/B(mu=1) > 0.005", "masked": ["htt_<ch>_<k>hi_<era>", ...],
     "categories": {"<hi bin>": {"lo": "<lo bin>", "edge": <score edge>, "s_over_b_first_hi_bin": <float>}},
     "asimov_data": true}

`masked` lists exactly the card bins whose data_obs is the Asimov background prediction. Every fit or limit on the
real data adds `--setParameters mask_<bin>=1` for each of them (the workspace built with --channel-masks); a fit to an
Asimov dataset (-t -1) needs none. A card without NN categories has no file or an empty `masked`: nothing is masked.
The combination of eras merges the files of its cards (merge()); the bin names carry the era already.

Unblinding (real data in the hi bins, no masks) needs the explicit --unblind of a command, which logs it loudly.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from shapesmith.cards import card_bins

logger = logging.getLogger(__name__)
FILE = "blinding.json"
HI_BIN = re.compile(r"^htt_.+hi_[^_]+$")  # the name pattern of a blinded channel, htt_<ch>_<k>hi_<era>


@dataclass(frozen=True)
class Blinding:
    """The masked bins of a card and the full content of its blinding.json (empty without the file)."""

    masked: tuple[str, ...] = ()
    content: dict = field(default_factory=dict)

    def assignments(self, value: int) -> list[str]:
        """`mask_<bin>=<value>` of every masked bin."""
        return [f"mask_{name}={value}" for name in self.masked]


def read(card_dir: Path) -> Blinding:
    """The blinding of the card in `card_dir`; a missing file means nothing is masked."""
    path = Path(card_dir) / FILE
    if not path.exists():
        return Blinding()
    content = json.loads(path.read_text())
    masked = content.get("masked", [])
    if not isinstance(masked, list) or not all(isinstance(name, str) for name in masked):
        raise ValueError(f"{path}: `masked` must be a list of card bin names, got {masked!r}")
    return Blinding(tuple(masked), content)


def check(blinding: Blinding, card: Path) -> None:
    """Every masked bin is a bin of the card (a mask of a missing channel is an error); a bin named like a blinded
    channel that is not masked is logged."""
    bins = card_bins(card)
    missing = [name for name in blinding.masked if name not in bins]
    if missing:
        raise ValueError(f"{card}: blinding.json masks bins the card does not have: {', '.join(missing)}")
    unmasked = [name for name in bins if HI_BIN.match(name) and name not in blinding.masked]
    if unmasked:
        logger.warning(f"{card}: the bins {', '.join(unmasked)} are named like blinded channels but blinding.json does not mask them")


def data_masks(card_dir: Path, unblind: bool = False) -> list[str]:
    """The `mask_<bin>=1` of a fit to the real data of the card in `card_dir`: those of its blinding.json, none with
    `unblind` (logged loudly)."""
    card_dir = Path(card_dir)
    blinding = read(card_dir)
    if (card_dir / "combined.txt").exists():
        check(blinding, card_dir / "combined.txt")
    if unblind:
        warn_unblind(card_dir, blinding)
        return []
    if blinding.masked:
        logger.info(f"{card_dir.name}: {len(blinding.masked)} blinded channels masked in the fit to data: {', '.join(blinding.masked)}")
    return blinding.assignments(1)


def warn_unblind(card_dir: Path, blinding: Blinding) -> None:
    banner = "!" * 100
    data = "the Asimov background prediction (no real data in the card)" if blinding.content.get("asimov_data", bool(blinding.masked)) else "the real data"
    logger.warning(f"\n{banner}\nUNBLINDED fit to data in {card_dir}: --unblind given, the masks of blinding.json are NOT applied; "
                   f"{len(blinding.masked)} hi channels ({', '.join(blinding.masked) or 'none'}) enter the fit with {data} as data_obs\n{banner}")


def merge(blindings: list[Blinding], sources: list[str] | None = None) -> dict:
    """The blinding.json of a combination: the masked bins of all cards in order, their categories, the edge rule
    (the list of the different rules if the cards disagree) and asimov_data of any card with masked bins."""
    masked = list(dict.fromkeys(name for blinding in blindings for name in blinding.masked))
    categories = {name: entry for blinding in blindings for name, entry in blinding.content.get("categories", {}).items()}
    rules = list(dict.fromkeys(blinding.content["edge_rule"] for blinding in blindings if "edge_rule" in blinding.content))
    if len(rules) > 1:
        logger.warning(f"the combined cards use different blinding edge rules: {'; '.join(rules)}")
    merged = {"edge_rule": rules[0] if len(rules) == 1 else rules or None, "masked": masked, "categories": categories,
              "asimov_data": any(blinding.content.get("asimov_data", True) for blinding in blindings if blinding.masked)}
    if sources is not None:
        merged["sources"] = sources
    return merged


def write_merged(card_dirs: list[Path], output: Path) -> Blinding:
    """<output>/blinding.json of the combination of the cards in `card_dirs`."""
    blindings = [read(directory) for directory in card_dirs]
    merged = merge(blindings, [str(Path(directory) / FILE) for directory, blinding in zip(card_dirs, blindings) if blinding.content])
    Path(output).mkdir(parents=True, exist_ok=True)
    (Path(output) / FILE).write_text(json.dumps(merged, indent=1))
    if merged["masked"]:
        logger.info(f"{len(merged['masked'])} blinded channels of {len(card_dirs)} cards in {Path(output) / FILE}")
    return Blinding(tuple(merged["masked"]), merged)
