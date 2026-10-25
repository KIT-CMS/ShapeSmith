"""The limit chain: combine on the datacards (expected limits, significance, best-fit r) and its summary.

The limits are prefit Asimov limits (-t -1), on the signal strength r (mu) first. With a signal cross section
(Analysis.signal_cross_section, spec N23), a second AsymptoticLimits with its theory nuisances frozen at 0 gives the
limit on the cross section, r x value_fb in fb (e.g. sigma(ggF HH) = r x 30.40 fb; the acceptance and shape
uncertainties of the signal stay); a card without those nuisances (e.g. only statistics) needs no second call. The
range of r, the precision of AsymptoticLimits and the minimizer options come from `combine` of the run configuration
(combine_options.py). limit_overview() collects the limits of several runs and combinations into one Brazil plot.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import uproot  # noqa: E402

from shapesmith import cmssw, combine_options  # noqa: E402
from shapesmith.cards import card_nuisances  # noqa: E402
from shapesmith.config import CombineConfig, RunConfig  # noqa: E402
from shapesmith.model import CrossSection  # noqa: E402

QUANTILES = {0.025: "exp_m2", 0.16: "exp_m1", 0.5: "exp_median", 0.84: "exp_p1", 0.975: "exp_p2", -1.0: "observed"}
COMMON = "-m {mass} -t -1"
CARD = "combined.txt"

logger = logging.getLogger(__name__)


def theory_nuisances(cross_section: CrossSection | None, combine: CombineConfig | None) -> tuple[str, ...]:
    """The theory nuisances of the cross section: `combine.xs_theory_nuisances` if set, else the analysis's."""
    if combine is not None and combine.xs_theory_nuisances is not None:
        return combine.xs_theory_nuisances
    return cross_section.theory_nuisances if cross_section else ()


def xs_frozen(cross_section: CrossSection | None, combine: CombineConfig | None, nuisances: list[str]) -> list[str] | None:
    """The theory nuisances among the card's `nuisances` (frozen in the cross-section limit); None without a cross
    section."""
    if cross_section is None:
        return None
    present = set(nuisances)
    return [name for name in theory_nuisances(cross_section, combine) if name in present]


def asymptotic_command(name: str, combine: CombineConfig | None, mass: str = "125", extra: str = "") -> str:
    return combine_options.command(f"combine -M AsymptoticLimits -d workspace.root {COMMON.format(mass=mass)} --expectSignal 0 -n {name} {extra}".strip(), combine, precise_limit=True)


def limit_commands(mass: str = "125", combine: CombineConfig | None = None, frozen: list[str] | None = None) -> list[str]:
    """Workspace, AsymptoticLimits, Significance and best fit on the Asimov dataset; with `frozen` (the theory
    nuisances of the cross section in the card) a second AsymptoticLimits with them frozen at 0."""
    common = COMMON.format(mass=mass)
    commands = [
        f"text2workspace.py {CARD} -o workspace.root -m {mass}",
        asymptotic_command(".Limit", combine, mass),
        combine_options.command(f"combine -M Significance -d workspace.root {common} --expectSignal 1 -n .Significance", combine),
        combine_options.command(f"combine -M MultiDimFit --algo singles -d workspace.root {common} --expectSignal 1 -n .Fit", combine),
    ]
    if frozen:
        commands.append(asymptotic_command(".LimitXS", combine, mass, f"{combine_options.parameters([f'{n}=0' for n in frozen])} {combine_options.frozen(frozen)}"))
    return commands


def _read(path: Path) -> list[tuple[float, float, float | None]]:
    with uproot.open(path) as f:
        tree = f["limit"]
        arrays = tree.arrays(library="np")
        r = arrays["r"] if "r" in arrays else [None] * len(arrays["limit"])
        return [(round(float(q), 3), float(limit), None if rr is None else float(rr)) for q, limit, rr in zip(arrays["quantileExpected"], arrays["limit"], r)]


def read_limit(path: Path) -> dict[str, float]:
    """The quantiles of an AsymptoticLimits output (exp_m2 ... exp_p2, observed)."""
    return {QUANTILES[q]: limit for q, limit, _ in _read(path) if q in QUANTILES}


def xs_limit(cross_section: CrossSection, frozen: list[str], limit: dict[str, float]) -> dict:
    """The limit on the cross section in fb from the r limit with the theory nuisances frozen (`limit`)."""
    return {"quantity": cross_section.quantity, "unit": "fb", "fb_per_r": cross_section.value_fb, "frozen": list(frozen),
            "limit": {key: value * cross_section.value_fb for key, value in limit.items()}}


def collect(card_dir: Path, mass: str = "125", cross_section: CrossSection | None = None, frozen: list[str] | None = None) -> dict:
    """The limits, significance and best fit of a card directory; with a cross section also its limit in fb (from
    .LimitXS, or from .Limit when no theory nuisance was frozen)."""
    card_dir = Path(card_dir)
    result: dict = {}
    limit_file = card_dir / f"higgsCombine.Limit.AsymptoticLimits.mH{mass}.root"
    significance_file = card_dir / f"higgsCombine.Significance.Significance.mH{mass}.root"
    fit_file = card_dir / f"higgsCombine.Fit.MultiDimFit.mH{mass}.root"
    for path in (limit_file, significance_file, fit_file):
        if not path.exists():
            logger.warning(f"{card_dir.name}: {path.name} missing")
    if limit_file.exists():
        result["limit"] = read_limit(limit_file)
    if significance_file.exists():
        result["significance"] = _read(significance_file)[0][1]
    if fit_file.exists():
        entries = _read(fit_file)
        best = [r for q, _, r in entries if q == -1.0]
        others = [r for q, _, r in entries if q != -1.0]
        result["r"] = {"best": best[0] if best else None, "low": min(others) if others else None, "high": max(others) if others else None}
    if cross_section is not None:
        frozen = frozen or []
        xs_file = card_dir / f"higgsCombine.LimitXS.AsymptoticLimits.mH{mass}.root"
        if frozen and not xs_file.exists():
            logger.warning(f"{card_dir.name}: {xs_file.name} missing")
        elif frozen or "limit" in result:
            result["xs_limit"] = xs_limit(cross_section, frozen, read_limit(xs_file) if frozen else result["limit"])
    logger.debug(f"{card_dir.name}: {json.dumps(result)}")
    return result


def _xs_header(results: dict[str, dict]) -> str | None:
    """`σ(<process>) / fb` of the cross-section limits of the results, None without any."""
    quantities = {res["xs_limit"]["quantity"] for res in results.values() if res.get("xs_limit")}
    return " / ".join(sorted(quantities)).replace("sigma(", "σ(") + " / fb" if quantities else None


def brazil(rows: list[tuple[str, dict]], path: Path, xlabel: str) -> Path:
    """Horizontal Brazil bands (68 %, 95 %, median; the observed limit where there is one) of the limits per row."""
    fig, ax = plt.subplots(figsize=(6, 0.8 * max(len(rows), 1) + 1.5))
    for i, (_, limit) in enumerate(rows):
        ax.barh(i, limit["exp_p2"] - limit["exp_m2"], left=limit["exp_m2"], color="#f5d000", height=0.6, label="95% expected" if i == 0 else None)
        ax.barh(i, limit["exp_p1"] - limit["exp_m1"], left=limit["exp_m1"], color="#00a651", height=0.6, label="68% expected" if i == 0 else None)
        ax.plot([limit["exp_median"]] * 2, [i - 0.3, i + 0.3], "k--", label="median expected" if i == 0 else None)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([name for name, _ in rows])
    ax.invert_yaxis()
    ax.set_xlim(left=0)
    ax.set_xlabel(xlabel)
    if rows:
        ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def xs_label(results: dict[str, dict]) -> str:
    quantity = next(res["xs_limit"]["quantity"] for res in results.values() if res.get("xs_limit"))
    return "95% CL upper limit on " + quantity.replace("sigma(", r"$\sigma$(") + " / fb"


def write_summary(results: dict[str, dict], output_dir: Path, name: str = "limits") -> tuple[Path, Path, Path]:
    """<name>.json, .md and .pdf (and <name>_xs.pdf with cross-section limits) in `output_dir`."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{name}.json"
    json_path.write_text(json.dumps(results, indent=2))
    xs_header = _xs_header(results)
    header = "| final state | expected limit on μ | 68% band | 95% band | significance | r best [68%] |" + (f" {xs_header}: median [68% band] |" if xs_header else "")
    lines = [header, "|---|---|---|---|---|---|" + ("---|" if xs_header else "")]
    for state, res in results.items():
        limit, r = res.get("limit", {}), res.get("r", {})
        xs = res.get("xs_limit", {}).get("limit")
        xs_cell = (f" {xs['exp_median']:.1f} [{xs['exp_m1']:.1f}, {xs['exp_p1']:.1f}] |" if xs else " - |") if xs_header else ""
        if limit:
            lines.append(
                f"| {state} | {limit['exp_median']:.2f} | [{limit['exp_m1']:.2f}, {limit['exp_p1']:.2f}] | [{limit['exp_m2']:.2f}, {limit['exp_p2']:.2f}] "
                f"| {res.get('significance', float('nan')):.2f} | {r.get('best')} [{r.get('low')}, {r.get('high')}] |" + xs_cell
            )
        else:
            lines.append(f"| {state} | - | - | - | - | - |" + xs_cell)
    md_path = output_dir / f"{name}.md"
    md_path.write_text("\n".join(lines) + "\n")
    rows = [(state, res["limit"]) for state, res in results.items() if res.get("limit")]
    pdf_path = brazil(rows, output_dir / f"{name}.pdf", r"95% CL upper limit on $\mu = \sigma/\sigma_\mathrm{SM}$")
    if xs_header:
        brazil([(state, res["xs_limit"]["limit"]) for state, res in results.items() if res.get("xs_limit")], output_dir / f"{name}_xs.pdf", xs_label(results))
    logger.info(f"summary of {len(results)} final states: {json_path}, {md_path.name}, {pdf_path.name}")
    return json_path, md_path, pdf_path


def run_limits(config: RunConfig, datacard_dir: Path, final_states: list[str], skip_combine: bool = False, cross_section: CrossSection | None = None) -> dict:
    """The prefit Asimov limits of every final state (no masks: no fit to data) and the summary in `datacard_dir`."""
    datacard_dir = Path(datacard_dir)
    results = {}
    for name in final_states:
        card_dir = datacard_dir / name
        frozen = xs_frozen(cross_section, config.combine, card_nuisances(card_dir / CARD)) if (card_dir / CARD).exists() else None
        if not skip_combine:
            logger.info(f"{name}: combine in {card_dir}")
            cmssw.run(limit_commands(combine=config.combine, frozen=frozen), config.combine, card_dir)
        results[name] = collect(card_dir, cross_section=cross_section, frozen=frozen)
    write_summary(results, datacard_dir)
    return results


def overview_rows(sources: list[tuple[str, Path]], key: str = "limit") -> list[tuple[str, dict]]:
    """(row label, limit) of every final state of every summary: (label, <dir>/limits.json) -> `<state> <label>`;
    `key` "xs_limit" takes the cross-section limits."""
    rows = []
    for label, path in sources:
        for state, res in json.loads(Path(path).read_text()).items():
            limit = res.get(key, {}).get("limit") if key == "xs_limit" else res.get(key)
            if limit:
                rows.append((f"{state} {label}".strip() if state != "combination" else label, limit))
    return rows


def limit_overview(sources: list[tuple[str, Path]], output: Path, name: str = "limit_overview") -> dict:
    """One table and Brazil plot of the limits of several summaries (per era: each channel and `all`; the
    combinations), on mu and, where present, on the cross section: <output>/<name>.{json,md,pdf}, <name>_xs.pdf."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    result: dict = {"limit": {row: limit for row, limit in overview_rows(sources)}, "xs_limit": {row: limit for row, limit in overview_rows(sources, "xs_limit")}}
    quantities = {res["xs_limit"]["quantity"] for _, path in sources for res in json.loads(Path(path).read_text()).values() if res.get("xs_limit")}
    result["xs_quantity"] = sorted(quantities)
    result["sources"] = {label: str(path) for label, path in sources}
    (output / f"{name}.json").write_text(json.dumps(result, indent=2))
    lines = ["| | expected limit on μ | 68% band | 95% band |" + (f" {' / '.join(sorted(quantities)).replace('sigma(', 'σ(')} / fb [68% band] |" if quantities else ""),
             "|---|---|---|---|" + ("---|" if quantities else "")]
    for row, limit in result["limit"].items():
        xs = result["xs_limit"].get(row)
        xs_cell = (f" {xs['exp_median']:.1f} [{xs['exp_m1']:.1f}, {xs['exp_p1']:.1f}] |" if xs else " - |") if quantities else ""
        lines.append(f"| {row} | {limit['exp_median']:.3f} | [{limit['exp_m1']:.3f}, {limit['exp_p1']:.3f}] | [{limit['exp_m2']:.3f}, {limit['exp_p2']:.3f}] |" + xs_cell)
    (output / f"{name}.md").write_text("\n".join(lines) + "\n")
    brazil(list(result["limit"].items()), output / f"{name}.pdf", r"95% CL upper limit on $\mu = \sigma/\sigma_\mathrm{SM}$")
    if quantities:
        label = "95% CL upper limit on " + " / ".join(sorted(quantities)).replace("sigma(", r"$\sigma$(") + " / fb"
        brazil(list(result["xs_limit"].items()), output / f"{name}_xs.pdf", label)
    logger.info(f"limit overview of {len(result['limit'])} rows from {len(sources)} summaries: {output / name}.md/.pdf")
    return result
