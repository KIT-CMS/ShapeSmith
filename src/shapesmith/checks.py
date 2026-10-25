"""The statistical checks of the NN phase (spec 7.2, SH11) on the datacard of a final state or of a combination of
eras. Each check runs in its own subdirectory of the card directory (`../combined.txt`, its own workspace) and writes
its summary next to the cards (<datacards>/<check>.json and .md, or <combination>/<check>.json and .md):

- gof: the saturated goodness of fit of the background-only model (r frozen at 0) to the data, against frequentist
  toys in jobs of TOYS_PER_JOB, `combine.parallel` at a time; p = the fraction of toys with a statistic at least the
  observed one. <card>/gof/, plot gof/gof.pdf.
- impacts: combineTool.py -M Impacts (initial fit, one fit per nuisance in parallel, impacts.json) and plotImpacts.py
  (impacts.pdf, with the pulls); on the data (the POI value hidden, --blind) or, `asimov`, on the Asimov dataset with
  r = 1. <card>/impacts/; the summary ranks the nuisances by their impact on r.
- a-posteriori expected limit: the background-only fit (MultiDimFit, r = 0) to the data with the blinded channels
  masked saves its snapshot; AsymptoticLimits -t -1 --toysFrequentist --bypassFrequentistFit on that snapshot with the
  masks set to 0 then builds the Asimov dataset of the postfit nuisances in every bin, the hi channels included. On r
  and on the signal cross section (limits.py). <card>/aposteriori/, summary aposteriori.json/md/pdf.
- injection: toys with r = mu for every mu (default 0, 1 and the expected limit), each fitted with FitDiagnostics:
  the mean fitted r should follow mu linearly and the pulls (r - mu) / sigma_r (the error towards mu) be unbiased with
  width 1. Toys of the prefit model (nuisances drawn from their constraints), or, `a_posteriori`, frequentist toys
  around the snapshot of the masked background-only fit. <card>/injection/, plot injection/injection.pdf.
- hybridnew: HybridNew (LHC-limits: CLs with toys) on the Asimov dataset, prefit or of the a-posteriori snapshot: the
  toy-based median expected limit, compared with the asymptotic one (the bins may go down to one background event,
  where the asymptotic formulae are in doubt). <card>/hybridnew/.

Masks: every fit to the real data masks the blinded channels of the card's blinding.json (masking.py), unless
`unblind`. Fits to the prefit Asimov dataset or to toys of the prefit model need none; the a-posteriori Asimov
dataset and toys come from the snapshot of the masked fit with the masks set to 0, so that the blinded channels count.
"""
from __future__ import annotations

import glob
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import uproot

from shapesmith import cmssw, combine_options, masking
from shapesmith.background_fit import workspace_command
from shapesmith.cards import card_nuisances
from shapesmith.config import CombineConfig
from shapesmith.limits import CARD, brazil, read_limit, xs_frozen, xs_limit
from shapesmith.model import CrossSection

logger = logging.getLogger(__name__)
MASS = "125"
TOYS_PER_JOB = 25
GOF_SEED = 1000  # job i of the GoF toys has the seed GOF_SEED + i
INJECTION_SEED = 2000  # job j of the injected value k: INJECTION_SEED + 100 k + j
HYBRIDNEW_SEED = 3000
SNAPSHOT_NAME = ".BackgroundOnlySnapshot"
SNAPSHOT = f"higgsCombine{SNAPSHOT_NAME}.MultiDimFit.mH{MASS}.root"
IMPACTS_LISTED = 30  # nuisances in the markdown summary of the impacts


@dataclass(frozen=True)
class Target:
    """A card to check: its name in the summaries (final state, or "combination") and its directory."""

    name: str
    card_dir: Path

    @property
    def card(self) -> Path:
        return Path(self.card_dir) / CARD

    def directory(self, check: str) -> Path:
        path = Path(self.card_dir) / check
        path.mkdir(parents=True, exist_ok=True)
        return path


def _workspace(masks: list[str]) -> str:
    return workspace_command(MASS, masks, f"../{CARD}")


def _chunks(total: int, size: int) -> list[int]:
    return [min(size, total - start) for start in range(0, total, size)]


def _logged(command: str, log: str) -> str:
    return f"{command} > {log} 2>&1"


def _off(masks: list[str]) -> list[str]:
    """The masks set to 0: the blinded channels count again (e.g. in the limit on the a-posteriori Asimov dataset)."""
    return [assignment.rpartition("=")[0] + "=0" for assignment in masks]


def _masked(masks: list[str]) -> list[str]:
    """The bins of the masks mask_<bin>=1."""
    return [assignment.partition("=")[0][len("mask_"):] for assignment in masks]


def snapshot_commands(combine: CombineConfig | None, masks: list[str]) -> list[str]:
    """The workspace and the background-only fit to the data (r frozen at 0, blinded channels masked) whose snapshot
    MultiDimFit is saved in SNAPSHOT."""
    fit = f"combine -M MultiDimFit -d workspace.root -m {MASS} {combine_options.parameters(['r=0', *masks])} --freezeParameters r --saveWorkspace -n {SNAPSHOT_NAME}"
    return [_workspace(masks), _logged(combine_options.command(fit, combine), "snapshot.log")]


def snapshot_source(masks: list[str], assignments: list[str] | tuple[str, ...] = ()) -> str:
    """The options that take the data of a call from the snapshot (Asimov or toys of its postfit nuisances, no new
    fit), with the masks set to 0 and the further parameter `assignments`."""
    return f"-d {SNAPSHOT} --snapshotName MultiDimFit --toysFrequentist --bypassFrequentistFit {combine_options.parameters([*_off(masks), *assignments])}".strip()


# -- GoF -------------------------------------------------------------------------------------------------------------


def gof_commands(combine: CombineConfig | None, masks: list[str], toys: int, toys_per_job: int = TOYS_PER_JOB) -> tuple[list[str], list[list[str]]]:
    """(workspace + observed statistic, [one command list per toy job]) of the saturated GoF of the background-only
    model; the toys are frequentist (generated around the masked fit to the data)."""
    base = f"combine -M GoodnessOfFit -d workspace.root -m {MASS} --algo saturated {combine_options.parameters(['r=0', *masks])} --freezeParameters r"
    first = [_workspace(masks), _logged(combine_options.command(f"{base} -n .obs", combine), "obs.log")]
    jobs = [[_logged(combine_options.command(f"{base} -n .toys -t {n} -s {GOF_SEED + i} --toysFrequentist", combine), f"toys_{GOF_SEED + i}.log")] for i, n in enumerate(_chunks(toys, toys_per_job))]
    return first, jobs


def _limit_values(path: Path) -> np.ndarray:
    with uproot.open(path) as handle:
        return np.asarray(handle["limit"]["limit"].array(library="np"), dtype=float)


def gof_toys(directory: Path) -> np.ndarray:
    """The test statistics of all toy files of a GoF directory."""
    files = sorted(glob.glob(str(Path(directory) / f"higgsCombine.toys.GoodnessOfFit.mH{MASS}.*.root")))
    return np.concatenate([_limit_values(Path(path)) for path in files]) if files else np.array([])


def gof_result(directory: Path) -> dict:
    observed = float(_limit_values(Path(directory) / f"higgsCombine.obs.GoodnessOfFit.mH{MASS}.root")[0])
    toys = gof_toys(directory)
    finite = toys[np.isfinite(toys)]
    p_value = float(np.mean(finite >= observed)) if len(finite) else None
    return {"statistic": observed, "toys": int(len(finite)), "failed_toys": int(len(toys) - len(finite)), "p_value": p_value,
            "toy_quantiles": {str(q): float(np.quantile(finite, q)) for q in (0.16, 0.5, 0.84)} if len(finite) else {}}


def run_gof(target: Target, combine: CombineConfig | None, toys: int, unblind: bool = False, skip_combine: bool = False) -> dict:
    from shapesmith.plotting.fit_plots import gof_plot

    directory = target.directory("gof")
    masks = masking.data_masks(target.card_dir, unblind)
    if not skip_combine:
        for stale in glob.glob(str(directory / f"higgsCombine.toys.GoodnessOfFit.mH{MASS}.*.root")):
            Path(stale).unlink()
        first, jobs = gof_commands(combine, masks, toys)
        logger.info(f"{target.name}: saturated GoF in {directory}, {toys} toys in {len(jobs)} jobs")
        cmssw.run(first, combine, directory)
        cmssw.run_parallel(jobs, combine, directory)
    result = {**gof_result(directory), "masked": _masked(masks), **({"unblinded": True} if unblind else {})}
    if result["p_value"] is not None:
        gof_plot(gof_toys(directory), result["statistic"], result["p_value"], directory / "gof.pdf", f"{target.name}: saturated GoF, background only")
    logger.info(f"{target.name}: GoF statistic {result['statistic']:.4g}, p = {result['p_value']} from {result['toys']} toys")
    return result


def gof_lines(results: dict[str, dict]) -> list[str]:
    lines = ["| card | statistic | toys | p-value | toy median [16%, 84%] | masked |", "|---|---|---|---|---|---|"]
    for name, res in results.items():
        q = res.get("toy_quantiles", {})
        quantiles = f"{q['0.5']:.4g} [{q['0.16']:.4g}, {q['0.84']:.4g}]" if q else "-"
        p_value = "-" if res["p_value"] is None else f"{res['p_value']:.3f}"
        lines.append(f"| {name} | {res['statistic']:.4g} | {res['toys']} | {p_value} | {quantiles} | {', '.join(res['masked']) or 'none'}{' (UNBLINDED)' if res.get('unblinded') else ''} |")
    return lines


# -- impacts ---------------------------------------------------------------------------------------------------------


def impacts_commands(combine: CombineConfig | None, masks: list[str], asimov: bool = False) -> list[str]:
    """Workspace, initial fit, the fits per nuisance (`combine.parallel` at a time), impacts.json and impacts.pdf;
    `asimov`: the Asimov dataset with r = 1 (no masks), else the data with the blinded channels masked."""
    parallel = (combine or combine_options.DEFAULT).parallel
    impacts = " ".join(part for part in (f"combineTool.py -M Impacts -d workspace.root -m {MASS}", "-t -1 --expectSignal 1" if asimov else combine_options.parameters(masks)) if part)
    return [
        _workspace([] if asimov else masks),
        _logged(combine_options.command(f"{impacts} --robustFit 1 --doInitialFit", combine), "initial_fit.log"),
        _logged(combine_options.command(f"{impacts} --robustFit 1 --doFits --parallel {parallel}", combine), "fits.log"),
        _logged(f"{impacts} -o impacts.json", "collect.log"),
        _logged("plotImpacts.py -i impacts.json -o impacts" + ("" if asimov else " --blind"), "plot.log"),
    ]


def impacts_result(directory: Path) -> dict:
    """The nuisances of impacts.json ranked by their impact on r: pull and constraint (against the prefit width)
    and the shifts of r."""
    content = json.loads((Path(directory) / "impacts.json").read_text())
    poi = content["POIs"][0]["name"]
    entries = []
    for param in content["params"]:
        prefit, fit, shifts = param.get("prefit"), param["fit"], param[poi]
        width = (prefit[2] - prefit[0]) / 2 if prefit and prefit[2] != prefit[0] else None
        entries.append({
            "name": param["name"], "impact": float(param.get(f"impact_{poi}", max(abs(shifts[2] - shifts[1]), abs(shifts[1] - shifts[0])))),
            "r_down": shifts[0] - shifts[1], "r_up": shifts[2] - shifts[1],
            "pull": (fit[1] - prefit[1]) / width if width else None, "constraint": (fit[2] - fit[0]) / 2 / width if width else None,
        })
    entries.sort(key=lambda entry: -entry["impact"])
    return {"poi": poi, "nuisances": len(entries), "ranking": entries}


def run_impacts(target: Target, combine: CombineConfig | None, asimov: bool = False, unblind: bool = False, skip_combine: bool = False) -> dict:
    directory = target.directory("impacts_asimov" if asimov else "impacts")
    masks = [] if asimov else masking.data_masks(target.card_dir, unblind)
    if not skip_combine:
        logger.info(f"{target.name}: impacts in {directory} ({'Asimov, r = 1' if asimov else 'data'})")
        cmssw.run(impacts_commands(combine, masks, asimov), combine, directory)
    result = {**impacts_result(directory), "data": "asimov" if asimov else "data", "masked": _masked(masks),
              "plot": str(directory / "impacts.pdf"), **({"unblinded": True} if unblind and not asimov else {})}
    logger.info(f"{target.name}: {result['nuisances']} impacts, largest {', '.join(e['name'] for e in result['ranking'][:5])}")
    return result


def impacts_lines(results: dict[str, dict]) -> list[str]:
    lines = []
    for name, res in results.items():
        lines += [f"## {name} ({res['data']}): {res['nuisances']} nuisances, plot {res['plot']}", ""]
        if res.get("masked"):
            lines += [f"Masked (blinded) channels: {', '.join(res['masked'])}", ""]
        lines += [f"| nuisance | impact on {res['poi']} | Δ{res['poi']} down | Δ{res['poi']} up | pull | constraint |", "|---|---|---|---|---|---|"]
        for entry in res["ranking"][:IMPACTS_LISTED]:
            pull = "-" if entry["pull"] is None else f"{entry['pull']:+.2f}"
            constraint = "-" if entry["constraint"] is None else f"{entry['constraint']:.2f}"
            lines.append(f"| {entry['name']} | {entry['impact']:.4g} | {entry['r_down']:+.4g} | {entry['r_up']:+.4g} | {pull} | {constraint} |")
        lines.append("")
    return lines


# -- a-posteriori expected limit -------------------------------------------------------------------------------------


def aposteriori_commands(combine: CombineConfig | None, masks: list[str], frozen: list[str] | None = None) -> list[str]:
    """The masked background-only fit with its snapshot, then AsymptoticLimits on the Asimov dataset of the snapshot
    with the masks set to 0; with `frozen` also the limit with those theory nuisances frozen at 0 (cross section)."""
    source = f"{snapshot_source(masks)} -m {MASS} -t -1 --expectSignal 0"
    commands = snapshot_commands(combine, masks)
    commands.append(_logged(combine_options.command(f"combine -M AsymptoticLimits {source} -n .APosteriori", combine, precise_limit=True), "limit.log"))
    if frozen:
        # the theory nuisances at 0, not at their postfit values of the snapshot; they join the masks set to 0
        xs_source = f"{snapshot_source(masks, [f'{name}=0' for name in frozen])} -m {MASS} -t -1 --expectSignal 0"
        commands.append(_logged(combine_options.command(f"combine -M AsymptoticLimits {xs_source} {combine_options.frozen(frozen)} -n .APosterioriXS", combine, precise_limit=True), "limit_xs.log"))
    return commands


def run_aposteriori(target: Target, combine: CombineConfig | None, cross_section: CrossSection | None = None, prefit: dict | None = None, unblind: bool = False, skip_combine: bool = False) -> dict:
    """The a-posteriori expected limit of the target; `prefit` the prefit Asimov limits (limits.json) to compare."""
    directory = target.directory("aposteriori")
    masks = masking.data_masks(target.card_dir, unblind)
    frozen = xs_frozen(cross_section, combine, card_nuisances(target.card))
    if not skip_combine:
        logger.info(f"{target.name}: a-posteriori expected limit in {directory}")
        cmssw.run(aposteriori_commands(combine, masks, frozen), combine, directory)
    result: dict = {"limit": read_limit(directory / f"higgsCombine.APosteriori.AsymptoticLimits.mH{MASS}.root"), "masked": _masked(masks)}
    if cross_section is not None:
        result["xs_limit"] = xs_limit(cross_section, frozen, read_limit(directory / f"higgsCombine.APosterioriXS.AsymptoticLimits.mH{MASS}.root") if frozen else result["limit"])
    if prefit and prefit.get("limit"):
        result["prefit_limit"] = prefit["limit"]
    if unblind:
        result["unblinded"] = True
    logger.info(f"{target.name}: a-posteriori expected limit on r {result['limit']['exp_median']:.4g} [{result['limit']['exp_m1']:.4g}, {result['limit']['exp_p1']:.4g}]")
    return result


def aposteriori_lines(results: dict[str, dict]) -> list[str]:
    quantities = sorted({res["xs_limit"]["quantity"] for res in results.values() if res.get("xs_limit")})
    xs = f" {' / '.join(quantities).replace('sigma(', 'σ(')} / fb [68% band] |" if quantities else ""
    lines = ["| card | a-posteriori expected limit on μ | 68% band | 95% band | prefit Asimov median | a-posteriori / prefit |" + xs, "|---|---|---|---|---|---|" + ("---|" if xs else "")]
    for name, res in results.items():
        limit, prefit = res["limit"], res.get("prefit_limit", {}).get("exp_median")
        xs_cell = ""
        if xs:
            entry = res.get("xs_limit", {}).get("limit")
            xs_cell = f" {entry['exp_median']:.1f} [{entry['exp_m1']:.1f}, {entry['exp_p1']:.1f}] |" if entry else " - |"
        compared = "- | -" if prefit is None else f"{prefit:.4f} | {limit['exp_median'] / prefit:.4f}"
        lines.append(f"| {name} | {limit['exp_median']:.4f} | [{limit['exp_m1']:.4f}, {limit['exp_p1']:.4f}] | [{limit['exp_m2']:.4f}, {limit['exp_p2']:.4f}] | {compared} |" + xs_cell)
    return lines


# -- signal injection and bias ---------------------------------------------------------------------------------------


def mu_label(mu: float) -> str:
    return "mu" + format(mu, "g").replace(".", "p").replace("-", "m")


def injection_commands(combine: CombineConfig | None, masks: list[str], mus: list[float], toys: int, a_posteriori: bool = False, toys_per_job: int = TOYS_PER_JOB) -> tuple[list[str], list[list[str]]]:
    """(workspace [+ snapshot], [one command list per toy job]): FitDiagnostics (s+b fit only) of toys with r = mu."""
    first = snapshot_commands(combine, masks) if a_posteriori else [_workspace([])]
    source = snapshot_source(masks) if a_posteriori else "-d workspace.root"
    jobs = []
    for k, mu in enumerate(mus):
        for j, n in enumerate(_chunks(toys, toys_per_job)):
            seed = INJECTION_SEED + 100 * k + j
            name = f".{mu_label(mu)}_s{seed}"
            fit = f"combine -M FitDiagnostics {source} -m {MASS} -t {n} --expectSignal {mu:g} -s {seed} --skipBOnlyFit -n {name}"
            jobs.append([_logged(combine_options.command(fit, combine), f"{name[1:]}.log")])
    return first, jobs


def injection_toys(directory: Path, mu: float) -> dict[str, np.ndarray]:
    """r, its errors and the pull (r - mu) / (upper error if r < mu else lower error) of the converged toy fits."""
    arrays = {key: [] for key in ("r", "rLoErr", "rHiErr", "fit_status")}
    for path in sorted(glob.glob(str(Path(directory) / f"fitDiagnostics.{mu_label(mu)}_s*.root"))):
        with uproot.open(path) as handle:
            if "tree_fit_sb" not in handle:
                continue
            tree = handle["tree_fit_sb"]
            for key in arrays:
                arrays[key].append(np.asarray(tree[key].array(library="np"), dtype=float))
    values = {key: np.concatenate(parts) if parts else np.array([]) for key, parts in arrays.items()}
    ok = values["fit_status"] == 0
    r, low, high = values["r"][ok], values["rLoErr"][ok], values["rHiErr"][ok]
    error = np.where(r < mu, high, low)
    with np.errstate(divide="ignore", invalid="ignore"):
        pull = np.where(error > 0, (r - mu) / error, np.nan)
    return {"r": r, "pull": pull[np.isfinite(pull)], "failed": np.array([int(np.sum(~ok))])}


def injection_summary(mu: float, toys: dict[str, np.ndarray]) -> dict:
    r, pull = toys["r"], toys["pull"]
    n, m = len(r), len(pull)
    sd = float(np.std(r, ddof=1)) if n > 1 else float("nan")
    width = float(np.std(pull, ddof=1)) if m > 1 else float("nan")
    mean_pull = float(np.mean(pull)) if m else float("nan")
    error = width / math.sqrt(m) if m > 1 else float("nan")
    return {"mu": mu, "toys": n, "failed": int(toys["failed"][0]), "r_mean": float(np.mean(r)) if n else float("nan"), "r_mean_error": sd / math.sqrt(n) if n > 1 else float("nan"),
            "r_median": float(np.median(r)) if n else float("nan"), "r_sd": sd, "pull_mean": mean_pull, "pull_mean_error": error, "pull_width": width,
            "biased": bool(m > 1 and abs(mean_pull) > 2 * error)}


def linearity(summary: dict[str, dict]) -> dict:
    """The straight line through the mean fitted r against mu (weighted by the errors of the means)."""
    points = [(entry["mu"], entry["r_mean"], entry["r_mean_error"]) for entry in summary.values() if np.isfinite(entry["r_mean"]) and entry["r_mean_error"] > 0]
    if len(points) < 2:
        return {}
    mu, r, error = (np.array(column) for column in zip(*points))
    (slope, offset), covariance = np.polyfit(mu, r, 1, w=1 / error, cov="unscaled")
    return {"slope": float(slope), "slope_error": float(math.sqrt(covariance[0, 0])), "offset": float(offset), "offset_error": float(math.sqrt(covariance[1, 1]))}


def run_injection(target: Target, combine: CombineConfig | None, mus: list[float], toys: int, a_posteriori: bool = False, unblind: bool = False, skip_combine: bool = False) -> dict:
    from shapesmith.plotting.fit_plots import injection_plot

    directory = target.directory("injection_aposteriori" if a_posteriori else "injection")
    masks = masking.data_masks(target.card_dir, unblind) if a_posteriori else []
    if not skip_combine:
        for mu in mus:
            for stale in glob.glob(str(directory / f"fitDiagnostics.{mu_label(mu)}_s*.root")):
                Path(stale).unlink()
        first, jobs = injection_commands(combine, masks, mus, toys, a_posteriori)
        logger.info(f"{target.name}: signal injection in {directory}, mu {', '.join(f'{mu:g}' for mu in mus)}, {toys} toys each in {len(jobs)} jobs")
        cmssw.run(first, combine, directory)
        cmssw.run_parallel(jobs, combine, directory)
    toys_per_mu = {mu_label(mu): injection_toys(directory, mu) for mu in mus}
    summary = {mu_label(mu): injection_summary(mu, toys_per_mu[mu_label(mu)]) for mu in mus}
    result = {"toys": "a-posteriori (frequentist around the masked background-only fit)" if a_posteriori else "prefit model", "injected": summary, "linearity": linearity(summary),
              "masked": _masked(masks), **({"unblinded": True} if unblind and a_posteriori else {})}
    injection_plot(summary, toys_per_mu, directory / "injection.pdf", f"{target.name}: signal injection ({result['toys']})")
    for entry in summary.values():
        logger.info(f"{target.name}: mu {entry['mu']:g}: mean r {entry['r_mean']:.4g} ± {entry['r_mean_error']:.2g}, pull {entry['pull_mean']:+.3f} ± {entry['pull_mean_error']:.3f}, width {entry['pull_width']:.3f} ({entry['toys']} toys, {entry['failed']} failed)"
                    + (" BIASED" if entry["biased"] else ""))
    return result


def injection_lines(results: dict[str, dict]) -> list[str]:
    lines = []
    for name, res in results.items():
        fit = res["linearity"]
        line = f"mean r = ({fit['slope']:.3f} ± {fit['slope_error']:.3f}) μ + ({fit['offset']:.3g} ± {fit['offset_error']:.2g})" if fit else "no linearity fit (fewer than two injected values)"
        lines += [f"## {name}: toys of the {res['toys']}", "", line, ""]
        lines += ["| μ injected | toys (failed) | mean r | median r | sd r | mean pull | pull width | biased |", "|---|---|---|---|---|---|---|---|"]
        for entry in res["injected"].values():
            lines.append(f"| {entry['mu']:g} | {entry['toys']} ({entry['failed']}) | {entry['r_mean']:.4g} ± {entry['r_mean_error']:.2g} | {entry['r_median']:.4g} | {entry['r_sd']:.4g} | "
                         f"{entry['pull_mean']:+.3f} ± {entry['pull_mean_error']:.3f} | {entry['pull_width']:.3f} | {'yes' if entry['biased'] else 'no'} |")
        lines.append("")
    return lines


# -- HybridNew -------------------------------------------------------------------------------------------------------


def hybridnew_commands(combine: CombineConfig | None, masks: list[str], toys: int, rel_acc: float, abs_acc: float, a_posteriori: bool = False) -> list[str]:
    """HybridNew (LHC-limits) on the Asimov dataset (prefit, or of the a-posteriori snapshot with the masks set to 0);
    r from 0 to the upper end of `combine.r_range`; its own accuracy, never PRECISE_LIMIT."""
    first = snapshot_commands(combine, masks) if a_posteriori else [_workspace([])]
    source = snapshot_source(masks) if a_posteriori else "-d workspace.root"
    high = (combine or combine_options.DEFAULT).r_range[1]
    limit = (f"combine -M HybridNew {source} -m {MASS} --LHCmode LHC-limits -t -1 --expectSignal 0 -T {toys} --rAbsAcc {abs_acc:g} --rRelAcc {rel_acc:g} "
             f"-s {HYBRIDNEW_SEED} -n .HybridNew")
    return first + [_logged(combine_options.merge(combine_options.command(limit, combine), f"--setParameterRanges r=0,{high:g}"), "hybridnew.log")]


def run_hybridnew(target: Target, combine: CombineConfig | None, toys: int, rel_acc: float, abs_acc: float, asymptotic: dict | None = None, a_posteriori: bool = False, unblind: bool = False, skip_combine: bool = False) -> dict:
    """The HybridNew median expected limit; `asymptotic` the asymptotic limits to compare (prefit: limits.json,
    a-posteriori: aposteriori.json)."""
    directory = target.directory("hybridnew_aposteriori" if a_posteriori else "hybridnew")
    masks = masking.data_masks(target.card_dir, unblind) if a_posteriori else []
    if not skip_combine:
        logger.info(f"{target.name}: HybridNew ({'a-posteriori' if a_posteriori else 'prefit'} Asimov) in {directory}, {toys} toys per point")
        cmssw.run(hybridnew_commands(combine, masks, toys, rel_acc, abs_acc, a_posteriori), combine, directory)
    with uproot.open(directory / f"higgsCombine.HybridNew.HybridNew.mH{MASS}.{HYBRIDNEW_SEED}.root") as handle:
        arrays = handle["limit"].arrays(["limit", "limitErr"], library="np")
    result = {"asimov": "a-posteriori" if a_posteriori else "prefit", "limit": float(arrays["limit"][-1]), "limit_error": float(arrays["limitErr"][-1]), "toys": toys,
              "rel_acc": rel_acc, "abs_acc": abs_acc, "masked": _masked(masks)}
    median = (asymptotic or {}).get("limit", {}).get("exp_median")
    if median:
        result["asymptotic_median"] = median
        result["ratio"] = result["limit"] / median
    logger.info(f"{target.name}: HybridNew median expected limit {result['limit']:.4g} ± {result['limit_error']:.2g}" + (f", asymptotic {median:.4g} (ratio {result['ratio']:.3f})" if median else ""))
    return result


def hybridnew_lines(results: dict[str, dict]) -> list[str]:
    lines = ["| card | Asimov | HybridNew median expected | asymptotic median | HybridNew / asymptotic | toys per point |", "|---|---|---|---|---|---|"]
    for name, res in results.items():
        median = res.get("asymptotic_median")
        compared = "- | -" if median is None else f"{median:.4g} | {res['ratio']:.3f}"
        lines.append(f"| {name} | {res['asimov']} | {res['limit']:.4g} ± {res['limit_error']:.2g} | {compared} | {res['toys']} |")
    return lines


# -- summaries -------------------------------------------------------------------------------------------------------

LINES = {"gof": gof_lines, "impacts": impacts_lines, "aposteriori": aposteriori_lines, "injection": injection_lines, "hybridnew": hybridnew_lines}


def write_summary(check: str, results: dict[str, dict], output_dir: Path, title: str = "") -> tuple[Path, Path]:
    """<output_dir>/<check>.json and .md; the results of earlier runs of other cards in the JSON are kept."""
    output_dir = Path(output_dir)
    json_path = output_dir / f"{check}.json"
    previous = json.loads(json_path.read_text()) if json_path.exists() else {}
    merged = {**previous, **results}
    json_path.write_text(json.dumps(merged, indent=2))
    md_path = output_dir / f"{check}.md"
    md_path.write_text("\n".join(([f"# {title}", ""] if title else []) + LINES[check.split("_")[0]](merged)) + "\n")
    if check == "aposteriori":
        brazil([(name, res["limit"]) for name, res in merged.items()], output_dir / "aposteriori.pdf", r"95% CL a-posteriori expected limit on $\mu = \sigma/\sigma_\mathrm{SM}$")
    logger.info(f"{check} of {len(results)} cards: {json_path}, {md_path.name}")
    return json_path, md_path
