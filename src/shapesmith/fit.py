"""Run combine inside the CMSSW environment and collect expected limits, significance and best-fit r (Spec §10.3)."""
from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import uproot  # noqa: E402

from shapesmith.config import RunConfig  # noqa: E402

logger = logging.getLogger(__name__)

QUANTILES = {0.025: "exp_m2", 0.16: "exp_m1", 0.5: "exp_median", 0.84: "exp_p1", 0.975: "exp_p2", -1.0: "observed"}
COMMON = "-m {mass} --setParameterRanges r=-40,40 -t -1"


def combine_script(cmssw_dir: str, scram_arch: str, card_dir: Path, mass: str = "125") -> str:
    return "\n".join(
        [
            "set -e",
            f"export SCRAM_ARCH={scram_arch}",
            "source /cvmfs/cms.cern.ch/cmsset_default.sh",
            f"cd {cmssw_dir}/src",
            "eval $(scramv1 runtime -sh)",
            f"cd {Path(card_dir).resolve()}",
            f"text2workspace.py combined.txt -o workspace.root -m {mass}",
            f"combine -M AsymptoticLimits -d workspace.root {COMMON.format(mass=mass)} --expectSignal 0 -n .Limit",
            f"combine -M Significance -d workspace.root {COMMON.format(mass=mass)} --expectSignal 1 -n .Significance",
            f"combine -M MultiDimFit --algo singles -d workspace.root {COMMON.format(mass=mass)} --expectSignal 1 -n .Fit",
        ]
    )


def run_combine(config: RunConfig, card_dir: Path) -> None:
    if config.combine is None:
        raise ValueError("RunConfig.combine (cmssw_dir) is required to run combine")
    script = combine_script(config.combine.cmssw_dir, config.combine.scram_arch, card_dir)
    logger.info(f"running combine in {card_dir}")
    subprocess.run(["bash", "-c", script], check=True)


def _read(path: Path) -> list[tuple[float, float, float | None]]:
    with uproot.open(path) as f:
        tree = f["limit"]
        arrays = tree.arrays(library="np")
        r = arrays["r"] if "r" in arrays else [None] * len(arrays["limit"])
        return [(round(float(q), 3), float(limit), None if rr is None else float(rr)) for q, limit, rr in zip(arrays["quantileExpected"], arrays["limit"], r)]


def collect(card_dir: Path, mass: str = "125") -> dict:
    card_dir = Path(card_dir)
    result: dict = {}
    limit_file = card_dir / f"higgsCombine.Limit.AsymptoticLimits.mH{mass}.root"
    if limit_file.exists():
        result["limit"] = {QUANTILES[q]: limit for q, limit, _ in _read(limit_file) if q in QUANTILES}
    significance_file = card_dir / f"higgsCombine.Significance.Significance.mH{mass}.root"
    if significance_file.exists():
        result["significance"] = _read(significance_file)[0][1]
    fit_file = card_dir / f"higgsCombine.Fit.MultiDimFit.mH{mass}.root"
    if fit_file.exists():
        entries = _read(fit_file)
        best = [r for q, _, r in entries if q == -1.0]
        others = [r for q, _, r in entries if q != -1.0]
        result["r"] = {"best": best[0] if best else None, "low": min(others) if others else None, "high": max(others) if others else None}
    return result


def write_summary(results: dict[str, dict], output_dir: Path) -> tuple[Path, Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "limits.json"
    json_path.write_text(json.dumps(results, indent=2))
    lines = ["| final state | expected limit | 68% band | 95% band | significance | r best [68%] |", "|---|---|---|---|---|---|"]
    for name, res in results.items():
        limit, r = res.get("limit", {}), res.get("r", {})
        if limit:
            lines.append(
                f"| {name} | {limit['exp_median']:.2f} | [{limit['exp_m1']:.2f}, {limit['exp_p1']:.2f}] | [{limit['exp_m2']:.2f}, {limit['exp_p2']:.2f}] "
                f"| {res.get('significance', float('nan')):.2f} | {r.get('best')} [{r.get('low')}, {r.get('high')}] |"
            )
        else:
            lines.append(f"| {name} | - | - | - | - | - |")
    md_path = output_dir / "limits.md"
    md_path.write_text("\n".join(lines) + "\n")
    pdf_path = output_dir / "limits.pdf"
    states = [name for name, res in results.items() if res.get("limit")]
    fig, ax = plt.subplots(figsize=(6, 0.8 * max(len(states), 1) + 1.5))
    for i, name in enumerate(states):
        limit = results[name]["limit"]
        ax.barh(i, limit["exp_p2"] - limit["exp_m2"], left=limit["exp_m2"], color="#f5d000", height=0.6, label="95% expected" if i == 0 else None)
        ax.barh(i, limit["exp_p1"] - limit["exp_m1"], left=limit["exp_m1"], color="#00a651", height=0.6, label="68% expected" if i == 0 else None)
        ax.plot([limit["exp_median"]] * 2, [i - 0.3, i + 0.3], "k--", label="median expected" if i == 0 else None)
    ax.set_yticks(range(len(states)))
    ax.set_yticklabels(states)
    ax.set_xlabel(r"95% CL upper limit on $\sigma/\sigma_\mathrm{SM}$")
    if states:
        ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(pdf_path)
    plt.close(fig)
    return json_path, md_path, pdf_path


def run_fit(config: RunConfig, datacard_dir: Path, final_states: list[str], skip_combine: bool = False) -> dict:
    datacard_dir = Path(datacard_dir)
    results = {}
    for name in final_states:
        card_dir = datacard_dir / name
        if not skip_combine:
            run_combine(config, card_dir)
        results[name] = collect(card_dir)
    write_summary(results, datacard_dir)
    return results
