"""correctionlib payloads (schema v2): node builders, provenance, the schema check and gzip I/O.

The provenance of a payload (what produced it) is a JSON object in `CorrectionSet.description`. Payloads are written
without unset fields, so that files written with a newer correctionlib (which adds a `$schema` key) stay readable by
correctionlib 2.6 (LCG_108) and by CROWN.
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Mapping, Sequence

import correctionlib
import correctionlib.schemav2 as cs
import numpy as np

Node = cs.Binning | cs.Category | cs.Transform | cs.Formula | float


def _leaf(value):
    """A number (also a numpy scalar) as a plain float; a node unchanged."""
    return float(value) if isinstance(value, (int, float, np.number)) else value


def variable(name: str, type: str = "real", description: str | None = None) -> cs.Variable:
    fields = {"description": description} if description is not None else {}
    return cs.Variable(name=name, type=type, **fields)


def binning(input: str, edges: Sequence[float], content: Sequence[Node], flow: str | Node = "clamp") -> cs.Binning:
    """A binned node; numbers (also numpy arrays) become plain floats."""
    return cs.Binning(nodetype="binning", input=input, edges=[float(e) for e in edges], content=[_leaf(c) for c in content], flow=flow)


def category(input: str, content: Mapping[str | int, Node], default: Node | None = None) -> cs.Category:
    items = [cs.CategoryItem(key=key, value=_leaf(value)) for key, value in content.items()]
    fields = {"default": _leaf(default)} if default is not None else {}
    return cs.Category(nodetype="category", input=input, content=items, **fields)


def formula(expression: str, variables: Sequence[str], parameters: Sequence[float] = ()) -> cs.Formula:
    """A TFormula of the inputs `variables` (x, y, ... in order) with the parameters [0], [1], ..."""
    fields = {"parameters": [float(p) for p in parameters]} if parameters else {}
    return cs.Formula(nodetype="formula", expression=expression, parser="TFormula", variables=list(variables), **fields)


def transform(input: str, expression: str, content: Node) -> cs.Transform:
    """`content` evaluated with `input` replaced by the TFormula `expression` of it (x), e.g. "abs(x)"."""
    rule = cs.Formula(nodetype="formula", expression=expression, parser="TFormula", variables=[input])
    return cs.Transform(nodetype="transform", input=input, rule=rule, content=content)


def correction(name: str, inputs: Sequence[cs.Variable], data: Node, version: int = 1, description: str | None = None, output: cs.Variable | None = None) -> cs.Correction:
    fields = {"description": description} if description is not None else {}
    return cs.Correction(name=name, version=version, inputs=list(inputs), output=output or variable("weight"), data=data, **fields)


def compound(name: str, inputs: Sequence[cs.Variable], stack: Sequence[str], output: cs.Variable | None = None) -> cs.CompoundCorrection:
    """The product of the corrections in `stack`, each evaluated on its own inputs (taken from `inputs` by name)."""
    return cs.CompoundCorrection(name=name, inputs=list(inputs), output=output or variable("weight"), inputs_update=[], input_op="*", output_op="*", stack=list(stack))


def correction_set(corrections: Sequence[cs.Correction], provenance: Mapping, compound_corrections: Sequence[cs.CompoundCorrection] = ()) -> cs.CorrectionSet:
    fields = {"compound_corrections": list(compound_corrections)} if compound_corrections else {}
    return cs.CorrectionSet(schema_version=2, description=json.dumps(provenance, sort_keys=True), corrections=list(corrections), **fields)


def provenance(cset: cs.CorrectionSet) -> dict:
    return json.loads(cset.description)


def dumps(cset: cs.CorrectionSet) -> str:
    """The payload JSON, checked: valid schema v2, loadable by the correctionlib evaluator, no `$schema` key."""
    text = cset.model_dump_json(exclude_unset=True)
    if "$schema" in json.loads(text):
        raise ValueError("payload carries a $schema key, which correctionlib 2.6 rejects")
    cs.CorrectionSet.model_validate_json(text)
    correctionlib.CorrectionSet.from_string(text)
    return text


def write(cset: cs.CorrectionSet, path: Path) -> Path:
    """Write the checked payload; a `.gz` path is gzip compressed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = dumps(cset).encode()
    path.write_bytes(gzip.compress(data, mtime=0) if path.suffix == ".gz" else data)
    return path


def read(path: Path) -> cs.CorrectionSet:
    data = Path(path).read_bytes()
    return cs.CorrectionSet.model_validate_json(gzip.decompress(data) if Path(path).suffix == ".gz" else data)


def _node_points(node, points: dict[str, set]) -> None:
    """Collect the bin edges (binning) and keys (category) of every input below a node, by input name."""
    if isinstance(node, dict):
        if node.get("nodetype") == "binning":
            points.setdefault(node["input"], set()).update(("edge", float(e)) for e in node["edges"])
        elif node.get("nodetype") == "category":
            points.setdefault(node["input"], set()).update(("key", item["key"]) for item in node["content"])
        for value in node.values():
            _node_points(value, points)
    elif isinstance(node, list):
        for value in node:
            _node_points(value, points)


def grid(correction: dict, fixed: Mapping[str, object] | None = None) -> list[tuple]:
    """Evaluation points of a correction (its JSON): per real input the centres between its bin edges, per category
    input its keys, `fixed` inputs at their given value; the cartesian product in input order."""
    import itertools

    fixed = dict(fixed or {})
    found: dict[str, set] = {}
    _node_points(correction["data"], found)
    axes = []
    for variable in correction["inputs"]:
        name = variable["name"]
        if name in fixed:
            axes.append([fixed[name]])
            continue
        entries = found.get(name, set())
        edges = sorted(value for kind, value in entries if kind == "edge")
        keys = sorted((value for kind, value in entries if kind == "key"), key=str)
        values = [0.5 * (a + b) for a, b in zip(edges[:-1], edges[1:]) if b > a] if edges else keys
        if not values:
            raise ValueError(f"{correction['name']}: no points for the input {name}")
        axes.append(values)
    return list(itertools.product(*axes))


def compare(first: Path, second: Path, fixed: Mapping[str, object] | None = None) -> dict[str, dict]:
    """Every correction of both payloads evaluated on the grid of both (their edges and keys): correction -> number
    of points, max and mean of |second / first - 1| (points where the first is 0 are left out) and the number of
    points with a category key that only one of them has (`unmatched`, left out too; e.g. the process Wjets of the
    fractions of a fake-factor payload with W+jets against one without)."""
    sets = [json.loads(gzip.decompress(Path(p).read_bytes()) if Path(p).suffix == ".gz" else Path(p).read_bytes()) for p in (first, second)]
    evaluators = [correctionlib.CorrectionSet.from_string(json.dumps(s)) for s in sets]
    by_name = [{c["name"]: c for c in s["corrections"]} for s in sets]
    result = {}
    for name in by_name[0]:
        if name not in by_name[1]:
            continue
        points = sorted(set(grid(by_name[0][name], fixed)) | set(grid(by_name[1][name], fixed)), key=str)
        deviations, unmatched = [], 0
        for point in points:
            try:
                a, b = (evaluator[name].evaluate(*point) for evaluator in evaluators)
            except IndexError:  # a category key of only one payload
                unmatched += 1
                continue
            if a != 0:
                deviations.append(abs(b / a - 1.0))
        result[name] = {"points": len(deviations), "max": max(deviations, default=0.0), "mean": float(np.mean(deviations)) if deviations else 0.0, "unmatched": unmatched}
    return result
