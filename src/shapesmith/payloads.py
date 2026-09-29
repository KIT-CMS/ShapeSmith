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

Node = cs.Binning | cs.Category | float


def variable(name: str, type: str = "real", description: str | None = None) -> cs.Variable:
    fields = {"description": description} if description is not None else {}
    return cs.Variable(name=name, type=type, **fields)


def binning(input: str, edges: Sequence[float], content: Sequence[Node], flow: str | Node = "clamp") -> cs.Binning:
    """A binned node; numbers (also numpy arrays) become plain floats."""
    content = [float(c) if isinstance(c, (int, float, np.number)) else c for c in content]
    return cs.Binning(nodetype="binning", input=input, edges=[float(e) for e in edges], content=content, flow=flow)


def category(input: str, content: Mapping[str | int, Node], default: Node | None = None) -> cs.Category:
    items = [cs.CategoryItem(key=key, value=float(value) if isinstance(value, (int, float, np.number)) else value) for key, value in content.items()]
    fields = {"default": float(default) if isinstance(default, (int, float, np.number)) else default} if default is not None else {}
    return cs.Category(nodetype="category", input=input, content=items, **fields)


def correction(name: str, inputs: Sequence[cs.Variable], data: Node, version: int = 1, description: str | None = None, output: cs.Variable | None = None) -> cs.Correction:
    fields = {"description": description} if description is not None else {}
    return cs.Correction(name=name, version=version, inputs=list(inputs), output=output or variable("weight"), data=data, **fields)


def correction_set(corrections: Sequence[cs.Correction], provenance: Mapping) -> cs.CorrectionSet:
    return cs.CorrectionSet(schema_version=2, description=json.dumps(provenance, sort_keys=True), corrections=list(corrections))


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
