import gzip
import json

import correctionlib
import numpy as np
import pytest

from shapesmith import payloads


def _payload():
    ff = payloads.binning("pt_2", np.array([30.0, 40.0, 150.0]), np.array([0.2, 0.15]))
    data = payloads.category("syst", {"StatShiftUp": payloads.binning("pt_2", [30.0, 150.0], [0.25])}, default=ff)
    by_dm = payloads.category("dm", {0: 1.1, 1: 0.9}, default=1.0)
    corrections = [
        payloads.correction("QCD_fake_factors", [payloads.variable("pt_2"), payloads.variable("syst", "string")], data, version=2),
        payloads.correction("tau_sf", [payloads.variable("dm", "int", "decay mode")], by_dm, description="per decay mode"),
    ]
    return payloads.correction_set(corrections, {"analysis": "mini", "era": "2018"})


def test_payload_evaluates_and_carries_its_provenance(tmp_path):
    path = payloads.write(_payload(), tmp_path / "out" / "fake_factors_mt.json.gz")
    text = gzip.decompress(path.read_bytes()).decode()
    assert "$schema" not in json.loads(text)
    evaluator = correctionlib.CorrectionSet.from_string(text)
    assert evaluator["QCD_fake_factors"].evaluate(35.0, "nominal") == pytest.approx(0.2)  # the default is the nominal
    assert evaluator["QCD_fake_factors"].evaluate(500.0, "nominal") == pytest.approx(0.15)  # clamped
    assert evaluator["QCD_fake_factors"].evaluate(35.0, "StatShiftUp") == pytest.approx(0.25)
    assert evaluator["tau_sf"].evaluate(1) == pytest.approx(0.9) and evaluator["tau_sf"].evaluate(10) == 1.0
    back = payloads.read(path)
    assert payloads.provenance(back) == {"analysis": "mini", "era": "2018"}
    assert back.corrections[0].version == 2 and back.corrections[1].inputs[0].description == "decay mode"


def test_plain_json_is_written_uncompressed(tmp_path):
    path = payloads.write(_payload(), tmp_path / "payload.json")
    assert json.loads(path.read_text())["schema_version"] == 2


def test_an_invalid_payload_is_rejected():
    broken = payloads.correction("x", [payloads.variable("pt")], payloads.binning("pt", [0.0, 1.0], [1.0]))
    broken.data.content.append(2.0)  # two values for one bin
    with pytest.raises(Exception):
        payloads.dumps(payloads.correction_set([broken], {}))


def test_transform_node():
    by_abs_eta = payloads.transform("eta", "abs(x)", payloads.binning("eta", [0.0, 1.5, 2.5], [1.0, 2.0]))
    cset = payloads.correction_set([payloads.correction("es", [payloads.variable("eta")], by_abs_eta)], {})
    evaluator = correctionlib.CorrectionSet.from_string(payloads.dumps(cset))
    assert evaluator["es"].evaluate(-2.0) == 2.0 and evaluator["es"].evaluate(0.5) == 1.0
