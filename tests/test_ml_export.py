import numpy as np
import pandas as pd
import pytest

from shapesmith.ml_export import class_weights, export_process, fold_masks, run_ml_export, tuple_column


def test_tuple_column_pads_to_five():
    assert tuple_column("Nominal", "weight") == ("Nominal", "weight", "", "", "")
    assert tuple_column("Nominal", "variables", "m_vis") == ("Nominal", "variables", "m_vis", "", "")


def test_fold_masks_match_the_legacy_convention():
    event = np.arange(8)
    masks = fold_masks(event, folds=2)
    assert masks["fold0"].tolist() == [False, True, False, True, False, True, False, True]  # odd events
    assert masks["fold1"].tolist() == [True, False, True, False, True, False, True, False]
    assert masks["fold0_training"].tolist() == [False, True, False, False, False, True, False, False]  # pattern T,T,F,F on rows
    assert (masks["fold0_training"] | masks["fold0_validation"]).tolist() == masks["fold0"].tolist()


def test_class_weights_balance_classes():
    weights = np.array([1.0, 1.0, 2.0, 6.0])
    labels = np.array([0, 0, 1, 1])
    cw = class_weights(weights, labels)
    assert cw[labels == 0].sum() == pytest.approx(cw[labels == 1].sum()) == pytest.approx(weights.sum())


def test_export_process_layout(mini_run):
    config, analysis = mini_run
    frame = export_process(config, analysis, "mt", "ZTT")
    assert frame.columns.nlevels == 5
    columns = set(frame.columns)
    assert tuple_column("Event", "id") in columns and tuple_column("Event", "event") in columns
    assert tuple_column("Labels", "is_Z") in columns and tuple_column("Labels", "is_HH") in columns and tuple_column("Labels", "is_jetFakes") in columns
    assert tuple_column("Nominal", "variables", "m_vis") in columns and tuple_column("Nominal", "variables", "score") in columns
    assert tuple_column("Nominal", "weight") in columns and tuple_column("Nominal", "cut") in columns
    assert (frame[tuple_column("Labels", "is_Z")] == 1).all() and (frame[tuple_column("Labels", "is_HH")] == 0).all()
    fakes = export_process(config, analysis, "mt", "jetFakes")  # data in the anti-isolated region, weighted with fake factors
    assert (fakes[tuple_column("Labels", "is_jetFakes")] == 1).all()
    assert (fakes[tuple_column("Nominal", "weight")] < 1.0).all()


def test_run_ml_export_writes_folds(mini_run):
    config, analysis = mini_run
    files = run_ml_export(config, analysis, ["mt"])
    names = sorted(p.name for p in files)
    assert names == ["fold0.feather", "fold0_training.feather", "fold0_validation.feather", "fold1.feather", "fold1_training.feather", "fold1_validation.feather"]
    fold0 = pd.read_feather(config.ml_dir / "mt" / "fold0.feather")
    assert fold0.columns.nlevels == 5
    assert (fold0[tuple_column("Event", "event")] % 2 == 1).all()
    labels = fold0["Labels"].to_numpy().argmax(axis=1)
    cw = fold0[tuple_column("Nominal", "class_weight")].to_numpy()
    sums = [cw[labels == c].sum() for c in np.unique(labels)]
    assert np.allclose(sums, sums[0])
