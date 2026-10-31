"""The binning functions of NN-score categories (nn_binning.py): every rule of the data-blind algorithm, the AutoRebin
replica against CombineHarvester (tests/reference/make_auto_rebin_reference.py) and the blinding edge."""
import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from shapesmith.nn_binning import (
    BACKGROUND_RULES, SIGNAL_RULES, THESIS_HH_EDGES, BinningRules, auto_rebin, blinding_index, category_rules, resolve, resolve_edges, snap,
)

REFERENCE = json.loads((Path(__file__).parent / "reference" / "auto_rebin_combineharvester.json").read_text())
FINE = np.round(np.linspace(0.0, 1.0, 101), 6)  # 100 fine bins
TOP_GROUPS = {"Top": ("Top",), "tautau": ("tautau",), "jetFakes": ("jetFakes",)}


def per_start_bin(values, fine_per_bin=10):
    """Fine-bin values from one value per start bin of `fine_per_bin` fine bins, spread evenly."""
    return np.repeat(np.asarray(values, dtype=float) / fine_per_bin, fine_per_bin)


def templates(background, sumw2=None, raw=None, **more):
    """One background `Top` (per start bin of 10 fine bins) with its sumw2 and raw count, plus other named ones."""
    background = per_start_bin(background)
    result = {"Top": (background, per_start_bin(sumw2) if sumw2 is not None else background * 1e-4, per_start_bin(raw) if raw is not None else np.full(100, 100.0))}
    for name, values in more.items():
        result[name] = (per_start_bin(values), np.abs(per_start_bin(values)) * 1e-4, np.full(100, 100.0))
    return result


RULES = BinningRules(n_start=10)  # 10 start bins of equal (uniform) signal: 0.1 wide
SIGNAL = np.full(100, 0.25)  # exact running sums: the start edges are multiples of 0.1 (n_start 10) or 0.02 (50)


def test_the_background_rule_merges_from_the_top():
    edges = resolve_edges(FINE, SIGNAL, templates([5, 5, 5, 5, 5, 5, 5, 5, 0.6, 0.5]), RULES)
    assert edges == (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0)
    assert resolve_edges(FINE, SIGNAL, templates([5, 5, 5, 5, 5, 5, 5, 5, 0.6, 0.5]), dataclasses.replace(RULES, min_background=0.4)) == tuple(np.round(np.linspace(0, 1, 11), 6))


def test_the_uncertainty_rule_merges_until_50_percent():
    # top bins: 10 events with sumw2 40 each -> relative uncertainty 0.63, 0.45, 0.37, 0.32, 0.28 for 1..5 merged
    sumw2 = [0.01] * 5 + [40] * 5
    edges = resolve_edges(FINE, SIGNAL, templates([10] * 10, sumw2), RULES)
    assert edges == (0.0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0)  # pairs from the top; [0.5, 0.6) needs a bin with sumw2 0.01
    assert resolve_edges(FINE, SIGNAL, templates([10] * 10, sumw2), dataclasses.replace(RULES, max_rel_uncertainty=0.3)) == (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0)
    assert resolve_edges(FINE, SIGNAL, templates([10] * 10, sumw2), dataclasses.replace(RULES, max_rel_uncertainty=0.1)) == (0.0, 1.0)


def test_the_raw_count_rule_needs_an_event_of_every_group():
    """Top has no unweighted event in the two highest start bins; the third brings one."""
    backgrounds = templates([5] * 10, raw=[3] * 8 + [0, 0], tautau=[5] * 10, jetFakes=[5] * 10)
    with_groups = dataclasses.replace(RULES, required=TOP_GROUPS)
    assert resolve_edges(FINE, SIGNAL, backgrounds, with_groups) == (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0)
    assert resolve_edges(FINE, SIGNAL, backgrounds, RULES) == tuple(np.round(np.linspace(0, 1, 11), 6))
    # the counts of the names of a group add up
    split = dataclasses.replace(RULES, required={"Top": ("Top", "jetFakes")})
    assert resolve_edges(FINE, SIGNAL, backgrounds, split) == tuple(np.round(np.linspace(0, 1, 11), 6))


def test_a_negative_template_merges_until_it_is_above_one_percent_of_the_background():
    backgrounds = templates([5] * 10, jetFakes=[1] * 7 + [1.0, 0.1, -0.3])  # -0.3, -0.2, +0.8 from the top (B 4.7, 9.8)
    assert resolve_edges(FINE, SIGNAL, backgrounds, RULES) == (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0)
    assert resolve_edges(FINE, SIGNAL, backgrounds, dataclasses.replace(RULES, negative_share=None)) == tuple(np.round(np.linspace(0, 1, 11), 6))
    negative_signal = SIGNAL.copy()
    negative_signal[90:] = -0.01  # the signal counts as a template too (start bins from the background here)
    background_start = dataclasses.replace(RULES, start="background")
    assert resolve_edges(FINE, negative_signal, templates([5] * 10), background_start)[-3:] == (0.7, 0.8, 1.0)
    assert resolve_edges(FINE, negative_signal, templates([5] * 10), dataclasses.replace(background_start, negative_share=None))[-3:] == (0.8, 0.9, 1.0)


@pytest.mark.parametrize("top,share,merged", [
    (-0.04, 0.01, False),  # -0.04 of B = 5 + 1 - 0.04 = 5.96: -0.67 %, allowed
    (-0.07, 0.01, True),  # -1.18 %; merged -0.02 of 11.98 passes
    (-0.04, 0.0, True),  # negative_share 0: every template >= 0, the rule before C'
    (-5e-10, 0.0, False),  # within NEGATIVE_TOLERANCE
    (-0.04, 0.005, True),
    (-0.5, None, False),  # no negative rule
])
def test_the_negative_rule_allows_a_share_of_the_bin_background(top, share, merged):
    """Only the top start bin has a small negative template (ZL), the next one +0.05: it merges where it fails."""
    backgrounds = templates([5] * 10, jetFakes=[1] * 10, ZL=[0.0] * 8 + [0.05, top])
    rules = dataclasses.replace(RULES, negative_share=share)
    resolution = resolve(FINE, SIGNAL, backgrounds, rules)
    assert resolution.edges[-2:] == ((0.8, 1.0) if merged else (0.9, 1.0))
    assert resolution.passed


def test_the_raw_count_rule_applies_to_the_groups_with_ten_percent_of_the_background():
    """Top is 0.3 of 10.3 per start bin (2.9 %) and has no unweighted event in the two highest start bins: exempt from
    the rule with the default share, merged with raw_min_share 0 (the rule before C') or a share below 2.9 %."""
    backgrounds = templates([0.3] * 10, raw=[3] * 8 + [0, 0], tautau=[5] * 10, jetFakes=[5] * 10)
    rules = dataclasses.replace(RULES, required=TOP_GROUPS)
    resolution = resolve(FINE, SIGNAL, backgrounds, rules)
    assert resolution.edges == tuple(np.round(np.linspace(0, 1, 11), 6)) and resolution.passed
    assert resolution.raw_exempt == ("Top",)
    assert resolution.raw_shares == pytest.approx({"Top": 0.3 / 10.3, "tautau": 5 / 10.3, "jetFakes": 5 / 10.3})
    assert resolution.bins[-1].raw == {"Top": 0.0, "tautau": 1000.0, "jetFakes": 1000.0}  # counted, not required
    merged = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0)
    assert resolve_edges(FINE, SIGNAL, backgrounds, dataclasses.replace(rules, raw_min_share=0.0)) == merged
    assert resolve_edges(FINE, SIGNAL, backgrounds, dataclasses.replace(rules, raw_min_share=0.029)) == merged
    assert resolve_edges(FINE, SIGNAL, backgrounds, dataclasses.replace(rules, raw_min_share=0.03)) != merged
    assert resolve(FINE, SIGNAL, backgrounds, dataclasses.replace(rules, raw_min_share=0.0)).raw_exempt == ()


def test_the_raw_count_rule_applies_from_a_share_equal_to_raw_min_share():
    """Top is exactly half of the background (50 of 100): raw_min_share 0.5 applies the rule, a share above does not."""
    backgrounds = templates([5] * 10, raw=[3] * 8 + [0, 0], tautau=[5] * 10)
    rules = dataclasses.replace(RULES, required={"Top": ("Top",)}, raw_min_share=0.5)
    assert resolve(FINE, SIGNAL, backgrounds, rules).raw_exempt == ()
    assert resolve(FINE, SIGNAL, backgrounds, dataclasses.replace(rules, raw_min_share=0.5000001)).raw_exempt == ("Top",)


def test_the_negative_rule_of_a_bin_without_positive_background():
    """No share of a negative background: a small positive template never fails the negative rule."""
    signal = np.full(100, 1e-4)
    resolution = resolve(FINE, signal, templates([-0.5] * 10), dataclasses.replace(RULES, start="background"))
    assert resolution.edges == (0.0, 1.0) and resolution.bins[0].failed == ("background", "uncertainty", "negative:Top")


def test_the_raw_count_share_counts_the_background_from_low_on():
    """Top is large below low = 0.5 and 2 % above it: its share is that of the scores the category can have."""
    top = [50] * 5 + [0.2] * 5
    backgrounds = templates(top, raw=[3] * 8 + [0, 0], tautau=[5] * 10, jetFakes=[5] * 10)
    rules = dataclasses.replace(RULES, required=TOP_GROUPS, low=0.5, n_start=5)
    resolution = resolve(FINE, SIGNAL, backgrounds, rules)
    assert resolution.raw_exempt == ("Top",) and resolution.raw_shares["Top"] == pytest.approx(1.0 / 51.0)
    assert resolution.edges[-3:] == (0.8, 0.9, 1.0)
    assert resolve(FINE, SIGNAL, backgrounds, dataclasses.replace(rules, low=0.0, n_start=10)).raw_exempt == ()
    # without a positive background of the category no group has a share, and none is required
    empty = templates([0.0] * 10, raw=[0] * 10, tautau=[0.0] * 10, jetFakes=[0.0] * 10)
    nothing = resolve(FINE, SIGNAL, empty, dataclasses.replace(RULES, required=TOP_GROUPS))
    assert nothing.raw_shares == {"Top": None, "tautau": None, "jetFakes": None} and nothing.raw_exempt == tuple(TOP_GROUPS)
    assert nothing.bins[0].failed == ("background", "uncertainty")
    every = resolve(FINE, SIGNAL, empty, dataclasses.replace(RULES, required=TOP_GROUPS, raw_min_share=0.0))  # the rule before C'
    assert every.raw_exempt == () and every.bins[0].failed == ("background", "uncertainty", "raw:Top")  # Top has no event


def test_the_record_of_a_resolution():
    backgrounds = templates([5] * 10, raw=[3] * 10, tautau=[0.2] * 10)
    resolution = resolve(FINE, SIGNAL, backgrounds, dataclasses.replace(RULES, required={"Top": ("Top",), "tautau": ("tautau",)}))
    record = resolution.record()
    assert list(record) == ["raw_shares", "raw_exempt", "start", "capped", "leftover", "passed", "bins"]
    assert record["raw_exempt"] == ["tautau"] and record["raw_shares"]["Top"] == pytest.approx(5 / 5.2)
    assert record["start"] == list(resolution.start) and record["passed"] and record["leftover"] is None
    assert record["bins"][0] == {**dataclasses.asdict(resolution.bins[0])}
    json.dumps(record, allow_nan=False)
    failing = resolve(FINE, SIGNAL, templates([0.0] * 10), RULES).record()
    assert failing["bins"][0]["rel_uncertainty"] is None and not failing["passed"]  # inf -> None (strict JSON)


def test_the_cap_keeps_the_highest_bins_and_merges_the_rest_into_the_lowest():
    """50 start bins of 0.02 all pass: 14 single bins from the top and the 15th with everything below."""
    resolution = resolve(FINE, SIGNAL, templates([50] * 10), SIGNAL_RULES)
    assert len(resolution.edges) - 1 == 15 and resolution.capped and resolution.passed
    assert resolution.edges == pytest.approx((0.0,) + tuple(np.arange(0.72, 1.0001, 0.02)))
    assert resolution.leftover == (0.0, 0.7)
    background = resolve(FINE, SIGNAL, templates([50] * 10), dataclasses.replace(BACKGROUND_RULES, n_start=10))
    assert len(background.edges) - 1 == 5 and background.capped and background.edges[1:] == (0.6, 0.7, 0.8, 0.9, 1.0)
    assert not resolve(FINE, SIGNAL, templates([50] * 10), RULES).capped


def test_the_background_categories_start_from_equal_background():
    background = np.linspace(100, 10, 10)  # falling: fine bins of 10, 9, ..., 1; quintiles at 110, 220, 330, 440 of 550
    resolution = resolve(FINE, SIGNAL, templates(background), BACKGROUND_RULES)
    assert resolution.start == (0.0, 0.12, 0.24, 0.39, 0.58, 1.0) and resolution.edges == resolution.start


def test_the_signal_start_bins_have_equal_signal():
    signal = np.repeat(np.linspace(0.0, 0.2, 10), 10)  # rising
    resolution = resolve(FINE, signal, templates([1000] * 10), dataclasses.replace(RULES, n_start=5))
    cumulative = np.cumsum(signal)
    for k, edge in enumerate(resolution.start[1:-1], start=1):  # each edge closes the fine bin that reaches k / 5
        index = int(np.searchsorted(FINE, edge - 1e-9))
        assert cumulative[index - 1] >= cumulative[-1] * k / 5 > cumulative[index - 2]


def test_scores_below_low_and_failing_low_bins_join_the_lowest_accepted_bin():
    rules = dataclasses.replace(RULES, low=0.2, n_start=8, required={"Top": ("Top",)})
    raw = [5, 5, 0, 0, 3, 3, 3, 3, 3, 3]  # the start bins [0.2, 0.3) and [0.3, 0.4) have no Top event
    resolution = resolve(FINE, SIGNAL, templates([5] * 10, raw=raw), rules)
    assert resolution.start == tuple(np.round(np.linspace(0.2, 1.0, 9), 6)) and resolution.start[0] == 0.2
    assert resolution.edges == (0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
    assert resolution.leftover == (0.0, 0.4) and resolution.passed


def test_no_bin_passes():
    resolution = resolve(FINE, SIGNAL, templates([0.05] * 10), RULES)
    assert resolution.edges == (0.0, 1.0) and not resolution.passed and not resolution.capped and resolution.leftover is None
    assert resolution.bins[0].failed == ("background",)


def test_the_record_names_the_rules_a_final_bin_fails():
    """A negative leftover makes the lowest bin fail again; the record says which template."""
    backgrounds = templates([5] * 10, jetFakes=[-30] + [1] * 9)
    resolution = resolve(FINE, SIGNAL, backgrounds, dataclasses.replace(RULES, max_bins=3))
    assert len(resolution.bins) == 3 and resolution.bins[0].failed == ("negative:jetFakes",) and not resolution.passed
    assert resolution.bins[0].smallest == ("jetFakes", pytest.approx(-30 + 7 * 1))
    assert all(not b.failed for b in resolution.bins[1:])


def test_category_rules():
    """Rule set C' of spec 6.3: the HH category <= 50 %, the background categories <= 10 %, both the raw-count rule for
    the groups with >= 10 % of the background and no template below -1 % of the bin's background."""
    rules = category_rules(5, True, TOP_GROUPS)
    assert (rules.low, rules.min_background, rules.max_rel_uncertainty, rules.max_bins, rules.start, rules.n_start, rules.required) == (0.2, 1.0, 0.5, 15, "signal", 50, TOP_GROUPS)
    background = category_rules(6, False)
    assert (background.low, background.min_background, background.max_rel_uncertainty, background.max_bins, background.start, background.n_start) == (1 / 6, 1.0, 0.1, 5, "background", 5)
    for r in (rules, background):
        assert (r.min_raw, r.raw_min_share, r.negative_share) == (1.0, 0.1, 0.01)


@pytest.mark.parametrize("backgrounds,rules,match", [
    ({"Top": (np.ones(100), np.ones(100), np.ones(99))}, RULES, "need 100 values"),
    ({"Top": (np.ones(100), np.ones(100), np.ones(100))}, dataclasses.replace(RULES, required={"Top": ("Top", "QCD")}), "QCD"),
    ({"Top": (np.ones(100), np.ones(100), np.ones(100))}, dataclasses.replace(RULES, required={"Top": ()}), "groups without names"),
    ({"Top": (np.ones(100), np.ones(100), np.ones(100))}, dataclasses.replace(RULES, low=1.0), "low"),
    ({"Top": (np.ones(100), np.ones(100), np.ones(100))}, dataclasses.replace(RULES, raw_min_share=1.5), "raw_min_share"),
    ({"Top": (np.ones(100), np.ones(100), np.ones(100))}, dataclasses.replace(RULES, negative_share=-0.01), "negative_share"),
])
def test_inconsistent_inputs_are_refused(backgrounds, rules, match):
    with pytest.raises(ValueError, match=match):
        resolve_edges(FINE, SIGNAL, backgrounds, rules)


@pytest.mark.parametrize("group", ["random", "thesis"])
def test_auto_rebin_is_that_of_combineharvester(group):
    """Every case of the reference (300 random histograms and 20 on the thesis edges) as CombineHarvester rebins it."""
    for case in REFERENCE[group]:
        assert list(auto_rebin(case["edges"], case["values"], case["sumw2"])) == case["result"], case


def test_auto_rebin_merges_towards_fewer_bins_and_into_one_without_solution():
    edges = [0.0, 1.0, 2.0, 3.0, 4.0]
    # the smallest bin (0.5) needs one neighbour either way; equally good: towards the peak, the last bin
    assert auto_rebin(edges, [10, 0.5, 0.6, 20], [1, 0.1, 0.1, 1]) == (0.0, 1.0, 3.0, 4.0)
    # the same towards the first bin; the merged 10.5 leaves 0.6, which merges again
    assert auto_rebin(edges, [10, 0.5, 0.6, 10], [1, 0.1, 0.1, 1]) == (0.0, 3.0, 4.0)
    assert auto_rebin(edges, [10, 10, 10, 10], [1, 1, 1, 1]) == tuple(edges)
    assert auto_rebin(edges, [0.1, 0.1, 0.1, 0.1], [0.01] * 4) == (0.0, 4.0)
    assert auto_rebin(edges, [10, 5, 10, 10], [1, 30, 1, 1]) == (0.0, 2.0, 3.0, 4.0)  # relative error 1.1 >= 0.9


def test_snap():
    assert snap(THESIS_HH_EDGES, FINE)[:4] == [0, 20, 30, 35]
    with pytest.raises(ValueError, match="not a subset"):
        snap([0.0, 0.333, 1.0], FINE)


@pytest.mark.parametrize("signal,background,index", [
    ([0.001, 0.01, 0.1], [10, 10, 10], 2),
    ([0.001, 0.06, 0.04], [10, 10, 10], 1),  # the first bin above, not the bins from which on every one is
    ([0.001, 0.01, 0.02], [10, 10, 10], None),
    ([0.1, 0.2], [10, 10], 0),
    ([0.0, 0.1], [10, 0.0], 1),  # signal without background
    ([0.05, 0.0], [10, 0.0], None),  # 0.5 % is not above 0.5 %
])
def test_blinding_index(signal, background, index):
    assert blinding_index(signal, background, 0.005) == index
