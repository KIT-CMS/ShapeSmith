# ShapeSmith

From CROWN ntuples to histograms, background estimates, combine datacards, fits, plots,
ML training data and measurements (correctionlib payloads). Columnar (uproot, pandas,
numexpr, numpy), no ROOT needed except for `shapesmith fit`, which calls combine in a CMSSW
subshell.

An analysis is a Python package that builds a `shapesmith.model.Analysis` from a small run
YAML; ShapeSmith runs the steps. Example analysis:
[BBTauTauAnalysis-ShapeSmith](https://github.com/KIT-CMS/BBTauTauAnalysis-ShapeSmith).

## Install

```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_108/x86_64-el9-gcc15-opt/setup.sh   # Python 3.12, uproot, pandas, mplhep, correctionlib, XRootD
python3 -m venv --system-site-packages ~/.venvs/shapesmith && source ~/.venvs/shapesmith/bin/activate
pip install -e ".[test]"          # editable: changes in this checkout are live
pytest                            # tests on synthetic ntuples, no network
```

## Steps

```bash
shapesmith validate  -c run.yaml              # build + validate the analysis, list processes and needed columns per channel
shapesmith skim      -c run.yaml [--channels mt] [--samples TT,SingleMuon] [--force] [--workers 8]   # ntuples (+ friends) -> Parquet
shapesmith hist      -c run.yaml [--control] [--regions all] [--skip-systematics] [--processes ZTT,data]  # histograms (ROOT + JSON index)
shapesmith estimate  -c run.yaml [--control]  # estimated processes and variations (Channel.estimators)
shapesmith measure   -c run.yaml [--suggest-binning] [--merge] [--refit]   # Analysis.measurement -> <output_dir>/<measurement>/<era>/
shapesmith plot      -c run.yaml [--control] [--region nominal] [--blind] [--log]
shapesmith publish   -c run.yaml --to DIR --variant KEY [--label TEXT] [--description TEXT] [--channels et,mt]   # plots + yields -> static web gallery DIR/index.html
shapesmith sync      -c run.yaml              # combine-style shape files per channel
shapesmith datacards -c run.yaml [--min-background 1.0] [--no-systematics]
shapesmith fit       -c run.yaml [--final-states mt,all] [--skip-combine]   # limits, significance, best fit
shapesmith fit       -c run.yaml --background-only [--final-states all]   # background-only fit to the data, pulls, yields
shapesmith combine   -c run_2017.yaml -c run_2018.yaml ... -o DIR [--final-state all] [--background-only] [--validate]   # cards of several eras combined
shapesmith combine   -c run.yaml --era 2017 --era 2018 ... -o DIR   # the same with one run YAML loaded per era ({era} filled)
shapesmith ml-export -c run.yaml              # training data: <ml_dir>/<channel>/events.feather + classes.json
shapesmith ml-merge  -c run.yaml --era 2017 --era 2018 ... -o DIR   # the exports of several eras in one per channel
shapesmith inspect   output/shapes.root [--unchanged]   # what a histogram file contains; variations equal to their nominal
```

`skim` reads the ntuples once into `<skim_dir>/<channel>/<nick>/*.parquet` plus a
`manifest.json`; everything else works on the skims. `skim`, `hist`, `measure` and `ml-export`
record what produced their outputs in `<directory>/versions/<analysis name>.json` (versions,
git hashes of core, analysis and sample database, the full run configuration and its hash).

`datacards` writes one card per final state, `<output_dir>/datacards/<final state>/combined.txt`;
`fit` runs combine on them in CMSSW (`combine.cmssw_dir`) and summarises the results in
`<output_dir>/datacards/limits.{json,md,pdf}`. `fit --background-only` instead fits the data with
r frozen to 0 (`combine -M FitDiagnostics --setParameters r=0 --freezeParameters r --skipSBFit
--cminDefaultMinimizerStrategy 0 --robustHesse 1 --saveNormalizations --saveShapes --saveWithUncertainties`; the
covariance comes from `robustHesse`, its quality is -1), by default on the combination of
the selected channels (`all`, or the only final state), and writes
`datacards/<final state>/fitDiagnostics.BackgroundOnly.root` and `datacards/background_fit.{json,md}`
(`--skip-combine`: from the stored values): per final state the fit status, per nuisance the prefit [mean, width] of its Gaussian constraint, postfit
[value, error], pull and constraint, and per channel, category and process the data and the
prefit and postfit [value, error] yields (with the totals `total_background`, `total_signal`,
`total`). It fits the observed data of every bin of the card.

`combine` (the combination of eras, one run YAML each, or one run YAML loaded once per `--era`) runs `combineCards.py` over the card of one final state of
every run (the bin names stay, they carry the era; two runs of one era are refused) into `DIR/combined.txt`, optionally
CombineHarvester's `ValidateDatacards.py`, then the limits or the background-only fit of `fit` there, with the same
summaries; `DIR/nuisances.md` lists every nuisance with the eras it appears in and flags names that end in another
run's era token.

`publish` copies the plots of `<output_dir>/plots/` into a web gallery (`DIR/index.html`, a static
viewer reading `manifest.json`; also works from `file://`): one column per variant (`--variant`,
`[a-z0-9_]+`, e.g. one run configuration), with the data, prediction and stacked-group yields of
every plotted region and category (from the `yield` control variable where there is one, computed
as the stack plot does) and the provenance of the histograms. Publishing a variant again replaces
its published channels and keeps the other variants and channels; only changed plots are copied
and everything is world-readable. Variants and channels keep the order of their first publication: a
new variant comes last, a new channel next to the channels of its run already in the gallery, else
last. `--title`/`--about` set the gallery's title and the text of its collapsed "About this gallery"
section. A directory that is neither empty nor a gallery is refused.
`web: {dir, variant, label, description, title, about, site}` in the run YAML sets the defaults of the
options. For an analysis with a measurement, `publish` takes the measurement's plots from
`<output_dir>/<measurement>/<era>/` instead (the fake factors: by process, split category and
quantity, with the p-value of each correction and the ttbar data/MC factors); a gallery holds
either control plots or one measurement.
`--site SITE` (`web.site`) makes the gallery one era of a site: `DIR` must be a subdirectory of
`SITE` (`web: {site: /web/USER/public_html/TAG, dir: "/web/USER/public_html/TAG/{era}"}`), and
`SITE/index.html` shows the galleries of every subdirectory with an era switch (the era in the
link, the most recent by default; channel, region, category, variable and columns stay where the
other era has them). Each publish rewrites the site's era list (`site.json`, `site.js`) from its
subdirectories; each era stays a complete gallery of its own. A gallery without a site stays as it is.

## The analysis model

Everything that differs by channel lives in its `Channel`: samples, cuts, processes, regions,
categories, control variables, variations and estimators.

- `Sample(nick, group, kind, xsec, nevents, generator_weight, cut, era, lumi_scale)`: kind `data`, `mc` or
  `embedding`; `norm_weight = xsec / (nevents * generator_weight) * lumi_scale * sign(genWeight)` for MC.
  A sample's own `cut` (e.g. one generator-level part of an inclusive sample) is applied at skim
  time; its normalisation stays that of the whole sample. `era` borrows the sample from another
  era, or places it in an analysis over several eras: its n-tuples are read from `<base>/<era>/<nick>` (the base of
  its kind), and it is
  normalised with the luminosity of the analysis that uses it (None: the era of the run) times `lumi_scale`, its
  era's share of that luminosity (e.g. 2016preVFP in a 2016 measurement of both halves; 1 otherwise).
- `Variable(name, expr, edges)`: fixed edges, or `EqualData(n_bins, low, high, cut)`, edges with equal data counts
  computed whenever histograms are filled, from the data of the category or, with `cut`, of that cut (common edges of
  a pass and a fail category), or, for a category, `ScoreBinning(fine, rules | thesis, blinding, clip)`, edges that
  `datacards` resolves from the nominal templates (see "Score categories").
- `Process(name, group, role, plot_group, selection)`: role `data`, `signal`, `background` or
  `auxiliary` (booked in the nominal region only, for an estimator; never in datacards, plots or
  the ML export). A process owns its whole weight set.
- `Region(name, replace_cuts, add_weights, replace_weights)`: cuts replaced by name, weights
  replaced where a process carries them, weights added to every process.
- `WeightVariation(name, replace_weights, applies_to, regions)` and
  `ColumnVariation(name, suffix | derived, applies_to, regions, groups)`: a column variation reads
  column `c` as `c + suffix` where that branch exists (a CROWN shift), or replaces it by
  `derived[c]`, an expression of nominal columns; `groups` restricts it to the samples of some
  groups (a CROWN shift produced for some samples only). Names ending in `Up`/`Down` need their
  partner and become datacard shapes; other names are templates.
- Estimators, run in order by `estimate`: `DataMinus(output, region, subtract, scale,
  clip_negative, target)`, `ABCD(output, b, c, d, subtract, target)`, both writing their output to
  the region `target` (default the nominal one; e.g. QCD in a diagnostic region),
  `TemplateShift(name, process, template, fraction)`, which also varies every template variation
  `t` of the process (as `<name>Up@t`), and `VariationSum(name, parts)`: `<name>Up/Down` =
  nominal + the sum of (part - nominal) for every histogram with a part, the part `p` being the column
  variation `<name>Up%p` / `<name>Down%p` (`histogram.part_of`), e.g. one nuisance from CROWN
  shifts produced per decay mode (exact where no event is changed by two parts). Parts never
  become datacard shapes; place the sum after the estimators whose outputs should carry it.
- `MLExportConfig(variables, classes, class_of, region_of, identifiers, mc_only, subtraction, profile)`: `ml-export`
  writes, per channel, the processes of `class_of` the channel has as signal, background or estimator output
  (`exported(channel)`), each in its class (`classes` in order: the class index); an estimator output is the data of
  its `region_of` region, which must not be the nominal one (see "ML export").
- `Analysis.control_channels`: channels that `datacards` and `fit` put into every final state
  (each selected channel and `all`) but never make a final state themselves, e.g. a ttbar
  control channel fitted together with the signal channels; `ml-export` skips them.
- `Analysis.lnn`: `LnN(name, processes, value, channels)`, `$CHANNEL` and `$ERA` expanded per bin;
  entries with the same name share one datacard line (at most one may apply to a column), e.g. one
  value on MC and another on the embedded sample. `Analysis.lnn_variations`: Up/Down variations
  (names without the direction) that datacards write as an lnN instead of a shape, `down/up` = the
  varied over the nominal yield per bin and process.

**The event rule** (`shapesmith.events`, used by hist, the ML export and measurements): cuts are
the channel cuts with the region's replacements, the process cuts and the skim cuts; weights are
the process weights with the region's replacements, then the region's added weights; a weight
variation replaces weights in place (and does not apply to a process lacking one of them), a
column variation rewrites every expression. Event weight: `(norm_weight * lumi) * product`,
`lumi` for MC only.

**Booking**: the signal in the nominal region, other processes in the nominal region and the
regions the estimators read and write (or `--regions`). A variation is filled where its
`applies_to` and `regions` allow; a column variation only where it changes an expression of the
histogram (elsewhere it equals the nominal). `DataMinus` builds its output for every column
variation of data or a subtracted process, taking the nominal of inputs that lack it; weight
variations are not propagated, ABCD is nominal only. `--skip-systematics` skips weight and column
variations. `hist` always refills the requested scopes and keeps the other histograms of its file.

## Score categories (NN binning and blinding)

A category whose variable has a `ScoreBinning` (e.g. an NN score) gets its edges in `datacards`, after `estimate`, from
the nominal templates; data are never read for them (`shapesmith.nn_binning`, `shapesmith.score_bins`).

- `hist` fills the `fine` edges (e.g. 1000 equal bins on [0, 1]) and, for the nominal histograms of every booked region,
  the unweighted counts of the events with a non-zero weight under the variable `<name>%raw`
  (`histogram.raw_variable`); `estimate` keeps them out of the estimators and gives a `DataMinus` or `ABCD` output
  those of the data it is made of.
- `rules` (`nn_binning.BinningRules`): the data-blind algorithm of the NN spec 6.3. Start bins of equal signal (`start:
  signal`, `n_start` 50) or equal total background (`background`, 5) in [`low`, upper fine edge] (`low` = 1/K for an
  argmax category of K classes), merged from the top until every bin has a total background >= `min_background` (1),
  a relative statistical uncertainty of the total background <= `max_rel_uncertainty` (0.30; background categories
  0.10), >= `min_raw` (1) unweighted events of every group of `required` (label -> background names, e.g. Top, tautau,
  jetFakes) and no negative template (signal included); at most `max_bins` (15; background categories 5) bins, the
  rest (and the scores below `low`) in the lowest accepted bin, one bin where none passes.
  `nn_binning.resolve_edges(fine_edges, signal_sumw, {name: (sumw, sumw2, raw_count)}, rules) -> edges` is the pure
  numpy function (`resolve` adds the record), `category_rules(n_classes, signal, required)` the rules of an HH or a
  background category.
- `thesis` (start edges, e.g. `nn_binning.THESIS_HH_EDGES`): the fine bins rebinned to them, then the AutoRebin of the
  master's thesis on the total background (`nn_binning.auto_rebin`, CombineHarvester's `AutoRebin` with
  SetBinThreshold(1.0), SetBinUncertFraction(0.9), SetRebinMode(1); tested against CombineHarvester).
- `clip`: processes whose negative bins left after the rebinning are set to zero (each template keeps its integral,
  nominal and variations); the outputs of estimators that clip (`ABCD`, `DataMinus(clip_negative=True)`) are clipped
  there too instead of in their fine bins. Every clipped bin is listed under `clipped` in `template_changes.json`.
- `blinding` (S/B(mu=1), 0.005 for N14): the bins from the first one with S/B above it (nominal templates, final
  edges) become the card channel `htt_<channel>_<k>hi_<era>` with `data_obs` = the sum of its background templates
  (Asimov), the bins below `htt_<channel>_<k>lo_<era>` with the data; no lo channel if the first bin is above, one
  unsplit channel if no bin is.

`datacards` writes per card `nn_binning.json` (per channel and category: method, rules, start and final edges, per bin
the signal, background, uncertainty, raw counts and failed rules, the blinding edge and S/B per bin, the card bins)
and `blinding.json`, the contract with every fit and limit:

```json
{"edge_rule": "S/B(mu=1) > 0.005", "masked": ["htt_mt_1hi_2018"],
 "categories": {"htt_mt_1hi_2018": {"lo": "htt_mt_1lo_2018", "edge": 0.912, "s_over_b_first_hi_bin": 0.0071}},
 "asimov_data": true}
```

`masked` lists exactly the card bins with Asimov `data_obs`: a fit or limit on real data sets `mask_<bin>=1` for each,
a prefit Asimov fit (`-t -1`) needs none. A card without a category with `blinding` has no `blinding.json` (a stale
one is removed): nothing is masked; one whose categories have no bin above the threshold has `"masked": []`. `datacards.bin_categories(analysis)` maps every possible card bin, the lo/hi parts
included, to its channel and category. `plot` and the yields of `publish` show a score category in the bins of its
card, with the prediction in place of the data in the blinded bins (shaded in the plot).

## Skims

Per sample, the skim keeps the columns of every expression its processes can use and the
shifted branches of its column variations. An event is kept if the skim cuts pass nominally or
under any column variation of its kind (hist re-applies the varied skim cuts). The skim adds the
bookkeeping columns `sample_nick`, `norm_weight`, `is_data`, `is_mc` and `is_embedding`; the kind
flags exist before the skim cuts, so a cut may depend on the kind (e.g. a trigger cut the
embedded sample does without). Where the channel
declares CROWN shifts for a sample (its kind and group), every file must carry a shifted branch for each of them,
and every shifted branch `c__X` of a needed column must be declared; otherwise the skim fails.

A friend is read for the samples of its kinds (`applies_to`) in its channels (`channels`, default
all), so a channel without that friend production (e.g. a control channel) does not look for it.

A sample is read from the base of its kind: `ntuples.bases` (e.g. `{mc: .../TAG_B/CROWNRun, embedding: ...}`, the
MC and embedding of a re-production beside the data of the first one), else `ntuples.base`; a friend split by kind is
two friends with their `applies_to`. A configuration without `bases` behaves, logs and is recorded (resolved
configuration, its hash) exactly as before.

The manifest records the skim contract. A stored skim is reused when its skim cuts,
normalisation, sample cut, n-tuple era (of a borrowed sample) and n-tuple base (of a kind in `ntuples.bases`) are unchanged, its recorded column variations and friends contain
the current ones, and its Parquet schema holds the needed columns; so one skim made with all
variations and friends serves every subset. Incompatible skims are listed together, with the
`skim --force --samples ...` fix. A new friend version gets a new friend base (a friend rewritten
in place is not detected). The manifest is written before the first file of a sample and after
every finished file, so an interrupted skim resumes with only the unfinished files. Parquet
files and manifests are replaced atomically.

## ML export

`ml-export` writes the training data of a run (`ml_dir`): per exported channel one flat Feather file
`<ml_dir>/<channel>/events.feather`, one row per event and exported process, and `<ml_dir>/classes.json`. The rows
are the selected events of every process of `MLExportConfig.class_of` in the nominal region, in its order; an
estimator output is the data of its `region_of` region with that region's weights (e.g. the fake factor) times the
scale of its `DataMinus`. With `subtraction` (default) the processes a `DataMinus` output subtracts follow, in its
region, with negated weights, in the class of the output (`is_subtraction`): the output's rows then add up to its
estimate. `--workers` loads the processes in parallel; the row order does not depend on it. The skims of a channel
hold the columns of the export (`skim.needed_columns`), except in a control channel, which is never exported.

| Column | Type | Content |
|---|---|---|
| identifiers (`MLExportConfig.identifiers`, e.g. `run`, `lumi`, `event`) | as in the n-tuple (e.g. uint32, uint32, uint64) | the event; (`era`, `channel`, `sample_nick`, identifiers) is its key |
| `era`, `channel`, `sample_nick`, `process` | category (dictionary-encoded string) | `process`: the exported process or estimator output; of a subtraction row the subtracted process |
| `row` | int64 | the position in the file (0, 1, ...): (`era`, `channel`, `row`) is unique |
| `class_index` | int8 | the position of the class in `classes` |
| `weight` | float64 | the event weight with its sign (normalisation, luminosity, every process and region weight) |
| `is_subtraction` | bool | a subtraction row |
| `thesis_training` | bool | the training half of a parity fold in the previous smhtt_ul export: the pattern True, True, False, False over the rows of each process (its `fold<k>_training` of K folds was `event % K == (k + 1) % K` and `thesis_training`) |
| the variables (`MLExportConfig.variables`) | float32 | the column, or the expression of a channel or category variable of that name; NaN for an `mc_only` variable in data and embedded rows |

There are no folds and no class weights in the files: a trainer takes its folds from the event number (`event % n`
on uint64) and its class weights from `weight`. `classes.json`: `profile`, `classes` (in class-index order),
`variables`, `mc_only`, `identifiers`, `regions` (`region_of`), `eras`, and per channel `processes` (class -> its
exported processes) and `subtraction` (estimator output -> the subtracted processes). A second export into the same
`ml_dir` must have the same class scheme (it adds its channels and eras), else it fails. The log lists the rows and the
signed weight of every class and warns about event keys that occur more than once.

`ml-merge` (one run YAML per era, or one run YAML with `--era` per era, like `combine`) concatenates the files of
every era's `ml_dir` per channel into `DIR/<channel>/events.feather` (the eras one after the other, `row` that of its
era) and writes `DIR/classes.json` with the eras and, per channel and class, the processes of any era. The eras must
have the same class scheme and the requested channels, and no era may occur twice. Weights keep the luminosity of
their era. The provenance records of each era (`versions/`) are copied to `DIR/versions/<era>/`.

## Samples

`shapesmith.samples.read_sample_list` reads a KingMaker sample list copied unchanged (one nick per
line); `normalisation(database, nicks)` looks every nick up in the KingMaker sample database
(`datasets.json`) and reports all missing nicks at once. The database renames nicks now and
then, so an older production needs the database checkout it was produced with.

## Measurements

`Analysis.measurement` is an object with a `name` and `run(context)`; `shapesmith measure` calls
it with a `MeasureContext` (configuration, analysis, channels, output directory,
`events(query)`, `provenance()`, and the flags `suggest_binning` and `merge`, the latter to combine
the results of earlier runs into one payload). Measurements:

- `shapesmith.measurements.tau_id_es`: tau-ID scale factor and energy scale of embedded taus, the
  chain of smhtt_ul `tauID_SFs_dev`. One run per working-point combination fills the mt channel
  and the mm control region, writes the shapes in the input format of MorphingTauID2017 and runs
  per category in CMSSW the datacards, T2W, a 2D likelihood scan and the profiles of both POIs (v7 spec 5.4, U11):
  ES from a 1D scan (41 points) with r profiled, r from the envelope of a 2D grid with both POIs fixed in the ES window
  of the ES profile (the likelihood has several minima in ES, which a floating ES follows); best value from the
  parabola at the lowest point, the 1σ interval from the interpolated ΔNLL = 0.5 crossings; a side without a crossing
  is rescanned on a widened range, and one that still has none is a problem naming its side. The postfit shapes come
  from a fit with both POIs fixed at the profile minima; the predecessor's singles fit runs as a cross-check
  (`report.md` flags deviations > 20 % of the interval width). `--refit` repeats the CMSSW part from the stored
  `synced/` shapes. The energy scale is a grid of template column variations in units of 0.1 % (`grid.py`);
  `--merge` writes the correctionlib payload of all combinations, its pT-binned ES changing linearly across 36-42 GeV
  (no gap in the corrected pT at 40 GeV), after checking every interval against its fit range.
- `shapesmith.measurements.fake_factors`: the fake-factor measurement (the SM method of
  TauFakeFactors). An analysis sets `Analysis.measurement = FakeFactorMeasurement(legs, name=...)`, built
  from its region and process names and per-channel tables (`Binned`, `Fit`, `Split`,
  `ProcessFF`, `DrSr`, `DataScale`, `SameSignQCD`, `Fractions`, `Leg`). A leg has QCD, ttbar and optionally W+jets
  fake factors: from data minus the MC and minus QCD from the same-sign version of each region (`SameSignQCD`), with a
  DR->SR correction from W MC (`DrSr(target=...)`), and three-process fractions. `shapesmith measure` writes
  `fake_factors_<ch>.json.gz` and `FF_corrections_<ch>.json.gz` (the conventions of the CROWN
  fake-factor friend), the intermediate DR->SR payload, `measurement_<ch>.json` (every ratio, fit
  input and result) and plots into `<output_dir>/<name>/<era>/` (one name per payload variant); `--suggest-binning`
  prints equipopulated edges instead, last as one JSON line; with `placeholder_edges` set (an era without its own
  edges) only `--suggest-binning` runs. The
  histograms (`hist.py`: centre of mass, MC-suppressed errors), fits (`fit.py`: bands,
  SystMCShift, the compatibility-with-1 reset, sparsify) and the kernel are checked against ROOT
  and TauFakeFactors (`tests/reference/make_fake_factor_reference.py`); the normative algorithm is
  the "Algorithm reference" of the design spec.
- `shapesmith.measurements.tau_trigger`: the efficiency of the tau leg of the di-tau trigger for the tt embedding (v7
  E33), μτh tag and probe per isolation class (`wp_class`) and selection variant: per (decay mode, pT) bin a pass and
  a fail category; a card with the signal (embedded taus) shape of pass + fail in both bins, the rateParams
  `eff` [0, 1] on it in pass and `(1-@0)` in fail and a common `r_<signal>`, lnN normalisations and the
  pass/fail anti-correlated fake trigger efficiency; a 1D profile scan of `eff` (an interval at 0 or 1 is a problem),
  the singles fit as a cross-check; the counting cross-check in a counting region and m_vis window, ε_MC and
  ε_EMB^mon. `--refit` from the stored shapes, `--suggest-binning` merges pT bins from the top until the statistical
  error of ε_data, counted in `suggest_regions` and `suggest_window`, stays below `max_error` (relative or
  absolute), `--merge` writes the `tau_trigger` payload (KIT/POG schema, flow clamp) with the variants
  and a significant fit-counting difference in up/down, after checking every combination; it refuses a class the
  consumer books (`wp_classes`) without a measurement. A signal shape may be a variation of the channel or an
  estimator output (a template shift of the signal, a summed variation). The ε(pT) plots draw any number of
  reference payloads (`references`, e.g. the POG and the KIT embedding file).

Building blocks:

- `shapesmith.payloads`: correctionlib schema-v2 builders; the provenance is JSON in
  `CorrectionSet.description`; payloads are checked (schema, evaluator, no `$schema` key, so
  correctionlib 2.6 reads them) and written gzip compressed.
- `shapesmith.measurements.smoothing`: TauFakeFactors' kernel smoothing without ROOT, a port of
  ROOT's `TGraphSmooth::SmoothKern` checked against ROOT and FF_Updated
  (`tests/reference/make_smoothing_reference.py`).
- `shapesmith.cmssw`: run commands in a CMSSW environment (`combine.cmssw_dir`), started from a
  clean environment so that the Python environment of ShapeSmith does not shadow CMSSW's.

## Run configuration

```yaml
analysis: my_analysis.analysis:build      # module:function returning an Analysis
era: "2018"
channels: [et, mt, tt]
switches: {jet_fakes: mc, embedding: false}   # free-form, interpreted by the analysis
sample_database: ../../KingMaker_sample_database/nanoAOD_v15/datasets.json   # relative paths: relative to this file
ntuples:
  server: root://cmsdcache-kit-disk.gridka.de
  base: /store/user/USER/CROWN/ntuples/TAG/CROWNRun      # <base>/<era>/<nick>/<channel>/*.root
  bases: {mc: /store/user/USER/CROWN/ntuples/TAG_B/CROWNRun}   # optional: the kinds with their own base (default: base)
  friends:                                # later friends override earlier ones; applies_to and channels default to all
    - {base: /store/user/USER/CROWN/ntuples/TAG/CROWNFriends/nn_v1, applies_to: [mc, data, embedding]}
    - {base: /store/user/USER/CROWN/ntuples/TAG/CROWNFriends/ff_v1, channels: [et, mt, tt]}
skim_dir: /ceph/USER/shapesmith/TAG
output_dir: output/TAG
ml_dir: /ceph/USER/shapesmith_ml/TAG
workers: 16
log_level: INFO                           # DEBUG | INFO | WARNING | ERROR
combine: {cmssw_dir: /work/USER/CMSSW_14_1_9, scram_arch: el9_amd64_gcc12}
web: {dir: /web/USER/public_html/TAG, variant: emb_ff, label: "embedding + FF"}   # shapesmith publish
```

`-s key=value` overrides an entry (values parsed as YAML). Then every `{era}` in a string of the configuration
(paths, friends, switches, web texts; not `era` itself) becomes the era of the run, so one YAML serves every era:
`skim_dir: /ceph/USER/shapesmith/TAG/skims/{era}`, `switches: {sample_lists: ["list_{era}"]}`, run with
`-s era=2017`. Inside a flow collection (`[...]`, `{...}`) a value with `{era}` is quoted. An era written as a
number (`era: 2018`, `-s era=2017`) is read as text.

## Logging

`log_level` sets the level of ShapeSmith and of the analysis package; one run changes it with
`-s log_level=DEBUG`. Other libraries (matplotlib, numexpr, uproot, ...) log from WARNING on.

- Console: the messages, on stderr (stdout keeps the reports of `validate` and `inspect`).
- Log file: every command but `validate` and `inspect` also writes
  `<output_dir>/logs/<command>_<YYYYmmdd-HHMMSS>.log` with time, level, process and logger, and the
  traceback of a failing command.
- INFO: the run (versions, configuration, channels, workers), the analysis per channel, what every
  stage does per channel, the progress of the parallel jobs (every tenth), the files written and
  the duration. DEBUG adds every ntuple file, booking, estimate, datacard bin, fit and plot, and
  the output of the CMSSW commands (at INFO it is shown only when they fail).
- The worker processes log through the main process, so their messages reach the console and
  the log file as well.

## Notes

- Expressions (cuts, weights, variables) are pandas `eval` syntax on ntuple columns, evaluated
  with numexpr: `(pt_1 > 25) & (abs(eta_1) < 2.1)`, `id_wgt_tau_1 * trg_wgt`.
- The edges of a variable are numbers or `EqualData(n_bins, low, high)`: `n_bins` bins with equal
  data counts in (low, high), as smhtt_ul `gof/build_binning.py` (`binning.py`). Every fill
  (`hist`, a measurement) computes them from the data of the category in the nominal region, uses
  them for every region and variation, and records them in `binning.json` next to the histogram
  file.
- Histograms are the small numpy `shapesmith.histogram.Histogram`; ROOT files are written and
  read through uproot (`boost-histogram`/`hist` are broken in LCG_108, their axis edges come out
  constant).
- `workers > 1` uses a spawn-based process pool (fork deadlocks with uproot/pyarrow threads).
  Scripts that call ShapeSmith functions with more than one worker therefore need the usual
  `if __name__ == "__main__":` guard; `--workers 1` runs everything inline for debugging.
- `tests/golden/` holds the mini-dataset outputs of core `aa5daa5`; the rebuilt core reproduces
  them bitwise (`tests/test_golden.py`).
