# ShapeSmith

From CROWN ntuples to histograms, background estimates, combine datacards, fits, plots,
ML training folds and measurements (correctionlib payloads). Columnar (uproot, pandas,
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
shapesmith measure   -c run.yaml [--suggest-binning]   # Analysis.measurement -> <output_dir>/<measurement>/<era>/
shapesmith plot      -c run.yaml [--control] [--region nominal] [--blind] [--log]
shapesmith sync      -c run.yaml              # combine-style shape files per channel
shapesmith datacards -c run.yaml [--min-background 1.0] [--no-systematics]
shapesmith fit       -c run.yaml [--final-states mt,all] [--skip-combine]   # limits, significance, best fit
shapesmith ml-export -c run.yaml              # Feather training folds
shapesmith inspect   output/shapes.root [--unchanged]   # what a histogram file contains; variations equal to their nominal
```

`skim` reads the ntuples once into `<skim_dir>/<channel>/<nick>/*.parquet` plus a
`manifest.json`; everything else works on the skims. `skim`, `hist`, `measure` and `ml-export`
record what produced their outputs in `<directory>/versions/<analysis name>.json` (versions,
git hashes of core, analysis and sample database, the full run configuration and its hash).

## The analysis model

Everything that differs by channel lives in its `Channel`: samples, cuts, processes, regions,
categories, control variables, variations and estimators.

- `Sample(nick, group, kind, xsec, nevents, generator_weight, cut)`: kind `data`, `mc` or
  `embedding`; `norm_weight = xsec / (nevents * generator_weight) * sign(genWeight)` for MC.
  A sample's own `cut` (e.g. one generator-level part of an inclusive sample) is applied at skim
  time; its normalisation stays that of the whole sample.
- `Process(name, group, role, plot_group, selection)`: role `data`, `signal`, `background` or
  `auxiliary` (booked in the nominal region only, for an estimator; never in datacards, plots or
  the ML export). A process owns its whole weight set.
- `Region(name, replace_cuts, add_weights, replace_weights)`: cuts replaced by name, weights
  replaced where a process carries them, weights added to every process.
- `WeightVariation(name, replace_weights, applies_to, regions)` and
  `ColumnVariation(name, suffix | derived, applies_to, regions)`: a column variation reads column
  `c` as `c + suffix` where that branch exists (a CROWN shift), or replaces it by `derived[c]`, an
  expression of nominal columns. Names ending in `Up`/`Down` need their partner and become
  datacard shapes; other names are templates.
- Estimators, run in order by `estimate`: `DataMinus(output, region, subtract, scale)`,
  `ABCD(output, b, c, d, subtract)` and `TemplateShift(name, process, template, fraction)`.

**The event rule** (`shapesmith.events`, used by hist, the ML export and measurements): cuts are
the channel cuts with the region's replacements, the process cuts and the skim cuts; weights are
the process weights with the region's replacements, then the region's added weights; a weight
variation replaces weights in place (and does not apply to a process lacking one of them), a
column variation rewrites every expression. Event weight: `(norm_weight * lumi) * product`,
`lumi` for MC only.

**Booking**: the signal in the nominal region, other processes in the nominal and the estimator
regions (or `--regions`). A variation is filled where its `applies_to` and `regions` allow; a
column variation only where it changes an expression of the histogram (elsewhere it equals the
nominal). `DataMinus` builds its output for every column variation of data or a subtracted
process, taking the nominal of inputs that lack it; weight variations are not propagated, ABCD
is nominal only. `--skip-systematics` skips weight and column variations. `hist` always refills
the requested scopes and keeps the other histograms of its file.

## Skims

Per sample, the skim keeps the columns of every expression its processes can use and the
shifted branches of its column variations. An event is kept if the skim cuts pass nominally or
under any column variation of its kind (hist re-applies the varied skim cuts). Where the channel
declares CROWN shifts for a sample kind, every file must carry a shifted branch for each of them,
and every shifted branch `c__X` of a needed column must be declared; otherwise the skim fails.

The manifest records the skim contract. A stored skim is reused when its skim cuts,
normalisation and sample cut are unchanged, its recorded column variations and friends contain
the current ones, and its Parquet schema holds the needed columns; so one skim made with all
variations and friends serves every subset. Incompatible skims are listed together, with the
`skim --force --samples ...` fix. A new friend version gets a new friend base (a friend rewritten
in place is not detected). The manifest is written before the first file of a sample and after
every finished file, so an interrupted skim resumes with only the unfinished files. Parquet
files and manifests are replaced atomically.

## Samples

`shapesmith.samples.read_sample_list` reads a KingMaker sample list copied unchanged (one nick per
line); `normalisation(database, nicks)` looks every nick up in the KingMaker sample database
(`datasets.json`) and reports all missing nicks at once. The database renames nicks now and
then, so an older production needs the database checkout it was produced with.

## Measurements

`Analysis.measurement` is an object with a `name` and `run(context)`; `shapesmith measure` calls
it with a `MeasureContext` (configuration, analysis, channels, output directory,
`events(query)`, `provenance()`). Building blocks:

- `shapesmith.payloads`: correctionlib schema-v2 builders; the provenance is JSON in
  `CorrectionSet.description`; payloads are checked (schema, evaluator, no `$schema` key, so
  correctionlib 2.6 reads them) and written gzip compressed.
- `shapesmith.measurements.smoothing`: TauFakeFactors' kernel smoothing without ROOT, a port of
  ROOT's `TGraphSmooth::SmoothKern` checked against ROOT and FF_Updated
  (`tests/reference/make_smoothing_reference.py`).
- `shapesmith.cmssw`: run commands in a CMSSW environment (`combine.cmssw_dir`).

`shapesmith.measurements.fake_factors` is the fake-factor measurement (the SM method of
TauFakeFactors): an analysis sets `Analysis.measurement = FakeFactorMeasurement(legs)`, built from
its region and process names and per-channel tables (`Binned`, `Fit`, `Split`, `ProcessFF`,
`DrSr`, `DataScale`, `Fractions`, `Leg`). `shapesmith measure` writes `fake_factors_<ch>.json.gz`
and `FF_corrections_<ch>.json.gz` (the conventions of the CROWN fake-factor friend), the
intermediate DR->SR payload, `measurement_<ch>.json` (every ratio, fit input and result) and plots;
`--suggest-binning` prints equipopulated edges instead. The histograms (`hist.py`: centre of mass,
MC-suppressed errors), fits (`fit.py`: bands, SystMCShift, the compatibility-with-1 reset,
sparsify) and the kernel are checked against ROOT and TauFakeFactors
(`tests/reference/make_fake_factor_reference.py`); the normative algorithm is the "Algorithm
reference" of the design spec.

## Run configuration

```yaml
analysis: my_analysis.analysis:build      # module:function returning an Analysis
era: "2018"
channels: [et, mt, tt]
switches: {jet_fakes: mc, embedding: false}   # free-form, interpreted by the analysis
sample_database: ../../KingMaker_sample_database/nanoAOD_v15/datasets.json   # relative paths: relative to this file
ntuples:
  server: root://cmsdcache-kit-disk.gridka.de
  base: /store/user/USER/CROWN/ntuples/TAG/CROWNRun
  friends:
    - {base: /store/user/USER/CROWN/ntuples/TAG/CROWNFriends/nn_v1, applies_to: [mc, data, embedding]}
skim_dir: /ceph/USER/shapesmith/TAG
output_dir: output/TAG
ml_dir: /ceph/USER/shapesmith_ml/TAG
workers: 16
combine: {cmssw_dir: /work/USER/CMSSW_14_1_9, scram_arch: el9_amd64_gcc12}
```

## Notes

- Expressions (cuts, weights, variables) are pandas `eval` syntax on ntuple columns, evaluated
  with numexpr: `(pt_1 > 25) & (abs(eta_1) < 2.1)`, `id_wgt_tau_1 * trg_wgt`.
- Histograms are the small numpy `shapesmith.histogram.Histogram`; ROOT files are written and
  read through uproot (`boost-histogram`/`hist` are broken in LCG_108, their axis edges come out
  constant).
- `workers > 1` uses a spawn-based process pool (fork deadlocks with uproot/pyarrow threads).
  Scripts that call ShapeSmith functions with more than one worker therefore need the usual
  `if __name__ == "__main__":` guard; `--workers 1` runs everything inline for debugging.
- `tests/golden/` holds the mini-dataset outputs of core `aa5daa5`; the rebuilt core reproduces
  them bitwise (`tests/test_golden.py`).
