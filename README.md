# ShapeSmith

From CROWN ntuples to histograms, background estimates, combine datacards, fits, plots and
ML training folds. Columnar (uproot, pandas, numexpr, numpy), no ROOT needed except for
`shapesmith fit`, which calls combine in a CMSSW subshell.

An analysis is a Python package that builds a `shapesmith.model.Analysis` (samples,
selections, processes, regions, categories, variables, systematics, style) from a small run
YAML; ShapeSmith runs the steps. Example analysis:
[BBTauTauAnalysis-ShapeSmith](https://github.com/KIT-CMS/BBTauTauAnalysis-ShapeSmith).

## Install

```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_108/x86_64-el9-gcc15-opt/setup.sh   # Python 3.12, uproot, pandas, mplhep, XRootD
python3 -m venv --system-site-packages ~/.venvs/shapesmith && source ~/.venvs/shapesmith/bin/activate
pip install -e ".[test]"          # editable: changes in this checkout are live
pytest                            # 67 tests on synthetic ntuples, no network
```

## Steps

```bash
shapesmith example-config run.yaml            # annotated run configuration
shapesmith validate  -c run.yaml              # build + validate the analysis, list the columns each channel needs
shapesmith skim      -c run.yaml [--channels mt] [--samples TT,SingleMuon] [--workers 8]   # ntuples (+ friends) -> Parquet
shapesmith hist      -c run.yaml [--control] [--skip-systematics] [--processes ztt,data]  # histograms (ROOT + JSON index)
shapesmith estimate  -c run.yaml [--control]  # data-driven processes (fake factors / ABCD), embedding variations
shapesmith plot      -c run.yaml [--control] [--blind] [--log]
shapesmith sync      -c run.yaml              # combine-style shape files per channel
shapesmith datacards -c run.yaml [--min-background 1.0] [--no-systematics]
shapesmith fit       -c run.yaml [--final-states mt,all] [--skip-combine]
shapesmith ml-export -c run.yaml              # Feather training folds
shapesmith inspect   output/shapes.root       # what a histogram file contains
shapesmith inventory datasets.json nicks.txt inventory.txt   # production inventory (nick + DBS path)
```

Two stages: `skim` reads the ntuples once (loose selection, all needed columns, normalisation
weight `norm_weight = xsec / (nevents * generator_weight) * sign(genWeight)`) into
`<skim_dir>/<channel>/<nick>/*.parquet` plus a `manifest.json`; everything else works on the
skims. `skim`, `hist` and `ml-export` write a `versions.json` (versions, git hashes of core,
analysis and sample database, full run configuration) next to their outputs.

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
- Systematics are weight variations from ntuple columns plus lnN normalisation uncertainties;
  NN scores and fake factors come from CROWN friend trees.
