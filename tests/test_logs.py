import logging

import pytest
from pydantic import ValidationError

from shapesmith import logs
from shapesmith.config import NtupleConfig, RunConfig
from shapesmith.parallel import run_jobs


@pytest.fixture
def restore_logging():
    """configure() changes the root logger and the package levels; restore them afterwards."""
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    levels = {name: logging.getLogger(name).level for name in ("shapesmith", "analysis_pkg", "matplotlib")}
    yield
    logs.reset()
    root.handlers[:] = handlers
    root.setLevel(level)
    for name, value in levels.items():
        logging.getLogger(name).setLevel(value)


def _config(tmp_path, **changes):
    return RunConfig(analysis="tests.mini_analysis:build", era="2018", channels=["mt"], ntuples=NtupleConfig(base=str(tmp_path)), skim_dir=tmp_path / "skims", output_dir=tmp_path / "out", **changes)


def test_log_level_of_the_config(tmp_path):
    assert _config(tmp_path).log_level == "INFO"
    assert _config(tmp_path, log_level="debug").log_level == "DEBUG"
    with pytest.raises(ValidationError):
        _config(tmp_path, log_level="verbose")


def test_configure_sets_the_package_levels_and_writes_the_file(tmp_path, restore_logging):
    path = logs.configure("DEBUG", tmp_path / "logs" / "run.log", ["analysis_pkg"])
    logging.getLogger("shapesmith.skim").debug("from the core")
    logging.getLogger("analysis_pkg.samples").debug("from the analysis")
    logging.getLogger("matplotlib.font_manager").info("third-party chatter")
    logging.getLogger("matplotlib").warning("third-party warning")
    logs.reset()
    text = path.read_text()
    assert "DEBUG   MainProcess shapesmith.skim: from the core" in text
    assert "analysis_pkg.samples: from the analysis" in text
    assert "third-party chatter" not in text and "third-party warning" in text


def test_level_filters_the_shapesmith_messages(tmp_path, restore_logging):
    path = logs.configure("WARNING", tmp_path / "run.log")
    logging.getLogger("shapesmith.fill").info("hidden")
    logging.getLogger("shapesmith.fill").warning("shown")
    logs.reset()
    assert "hidden" not in path.read_text() and "shown" in path.read_text()


def test_tracebacks_go_to_the_file_only(tmp_path, restore_logging, capsys):
    path = logs.configure("INFO", tmp_path / "run.log")
    try:
        raise RuntimeError("boom")
    except RuntimeError:
        logging.getLogger("shapesmith.cli").error("failed: boom", exc_info=True)
    logs.reset()
    assert "Traceback" in path.read_text() and "RuntimeError: boom" in path.read_text()
    console = capsys.readouterr()
    assert "failed: boom" in console.err and "Traceback" not in console.err and console.out == ""


def test_log_path(tmp_path):
    path = logs.log_path(tmp_path, "ml-export")
    assert path.parent == tmp_path / "logs" and path.name.startswith("ml-export_") and path.suffix == ".log"


def _logging_job(job):
    logging.getLogger("shapesmith.test_worker").debug(f"worker message {job[0]}")
    return job[0]


def test_worker_messages_reach_the_main_process(tmp_path, restore_logging):
    path = logs.configure("DEBUG", tmp_path / "run.log")
    results = sorted(result for _, result, _ in run_jobs(_logging_job, [(1,), (2,)], workers=2, label="test"))
    logs.reset()
    text = path.read_text()
    assert results == [1, 2]
    assert "worker message 1" in text and "worker message 2" in text
    assert "MainProcess" not in [line.split()[3] for line in text.splitlines() if "worker message" in line]
    assert "test: 2/2 jobs done" in text


def test_progress_steps():
    assert logs.progress_steps(3) == [1, 2, 3]
    assert logs.progress_steps(100) == [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    assert logs.progress_steps(0) == []


def test_duration():
    assert logs.duration(4.24) == "4.2 s"
    assert logs.duration(125) == "2 min 05 s"
    assert logs.duration(3725) == "1 h 02 min"
