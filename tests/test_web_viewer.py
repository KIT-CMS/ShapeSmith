import re
import shutil
import subprocess
from pathlib import Path

import pytest

import shapesmith

STATIC = Path(shapesmith.__file__).parent / "web" / "static"
NODE_TESTS = sorted((Path(__file__).parent / "web").glob("*.test.mjs"))
node = pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")


def test_static_files_and_stamp_placeholder():
    for name in ("index.html", "viewer.js", "viewer.css"):
        assert (STATIC / name).is_file(), name
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    for name in ("manifest.js", "viewer.js", "viewer.css"):
        assert f'"{name}?v=__SHAPESMITH_STAMP__"' in html, name
    # manifest.js defines the gallery before the viewer runs
    assert html.index('src="manifest.js?') < html.index('src="viewer.js?')
    # no external resources: the page works offline and from file://
    assert not re.search(r'(src|href)="(https?:)?//', html)
    # the publisher makes the site page from it (gallery.SITE_PAGE): one <html lang="en"> and one manifest.js script
    assert html.count('<html lang="en">') == 1 and html.count('src="manifest.js?') == 1 and "data-level" not in html


@node
def test_viewer_script_syntax():
    subprocess.run(["node", "--check", str(STATIC / "viewer.js")], check=True, capture_output=True, text=True)


@node
def test_viewer_pure_logic():
    assert len(NODE_TESTS) >= 2  # viewer.test.mjs (pure logic) and site.test.mjs (the era switch of a site page)
    result = subprocess.run(["node", "--test", *map(str, NODE_TESTS)], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
