"""Exercise the public documentation build and its navigation checks."""

from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.packaging
sys.path.insert(0, str(ROOT / "scripts"))
from build_docs import PageLinks, validate_links


def test_site_publishes_documentation_and_preserves_deep_links():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/build_site.py")],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    output = ROOT / "site/_build"
    homepage = PageLinks((output / "index.html").read_text(encoding="utf-8"))
    assert "./docs/zh/index.html" in homepage.targets
    assert "./docs/en/index.html#using-the-python-library" in homepage.targets
    assert not any("github.com" in target and "/blob/" in target for target in homepage.targets)
    chinese = (output / "docs/zh/index.html").read_text(encoding="utf-8")
    assert "安装与启动" in PageLinks(chinese).anchors
    assert "gui-vector" in PageLinks(chinese).anchors
    assert "../en/index.html" in PageLinks(chinese).targets
    assert "<table>" in chinese and "<pre><code" in chinese
    assert (output / "docs/en/development.html").is_file()
    assert (output / "docs/zh/development.html").is_file()
    assert "downloads" in PageLinks((output / "docs/releases/v0.2.2.html").read_text()).anchors
    assert (output / "src/dichromatic_map/crystal.py").read_bytes() == (ROOT / "src/dichromatic_map/crystal.py").read_bytes()
    online = (output / "use.html").read_text(encoding="utf-8")
    assert './tutorial.js' in PageLinks(online).targets
    assert './render_data.js' in PageLinks(online).targets
    for name in ("tutorial.js", "render_data.js", "worker_queue.mjs"):
        assert (output / name).read_bytes() == (ROOT / "site" / name).read_bytes()
    validate_links(output)


@pytest.mark.parametrize("target", ["missing.html", "#missing", "https://github.com/login?return_to=/Yazhuo-Liu/DichromaticMap"])
def test_site_check_rejects_broken_or_authentication_links(tmp_path, target):
    (tmp_path / "index.html").write_text(f'<a href="{target}">Documentation</a>')
    with pytest.raises(ValueError):
        validate_links(tmp_path)
