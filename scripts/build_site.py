"""Build the static Pages site with the current numerical package sources."""

from pathlib import Path
import json
import shutil
import re
from zipfile import ZipFile, ZIP_DEFLATED

from build_docs import build_docs, validate_links


ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
OUTPUT = SITE / "_build"
PACKAGE = ROOT / "src" / "dichromatic_map"


def build() -> None:
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    OUTPUT.mkdir(parents=True)
    for name in (
        "index.html", "index.js", "style.css",
        "use.html", "use.css", "use.js", "tutorial.js", "use_worker.js", "web_bridge.py", "docs.css",
        "sitemap.xml",
    ):
        if name == "index.html":
            match = re.search(r'^version = "([^"]+)"$', (ROOT / "pyproject.toml").read_text(), re.M)
            if match is None:
                raise ValueError("Project version is missing from pyproject.toml")
            version = match.group(1)
            (OUTPUT / name).write_text((SITE / name).read_text().replace("__VERSION__", version))
        else:
            shutil.copy2(SITE / name, OUTPUT / name)
    assets = OUTPUT / "assets"
    assets.mkdir()
    for name in (
        "gui-overview.png", "dichromatic_pattern_example.png",
        "dichromaticmap_logo.svg",
        "dichromaticmap_logo_with_title.svg",
        "dichromaticmap_logo_with_title_light.svg",
    ):
        shutil.copy2(ROOT / "docs" / "images" / name, assets / name)
    vendor = OUTPUT / "vendor"
    vendor.mkdir()
    with ZipFile(vendor / "dichromatic_map.zip", "w", ZIP_DEFLATED) as archive:
        for source in sorted(PACKAGE.glob("*.py")):
            archive.write(source, f"dichromatic_map/{source.name}")
    build_docs(ROOT, OUTPUT)
    validate_site()
    validate_links(OUTPUT)


def validate_site() -> None:
    expected = {
        "index.html": "https://yazhuoliu.com/DichromaticMap/",
        "use.html": "https://yazhuoliu.com/DichromaticMap/use.html",
    }
    for filename, canonical in expected.items():
        html = (OUTPUT / filename).read_text()
        if html.count(f'<link rel="canonical" href="{canonical}">') != 1:
            raise ValueError(f"{filename} must declare one canonical URL: {canonical}")
        if '<meta name="robots" content="index, follow' not in html:
            raise ValueError(f"{filename} is missing an indexable robots directive")
        blocks = re.findall(
            r'<script type="application/ld\+json">\s*(.*?)\s*</script>', html, re.S
        )
        if not blocks:
            raise ValueError(f"{filename} is missing JSON-LD")
        for block in blocks:
            json.loads(block)

    sitemap = (OUTPUT / "sitemap.xml").read_text()
    for canonical in expected.values():
        if f"<loc>{canonical}</loc>" not in sitemap:
            raise ValueError(f"sitemap.xml is missing {canonical}")


if __name__ == "__main__":
    build()
    print(f"Built {OUTPUT}")
