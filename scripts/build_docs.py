"""Publish repository documentation as public, self-contained HTML pages."""

from html import escape, unescape
from html.parser import HTMLParser
import os
from pathlib import Path
import re
import shutil
from urllib.parse import quote, unquote, urlsplit, urlunsplit
import xml.etree.ElementTree as ET

import markdown


SITE_URL = "https://yazhuoliu.com/DichromaticMap/"
GITHUB_FILES = (
    "https://github.com/Yazhuo-Liu/DichromaticMap/blob/main/",
    "https://raw.githubusercontent.com/Yazhuo-Liu/DichromaticMap/main/",
)


def heading_slug(value: str, separator: str) -> str:
    """Keep Chinese headings and the existing GitHub-style deep links."""
    return re.sub(r"[^\w\s-]", "", value.strip().lower()).replace(" ", separator)


def build_docs(root: Path, output: Path) -> None:
    shutil.copy2(root / "LICENSE", output / "LICENSE")
    sources = [root / "README.md", root / "CONTRIBUTING.md", *sorted((root / "docs").rglob("*.md"))]
    pages = {}
    for source in sources:
        relative = source.relative_to(root)
        if relative == Path("README.md"):
            destination = Path("docs/project.html")
        elif relative == Path("CONTRIBUTING.md"):
            destination = Path("docs/contributing.html")
        elif source.name == "README.md":
            destination = relative.with_name("index.html")
        else:
            destination = relative.with_suffix(".html")
        pages[source] = destination

    def relative_url(target: Path, page: Path) -> str:
        return quote(Path(os.path.relpath(target, page.parent)).as_posix(), safe="/-._~")

    for source, page in pages.items():
        body = markdown.markdown(
            source.read_text(encoding="utf-8"),
            extensions=["extra", "toc"],
            extension_configs={"toc": {"slugify": heading_slug}},
        )

        def rewrite(match: re.Match) -> str:
            attribute, url = match.groups()
            link = urlsplit(unescape(url))
            repository_path = None
            for prefix in GITHUB_FILES:
                if unescape(url).startswith(prefix):
                    repository_path = root / unquote(link.path[len(urlsplit(prefix).path):])
                    break
            if repository_path is None:
                if link.scheme or link.netloc or not link.path:
                    return match.group(0)
                repository_path = (source.parent / unquote(link.path)).resolve()
            if not repository_path.is_relative_to(root) or not repository_path.is_file():
                raise ValueError(f"Missing documentation target: {source.relative_to(root)}: {url}")
            if repository_path in pages:
                target = pages[repository_path]
            else:
                target = repository_path.relative_to(root)
                destination = output / target
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(repository_path, destination)
            rewritten = urlunsplit(("", "", relative_url(target, page), link.query, link.fragment))
            return f'{attribute}="{escape(rewritten, quote=True)}"'

        body = re.sub(r'\b(href|src)="([^"]*)"', rewrite, body)
        title_match = re.search(r"<h1\b[^>]*>(.*?)</h1>", body, re.S)
        title = unescape(re.sub(r"<[^>]+>", "", title_match.group(1))) if title_match else source.stem
        language = "zh-CN" if "zh" in page.parts else "en"
        links = [("index.html", "Website"), ("use.html", "Use online"),
                 ("docs/en/index.html", "English"), ("docs/zh/index.html", "中文")]
        navigation = "".join(f'<a href="{relative_url(Path(target), page)}">{label}</a>' for target, label in links)
        html = f'''<!doctype html>
<html lang="{language}">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)} · DichromaticMap</title>
  <link rel="canonical" href="{SITE_URL}{page.as_posix()}">
  <meta name="robots" content="index, follow">
  <link rel="stylesheet" href="{relative_url(Path('docs.css'), page)}">
  <link rel="icon" type="image/svg+xml" href="{relative_url(Path('assets/dichromaticmap_logo.svg'), page)}">
</head>
<body><nav aria-label="Documentation navigation">{navigation}</nav><main>{body}</main></body>
</html>
'''
        destination = output / page
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(html, encoding="utf-8")

    namespace = "http://www.sitemaps.org/schemas/sitemap/0.9"
    ET.register_namespace("", namespace)
    sitemap = ET.parse(output / "sitemap.xml")
    for page in pages.values():
        entry = ET.SubElement(sitemap.getroot(), f"{{{namespace}}}url")
        ET.SubElement(entry, f"{{{namespace}}}loc").text = SITE_URL + page.as_posix()
    sitemap.write(output / "sitemap.xml", encoding="utf-8", xml_declaration=True)


class PageLinks(HTMLParser):
    def __init__(self, html: str):
        super().__init__()
        self.targets = []
        self.anchors = set()
        self.feed(html)

    def handle_starttag(self, tag, attributes):
        attributes = dict(attributes)
        for attribute in ("id", "name"):
            if attribute in attributes:
                self.anchors.add(attributes[attribute])
        for attribute in ("href", "src"):
            if attributes.get(attribute):
                self.targets.append(attributes[attribute])


def validate_links(output: Path) -> None:
    """Fail the build on broken local targets, deep links, or login routes."""
    output = output.resolve()
    pages = {path: PageLinks(path.read_text(encoding="utf-8")) for path in output.rglob("*.html")}
    for page, parsed in pages.items():
        for url in parsed.targets:
            link = urlsplit(url)
            if link.netloc == "github.com" and re.search(r"/(login|signup|settings)(/|$)|/(issues|pull)/new(/|$)", link.path):
                raise ValueError(f"Authentication link in {page.relative_to(output)}: {url}")
            if url.startswith(SITE_URL):
                target = output / unquote(link.path[len(urlsplit(SITE_URL).path):])
            elif link.scheme or link.netloc:
                continue
            else:
                target = (page.parent / unquote(link.path)).resolve() if link.path else page
            if target.is_dir():
                target /= "index.html"
            if not target.is_relative_to(output) or not target.is_file():
                raise ValueError(f"Broken site link in {page.relative_to(output)}: {url}")
            if link.fragment and target in pages and unquote(link.fragment) not in pages[target].anchors:
                raise ValueError(f"Broken site anchor in {page.relative_to(output)}: {url}")
