# Building and publishing a release

Use Python 3.12 in a clean environment for release builds. The numerical library
still supports Python 3.10 and later. Native executables must be built on each
target OS and CPU architecture; PyInstaller is not a cross-compiler.

## Prepare and validate

1. Update `project.version` in `pyproject.toml` and add
   `docs/releases/v<version>.md` (for this release, [v0.2.2](v0.2.2.md)).
2. Install `.[test,gui,bundle]`, run `python scripts/run_tests.py`, and review
   the changes. Include the new source modules, tests and packaging files in
   the commit; generated `build/` and `dist/` files are ignored.
3. Commit and push the changes. **Actions → Release packages → Run workflow**
   builds the branch without creating a GitHub release. Download `release-all`
   from that run to inspect all six packages and their checksums.
4. When ready, create and push a version tag matching `pyproject.toml`:

   ```bash
   git tag -a v0.2.2 -m "DichromaticMap v0.2.2"
   git push origin v0.2.2
   ```

The tag workflow runs the complete existing test matrix, builds wheel/sdist and
four native archives, and tests each extracted application with real spawned
workers. Only after every job succeeds does it create a **draft** GitHub release
with the release notes, all packages and `SHA256SUMS.txt`. Review it on GitHub,
then click **Publish release** when ready. Re-running the tag workflow can update
an existing draft; it refuses to replace an already published release.

No PyPI credentials are required by the workflow, and it does not upload to PyPI.
To publish the verified Python artifacts separately, use your existing PyPI
account or trusted-publishing setup. For a manual upload from `dist/`:

```bash
python -m pip install twine
python -m twine check --strict dist/dichromatic_map-0.2.2*
python -m twine upload dist/dichromatic_map-0.2.2-py3-none-any.whl dist/dichromatic_map-0.2.2.tar.gz
```

## Local build

From the repository or an unpacked source distribution:

```bash
python -m pip install ".[test,gui,bundle]" twine
python scripts/run_tests.py
python -m build
python -m twine check --strict dist/dichromatic_map-0.2.2*
python scripts/build_executable.py
python scripts/release_checksums.py dist
```

`build_executable.py` writes a versioned native archive to `dist/`, extracts it
to a temporary directory, removes Python and Qt path overrides, and checks
bundled icons/styles, actual multiprocessing, FCC/BCC/SC rendering, PNG export,
sessions and completion previews. A failed smoke test fails the build command.
Recheck an existing archive with `python scripts/smoke_executable.py <archive>`.
Pass `--complete` to `release_checksums.py` only when all six release assets are
present; it rejects missing, unexpected or stale-version files.

Windows uses a ZIP with an `.exe` and supporting libraries. macOS uses a ZIP
with an `.app`; `ditto` preserves framework symlinks when packing and extracting.
Linux uses a tar.gz to preserve symlinks and executable permissions. These are
directory bundles, not single-file binaries or installers. The build matrix
uses Ubuntu 22.04 x64, Windows 2022 x64, macOS 15 arm64 and macOS 15 Intel.
Linux builds on newer local systems may require a newer glibc than the official
Ubuntu 22.04 build. No Windows signing certificate or Apple notarization is
configured; do not describe the downloads as signed/notarized applications.

Build configuration: [PyInstaller spec](../../packaging/DichromaticMap.spec).
PyInstaller documents [native builds and bundles](https://pyinstaller.org/en/stable/usage.html)
and [multiprocessing and symlink requirements](https://pyinstaller.org/en/stable/common-issues-and-pitfalls.html).
Runner architectures follow the [GitHub runner reference](https://docs.github.com/en/actions/reference/runners/github-hosted-runners).

## 中文发布步骤

先提交版本号、发布说明、功能代码、测试和打包脚本。可在 GitHub Actions 手动运行
**Release packages**，下载 `release-all` 试用；手动运行分支不会创建 Release。
准备好后推送 `v0.2.2` 标签。完整测试与各平台构建通过后，工作流会创建包含六个安装包
和校验文件的**草稿 Release**，由维护者检查后发布。

PyPI 上传独立进行，工作流不会自动上传。Windows/macOS 文件未配置发布者签名，
macOS 也未进行 notarization；系统阻止运行时可使用 Python 安装方式或自行构建。
