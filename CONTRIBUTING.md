# Contributing to DichromaticMap

Bug reports, documentation improvements and code contributions are welcome.
This is a volunteer-developed project. External donations are not currently
accepted.

## Report a problem or request a feature

Open a [GitHub issue](https://github.com/Yazhuo-Liu/DichromaticMap/issues) or
email [yliu3500@gatech.edu](mailto:yliu3500@gatech.edu).

For a bug report, include:

- The DichromaticMap version or commit, Python version and operating system.
- The relevant settings: lattice, tilt axis, misorientation, visible layers,
  Near-CSL method and parameters, and worker count when applicable.
- A small reproducible example or numbered GUI steps.
- The expected result and the actual result, including any error message.
- A screenshot or exported figure when it helps explain a display problem.

For a numerical discrepancy, include the input values, units and reference
result or calculation used for comparison. For a feature request, explain the
intended task and the behavior that would help accomplish it.

## Propose a change

Submit a [pull request](https://github.com/Yazhuo-Liu/DichromaticMap/pulls), or
email a description of the change and a patch or repository link to
[yliu3500@gatech.edu](mailto:yliu3500@gatech.edu). An issue or email can also be
used to discuss a substantial algorithm or API change before implementation.

Keep each contribution focused and describe the problem, the resulting
behavior and how it was checked. For code changes:

- Preserve scientific conventions, including units, layer identity, coordinate
  frames, strain definitions and numerical tolerances. Explain any intended
  change to those conventions and provide a reference or derivation.
- Add or update meaningful tests for changed behavior. Numerical changes should
  cover representative geometries and relevant boundary cases; GUI changes
  should cover the affected interaction or state transition.
- Update the English and Chinese user guides when usage or results change, and
  the implementation documents when algorithms or conventions change.
- Run the relevant test suites and report the results in the pull request,
  including any checks that could not be run.

Algorithm descriptions are available in the
[English implementation details](docs/en/development.md) and
[中文开发细节](docs/zh/development.md).

## Install and run the tests

From the repository root, install the editable package with its test and GUI
requirements:

```bash
python -m pip install -e ".[test,gui]"
python scripts/run_tests.py
```

Run individual suites when working on a specific part of the project:

```bash
python scripts/run_tests.py --suite core
python scripts/run_tests.py --suite gui
python scripts/run_tests.py --suite packaging
```

The `core` suite checks numerical behavior and does not depend on the GUI.
The `gui` suite requires PySide6 and PyQtGraph and uses Qt's `offscreen`
platform, so it does not require a display server. The `packaging` suite checks
the distributable package and keeps build artifacts in temporary directories.
The default command runs all suites.

Tests remain in the source repository for contributors; they are not installed
with the end-user wheel. Keep generated build artifacts out of contributions.

## 中文说明

问题反馈请使用 [GitHub Issues](https://github.com/Yazhuo-Liu/DichromaticMap/issues)
或邮件 [yliu3500@gatech.edu](mailto:yliu3500@gatech.edu)，并提供版本、平台、复现步骤、
预期与实际结果。贡献可提交 PR，也可邮件发送修改说明及补丁或仓库链接。
涉及算法的修改请说明科学依据，补充相关测试，并同步中英文文档。
本项目由志愿开发维护，暂不接受外部捐赠。
