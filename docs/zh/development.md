# 开发与维护

[English](../en/development.md) · [使用手册](README.md) · [项目首页](../../README.md)

## 目录与职责

```text
DichromaticMap/
├── main.py                     # 唯一根目录源码启动入口
├── pyproject.toml              # 包元数据、依赖、可选安装入口和资源清单
├── README.md                   # 英文项目首页，链接到中文文档
├── docs/
│   ├── en/                     # 英文使用手册与开发说明
│   └── zh/                     # 中文使用手册与开发说明
├── src/dichromatic_map/
│   ├── __init__.py             # 常用数值 API
│   ├── __main__.py             # 命令行参数与程序启动
│   ├── crystal.py              # 晶格、整数轴、投影、晶粒向量坐标
│   ├── cells.py                # 胞几何、胞结果数据、完整区域计数
│   ├── matching.py             # 精确 CSL 与局部同层匹配
│   ├── strain.py               # 均匀应变搜索与所选胞应变拟合
│   ├── compute.py              # worker、执行器、异步 NearSearch
│   ├── state.py                # 参数、选择、过滤和数值结果状态
│   └── ui/
│       ├── __init__.py         # 颜色和图形约定，不初始化 Qt
│       ├── controls.py         # 控件、布局、应用主题
│       ├── plot.py             # 图元、显示变换、标注和导出
│       ├── window.py           # 交互流程与上述对象的协调
│       └── resources/          # theme.qss 与箭头 SVG
├── test/                       # 数值、界面、进程与入口回归测试
└── legacy/                     # 只保留历史 Matplotlib 对照实现
```

`src` 是源码容器，`dichromatic_map` 才是包名，导入时不写 `src.`。
根目录旧兼容模块已删除，调用方应使用正式包，而不是导入 `main.py` 获取计算函数。
`main.py` 只为源码启动添加 `src` 到导入路径，再调用包的 `main()`；不包含另一套程序逻辑。
安装后的命令行入口调用同一个函数。启动保护确保子进程导入入口时不会再开窗口。

## 依赖与扩展边界

数值模块只依赖 NumPy 和标准库，不反向导入 UI。`state.py` 不保存 Qt 控件，
`compute.py` 不操作绘图。工作函数在模块顶层定义，便于使用 `spawn`；
进程间传递数组和数值结果，不传 QWidget 或窗口对象。
如果环境中有 `threadpoolctl`，worker 会限制内部 BLAS 线程；该依赖是可选的。

`DichromaticPatternWindow` 通过四个对象组合功能：

| 对象 | 持有的数据或职责 |
| --- | --- |
| `window.state` | 物理构型、选点、显示过滤、计算结果及缓存标识 |
| `window.compute` | 进程/线程池、任务句柄和任务代次 |
| `window.controls` | 控件和布局 |
| `window.plot` | 绘图项、图形坐标转换、标注与导出 |

窗口负责用户动作、状态更新和定时轮询的衔接。仍保留的窗口属性转发只是对同一份
组件数据的访问；新代码直接访问组件，不再扩大这些历史转发接口。
不使用多重继承，也不提前添加插件注册器、通用基类或每类一个文件的层级。

扩展建议：

1. 新算法先放入 `matching.py`、`strain.py` 等对应数值模块，添加独立数值测试。
2. 需要后台计算时在 `compute.py` 添加顶层 worker，由窗口协调提交和轮询。
   请求变化时应使旧结果失效，不允许旧任务覆盖当前构型；关闭窗口时释放执行资源。
3. 新控件放 `ui/controls.py`，新图元或标注放 `ui/plot.py`，事件流程接入 `ui/window.py`。
4. 修改主题时只编辑 `ui/resources/`。新资源类型还需同步 `pyproject.toml` 的包资源清单。
5. 修改行为后同步两个语言版本的文档。确实出现第二种实现时再决定是否增加抽象层。

显示旋转、Layers 可见性和视野裁剪不能改写物理原子坐标。
计数应重新覆盖完整选区，不使用当前画面中的点数作为计数结果。
手动胞的 G1/G2 实际顶点不可替换成局部匹配中点。

## Python API

若要在项目之外导入，在选定 conda 环境中、项目根目录下做一次可编辑安装：

```bash
python -m pip install -e .
# 同时安装界面依赖时使用：
python -m pip install -e ".[gui]"
```

之后修改 `src/` 下的代码无需重新安装；更改包元数据、依赖或入口配置时需重新安装。

```python
import numpy as np
from dichromatic_map import get_geometry, projected_columns, count_cell_atoms
from dichromatic_map.crystal import csl_angle_deg
from dichromatic_map.matching import exact_csl_cell, local_near_pairs

geometry = get_geometry("FCC", "110")
angle = csl_angle_deg(4, 1, "110")
g1 = projected_columns(12, 9, angle / 2, lattice="FCC", axis="110")
g2 = projected_columns(12, 9, -angle / 2, lattice="FCC", axis="110")
pairs = local_near_pairs(g1, g2, 0.05)
cell = exact_csl_cell(angle, lattice="FCC", axis="110")
assert cell is not None
vertices = np.array([[0, 0], [1, 0], [1, 1], [0, 1]]) @ cell.cell.T
counts = count_cell_atoms(vertices, angle, (cell.f1, cell.f2), layer=1)
print(geometry.layer_count, counts.half_open[:, 1])
```

| API / 数据 | 约定 |
| --- | --- |
| `get_geometry(lattice, axis)` | 标准化晶轴并返回几何；层数、层间距和轴向周期由几何计算 |
| `projected_columns(...)` | 返回 `ProjectedGrain`：`positions` 为 `(N,2)`，`layers` 为 `(N,)`，`half_indices` 为 `(N,3)` |
| `half_indices` | 整数坐标单位是 a₀/2，不是归一化二维坐标 |
| `same_layer_coincidence_sites(g1, g2, tolerance)` | 由 `matching.py` 提供，逐层返回精确重合坐标 |
| `local_near_pairs(g1, g2, distance, layers=None)` | 同层互为最近邻配对；只返回分析结果，不移动输入原子 |
| `exact_csl_cell(angle, lattice, axis)` | 返回保层共同周期胞或 `None`；返回值不是任意手动框 |
| `count_cell_atoms(vertices, angle, deformations, ...)` | 顶点为共享 `(4,2)` 或两晶粒独立 `(2,4,2)`；结果按晶粒和层索引 |
| `layer=-1` | 计数 API 默认统计全部层；GUI 明确传入手动胞的选定层，不采用这个默认值 |
| `strain_selected_cell(...)` | 从 `strain.py` 导入；输入原始配对顶点 `(2,4,2)`，返回变形、平移与应变信息；不直接修改窗口 |

长度均以 a₀ 表示，角度参数为度，二维变形矩阵为 `(2,2)`。
不合法输入通常抛出 `ValueError`；超过交互资源限制抛出其子类 `GeometryLimitError`。
具体签名和数值约定以对应函数的 docstring 为准。

## 测试与文档维护

完整测试依赖 NumPy、PySide6、PyQtGraph 和用于历史对照的 Matplotlib。
依赖已具备时，不必安装本项目，直接从根目录执行：

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest discover -s test -v
```

也可用 `python -m pip install -e ".[test]"` 安装测试依赖。
测试会为源码导入配置 `src`，使用 offscreen Qt；生成的临时导出使用临时目录。
覆盖晶格枚举、CSL 与应变、选区计数、可见性与旋转、向量坐标、旧任务失效、
进程返回结果、根入口在其他目录的独立启动和 PNG 导出。

文档是可直接在编辑器或代码托管平台阅读的 Markdown，没有 Sphinx/MkDocs 依赖，
也不需要生成 `docs/_build/`。链接采用相对路径，更新文件位置时检查中英文互链。

## 为什么有 build？

运行这个项目不需要先 build。`python main.py` 直接运行源码。
之前出现的 `build/` 来自验证 `pip wheel` 安装包的过程，而不是应用运行的需求。
本项目没有自定义 C/C++ 扩展；构建 wheel 主要是把 Python 源码、主题/图标及安装元数据
组织为可分发包，不是把本项目编译成原生可执行程序。

`pyproject.toml` 中的 `[build-system]` 指定安装/打包时使用的后端，
不表示每次运行都要构建。该配置还让项目可以规范安装、导入并携带 UI 资源，因此保留。

| 产物 | 用途与清理注意 |
| --- | --- |
| `build/` | 打包中间副本；完成构建后通常可删除，不是源码 |
| `dist/` | 显式打包输出的 wheel/sdist；删除前确认不需要保留发布产物 |
| `__pycache__/`、`.pyc` | Python 导入时生成的字节码缓存；可删除，下次运行可能自动再生成 |
| `*.egg-info/` | 包的安装/构建元数据；构建残留可清理，但活跃的可编辑安装可能依赖它 |
| `.pytest_cache/` 等 | 测试工具缓存；可重新生成 |

如果删除了可编辑安装所需的元数据，重新执行对应的 `pip install -e ...`。
不要把清理项目缓存变成删除 conda 环境或 `site-packages`。
源码、`pyproject.toml`、QSS、SVG 和文档不是临时产物。

临时运行时若不希望写入字节码缓存，可使用 `python -B main.py` 或设置
`PYTHONDONTWRITEBYTECODE=1`。这只是可选的缓存策略，不应把出现 `__pycache__`
误认为项目必须进行一次构建。
