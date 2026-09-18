# DichromaticMap 使用手册

[English](../en/README.md) · [项目首页](../../README.md) · [开发细节](development.md)

DichromaticMap 包含数值计算 Python 库和交互式查看器，用于 SC/FCC/BCC 倾转晶界双色图。
它按轴向层显示两个晶粒，支持精确重合点、局部近邻配对、晶体向量测量、原子计数、
均匀应变共同胞搜索、PNG 导出，以及会话保存/恢复和数值表导出。

使用指南：[认识界面](#gui-overview) · [第一次操作](#gui-quick-start) · [晶体与角度](#gui-orientation) ·
[显示层](#gui-layers) · [视野](#gui-view) · [晶界](#gui-boundary) · [向量](#gui-vector) ·
[Near-CSL](#gui-near-csl) · [手动胞与计数](#gui-manual-cell) · [所选胞应变](#gui-selected-strain) · [导出](#gui-export) · [会话与数值表](#gui-session) · [常见问题](#gui-troubleshooting)。

## 安装与启动

需要 Python 3.10 或更新版本。下载项目后，在项目根目录执行以下命令，安装数值库和查看器：

```bash
python -m pip install ".[gui]"
python -m dichromatic_map
```

数值库依赖 NumPy，查看器还使用 PySide6 和 PyQtGraph。
如果只需数值计算，执行 `python -m pip install .`。
`dichromatic-map` 是等效的查看器启动命令；在项目根目录也可使用 `python main.py` 启动。

```bash
python -m dichromatic_map --lattice SC --axis 100
python -m dichromatic_map --lattice BCC --axis 100
python -m dichromatic_map --axis "1 -1 3" --workers 4
python -m dichromatic_map --help
```

图中坐标、视野尺寸和匹配距离均以参考晶格常数 a₀ 为单位，坐标 1 表示一个晶格常数。
晶格常数设置给出 a₀ 的 Å 数值，不会改变归一化的图中坐标。

支持三种立方 Bravais 晶格：简单立方 `SC`、面心立方 `FCC`（默认）和体心立方 `BCC`，
以及预定义 ⟨100⟩、⟨110⟩、⟨111⟩、⟨112⟩ 晶轴和自定义整数晶轴。
每种几何对应各自的轴向层数和重复周期。这里表示每个原胞含一个原子的点阵，
不包含金刚石等带额外多原子基元的晶体结构。

对于约分后的 SC 晶轴 [h k l]，一个轴向周期包含 h²+k²+l² 层，
周期长度为 a₀√(h²+k²+l²)，层间距为 a₀/√(h²+k²+l²)。
因此 SC [100]、[110]、[111]、[112] 分别有 1、2、3、6 层，SC [100] 只显示 A 层。
三种晶格采用相同的逐层匹配、胞计数和应变操作规则。

### 命令行参数

| 参数 | 含义与默认值 |
| --- | --- |
| `--lattice` | `FCC`（默认）、`BCC` 或 `SC` |
| `--axis` | 整数晶轴，默认 `110`；负数或多位数指标用引号和空格分隔，例如 `"1 -1 3"` |
| `--angle` | 参考错取向角，范围随晶轴变化：⟨100⟩ 为 0–45°、⟨110⟩ 为 0–90°、⟨111⟩ 为 0–60°，其他立方晶轴为 0–180°；未指定时 `110` 选 Σ9、`100` 选 Σ5，其他轴选最低 Σ 预设，无预设则为 0° |
| `--lattice-constant` | 参考晶格常数 a₀，单位 Å，默认 3.52 |
| `--width`、`--height` | 基础视野尺寸，单位 a₀，默认 12、9 |
| `--marker-size` | 原子标记尺寸参数，默认 32 |
| `--view-scale` | 初始视野倍率，范围 0.1–5，默认 1 |
| `--workers` | 计算进程数，默认最多 4 个并受可用 CPU 数限制；设为 1 使用单进程 |
| `--save` | 导出初始图为 PNG 后退出 |

### PNG 导出

```bash
python -m dichromatic_map --angle 22 --save pattern.png
```

无显示服务器时，将 `QT_QPA_PLATFORM` 设为 `offscreen` 即可导出 PNG。
在支持行内环境变量的命令行中：

```bash
QT_QPA_PLATFORM=offscreen python -m dichromatic_map --workers 1 --save pattern.png
```

也可以在查看器中用 `Export PNG…` 导出当前图，详见 [GUI 导出步骤](#gui-export)。

<a id="gui-overview"></a>

## 认识 GUI

![DichromaticMap 图形界面](../images/gui-overview.png)

示例为 FCC ⟨110⟩ Σ9。左侧是双色图，右侧 `Controls` 是可滚动的控制面板。
本文保留英文控件名，便于在界面中查找。控制面板内容超出窗口时，向下滚动即可找到其余控件。

| 位置 / 面板 | 用途 |
| --- | --- |
| 图标题、坐标轴和图例 | 查看晶格、晶轴、角度、当前 Near-CSL 状态，以及颜色和层符号 |
| `ORIENTATION` / `LAYERS` 页签 | 切换晶体取向设置与逐层显示设置，默认打开 `ORIENTATION` |
| `GB / VECTOR` | 选择 GB 参考线、测量向量；相关选项随选点操作出现 |
| `VIEW / PERFORMANCE` | 展开后用 `VIEW` 调整视野和晶粒参考轴显示，或用 `PERFORMANCE` 设置 `CPU workers` |
| `NEAR-CSL` | 局部近邻配对，或自动均匀应变共同胞搜索 |
| `MANUAL COMMON CELL` | 选四点、计数，以及对局部配对胞施加应变 |
| 面板底部状态区 | 当前选择提示、可见 G1/G2 原子数、同层 CSL 数和计算状态 |

点击带小三角的标题可以展开或折叠面板。`VIEW / PERFORMANCE`、`NEAR-CSL`、
`MANUAL COMMON CELL` 初始折叠；折叠只收起控件，不关闭已经启用的功能。

| 图中样式 | 含义 |
| --- | --- |
| 蓝色原子 / 橙红色原子 | 晶粒 G1 / G2 |
| 圆、菱形、三角形等标记 | 不同轴向层；颜色相同而形状不同表示同一晶粒的不同层 |
| 金色标记 | 同层精确 CSL 重合点 |
| 紫色中点和短连线 | Local Near-CSL 配对及其两个实际原子位置 |
| 绿色胞框 | 可用的自动共同平移胞 |
| 左下角蓝色 G1 / 橙色 G2 箭头 | 带 `[uvw]` 标签的两组面内正交晶粒参考方向 |
| `B1`、`B2` / `P1`、`P2` / `C1`–`C4` | GB 参考点 / 测量端点 / 手动胞顶点 |

多层几何的图例最多列出前 6 个已启用的层；更多层可在 `LAYERS` 中查看。
层数较多时形状会循环使用，应结合层名称和勾选项判断。

<a id="gui-quick-start"></a>

### 第一次操作：显示一个精确 CSL 胞并测量向量

1. 用 `python -m dichromatic_map` 启动。在 `ORIENTATION` 中确认 `Structure` 为 `FCC`，
   `Tilt / viewing axis` 为 `⟨110⟩`，从 `CSL preset` 选择 Σ9。
2. 等待状态区的 CSL `updating…` 消失。切换到 `LAYERS`，点击 `No layers`，
   再只勾选 `G1 A` 和 `G2 A`，使选点只涉及 A 层。
3. 勾选 `Automatic common cell`，点击旁边的 `Fit cell`，查看该角度的共同胞。
4. 在 `GB / VECTOR` 点击 `Measure vector   V`，先点击一个蓝色原子，再点击另一个位置的原子。
   图中出现 P1→P2 箭头和向量读数。两次点击之间仍可拖动平移或滚轮缩放。
5. 若要练习计数，展开 `MANUAL COMMON CELL`，点击 `Pick 4 CSL vertices   M`，
   沿一个凸四边形的边界依次选 4 个金色标记，等待该层的 G1/G2 计数显示。
6. 调整好视野后，滚动到面板底部点击 `Export PNG…` 保存图片。

<a id="gui-orientation"></a>

## 设置晶体、晶轴和角度

在 `ORIENTATION` 页签中操作：

1. 在 `CRYSTAL / AXIS` 的 `Structure` 选择 `FCC`、`BCC` 或 `SC`，选择后立即更新。
2. 在 `Tilt / viewing axis` 选择预定义晶轴；如需自定义，选 `Custom [h k l]`，
   输入 `1 -1 3` 这样的 3 个整数，点击 `Apply axis` 或在输入框按 Enter。
   负数或多位数指标用空格分隔；无效输入会在该区域显示原因，当前晶体保持不变。
3. 从 `CSL preset` 选择预设，或者修改 `Misorientation` 数值框 / 滑块。
   数值框、滑块及附近的范围提示随晶轴更新；预设菜单只保留该范围内的角度。
   数值框显示两位小数；预设保留精确角度，下面的 `Exact θ` 显示更多小数位。
   要使用某个精确 CSL 角，应直接选预设，不必抄写显示为两位小数的角度。
4. 等待晶格和重合点更新后再选点。更换晶轴会选用该轴的默认角度，随后可重新选择预设。

三种立方点阵 SC、FCC、BCC 使用相同的参考错取向角范围：

| 倾转轴族 | 错取向角范围 | 绕该轴的旋转周期 |
| --- | --- | --- |
| ⟨100⟩ | 0–45° | 90° |
| ⟨110⟩ | 0–90° | 180° |
| ⟨111⟩ | 0–60° | 120° |
| 其他晶轴，包括 ⟨112⟩ | 0–180° | 360° |

自定义 `[h k l]` 会先约分，再按立方对称性确定范围，支持负指标和分量置换；
例如 `0 -2 2` 使用 ⟨110⟩ 的范围。表中的范围上限是旋转周期的一半，
因为交换两个晶粒会把正、负相对转角视为等价。这里始终保持所选倾转轴，
不通过改变晶轴表示来求所有立方对称关系下的最小错取向角。
若无法确定晶体对称性，则回退到通用的 0–180°。

命令行 `--angle` 使用相同范围；超出范围会报错，不会自动改成对称等价角。
选择 `Custom angle` 后还需调节角度数值；仅切换该选项不会改变当前角度。
改变晶格、晶轴或参考错取向角会清除原有 GB、向量和手动胞选择，并重新计算已启用的 Near-CSL。
如需保留当前图像，应在修改前导出。

`Display rotation` 数值框和滑块用于旋转观察方向，范围为 −180° 到 +180°；点击 `0°` 复位。
它旋转两晶粒及全部叠加标记，网格和屏幕坐标轴保持固定。
显示旋转不改变参考错取向角、实际应变或晶粒坐标下的向量分量；PNG 导出包含该旋转。

<a id="gui-layers"></a>

## 显示晶粒和轴向层

切换到 `LAYERS` 页签。顶部给出每晶粒层数、层间距和完整轴向重复周期，长度单位均为 a₀。

- `All layers`：显示两晶粒的全部层。
- `No layers`：隐藏两晶粒的全部层，便于随后只开启所需层。
- `G1 A`、`G2 A` 等复选框：分别控制某一晶粒的某一层；图标与图中层符号一致。
  层较多时，层列表本身也可以滚动。
- `Automatic common cell`：显示可用的自动共同胞，默认关闭。`Fit cell` 将视野调整到该胞；
  没有识别出精确共同胞、也没有已应用的应变共同胞时，`Fit cell` 不可用。

例如只看 A 层，先点 `No layers`，再同时勾选 `G1 A` 和 `G2 A`。
若只想测量 G2 的向量，可只勾选所需的 G2 层；完成测量后再恢复 G1 显示。

隐藏原子不可选取。某层必须在两晶粒中都可见，才显示该层的精确 CSL 和局部 Near-CSL 标记。
GB 两侧显示开关还会进一步筛选这些原子和共同点。手动胞选定后，显示层开关不改变其计数。
自动胞框表示共同平移周期，其显隐与原子层复选框独立；`Fit cell` 不会自动勾选胞框。

<a id="gui-view"></a>

## 调整视野与计算进程

在图中拖动平移、滚轮缩放。选点时也可使用这些操作；拖动不会被当作选点。
展开 `VIEW / PERFORMANCE`，在 `VIEW` 页签中：

- `Field size` 控制显示范围相对于基础视野的倍率；倍率越大，显示区域越广。
  可拖动滑块，或直接点 `0.1×`、`0.5×`、`1×`、`2×`、`3×`、`5×`。
- `Center view   C` 将当前视野中心移回原点，并保留当前视野大小。
- `Grain reference axes` 默认勾选，在绘图区左下角显示两晶粒的参考方向；取消勾选可隐藏。
- 要查看一个具体选区，自动胞使用 `LAYERS` 的 `Fit cell`，手动胞使用其面板内的 `Fit`。

参考轴中 G1 为蓝色、G2 为橙色，每个晶粒显示两条互相垂直的面内方向，标注晶体方向 `[uvw]`。
例如沿 [110] 观察时，标签为 `[-1 1 0]` 和 `[0 0 1]`。四支箭头共用视野左下角的一个固定原点，
通过颜色区分晶粒，不显示 G1/G2 标题。箭头旋转时，原点位置和面板尺寸保持不变；
平移和缩放也不改变其屏幕尺寸与位置。它们不跟随原子平移，也不表示实际位置或向量长度。

箭头随各晶粒的参考转角（错取向角的一半，分别取正负）和 `Display rotation` 旋转。
施加应变后，还会跟随各晶粒极分解中的刚体旋转，保持参考轴相互垂直；
它们用于表示正交参考方向，不代表剪切或拉伸后的实际晶格矢量。
向量测量仍使用真实位移，其读数在参考轴可见时移到上方；隐藏参考轴可释放左下角空间。
启用时，PNG 导出也包含这些参考轴。

切换到 `PERFORMANCE` 页签可修改 `CPU workers`；`1` 表示单进程。
较大的视野或搜索范围可能需要等待，底部 `Compute` 显示正在进行的计算，Near-CSL 面板显示匹配或搜索进度。
等待时可继续平移和缩放。看到 `Lattice positions are updating` 时，等新原子显示后再选点；
看到 CSL `updating…` 或 `Matching same-layer neighbors…` 时，等相应标记更新后再选手动胞顶点。
改变进程数不会改变已应用的所选胞应变结果。

状态区的 `Visible G1 / G2` 和 `Same-layer CSL` 描述当前视野及可见性条件下的数量，
不能用来代替手动胞内的完整原子计数。

<a id="gui-boundary"></a>

## 绘制晶界参考线

1. 在 `GB / VECTOR` 点击 `Pick GB   R`，或按 `R`。
2. 左键点击一个可见原子作为 B1，再点击另一位置的可见原子作为 B2。
   两点可来自任一晶粒。点击空白处不会增加顶点，也不会清除已选点。
3. 第二点选定后显示 GB 直线与左右侧标注，并出现 `GB side visibility`。
   左右以有向直线 B1→B2 为准，不一定等于屏幕左、右。
4. 用 `G1 · Left`、`G1 · Right`、`G2 · Left`、`G2 · Right` 独立控制两侧，
   或点击 `G1:L  /  G2:R`、`G1:R  /  G2:L` 选择两种双晶显示方式。
5. 点击 `Show all sides   F` 恢复全部侧，但仍保留 GB 线和层开关状态。

重新点击 `Pick GB   R` 会清除旧 GB 线并开始新选择；如仅需清除，点击后按 `Esc`。
`Esc` 本身只退出选点模式，保留已选点和标注。

### 快捷键

| 操作 | 效果 |
| --- | --- |
| 拖动 / 滚轮 | 平移 / 缩放，保留未完成的选择 |
| `R` | 清除旧 GB 选择，开始选择 B1、B2 |
| `V` | 清除旧向量选择，开始选择 P1、P2 |
| `M` | 开始 / 暂停 / 继续手动胞选点；已有四点时开始新胞 |
| `C` | 居中视野，保留当前视野大小 |
| `F` | 显示两晶粒 GB 线两侧，显示层开关仍有效 |
| `1` / `2` | 显示 G1 左/G2 右，或 G1 右/G2 左 |
| `Esc` | 返回空闲模式，保留当前点和标注 |

输入晶轴或数字时，可直接使用面板按钮执行操作。

<a id="gui-vector"></a>

## 测量向量

在 `GB / VECTOR` 点击 `Measure vector   V`（或按 `V`），依次左键选择 P1、P2。
两个端点必须处于不同的投影位置；两次选点之间可以平移或缩放。
选中后，端点旁的 `G1-A`、`G2-B` 等标签指明其晶粒和层。
若原子投影重叠，选取屏幕上最近的可见原子；两晶粒严格等距时优先 G1。
需要指定晶粒或层时，先在 `LAYERS` 只保留目标显示项，再选对应端点。
可以选完 P1 后调整显示层，再选 P2。

箭头表示 P1→P2，读数显示在绘图区左下角；晶粒参考轴可见时，读数位于其上方。
同晶粒选点显示该晶粒坐标，跨晶粒选点同时显示 G1、G2 坐标，
两套坐标表示的是同一个实际空间位移。

| 读数 | 含义 |
| --- | --- |
| `G1/G2 current (polar)` | 实际位移在随晶粒刚体旋转的正交立方框架中的分量，包含应变对长度和方向的影响 |
| `G1/G2 lattice [uvw]` | 有应变时显示实际位移在该晶粒变形后常规晶格基矢中的系数；同晶粒同一对原子的系数可以保持不变 |
| `Current \|Δr\|/a₀` | 当前三维向量长度 |
| `View Δxy/a₀` | 跨晶粒测量时，向量在当前显示投影中的分量 |

`current` 分量和实际向量长度以参考 a₀ 归一化；`lattice [uvw]` 是变形后常规晶格基矢中的无量纲系数。
`[uvw]` 表示直接晶格方向，不是倒易晶面指标 `(hkl)`。
简单分量保留 `a₀/2[1 1 2]` 等形式；一般方向用 `≈ a₀[...]` 的实数分量表示。

位移包含所选原子当前的变形、旋转、相对平移和轴向层高差。
开始测量后，`GB / VECTOR` 中出现 `P2 axial periodic image`，用于选择 P2 的轴向周期像，
范围 −4 到 +4，默认值为 0。选完两点后调节该数值，图中的三维向量读数会更新。
它改变所表示的三维向量，不增加样品厚度，也不复制图中的晶格；箭头仅显示其二维投影。
平移和显示旋转不改变晶粒坐标下的读数，跨晶粒的显示 x/y 分量则随显示旋转更新。

再次点击 `Measure vector   V` 或按 `V` 会清除旧向量并开始新测量。
若只需清除旧箭头，开始新测量后按 `Esc`；仅按 `Esc` 会保留现有结果。

<a id="gui-near-csl"></a>

## 选择 Near-CSL 方法

展开 `NEAR-CSL`，选择方法，再点击 `Enable Near-CSL`。此功能默认关闭。

| 方法 | 作用 | 默认值 |
| --- | --- | --- |
| `Local matching · no bulk strain` | 将距离阈值内、同层且互为最近邻的原子配对，保留原子位置 | 距离 0.05 a₀ |
| `Homogeneous strain + periodic cell` | 在不增加刚体旋转的均匀面内应变下搜索共同平移胞 | 主应变上限 2%；整数搜索范围 ±12 |

### 局部匹配：查看原子配对而不改变结构

1. 在 `Method` 选择 `Local matching · no bulk strain`，点击 `Enable Near-CSL`。
2. 设置 `Local pair distance`，默认 0.05 a₀，范围 0.0001–0.5 a₀；参数改变后自动重新匹配。
3. 在 `LAYERS` 中使目标层同时在 G1/G2 可见，等待紫色标记和连线出现。
4. 阅读 Near-CSL 面板的 `Visible near pairs` 和 `d/a₀ min / mean / max`，分别为当前视野内
   近邻对数和配对距离的最小值、均值、最大值；精确 CSL 对不计入紫色配对。
5. 如需手动定义选区，保持此方法启用，按下一节选择四个共同点；
   点击 `Disable Near-CSL` 可关闭局部匹配并移除其标记。

局部匹配中，紫色中点表示近邻配对，紫色连线连接两个实际原子位置，标记形状对应所属层。
金色标记表示精确 CSL 点。局部配对仅在同层进行，不是跨层三维近邻搜索。
距离阈值以 a₀ 为单位，查看器中可设置为 0.0001–0.5 a₀，不是应变百分数。

局部匹配不建立周期胞，也不移动原子。若没有紫色点，可检查层开关、放大距离阈值或平移到其他区域。
金色精确 CSL 点与紫色配对分别统计；在精确重合较多的图中，紫色点较少并不表示计算失败。

### 自动均匀应变：搜索并应用共同周期胞

1. 在 `Method` 选择 `Homogeneous strain + periodic cell`，点击 `Enable Near-CSL`。
   已经启用时直接切换方法即可开始搜索。
2. 设置 `Max principal strain`（默认 2%，范围 0.01–10%）和 `Search index bound`
   （默认 12，即整数范围 ±12；可设 2–40）。修改后自动重新搜索，无需另外点击搜索按钮。
3. 等待面板中的 `Searching strained periodic cells…` 及进度更新。搜索期间可平移、缩放。
4. 有结果时，候选下拉列表出现，**第一项自动应用到两晶粒**；选择其他项也立即应用对应应变。
   项目格式为 `G1原子数/G2原子数 atoms | max strain …%`，原子数覆盖完整轴向周期，
   与手动胞的单层计数不同。列表可用于比较胞大小与应变大小。
5. 在下方详情查看变形、主应变和共同胞矢量；切到 `LAYERS`，勾选 `Automatic common cell`，
   再点 `Fit cell`，查看当前候选的胞框。仅隐藏胞框不会撤销应变。
6. 点击 `Disable Near-CSL` 恢复未应变晶格，或切换到局部匹配方法。

更换候选会清除 GB、向量和手动胞选择，建议确定候选后再测量。
没有结果时显示 `No compatible cell found within this bounded search`，晶格保持未应变；
可以增加搜索范围或主应变上限后重试。自动搜索无需手选四个顶点，也没有单独的 Apply 按钮。
搜索只覆盖有限的候选范围，不保证全局最优。施加的应变是指定的几何变化，不是原子弛豫或弹性能最小化。

<a id="gui-manual-cell"></a>

## 选择手动共同胞并计数

展开 `MANUAL COMMON CELL`：

1. 在 `LAYERS` 中只显示准备计数的同一层 G1/G2；点击 `Pick 4 CSL vertices   M`（或按 `M`）。
   使用金色精确 CSL 顶点不要求启用 Near-CSL；使用紫色顶点则需要启用局部匹配。
2. 沿边界顺时针或逆时针依次选择四个同一轴向层的顶点。可以选金色精确 CSL 标记，
   也可以选紫色局部近邻配对中点，不能选普通原子。第一个顶点确定层，后续不同层的点击会被拒绝。
   隐藏点不可选。两晶粒各自的实际顶点均须构成非退化、不自交的凸四边形。
3. 第四个顶点选定后自动闭合，蓝/红轮廓分别连接 G1/G2 的实际原子顶点。
   图右下角显示计数，面板中列出更多统计。
4. 使用 `Undo vertex` 撤销最后一点并继续选点、`Clear` 清除胞与计数、`Fit` 缩放到两晶粒选区。
   `Fit` 仅在四点齐全时可用。`Esc` 暂停选择；未满四点时再按 `M` 可继续，已满四点时则开始新胞。

若不同层的标记投影重叠，点击会被拒绝；先隐藏其他层，再重选该点。
顶点重复、层不符或第四点导致无效多边形时，不会接受该点，面板会解释原因；可用 `Undo vertex` 修正。
计数期间查看面板提示，待结果就绪后再读取图右下角的摘要。

### 计数含义

只统计顶点所属层中的全部原子，不只统计 CSL 点。G1/G2 分开计数，重合原子不会合并。
对于局部近邻配对顶点，每个晶粒都使用自己的四个实际原子位置作为边界，紫色中点只用于选点。
两个多边形的面积可以不同，计数也分别进行。

| 计数 | 含义 |
| --- | --- |
| `Interior` | 严格位于多边形内部的原子 |
| `Boundary` | 位于边或顶点上的原子 |
| `Closed` | 内部与边界原子之和 |
| 半开计数 | 对平行四边形，包含相邻的两条下界边、排除对面的两条上界边，避免重复平铺时重复计入边界原子 |

半开区域定义为 `C1 + u(C2−C1) + v(C4−C1)`，其中 `0 ≤ u,v < 1`。
非平行四边形只报告内部、边界和闭合计数。
例如图中的 `40 atoms · 34 inside + 5 edge + 1 corner` 表示半开计数，
边原子和顶点数只包含该约定保留的边界原子。`◇ layer` 表示菱形标记对应的层。

计数覆盖完整的选定多边形，包括视野以外的部分，不受显示层开关影响。
勾选 `Apply GB side visibility to counts` 可进一步仅计入所选 GB 侧的原子。
平移、缩放和显示旋转不改变计数，也不清除未完成的选择。

包括平行四边形在内的几何选区本身不构成周期性证明。
局部近邻配对胞可能只是近似重复区域，不能据此认定它是严格 CSL 原胞或确定 Σ 值。

改变晶格、倾转轴、参考错取向角或实际应变会使原选区失效。
包含局部近邻配对顶点的胞，在距离阈值改变或退出局部方法后也会清除。
下述应用/恢复流程会保存自身的配对顶点。

<a id="gui-selected-strain"></a>

## 对所选胞施加应变

在局部匹配中选择四个同层顶点后，`MANUAL COMMON CELL` 中的 `Apply bulk strain to selected cell` 才可用。
至少需要一个局部近邻配对顶点，也可混选同层精确 CSL 顶点。单独选胞不会使其变形。
此操作与自动 `Homogeneous strain + periodic cell` 搜索相互独立。

1. 设置 `Selected-cell strain limit`（默认 2%）和 `Rotation limit / grain`（每晶粒默认 1°）。
   应变与旋转分别限制；旋转上限设为 0° 时使用纯对称应变。
2. 点击 `Apply bulk strain to selected cell` 应用拟合。每个完整晶粒接受一个均匀变换和平移，使配对顶点对齐。
   结果优先减小相对原始晶粒的变化，再检查上限，并非搜索所有符合上限的拟合。
   若第四对顶点不相容、变换退化或超过上限，会显示原因并保持原状态。
3. 查看更新后的原子、精确 CSL 点和所选层计数。共同平移胞从对齐后的 C1 开始，不一定是原胞。
4. 在手动胞面板下方出现的只读文本框中查看或选择复制应变详情，包括带正负号的主应变、主伸长、刚体旋转、变形/旋转/伸长张量
   （`F/R/U`）、Green–Lagrange 应变（`E`）、平移、面积变化、对齐残差和共同胞矢量。
   百分数与无量纲张量分别标明；共同胞矢量使用未经显示旋转的分析坐标。
5. 点击 `Restore original local structure` 恢复原子与配对顶点。
   修改顶点或局部距离阈值前需要先恢复。

应用成功后，按钮变成 `Restore original local structure`，方法显示为 `Local cell · bulk strain applied`。
`Pick 4 CSL vertices   M`、`Undo vertex`、`Clear`、局部距离与应变上限控件暂时不可用；
先恢复结构，再调整这些参数或顶点。若应用按钮一开始就是灰色，请检查局部匹配已启用、
顶点已满四个，且至少有一个来自紫色近邻对；全为精确 CSL 顶点时不启用此操作。

`ORIENTATION` 保留参考错取向角。详情同时显示当前极分解框架的错取向角，
等于参考角加 G1 刚体旋转角减 G2 刚体旋转角。显示旋转独立于这些物理量。

平移、缩放、显示旋转和 worker 设置变化会保留已应用的拟合。
应用/恢复会清除旧 GB 线和向量选择；改变晶格、晶轴、参考角或方法会清除已应用状态。
这是人为施加的几何变换，不代表无应力构型或能量弛豫。

<a id="gui-export"></a>

## 从 GUI 导出 PNG

1. 选好晶格、角度、可见层和 GB 两侧；按需要显示胞框、向量或手动计数。
2. 调整平移、缩放和 `Display rotation`，使目标区域与标注出现在当前视野中。
   等待晶格、CSL、局部匹配及手动计数更新结束。
3. 滚动到右侧面板底部，点击 `Export PNG…`。
4. 在 `Export dichromatic pattern` 对话框中选择目录和 `.png` 文件名；默认名为
   `dichromatic_pattern.png`，保存类型为 `PNG image (*.png)`。确认保存，或取消以返回查看器。

导出的是当前绘图区，宽度为 1800 像素，包含图标题、坐标轴、图例、显示旋转、启用的晶粒参考轴及当前可见标注。
右侧控制面板和其中的详细文本不在图片内；需要保存应变详情时，可从其只读文本框选择复制。
自动胞或手动胞超出当前视野时，先用对应的 `Fit cell` 或 `Fit` 再导出。

<a id="gui-session"></a>

## 保存/恢复会话与导出数值表

1. 完成需要保留的选点和应变操作；若要保存正在计算的结果，请先等待计算完成。
2. 滚动到右侧 `Controls` 底部，点击 `Export PNG…` 旁的 `Save session…`。
3. 在文件对话框中选择目录和以 `.dmap` 结尾的文件名并保存。只生成一个文件；
   取消对话框不会改变当前会话。
4. 恢复时，在 `ORIENTATION` 页点击 `Import session…` 并选择该 `.dmap` 文件。
   导入会替换当前窗口的会话，需要保留现有工作时请先保存。文件损坏或版本不支持时，
   当前会话保持不变。

保存内容包括晶格、整数 tilt axis、完整精度的参考角度、晶格常数、GB 端点、向量端点及轴向周期像、
手动胞顶点、两晶粒的变形梯度和平移、已应用的共同胞，以及用于 `Restore original local structure`
的原始顶点。可见晶粒/层、GB 两侧筛选、手动计数选项、Near-CSL 和应变限值、显示旋转、
参考轴显隐及视野也会恢复。窗口宽高比不同时，恢复的视野保留中心并按需扩展，以覆盖原区域且不拉伸晶格。
尚未完成的选点可以继续；原子缓存和计数在导入后重新计算。
CPU workers 保留当前计算机的设置。应变搜索只保存已应用的候选胞，不保存整个候选列表；
若保存时启用了搜索但未应用任何胞，导入后保持未应变状态，修改搜索参数即可重新搜索。

`.dmap` 是标准 ZIP 压缩文件，可用 ZIP 工具打开，或复制一份并将副本扩展名改为 `.zip` 后解压：

| 文件 | 内容 |
| --- | --- |
| `session.json` | 供导入恢复使用、带格式版本的状态数据 |
| `counts.csv` | 两晶粒在手动选定层中的内部、边界、闭合及可用的半开计数，面积与 GB 筛选标记 |
| `vectors.csv` | P1→P2 在分析/显示坐标及相关晶粒极分解/晶格坐标中的分量，轴向周期像与长度 |
| `strain.csv` | 各晶粒的变形梯度、极分解旋转/伸长、Green–Lagrange 应变、主应变、平移及参考/当前角度 |
| `README.txt` | 各表的单位、坐标约定及缺失结果规则 |

CSV 可用表格软件或 Python 的 `csv` 模块读取。长度标明 a₀ 或 Å，面积标明 a₀² 或 Å²，
应变张量为无量纲。向量表区分当前空间分量与晶格 `[uvw]` 坐标。
计数针对完整选区重新计算，应用所选的 GB 两侧筛选，不依赖当前视野。
手动胞不足四个顶点时，`counts.csv` 只有表头；向量不足两个端点时，`vectors.csv` 只有表头。
未应变晶粒的变形梯度为单位矩阵、应变为零。修改 CSV 不会改变导入恢复的状态。

PNG 仍单独保存图像；`.dmap` 保存数值状态和表格，需要图片时请另外导出 PNG。

## 使用 Python 数值库

下面的示例只使用数值库和 NumPy。它生成参考错取向角为 22° 的两个 FCC [110] 晶粒，
查找精确重合点和局部近邻配对，并统计正方形内第 0 层的原子。

```python
import numpy as np
from dichromatic_map import (
    get_geometry,
    projected_columns,
    same_layer_coincidence_sites,
    local_near_pairs,
    count_cell_atoms,
)

angle = 22.0
geometry = get_geometry("FCC", "110")
grain1 = projected_columns(12, 9, angle / 2, lattice="FCC", axis="110")
grain2 = projected_columns(12, 9, -angle / 2, lattice="FCC", axis="110")

sites = same_layer_coincidence_sites(grain1, grain2, tolerance=1e-6)
pairs = local_near_pairs(grain1, grain2, distance=0.05)
print("Exact sites per layer:", [len(layer_sites) for layer_sites in sites])
print("Local near pairs:", len(pairs.layers))

vertices = np.array([[-2, -2], [2, -2], [2, 2], [-2, 2]], dtype=float)
counts = count_cell_atoms(
    vertices,
    angle,
    deformations=(np.eye(2), np.eye(2)),
    lattice="FCC",
    axis="110",
    layer=0,
)
print("Axial layers:", geometry.layer_count)
print("G1/G2 layer-0 closed counts:", (counts.interior + counts.boundary)[:, 0])
print("G1/G2 layer-0 half-open counts:", counts.half_open[:, 0])
```

这里的正方形只是计数区域，不表示已经证明晶体周期性。
匹配使用传入的投影原子柱；计数覆盖完整多边形，与前面生成的观察矩形无关。

简单立方示例：未旋转的 SC [100] 晶格中，边长为 2 a₀ 的正方形按半开约定
每晶粒包含 4 个原子；若把全部边界原子也计入，则每晶粒有 9 个：

```python
import numpy as np
from dichromatic_map import get_geometry, count_cell_atoms

geometry = get_geometry("SC", "100")
vertices = np.array([[0, 0], [2, 0], [2, 2], [0, 2]], dtype=float)
counts = count_cell_atoms(
    vertices, 0, (np.eye(2), np.eye(2)), lattice="SC", axis="100", layer=0
)
print(geometry.layer_count)                         # 1
print(counts.half_open[:, 0])                       # [4 4]
print((counts.interior + counts.boundary)[:, 0])     # [9 9]
```

### API 说明

以下六个函数可直接从 `dichromatic_map` 导入：

| 函数 | 输入与结果 |
| --- | --- |
| `get_geometry(lattice="FCC", axis="110")` | 返回平面基矢、轴向周期、层数及相关晶体几何信息 |
| `misorientation_range(axis="110", lattice="FCC")` | 返回不可变的 `AngleRange`，包含 `maximum_deg`、`period_deg`、`symmetry_order`；[110] 分别返回 90°、180°、2 |
| `projected_columns(width, height, rotation_deg, ...)` | 返回 `positions`（N×2，单位 a₀）、从 0 开始的 `layers` 和参考 `half_indices`（N×3，单位 a₀/2）；可指定视野中心、2×2 变形和平移 |
| `same_layer_coincidence_sites(grain_1, grain_2, tolerance)` | 按层返回重合位置，每层一个 N×2 数组；容差单位为 a₀ |
| `local_near_pairs(grain1, grain2, distance=0.05, ...)` | 返回实际端点 `first`/`second`、层标签、`midpoints` 和 `distances`；局部配对结果不包含精确重合对 |
| `exact_csl_cell(angle, max_denominator=128, lattice="FCC", axis="110")` | 对识别出的公度角返回保持层相位的共同平移胞，否则返回 `None`；`max_denominator` 同时限制四元数互素系数 m、n；共同胞在保层平面内为原胞，不一定是三维原胞 |
| `count_cell_atoms(vertices, angle, deformations, ...)` | 接收共享的 4×2 多边形或两晶粒各自的 2×4×2 多边形、两个 2×2 变形、可选平移和层索引；返回分晶粒、分层计数及每晶粒面积 |

参考错取向角 `angle` 的单位为度，G1/G2 分别旋转 `+angle/2`、`−angle/2`。
数值库中的几何和旋转函数仍按传入角度计算，不受查看器中约化输入范围的限制。
投影原子柱的变形作用于晶粒旋转之后，平移作用于变形之后。
`exact_csl_cell(...).cell` 的两列分别为共同胞的两个矢量，单位为 a₀。

`count_cell_atoms` 默认 `layer=-1`，统计一个完整轴向周期内的所有层。
指定从 0 开始的层索引，可以采用与界面手动胞相同的计数范围。
计数结果形状为 `(2, layer_count)`，未选中的层为 0。
使用半开计数前，应分别检查各晶粒的 `half_open_available`。
`areas` 给出各晶粒的多边形面积，单位为 a₀²。

所选胞应变拟合函数 `strain_selected_cell` 需要从 `dichromatic_map.strain` 导入。
输入形状为 `(2, 4, 2)` 的实际配对原子顶点 `vertices`、参考角度和所选层。
可选参数 `percent`、`max_rotation_deg` 分别设置应变和旋转上限。
返回结果包含共同胞、各晶粒的变换和平移、对齐后的顶点及拟合残差。
算法和张量约定见[开发细节](development.md)。

<a id="gui-troubleshooting"></a>

## 限制与常见问题

约分后的轴指标绝对值不超过 64，每种几何最多 256 个轴向层，
请求区域每晶粒最多枚举 250,000 个候选原子柱。
选区超过限制时会报告错误，不会返回部分计数。

| GUI 现象 | 处理方式 |
| --- | --- |
| 没有金色或紫色标记 | 等待计算结束；确认同一层在 G1/G2 中均显示，GB 两侧未将其隐藏。精确 CSL 可选预设角，局部配对需启用 Near-CSL |
| 点击原子没有反应 | 先选 GB、向量或手动胞工具，再左键点击；手动胞只能点金色或紫色共同点。晶格正在更新时等待后重选 |
| 总是选到 G1，或提示层重叠 | 在 `LAYERS` 隔离目标晶粒 / 层；手动胞需保留两晶粒的同一层 |
| 选完四点仍无计数 | 查看手动胞面板；第四点可能因顺序、共线、自交或层不符被拒绝，修正后重新选取 |
| 没有自动胞框 / `Fit cell` 不可用 | 勾选 `Automatic common cell`，并确认当前角度有精确胞或已有应用的应变候选。仅开启局部匹配不产生周期胞 |
| 应变搜索没有候选 | 增加 `Search index bound` 或 `Max principal strain`，等待新搜索结果；有限范围内无结果不等于不存在共同胞 |
| 所选胞应变按钮不可用 | 启用局部匹配，选四个同层共同点且至少包含一个紫色近邻对；已施加应变时先用恢复按钮再编辑 |
| 改角度后原来的线、箭头或胞消失 | 这些选择依赖原结构，改变晶格、晶轴、角度或应变候选后需要重新选择 |
| 选点模式退出后想继续 | 手动胞未完成时用 `M` 继续；`R`、`V` 会开始全新的 GB / 向量选择 |
| 画面或统计仍在更新 | 查看底部 `Compute` 和对应面板的进度，停止改参数后等待；较大区域可缩小视野，或调整 `CPU workers` |

- **缺少查看器依赖：** 在下载的项目根目录执行 `python -m pip install ".[gui]"`。
- **无法导入 `dichromatic_map`：** 在项目根目录执行 `python -m pip install .` 安装数值库，
  或执行 `python -m pip install ".[gui]"` 同时安装查看器。
- **没有显示服务器：** 使用上面的 offscreen PNG 导出命令。
- **进程启动受限：** 启动时添加 `--workers 1`。
- **视野、晶轴或选区超过限制：** 缩小视野或选区，或使用低指数晶轴。
  局部视野中的原子数不能代替完整选区计数。
