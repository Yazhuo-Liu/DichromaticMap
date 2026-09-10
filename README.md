# DichromaticMap

独立的 FCC/BCC tilt-GB 双色图 Qt 查看器，不依赖 GBClaw。
主程序为 `tilt_gb_dichromatic_pattern_qt.py`，计算模块为
`tilt_gb_crystallography.py` 和 `tilt_gb_near_csl.py`。
`legacy/` 中的 Matplotlib 版本仅保留兼容，不再增加功能。

## 启动

依赖：NumPy、PySide6、PyQtGraph。在安装好依赖的 conda base 环境中：

```bash
conda activate base
python tilt_gb_dichromatic_pattern_qt.py --workers 4
python tilt_gb_dichromatic_pattern_qt.py --lattice BCC --axis 100
python tilt_gb_dichromatic_pattern_qt.py --axis "1 -1 3"
```

坐标、尺寸和局部匹配距离均以 a₀ 为单位。支持 FCC/BCC、预定义
⟨100⟩/⟨110⟩/⟨111⟩/⟨112⟩ 及自定义整数轴，层数与轴向周期自动计算。
拖动平移、滚轮缩放；选择第一个点后可以拖动再选第二个点。

## 手动 common cell

自动周期框继续保留，`Show automatic cell` 独立控制它的显示。
新增 `MANUAL COMMON CELL` 面板：

1. 点击 `Pick 4 CSL vertices`（快捷键 M）。
2. 沿边界顺时针或逆时针依次选择四个**同一轴向层**的顶点。
   可选择金色精确 CSL 标记，也可选择局部 Near-CSL 的紫色配对中点；不吸附普通原子。
   第一顶点确定层，隐藏点不可选。四点必须构成不自交、非退化的凸四边形。
3. 第四点选定后自动闭合，图上显示 G1/G2 计数，面板列出逐晶粒、逐层的详细统计。
   `Undo vertex` 撤销最后一点，`Clear` 清除，`Fit` 缩放到手动胞；Esc 暂停选择。

计数约定：

- 默认统计两个晶粒、全部层、一个完整轴向平移周期内的原子。G1/G2 分开计数，
  两晶粒的重合原子不会被静默合并；统计的是原子，不只是 CSL 标记的数量。
- `Interior` 为严格胞内原子，`Boundary` 为边和顶点上的原子，`Closed` 为两者之和。
- 四点形成平行四边形时，另给出 `Half-open` 计数，并在图中优先显示它。
  定义为 `C1 + u(C2−C1) + v(C4−C1)`，`0 ≤ u,v < 1`；排除两条上界边，
  避免重复平铺时反复计入边界原子。非平行四边形只给胞内/边界/闭合计数。
- 勾选 `Count visible regions / layers only` 后按当前 GB 两侧开关及层选择过滤，
  **仍不按当前屏幕视野裁剪**。关闭此选项恢复所有层、两个完整晶粒的统计。
- 计数为完整选区重新生成原子，在后台使用现有进程池（单 worker 时使用后台线程），
  不依赖绘图缓存。拖动、缩放和显示旋转不改变计数，不会清除选到一半的顶点。
- **手动选区和几何平行四边形都不是周期性证明**。尤其是局部 Near-CSL 中点构成的胞，
  可能只是近似重复区域，不能据此宣称得到了严格 CSL 原胞或 Σ 值。
- 晶格、tilt axis、misorientation 或实际应变改变时旧手动胞会清除。
  若使用了局部 Near-CSL 顶点，改变距离阈值或退出局部方法也会清除它，避免使用失效标记。

## 整体显示旋转

`ORIENTATION` 内的 `Display rotation` 滑条与数值框范围为 −180° 到 +180°，
`0°` 按钮复位。旋转的是图中两晶粒、CSL 点、自动/手动胞、GB 线及测量箭头，
网格与屏幕 x/y 坐标轴保持固定；当前视野中心跟随所查看的构型一起转动。

这是显示变换，不改变 misorientation、晶粒应变、原始坐标或参考 Miller 指数。
旋转后鼠标吸附和隐藏侧判定仍在正确的物理坐标中执行。跨晶粒向量的 x/y 分量
随显示方向更新；应变面板的共同胞矢量明确标为未做显示旋转的分析坐标。
PNG 导出包含当前旋转和手动胞标注。

## Near-CSL 两种方法

功能默认关闭，选择方法后点击 `Enable Near-CSL`：

- `Local matching · no bulk strain`：默认方法，不改变原子坐标。
  同层且互为最近邻的两原子，在可调距离阈值内配对，默认 `0.05 a₀`。
  紫色中点是候选对齐位置，紫色连线连接原始原子；金色仅表示精确 CSL。
  本方法不实际移动原子、不求整体应变、不推断周期晶胞，也不是 stress-free 弛豫计算。
- `Homogeneous strain + periodic cell`：保留均匀应变方法，对两晶粒求对称正定的面内变形，
  不增加额外刚体旋转，寻找共同周期胞。默认主应变上限 2%、整数搜索上限 ±12。
  下拉框给出胞大小与应变的折中解；这不是弹性能最小化或全局最优证明。

局部搜索和精确检测保留全部层相位；当前局部方法不跨层找三维近邻。
多个 worker 使用共享进程池，局部查询不构造全原子两两距离矩阵。
交互资源限制：约分后的轴指标绝对值不超过 64、最多 256 层，每个晶粒最多枚举
250,000 个候选列。手动选区过大时不报告部分计数，应选更小的胞。

## 测试

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest discover -s test -v
```

测试使用无窗口 Qt 后端，覆盖独立三维晶格枚举、精确/均匀应变 CSL、局部匹配、
多进程取消、隐藏点选择、手动胞边界计数、超出视野的完整计数与旋转后的交互。
旧 Matplotlib 兼容测试另需要 Matplotlib；Qt 主程序不需要它。
