# Marker 密集位置/场查表试验（2026-10-09）

> 后续同日版本见 [固定拓扑刷新与原生 cubic 确认](native-marker-topology-cache-2026-10-09.md)。
> 本文的时间、误差、源码 SHA 与机器产物属于首次 bilinear 版本，未被覆盖；
> 当前实验源码已继续优化，文末命令在当前工作树运行会产生新版本结果，而非
> 逐值复现本文旧 SHA 的实现。默认原生 source 不变。

## 问题与范围

用户允许以数 GiB 内存换速度，并选择直接查表的激进路径。本轮实现的是显式
`EXPERIMENT ONLY` 数值近似，不改变默认 `NativeMarkerSource`、JOREK、原始数据、
roadmap/ADR，不训练或更新 optimizer。使用 `diagnosing-bugs` 的独立计时与误差
反馈环，使用 `tdd` 的已确认公共 `NativeMarkerSource.advance` 验收入口。
这不是交互式 HTML 逻辑原型；numeric experiment 独立于生产模块，结果保存为
本地数值证据，不创建分支/提交/issue 或把试验默认启用。

真实验证范围是两个 phaseA 的首次完整 source 窗口：完整初始场、rawD、前投影、
百万槽位、全部活 marker 和同一次出生结果。每组 reference/candidates 使用
相同初态及出生后 RNG。不执行九场网络预测、shard 后续推进或多步 BPTT，
不能把 source 加速写成完整训练加速，也没有真实未来场回授。

## 当前 marker 与 JOREK 一手证据

marker 保存 `x=(R,Z,phi)`、`i_elm/st`、原子权重、11 电荷态概率、`V_prev`、
寿命/槽位身份与原生出生字段。`V_prev` 是推进历史，不是任意可忽略的诊断。
每子步采样旧位置的场与前投影，电荷态更新/载荷在旧位置沉积，然后移动、重新定位。

JOREK 只读源码 commit `54c675bcc39c33b082c0c35046e28122dc53b3aa`：

- `/home/ac_desktop/JOREK_code/jorek/particles/examples/marker_spi_source.f90:482-519`
  读取旧 x/st/单元并采样；`:616-643` 旧位置基函数沉积；`:655-663` push 后定位。
- `particles/pushers/mod_fieldline_euler.f90:56` 用当前/上一速度方向的组合推进，
  不是直接 `1.5V-.5V_prev`；保存 `V_prev`。
- `grids/grid_utils/mod_find_rz_nearby.f90:92-185` 从旧单元起步，最多 8 轮 Newton，
  触边后邻接变换；R/Z 残差平方 `<1e-24`。失败进入 `find_RZ` 全局候选搜索。
  该实际实现不是当前 Python 参考定位器的 25 轮/六档 damping，因此本轮
  Python reference 对照不等于 JOREK Fortran 逐值认证。
- ADAS 在启动 `init_imp_adas` 时加载入内存（marker 源 `:72`、
  `models/mod_impurity.f90:48`、`tools/mod_openadas.f90:104`）。本仓库也在初始化
  `load_neon_tables/parse_adf11`，循环只是内存插值/指数/单位转换，本轮未改 ADAS。

marker Fortran SHA256：`4f6e832b107569d7a8a0155ce5995d04d59cc50771d256cdf99c1e692318991c`；
nearby 定位 Fortran：`b1cbbd3b8cbc3250a81132ac9c8f638561d60e832c0113fa13dd984f8999dcf4`。
默认 Python source 保持 SHA256
`f6dd533258fe83451c301888297a1154cf0b9d5fcdfb9c7ade23432d90bd551b`。

## 实验算法

实现：`src/models/native_lookup_experiment.py`。

1. 每个原生单元建立 g×g 均匀 `(s,t)` 网格，利用完整 cubic 自由度预计算 R/Z。
   几何表与所有网格节点的空间 KD-tree 常驻；不是规则 R/Z 网格的纯 O(1) 数组定位。
2. 旧 hints 优先试所在微单元；对四角定义的 bilinear 曲面解析求逆（稳定二次根及
   线性退化），不进行 Newton 迭代。未解出时查询最近节点并试相邻四个微单元，
   最后才回退原生定位。最近点仅提供候选，不等价于完整曲面域归属证明。
3. 每个 marker 窗口从实际当前 U 与前投影建立九场/五反馈值表。一次准备覆盖
   整个网格，所有子步/活粒子共享；不是永久冻结初始 marker 的场信息。
4. 当前 `(s,t)` 上 bilinear 求值；R/Z 梯度来自**同一 bilinear 值表示**及同一几何
   Jacobian。兼容性只在微单元内部，界面梯度可跳变，不是原生 cubic 精确梯度。
5. 电荷态、ADAS、速度 pusher、时间子步、旧位置 cubic 沉积、稀疏滤波投影不变。
   原生 fallback 得到的 st 随后也进入 bilinear 采样，这是显式混合近似。

新版本/分辨率/近似/geometry 字节 SHA 写入实验 configuration，与默认版本不同；
没有把旧模型检查点静默切到此模式，也未实现生产检查点/训练接线。
源公开入口拒绝几何内容在建表后改变；调用中并发改网格不受支持。
采样仅消费事件拥有的只读快照，换快照重建；准备开始先失效 owner，失败不会
让部分新表冒充旧缓存。物理状态仍按原 Source.advance 候选副本与失败不提交规则。
本轮真实路径为 `jorek_raw_deltas_v1 + field_static=True`；不支持动态 rawD 场的此试验模式。

`max_bytes` 是持续表空间的保守估计，不包含所有树节点、临时数组、模型和云副本；
`table_bytes` 统计列出的数组，真实资源以整个进程峰值 RSS 为准。

## 测试与测量

公共事件测试 `tests/test_native_lookup_experiment.py`：仿射/常量场、cubic 采样
随分辨率误差下降、输入 ndarray 内容变化后的刷新、跨单元/丢失、身份/时钟、
原子率失败不提交，以及网格变更拒绝。模块不存在、网格变更未拒绝两例先观察
失败再实现/修复；其余为已实现公共行为回归，不虚构所有测试均红→绿。
另用非零持久五反馈的常量 fixture 核验电荷闭合及输入不变。未专门注入
第二张表准备失败后重试，不能把 owner 失效修复写成所有缓存失败路径的实测认证。

真实 CLI：`tools/benchmark_native_lookup.py`。每组 g=17/33/65 串行比较；
source 计时含当前窗口场表准备，几何建表另计（只对静态网格可摊销）。
出生只做一次供所有路径共用，出生耗时单列，不计入 source 查表加速。
没有 profiler 或 CUDA 使用；自己的重测试/作业不与性能运行重叠。
每组每配置一次观测，不声称统计显著性。

每组保存完整槽位/身份差异和匹配活身份的原生单位相对误差，零参考分量不加
epsilon。五通道投影及 deltas 分开报告；还保存状态增量相对误差，避免初始位置
很大掩盖移动误差。指标为 reference-vs-lookup，不是 JOREK 真实演化预测精度。
初态登记 restart 前投影为零，因此这两个真实窗口不能验证非零反馈历史的查表误差。

独立采样 probe 覆盖全部初始活 marker＋全部出生，在实际事件表上调用实验
求值器，对照原生公共完整 `points` 的九场值/RZ梯度/五反馈。probe 不经过
委托原生的 proxy.points；不重建表、不消耗 RNG，独立计时，不含于 source wall。
它是初始位置的采样诊断，不是全部子步轨迹的误差采集。

两组完整首窗口成功，参考分别执行 40/81 个粒子子步；初始活粒子
11168/52604，共享出生 29/56，百万槽位均保留。所有候选时钟、gather、身份寿命、
RNG 与共享输入检查通过；存活与单元编号差异单独记录，不能称“全状态保真”。

| run / 每单元每方向网格点数 | source 秒（含窗口场表） | 几何首次建表秒 | 五通道投影相对L2范围 | 单元编号不同槽位 | 比参考额外丢失 |
|---|---:|---:|---:|---:|---:|
| v400 native | 6.531 | — | — | — | — |
| v400 g17 | 4.093 | 0.433 | 0.0120%–0.0174% | 2 | 0 |
| v400 g33 | 5.452 | 1.666 | 0.00309%–0.00411% | 1 | 0 |
| v400 g65 | 10.763 | 6.824 | 0.000666%–0.000966% | 0 | 0 |
| injInner native | 44.129 | — | — | — | — |
| injInner g17 | 26.360 | 0.437 | 0.0808%–0.4635% | 18 | 4 |
| injInner g33 | 28.302 | 1.680 | 0.0196%–0.1352% | 7 | 1 |
| injInner g65 | 34.393 | 6.859 | 0.00513%–0.1338% | 2 | 1 |

表中 source 不含 geometry build；首次冷使用需把两者相加。g17 的 source 耗时
减少约 37.33%/40.27%；含首次几何成本后约 30.70%/39.28%。几何成本使用外层
`timings_seconds.g*_geometry_build`，包含整个构造，不仅内部建表/建树段。
g65 在 v400 更慢，不能泛称“内存越大越快”。两组 g65 的窗口场表准备仍要约
6.99/6.88 秒，每个新场都要重新支付；几何一次性构表可摊销，场表不能永久复用。

持续表数组（含坐标、值、树索引，不含全部树节点）约 0.146/0.550/2.134 GiB。
两组整个进程最终峰值 RSS 约 4.48/4.50 GiB，包含云副本、表、构建和指标诊断。
不是各配置的独立峰值或 CUDA 显存，未使用 GPU。

初始全活/birth 位置的九场采样误差随加密下降，但完整轨迹并非单调：

| run / grid | 九场最大值相对L2 | 九场R/Z梯度最大相对L2 | R位移增量相对L2 | 末R速度相对L2 |
|---|---:|---:|---:|---:|
| v400 g17 | 0.00660% | 1.78% | 0.163% | 0.255% |
| v400 g33 | 0.00165% | 0.893% | 0.0457% | 0.131% |
| v400 g65 | 0.000408% | 0.450% | 0.00947% | 0.0660% |
| injInner g17 | 0.0212% | 3.01% | 0.490% | 0.910% |
| injInner g33 | 0.00520% | 1.50% | **4.55%** | **5.78%** |
| injInner g65 | 0.00131% | 0.741% | **2.49%** | **5.08%** |

增量只比较初始及双方末态共同存活的身份，不包括丢失样本；不同分辨率的
共同存活集略不同。g33 的最大单粒子 R 位移差为 12.22 mm，贡献约 91.1% 的
全匹配集 R 增量误差平方和，说明整体值/投影误差小不能证明轨迹尾部安全。
这些毫米仅用于定位极端偏离，验收主指标仍为未经量纲重缩放的相对误差。

定位未解而回退比例分别约 v400 1.07%/1.67%/2.66%，injInner 0.83%/1.48%/2.84%。
加密后 microcell 更多，实际 tree/fallback 查询增加；是否来自跨 microcell 及
候选路径变化仍是待验证解释。回退比例不等价于误接受率。
不能仅凭大部分调用无迭代就宣布边界定位合格。

结论：**查表提速可行，但本轮不把 bilinear 方案设为默认或训练合格 backend。**
仍需处理少数轨迹偏离/额外丢失，研究兼容高阶采样或敏感区域原生回退，以及
每窗口构表摊销。原生默认路径和旧训练产物均未更改。

机器证据：`output/native-lookup-2026-10-09/v400.json`、`inner.json`。
实验模块 SHA256 `e2199c36545dc1d2adaf3a9f8516cacd3324d93e7a138b26572a0a55d9bebae2`。
实验公共事件专项 **8 passed**；定位、ADAS、共享空间、生产源/出生/密度、
联合源、Torch 多步及性能诊断相关回归 **231 passed**（29.39 秒；1 条既有 Torch
profiler 周期事件清理警告）。曾有一项新测试因 expected 数组 shape 不匹配失败，
修正断言索引后重新运行完整相关集通过；不是运行时物理数值失败。

初次两组计时 CLI SHA256
`34617baa7618d0faa5308dfb550aa59f5eaf9b4b00e2d011136dea3eaa594579`。
随后追加末态诊断与 CLI 自身 SHA 记录，诊断版本 SHA256
`14affba913e294f9970e28ba7095939e3d1c23e2cb14db34c82cfad2fb914e1c`；
同一实验数学模块未改变。追加诊断在 source 计时结束后执行，不改写原始性能证据。

## injInner g33 末态补诊断

独立重跑文件 `output/native-lookup-2026-10-09/inner-g33-outliers.json` 成功，
保持同 seed/初态/出生/当前场：native 44.142 秒、g33 source 28.199 秒，
外层几何 1.679 秒，末态诊断另花 0.152 秒。R 位移最差槽位仍为 500558，
差异 12.22 mm，占 R 增量误差平方和 91.109%；第二个槽位 400633 占 8.702%，
前十占 99.966%。不是均匀全云偏差，也不能因平均投影误差小就忽略。

独立用原生 cubic 几何反求每个匹配活末态的 R/Z，并与其实际 x 比较：
52657 个匹配活身份全部有限；native 最大残差约 `1.00e-10 m`，lookup
最大 `4.10e-5 m`、p99 `2.47e-6 m`、RMS `9.67e-7 m`。这直接证明近似局部
坐标与原生 cubic 几何不再完全一致；这些绝对空间残差仅诊断映射，非主验收量。
它不证明这点几何误差单独导致 12.22 mm 轨迹误差。

最差槽位 native 末单元 3354、lookup 3370；原生末态场的 impurity 梯度也不同，
但两者是在**不同位置**求值，不能把该差直接解释为同位置插值误差。当前证据
支持继续检查单元/微单元路径、兼容梯度及敏感密度区域，尚未捕获首个分歧子步，
也未通过单独切换几何/场采样完成因果隔离。未尝试据此修正物理推进公式。

独立只读复核确认原始表格与机器结果一致，并纠正了完整构造计时口径及未证实
的路径归因。默认 source SHA 再核验未变。本轮结论仍为实验提速可行、默认替换
不合格；无训练、无 GPU 性能或完整 BPTT 加速结论。

## 复现

输出文件必须不存在，两个命令串行运行。

```bash
.torch-cuda-env/bin/python -m pytest -q tests/test_native_lookup_experiment.py
.torch-cuda-env/bin/python -m tools.benchmark_native_lookup \
  --run-index 0 --output output/native-lookup-2026-10-09/v400.json
.torch-cuda-env/bin/python -m tools.benchmark_native_lookup \
  --run-index 1 --output output/native-lookup-2026-10-09/inner.json
.torch-cuda-env/bin/python -m tools.benchmark_native_lookup \
  --run-index 1 --grid-points 33 \
  --output output/native-lookup-2026-10-09/inner-g33-outliers.json
```

CPU、FP64、2 个 Torch CPU threads、chunk 2048、seed 20261009；每命令总 wall
预算 600 秒、进程 RSS 8192 MiB、持续表估计 4 GiB。机器结果记录实际输入、
模块 SHA、配置指纹、时钟/gather/子步、RNG、内存、采样与状态误差及失败阶段。
