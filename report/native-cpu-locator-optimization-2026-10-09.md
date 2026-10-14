# CPU marker 定位器：接受试探几何复用（2026-10-09）

## 范围与结论

这是上一轮采样优化后的有界性能切片，不是训练或新物理模型。只消除
Newton 定位中“已接受 damping trial 的几何，下一轮立即重新求值”的重复工作。
未改候选顺序、25 轮上限、收敛容差、damping 顺序、Jacobian/R 合法性检查、
粒子子步、场回授、投影累加、出生/RNG、完整云、历史或失败回滚。
没有 optimizer 更新，未修改 JOREK、原始数据、roadmap 或 ADR。

使用 `diagnosing-bugs` 分离带 profiler 的热点证据与无 profiler 的独立计时；
使用 `tdd` 的公开入口回归约束重构。旧路径先通过冻结搜索算法的测试，
性能重构没有虚构功能性红→绿。子代理分别补公共事件测试、诊断工具和独立审查。

## 实现与边界

`src/models/native_coupling.py` 的 `NativeSourceMesh.locate/locate_batch` 保留
已接受 trial 的完整 `(rz, jac)`，仅在下一轮相同局部坐标处复用。仍在下一轮
原位置判断收敛，不提前接受第 25 轮末尚未重新检查的试探点。
批量缓存按 accepted rows 的原顺序压缩；整批失败与标量 fallback 的原始 hints
不变。每次调用与每个标量候选重新初始化，不跨调用缓存可变几何。

优化只对原生 mesh 类、无相应实例 geometry override 的路径启用。子类或实例
override 保留旧求值行为。批量路径额外要求 local hints 为 float64：float32/int
写回会舍入 trial，不能复用写回前的几何；这些输入保持旧路径而非偷偷转换。
不保证类级 monkeypatch、模块 helper 动态替换或调用中并发修改几何的等价性。
完整几何公式/求解方式/守卫未改，没有新增 GPU backend、编译器或依赖。

一个既有测试原先依赖 `_cubic_basis` 的调用 trace，重构后输出相同但调用减少。
本轮将它改为同一 stiff damping 案例的公共定位结果检查；未放宽坐标容差。

## 保真契约与证据

冻结本轮旧完整模块：
`output/native-cpu-locator-2026-10-09/native_coupling_before.py`，SHA256
`1471f8567f1abcc0ccce115a996dda50fee549317cc2c76bbc774bc9c2d45c40`。
新生产模块 SHA256：
`f6dd533258fe83451c301888297a1154cf0b9d5fcdfb9c7ade23432d90bd551b`。
ADAS helper 本轮未改，SHA256：
`23fc174293056877aeab0eb2ec1f5aee77263ae5b02f4d0e843c4b825df27618`。

两组四事件的 old 端点证据复用上一轮的 `after-{v400,inner}-capture`；
其生产 SHA 与本轮冻结模块完全相同。new 端点重新执行本轮
`after-{v400,inner}-capture`，不把上一轮更早的采样优化前结果当本轮基线。
比较器绑定完整初态、模型/buffers、配置、控制和 RNG 后要求 array_equal，
不允许 backend 差异白名单。结果见本轮 `{v400,inner}-exact-comparison.json`。

两组均 `all_physical_evidence_exact=true`：完整九场/rawD/memory/前场、全部
百万 marker 槽位、五通道投影及 deltas、独立 density source、全部 shard 状态、
身份/时钟/RNG 精确一致。定义有效的原生 JOREK 单位相对误差为 0；零参考处
也没有不匹配。每组全部 846 shards，不裁云。

严格恢复既有 `seed1-run3/epoch-3-parameters-optimizer.pt`（3147 参数，28 个
可训练参数张量及全部 buffers）。seed 1030/1031、FP64、Torch deterministic、
两个 Torch CPU threads、chunk 2048。两步无梯度完整物理前缀，B=0 重置记忆，
两步自主预测及一次真实 loss backward；不是四步 BPTT。未来场/Ψ 不回授，
未来标签只在预测完成后作监督，控制仍是 recorded schedule replay。
两组 loss 分别为 0.2046684205563342 和 1.3075729366752518。

所有参数张量均有有限非零梯度，第二预测节点 loss 对第一节点 U/rawD/memory
跨步梯度验证通过。反向比较范围是 loss、逐场报告及已报告的梯度统计，
没有捕获全梯度数组，不宣称全梯度逐元素一致。NumPy 源/shard 仍 stop-gradient，
不宣称端到端可微。`retain_graph=True` 用于额外诊断，不等价于普通训练释放成本。

## 定位诊断与测试

`tools/benchmark_native_locator_geometry.py` 在可信冻结模块的独立 namespace
比较 scalar/batch 与无/正确/错误/无效 hints。74 个点来自曲面两单元 fixture，
不是生产轨迹采样。每案例重复三次交替运行顺序，计时与单独 cProfile 分开；
全部 8 案例的 element ids/local coordinates 逐位一致。
无 hint 标量几何调用从 539 降到 309；错误 cell hints 从 559 降到 326。
正确 hints 的批量几何调用从 11 降到 9。微基准不替代真实整链性能证据。

公共 Source.advance 测试用冻结旧搜索算法验证完整事件/历史/时钟与失败回滚；
旧 oracle 的搜索控制流独立，几何通过未改变的公共完整求值器计算。
alias loader 测试验证 namespace 隔离，不把加载当前代码的测试冒充旧版本数值 oracle。
实际全云对照另覆盖生产多行批次；专门的 25 轮极限 adversarial fixture 尚未新增。
补强的公共事件测试覆盖 5 个活粒子的混合 batch（立即完成、继续 Newton、
不同 hints、跨单元及丢失），以及必须拒绝全步/半步 damping 的 stiff cubic
消融事件。多行中部分完成后另一行 trial geometry 抛异常的专门组合尚未覆盖。

injInner 真实四事件 profile 中，调用次数如下。prefix/两次 forecast 的 profile
合并统计；父子累计时间重叠，不能相加，profiler 时间也不用于独立提速比。

| 调用 | before | after |
|---|---:|---:|
| locate_batch | 3354 | 3354 |
| scalar locate（含 fallback） | 6161 | 6161 |
| 批量几何求值 | 69920 | 57522 |
| 标量几何求值 | 274822 | 203125 |
| source sample | 3354 | 3354 |
| source.advance | 4 | 4 |

减少的是同一次定位内部的重复几何工作，不是粒子采样/事件数量。
批量定位自身 profile self time 上升（2.734→3.408 秒），提示新增缓存数组有成本；
是否净加速由下面无 profiler 的完整路径计时判断。

相关 CPU 回归 **223 passed**，一个既有 Torch profiler cycle 警告。包括新 locator
事件/loader 测试、既有 locator latency、marker sampling、ADAS、shared spatial、
source batch/time、辐射/nonlocal/出生/密度、完整状态/恢复/回滚、联合场步、
跨步反传及抓取/比较器测试。修改的生产/工具/测试编译与 `git diff --check` 通过。
新增公共事件文件单独为 23 passed；stiff 公共定位例另以独立冻结搜索算法对照。

## 独立整链计时

四个 `paired-{before,after}-{v400,inner}-timing` 在新进程串行执行，关闭 capture、
cProfile、torch.profiler。自己的重测试/比较/性能作业不与计时重叠；不保证整机
其它进程或 GPU 独占。保留阶段计时/sync wrappers，不是零插桩裸运行。
Torch `2.11.0+cu130`、NumPy `2.5.2`，其它模型/数据/精度/线程设置见上。

每个配对的完整初态/model/RNG/config/control binding 一致，并分别绑定对应
独立 capture 的相同源码 SHA。两组 loss/逐场报告/参数及跨步梯度统计、末 RNG
均 exact。所有运行 optimizer 更新为 0。

| 阶段（秒） | v400 before | v400 after | injInner before | injInner after |
|---|---:|---:|---:|---:|
| 两步完整物理前缀 | 33.3747 | 31.6931 | 75.5570 | 64.7015 |
| B=0 准备＋两步完整预测 | 13.5033 | 13.0461 | 31.8559 | 27.9780 |
| 标签读取 | 0.1329 | 0.1333 | 0.1372 | 0.1367 |
| 四项 loss＋指标 | 1.1380 | 1.0427 | 1.2554 | 0.9760 |
| loss backward | 0.0941 | 0.0881 | 0.0644 | 0.0708 |
| 主链合计（不含标签读取） | 48.1101 | 45.8699 | 108.7326 | 93.7263 |
| 嵌套子项：四次 marker source.advance | 10.3430 | 8.6369 | 83.4992 | 69.3628 |

主链合计为前缀、预测、loss/指标及 backward；不含构建/恢复/标签 I/O/额外
跨步诊断。marker 是父阶段中的子项，不能再次加进合计。
marker observed ratio 为 **1.1975 / 1.2038**（耗时减少 **16.49% / 16.93%**）；
主链为 **1.0488 / 1.1601**（减少 **4.66% / 13.80%**）。
这不是 CPU 并行测试或 CPU/GPU backend 对比；未新增线程/GPU迁移。

共享定位器也供 shard update 使用，该子项由 3.0139→2.2458 / 2.3563→1.6969 秒。
其它未改的 loss/backward 等阶段有单次波动，不能把全部变化归因于定位优化。
每组仅一个配对，未建立统计显著性或跨窗口普适保证，也不能据此声称 GPU 跑满。
Torch 峰值分配均为 413615616 bytes（394.45 MiB），峰值 RSS 约 3.44 GiB，
没有显著内存改善。NumPy 源/shard 的梯度边界未变。

机器汇总：`output/native-cpu-locator-2026-10-09/paired-performance-summary.json`；
保留全部四组原始 `result.json` 与 `chain/result.json`、两组 full-slot capture、
两组 exact-comparison 和 before/after-inner-profile 各三份、共六份 `.prof`。
旧端点证据沿用上一轮相同源码 capture；计时与 profile 的 before 均本轮新跑。
本轮输出约 1.4 GiB，保留完整证据，没有删除上一轮产物。

## 复现

以下输出必须使用尚不存在的新目录；同一 GPU/CPU 上串行运行，勿并发测试。
可信冻结 `.py` 与 checkpoint 为本地可信代码/pickle，不用于未知外部输入。

```bash
.torch-cuda-env/bin/python -m tools.profile_native_coupled_backward \
  --run-index 0 --capture-endpoints --deterministic \
  --output-dir output/native-cpu-locator-2026-10-09/after-v400-capture
.torch-cuda-env/bin/python -m tools.compare_native_cpu_sampling \
  --reference output/native-cpu-sampling-2026-10-09/after-v400-capture \
  --candidate output/native-cpu-locator-2026-10-09/after-v400-capture \
  --output output/native-cpu-locator-2026-10-09/v400-exact-comparison.json
.torch-cuda-env/bin/python -m tools.benchmark_native_locator_geometry \
  --output output/native-cpu-locator-2026-10-09/curved-comparison.json --repeats 3
```

injInner 使用 run-index 1，目录中的 v400 改为 inner。性能运行去掉 capture/profile
选项，before 另加 `--reference-coupling-file` 指向本轮冻结模块；after 省略该选项。
抓取和 profile 含同步/hash/I/O 或 profiler 开销，不能拿其耗时宣称加速。
本轮不评估新增物理精度、预测稳定性或 C0；后续候选是静态准备/配置验证成本，
须另立 mutable 输入失效与完整状态保真验收，不直接承诺新缓存安全。
