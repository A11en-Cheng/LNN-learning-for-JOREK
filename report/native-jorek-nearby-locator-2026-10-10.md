# 保存邻接表的 find_RZ_nearby 路径试验（2026-10-10）

> 后续勘误与归因：[运动增量诊断](native-nearby-motion-increment-diagnosis-2026-10-10.md)。本文第二窗18.35%来自两臂共减旧臂起点，属于累计终点差/旧臂本窗增量；修正双方各减自己的起点后，本窗速度增量误差为26.02%。旧机器产物及本文旧数值保留为历史执行证据，不据此称各自本窗增量。首窗2.63%不变；主误差集中在三个低密度扩散敏感粒子，并非已证明的病态几何。

本轮实现可切换的 CPU/H=1 marker 移动定位路径，不训练、不更新 optimizer、不修改 JOREK、原始数据或 roadmap。采用 `diagnosing-bugs` 建立完整源事件的红灯工作回归，再做源码对照、实际 Fortran oracle、完整真实云配对计时及历史续跑。**不是此前重建邻接的实验，也不是整套 JOREK 全局定位器的移植。**

## 实现和使用范围

- [native_nearby_locator.py](../src/models/native_nearby_locator.py)：每批从保存的旧单元/局部坐标起步，显式 2×2 Newton 更新；跨界时整步限到边上，读取**原生保存的 neighbours**，按互反边和节点方向转换坐标，然后继续。
- 源文件中 `R_old/Z_old` 初始化后被 `try_interp` 覆盖。因此调用接口使用旧 `i_elm/st` 即可，不需要另外缓存旧物理位置。
- 使用 JOREK 的 8 轮总预算、平方 R/Z 残差 `<1e-24`、行列式绝对值小于 `1e-8` 时的带符号逆值上限、方向相关边界距离和 `1-1e-12` 跨边判断。没有六级阻尼、逐粒子 Python 主循环或先失败再寻找邻居。
- 零邻居返回负的**当前**单元编号、边界局部坐标和丢失标记；保持已推进的物理位置。粒子不再参与后续子步。旧 Python 全局路径通常写 0，这是一项有意的状态语义变化。
- 只替换移动后定位；场求值、ADAS、电荷、源出生、权重、沉积、投影和时间历史保持原方法。出生和 shard 的无 hint 定位未随本轮更换。
- `NativeNearbyMarkerSource` 只在调用内浅复制 mesh/source 并替换副本定位入口；原对象方法不修改。保留网格/源配置快照守卫、并发拒绝和失败不提交，局部坐标要求 float64。邻接表常驻只读数组，内容进入配置指纹；不跨步缓存场值。
- [真实 builder](../tools/native_real_chain_builder.py) 支持 `args.marker_locator='jorek_nearby_v1'`，从初始化场文件读取 `neighbours`，缺失则失败，不根据共顶点重建。该选项要求 CPU `batched_v1`。原默认仍为 `native`，旧检查点不静默换算法。
- [完整事件诊断入口](../tools/profile_native_coupled_latency.py) 暴露 `--marker-locator jorek_nearby_v1`。本环境 CUDA 不可见，未执行真实 GPU 九场完整事件；小网格两事件/跨步反传测试已执行。NumPy 源的梯度边界不变。

## 源码事实与明确差异

一手只读源树：`/home/ac_desktop/JOREK_code/jorek`。

| 位置 | 行号 | 对照内容 |
|---|---:|---|
| `particles/examples/marker_spi_source.f90` | 655–663 | Adams–Bashforth 推进后携带旧单元和局部坐标调用 nearby |
| `grids/grid_utils/mod_find_rz_nearby.f90` | 61–62,92–161 | 8 轮、平方阈值、显式 Newton、限步跨边与丢失 |
| 同文件 | 165–185,205–209 | NaN/预算耗尽转全局，逆行列式 floor |
| `grids/mod_neighbours.f90` | 77–120,140–200 | 首个互反边、节点方向、角点象限规则和局部坐标转换 |

nearby 源 SHA256：`b1cbbd3b8cbc3250a81132ac9c8f638561d60e832c0113fa13dd984f8999dcf4`；neighbours 源：`9e1be99b70f3319db5677b1e77d0ab25bbffbf3435ff2d9e1c7c743d86b6f203`。当前 Python 旧路径源 SHA256：`42d64eee5610b020c4bfe22e225d30cc707db9644161d498e513eabdc0137f0e`。完整实现/工具哈希及输入内容指纹在机器报告中。

不能宣称整段逐操作复刻：

1. **全局 fallback** 仍是现有 Python 方法，不是 JOREK 的 rtree 排序/五初值 `find_RZ`。负邻居采用终止式全局结果；JOREK 那个分支成功后还用原边界 `st_new` 继续求值。失败的负邻居，Python seam 返回 0/99，而 JOREK 转负单元边界丢失。配置明确登记此差异。
2. **几何守卫** 沿用 Python 的正 Jacobian/RZ 守卫；JOREK `try_interp` 的带符号 floor 本身允许非正行列式。无效点转全局，不静默替换几何。两组本轮真实窗口均未触发这条 seam。
3. **拓扑校验** 对非法正邻居/互反边在构建时失败；JOREK 在部分异常跨边时转丢失。正邻居方向还要求共享节点 ID；实际生产表满足要求，不能推广到任意坐标重合但不同 ID 的表。
4. 初态不提前用残差退出；角点按源码象限比较选择边，不增加“对角邻居”或任意角点回退。

真实表有 198 个零邻居、68 个负邻居，正邻居全部互反；保持 0 和负值的区别，尤其轴相关负邻居不得改写为普通外边界。

## 紧反馈回归与独立 Fortran 对照

红灯入口（在接入前执行）：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .torch-cuda-env/bin/python -m pytest -q tests/test_native_nearby_locator.py
```

完整源事件中近公共曲面边的单粒子移动确实进入原 `_locate`，输出 `crossing restarted global candidate search 1 times`，0.89s 失败。随后同一测试切换明确的新 adapter，要求通过相邻单元且全局重启次数为 0；跨边工作回归已绿。

独立 oracle：[Fortran harness](../tests/native_nearby_reference.f90)、[编译/解析输入](../tests/native_nearby_oracle.py)、[oracle tests](../tests/test_native_nearby_oracle.py)。在 pytest 临时目录编译**实际 JOREK nearby 文件和原样提取的两段邻居子程序**，测试几何用独立 tensor 单项式，不调用 Python 基函数求期望。覆盖曲面、逆向边、多单元、角点、丢失、32 种边方向组合、八轮预算和符号 floor。全局定位是受控 stub，因此这些测试不证明全局 seam 的数值等价。

Python/Fortran 的 seven nearby 解析案例单元号一致，局部坐标 `rtol=0, atol=1e-12`；单独登记全局失败和非法拓扑差异。不能将“oracle 有 68 项”解读为 68 个完整真实云对照。

## 两组完整真实云计时

工具：[benchmark_native_nearby.py](../tools/benchmark_native_nearby.py)。当前生产 CPU 方法是基线，含上一轮已完成的保序候选筛选和调用内几何复用，**不是更老慢版本**。每臂预热一次，正式两轮反序配对，再独立跑计数；完整 100 万槽、相同初态/场/rawD/固定一次 seeded 出生和 RNG。构建、出生、克隆、比较和额外计数不在正式源推进时间里。

| 首个完整源窗口 | 当前 native 中位数 | nearby 中位数 | 耗时减少 |
|---|---:|---:|---:|
| v400 | 4.93635 s | 3.14335 s | 36.32% |
| injInner | 38.40825 s | 28.68962 s | 25.30% |

原始配对：v400 baseline 4.91984/4.95286、nearby 3.10225/3.18445；inner baseline 38.24697/38.56953、nearby 28.77427/28.60497。这是两个样本的本机观察，不是统计显著性或整训练吞吐结论。小型回归与源码 oracle 可产生短暂 CPU 活动，未并发第二个完整真实性能作业；不宣称专属隔离机器计时。

v400 初活 11,168、出生 29、40 子步、447,880 次粒子访问；nearby 1,897 次跨边，0 lost，0 全局 fallback。inner 初活 52,604、出生 56、81 子步、4,265,327 次访问；6,462 次跨边、2 lost、0 fallback。两组源码和其它输入前后相同。进程峰值 RSS 2.518/2.438 GiB，包含比较临时内存，不是邻接缓存新增成本。

## 误差和目前结论

统一相对 L2：`||candidate-reference||₂ / ||reference||₂`，在原生数组上计算，不给分母补 epsilon；全零相同报 0，全零不同比值未定义并单报不匹配。单元 ID 不用其数值 L2 判断对错，而按逐粒子一致性/丢失身份核对。运动另比对相对初态的真实位移和速度变化，避免大位置本底掩盖误差。

| 指标 | v400 | injInner |
|---|---:|---:|
| 位移增量相对 L2 | 7.61e-11 | 1.041e-3 |
| 速度增量相对 L2 | 3.29e-10 | 2.630e-2 |
| 完整 P_imp 相对 L2 | 5.65e-13 | 2.73e-9 |
| 电离投影相对 L2 | 6.69e-10 | 1.31e-7 |
| 辐射投影相对 L2 | 8.44e-10 | 3.69e-6 |
| 电荷平方矩投影相对 L2 | 6.07e-10 | 4.26e-5 |
| 电荷矩投影相对 L2 | 8.15e-10 | 1.94e-4 |
| 数密度矩投影相对 L2 | 1.30e-9 | 5.40e-4 |

两组 alive、权重、寿命、出生、时钟、gather 和 RNG 都一致。v400 单元编号也全部一致；inner 严格 ID 检查有 2 项不一致，所以 `inner.json success=false`，不是推进崩溃。反序配对和独立计数均实际完成，`experiment_executed=true`。

同目标跨边诊断进一步确认：slot 973、701199 在边界丢失，旧结果分别是 0，新结果是 -3377/-3376。新路径保留边界局部坐标，符合 JOREK；**不是两个活粒子被选到不同单元**。6,462 个检查目标中仅这两个 ID 不同。然而此检查只覆盖跨边/异常点，不能排除同单元内的局部坐标漂移，也不能拿该结果解释掉 2.63% 的速度增量差异。

误差是相同 RNG 下算法变更的确定性差异，不是出生抽样统计噪声。旧定位接受 `<1e-10`，新定位接受 `<1e-12`，以及 damped solve/显式限步不同，会改变下一子步的采场；病态几何可将很小 RZ 残差放大为 st/梯度差异。这一归因需要全点同目标证据，不以推测代替结论。

**当前新路径保持 opt-in；未强制替换训练默认、未声称完整生产/JOREK 逐值等价。**

## 连续窗口与续跑

[validate_native_nearby_history.py](../tools/validate_native_nearby_history.py) 让两臂各自延续完整云、非零投影历史和 RNG，各自产生下一窗出生；不共用未来出生。第一窗完整 MarkerRuntime+RNG 序列化后，恢复重做第二窗，要求同臂所有数组、稳定物理诊断、定位计数、出生和 RNG 逐值一致。

v400 4111→4112：29→2 出生；跨臂相对反馈 L2 为 6.15e-10→1.10e-9，离散契约通过；两臂恢复均完整一致。该验证是给定记录场/rawD/shard 的条件源诊断，不是 NN/shard 自主推进，不读取未来云或投影替换状态。

injInner 5681→5682：56→11 出生，两臂出生本身逐值一致。两臂各自恢复均完整一致，包括定位计数和 RNG；跨臂仍有丢失编号差异，`inner-history.json success=false` 保留，不将正确的同臂续跑当作跨臂等价。第一窗/第二窗完整反馈相对 L2 为 4.93e-5/2.50e-4；第二窗位移增量相对 L2 为 8.136e-3，速度增量相对 L2 为 **1.835e-1**。因此不能在无完整运动核查的前提下自动切为训练默认。

## 全点同目标补充：定位差异与动力学差异分开

[inner-all-points.json](../output/native-jorek-nearby-2026-10-10/inner-all-points.json) 检查所有 **4,265,327** 个定位目标，而非仅跨边点。对每个 nearby 轨迹的目标 `(R,Z)`、旧单元和旧 st，另外调用当前定位器；不改变新路径实际输出。这个诊断不是两臂同轨迹演化，也不是计时基准。

- 同目标只有两个 ID 差异，仍是 slot 973、701199 的丢失编码。没有发现活目标在此对照中落入不同单元。
- 全点同单元坐标差异的最大范数约 `2.9503e-8`。最大例 slot 800403、第80子步：旧物理残差 `6.84e-11`，新残差 `2.24e-16`，Jacobian 条件数约22.23；同点重采样的速度相对 L2 差异约 `5.16e-10`。
- 首个超过 **诊断阈值** `1e-8` 的局部坐标差异在 slot 201150、第0子步；旧物理残差 `9.92e-11`，新为 `2.24e-16`，条件数约14.02，同点速度差异 `5.24e-10`。这里的 `1e-8` 只用于选择展示事件，不是定位验收容差或“第一个浮点差异”。
- 这些观测不支持把已展示的点直接归为严重病态逆映射；更不能用同点微小速度差异解释掉跨时间轨迹/速度增量差异。完整推进后的误差显著大于同点定位误差，提示多子步输运/场采样敏感性，但尚未做两臂首个动力学分叉的因果归因。
- 展示名单按局部坐标差异选取，不是全粒子速度敏感度最大值；不能将列表中的 `5e-10` 当作所有目标速度误差的上界。

下一安全工作是隔离两臂输运差异：同初态逐子步记录高误差粒子的 st、采样速度、`rho_imp`/扩散分母及位移增量，再与相同当前位置的实际 JOREK 求值/推进对照。不要先修改容差、屏蔽边界或用新路径“更像源码”替代误差证明。

## 验证、复现与机器证据

最终主代理合并回归 **18 文件，267 passed in 59.74s**；包含完整源、几何/邻接、时间/历史/投影、真实链 builder、模型事件、实际 Fortran oracle、诊断 CLI 和连续历史接口。编译与 `git diff --check` 通过；不重复累计此前 249/86/21 等子集。新增测试无需训练或 optimizer。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .torch-cuda-env/bin/python -m tools.benchmark_native_nearby --run-index 0 --repeats 2 --output output/native-jorek-nearby-2026-10-10/v400.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .torch-cuda-env/bin/python -m tools.benchmark_native_nearby --run-index 1 --repeats 2 --output output/native-jorek-nearby-2026-10-10/inner.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .torch-cuda-env/bin/python -m tools.validate_native_nearby_history --run-index 0 --output output/native-jorek-nearby-2026-10-10/v400-history.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .torch-cuda-env/bin/python -m tools.validate_native_nearby_history --run-index 1 --output output/native-jorek-nearby-2026-10-10/inner-history.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .torch-cuda-env/bin/python -m tools.diagnose_native_nearby_difference --run-index 1 --all-points --output output/native-jorek-nearby-2026-10-10/inner-all-points.json
```

输出独占创建，复跑需使用新路径。真实诊断 600s 协作 SIGALRM/8GiB 单进程 RSS 上限，底层 native 调用可推迟中断，不是硬 watchdog。无训练或超参搜索。条件历史/同点差异诊断可并行执行，因此它们的 job wall 不用作速度比较。

机器文件：[v400](../output/native-jorek-nearby-2026-10-10/v400.json)、[inner](../output/native-jorek-nearby-2026-10-10/inner.json)、[v400 历史](../output/native-jorek-nearby-2026-10-10/v400-history.json)、[inner 历史](../output/native-jorek-nearby-2026-10-10/inner-history.json)、[inner 跨边诊断](../output/native-jorek-nearby-2026-10-10/inner-difference-r2.json)、[inner 全点诊断](../output/native-jorek-nearby-2026-10-10/inner-all-points.json)。首次诊断缺 `args.repeats` 的失败文件 `inner-difference.json` 保留，修复入口并新增 CLI 回归后成功执行 r2；不隐藏失败或将其计为物理执行。成功执行诊断不等于跨臂合同通过：inner 各机器文件的 `success=false` 均按严格原生编号条件保留。
