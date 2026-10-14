# CPU marker 采样优化与完整耦合保真（2026-10-09）

## 范围

这是 CPU 算法优化的第一个有界切片，不是新增训练或物理模型。修改只涉及
CPU `batched_v1` marker 采样和 ADAS 查询的重复工作，保留公式、粒子子步、
定位容差、投影累加、全部槽位、出生/身份/RNG、反馈历史、事件顺序及失败回滚。
没有更新 optimizer、JOREK、原始数据、roadmap 或 ADR。

三个公开保真入口经用户确认：`NativeNeonAtomicRates`、
`NativeMarkerSource.advance`、`TorchRecomputedSourceEvents`。
本轮使用 `tdd` 做公开入口红→绿与回归，使用 `diagnosing-bugs` 将保真运行与
无快照性能运行分开；子代理分别负责 ADAS、证据工具和独立只读审查。

## 实现

1. **marker 消费专用求值**：仅计算 transport 真正读取的值通道
   `[4,6,7,8]`、R/Z 梯度通道 `[0,1,7]` 与前投影值通道 `[3,4]`。
   入口仍校验完整九场和完整五通道反馈/历史；每批几何检查保留。
   公共完整 FEM 求值器、九场状态、神经网络的值/梯度输入均未裁剪。
2. **事件局部 owned snapshot**：静态 rawD 采样每事件准备一次场快照；
   动态采样每个实际粒子采样时刻准备一次，供该时刻的 chunks 复用。
   前投影在事件入口复制并只读，不使用本事件刚生成的投影。
   不跨事件缓存，不按 ndarray 身份假设内容不变。
3. **ADAS 逐表共享搜索**：同一张表的十个电荷态共享 clip、searchsorted、
   索引及插值坐标；SCD/ACD/PLT/PRB 各自使用自己的轴。保留逐行四项
   乘加和指数顺序、默认 left 搜索、单位转换、PLT0–9/PRB1–10 映射与版本指纹。

一个实际修正：初次采用 advanced field indexing 改变了局部数组 strides，
在完整 marker 测试中引入 1 ULP 差异。没有放宽精度标准；改用保留原布局的
连续 slice views 后，冻结 full evaluator 的事件级精确测试通过。
独立只读审查另覆盖 C/F/非连续布局、空批、重复元素及 2048 粒子批。

源码入口：`src/models/native_coupling.py` 的 `_marker_points`、`advance`、
`NativeNeonAtomicRates.__call__`；共享查询 helper 为
`src/models/native_atomic_interpolation.py`。

## 对照契约

冻结旧完整模块：
`output/native-cpu-sampling-2026-10-09/native_coupling_before.py`；SHA256
`282efa5c22fd9837ce128f18a761416f2b566bd9705e58eeb0d1087829214b3f`。
诊断 CLI 在新进程、生产 consumers 导入前以同一个模块名加载它；不是把新
helper 再称为旧 oracle，也不改物理 configuration。

本轮生产 `native_coupling.py` SHA256 为
`1471f8567f1abcc0ccce115a996dda50fee549317cc2c76bbc774bc9c2d45c40`；
`native_atomic_interpolation.py` 为
`23fc174293056877aeab0eb2ec1f5aee77263ae5b02f4d0e843c4b825df27618`。

v400 与 injInner 的 before/after 都严格恢复
`output/native-cuda-curriculum-2026-10-09/seed1-run3/epoch-3-parameters-optimizer.pt`：
3147 参数、全部 run-specific buffers、训练支撑/归一化、完整初态及控制指纹。
seed 为 1030/1031，Torch deterministic 开启，CUBLAS `:4096:8`，FP64，
2 个 Torch CPU threads，marker chunk 2048。

每组为两步无梯度完整物理前缀、B=0 记忆重置、两步自主耦合预测与一次真实
loss backward；不把前缀宣称为四步 BPTT。监督包含完整原生系数、rawD、
Gauss 值、兼容梯度四项。未来标签仅在两个预测事件完成后读取；场、Ψ 与
其它诊断回授均来自预测路径。控制是 recorded schedule replay，不是自主控制预测。
诊断 seed 在 builder 前设置，未重放 checkpoint optimizer/controller/RNG 轨迹。

每事件抓取完整九场/rawD/memory/前场、全部百万 marker 槽位的所有数组、
五通道投影值及 deltas、实际传入场网络的独立 density source、全部 shard
数组/非数组状态、时钟、历史来源、配置及 Python/NumPy/Torch/CUDA RNG。
比较器拒绝绑定不一致、缺键、错误 shape/dtype 和组件 digest 不一致；没有
backend-change 白名单，连续数组也要求 array_equal。

相对误差直接用原生 JOREK 单位计算。零参考分量另登记不匹配数，不靠 epsilon
掩盖。抓取是事件端点而非每个粒子子步轨迹；没有抓取全梯度数组，反向对照
范围为 loss/逐场报告与梯度统计、连通性，不宣称全梯度逐元素一致。
捕获运行含同步/hash/I/O，因此不作为速度证据。

## 验证结果

两组各四事件的完整数组、离散身份、metadata、clock 与 RNG 全部逐位一致，
`all_physical_evidence_exact=true`。相对 L2/逐点误差在有定义的分量上均为 0；
零参考处也没有新增不匹配。每组保留 1000000 槽位与全部 846 shards。

| run | 初始活粒子 | 四事件末活粒子 | loss（before=after） |
|---|---:|---:|---:|
| v400 | 11168 | 11197 / 11199 / 11201 / 11203 | 0.2046684205563342 |
| injInner | 52604 | 52658 / 52668 / 52686 / 52704 | 1.3075729366752518 |

两组 28 个可训练参数张量均获得有限非零梯度；第二预测节点 loss 对第一节点
U/rawD/memory 的梯度有限非零。loss、逐场报告、参数梯度统计、跨步梯度统计
before/after 全部 exact，模型/buffers 与输入初态 digest 均未变。
这里的 loss 是冻结既有权重的重算结果，不是一次新增训练的成绩。

相关 CPU 回归 **175 passed**（一个既有 Torch profiler cycle 警告）。覆盖静态/
动态采样、chunk 复用、跨事件内容变化、未消费字段 NaN/Inf 拒绝、全部 cubic
自由度和非零投影历史的冻结 full evaluator 对照、独立 ADAS public-provider
公式/真实表/不同轴、生产出生/密度/nonlocal/辐射、状态恢复/回滚、联合场步及
跨步反传、抓取/比较器完整性。修改的生产/工具/测试文件编译通过；
`git diff --check` 通过。

```bash
.torch-cuda-env/bin/python -m pytest -q \
  tests/test_native_marker_sampling.py tests/test_native_atomic_interpolation.py \
  tests/test_native_shared_spatial.py tests/test_native_source_batch.py \
  tests/test_native_source_time_integration.py tests/test_native_production_radiation.py \
  tests/test_native_coupling.py tests/test_native_production_events.py \
  tests/test_native_production_nonlocal.py tests/test_native_production_birth.py \
  tests/test_native_production_density.py tests/test_native_source_time.py \
  tests/test_native_bounded_cloud.py tests/test_native_joint_source.py \
  tests/test_native_production_torch_integration.py tests/test_native_state.py \
  tests/test_native_temporal.py tests/test_profile_native_coupled_backward.py \
  tests/test_compare_native_cpu_sampling.py
```

机器证据：`output/native-cpu-sampling-2026-10-09/{v400,inner}-exact-comparison.json`，
四套 `{before,after}-{v400,inner}-capture/chain/` 的逐事件 NPZ/metadata 与 result。

## 性能口径

串行独立运行 `paired-{before,after}-{v400,inner}-timing`，每组仅一个配对，
不启用 capture、cProfile 或 torch.profiler，deterministic 与其它设置一致。
计时仍有同步及阶段 wrappers，不等于零插桩；父子时间不能相加。
自己的重测试/比较/GPU 作业不与计时重叠，GPU 并非独占。期间一次整卡查询为
RTX5070Ti、9% utilization、3002/16303 MiB，不能据此推导平均利用率。

两组配对均完成。Torch `2.11.0+cu130`、NumPy `2.5.2`。before/after 和独立
capture 的初态/model/RNG/configuration binding 一致，配对 loss/逐场报告和
梯度统计也 exact；全部运行 optimizer 更新为 0。

| 阶段（秒） | v400 before | v400 after | injInner before | injInner after |
|---|---:|---:|---:|---:|
| 两步完整物理前缀 | 34.1272 | 33.0859 | 80.8696 | 75.3520 |
| B=0 准备＋两步完整预测 | 13.5573 | 13.4654 | 33.4543 | 31.7052 |
| 标签读取 | 0.1330 | 0.1340 | 0.1407 | 0.1346 |
| 四项 loss＋逐场/源/shard 指标 | 2.1056 | 1.4421 | 2.6942 | 1.0934 |
| loss backward | 0.0737 | 0.1131 | 0.0669 | 0.0600 |
| 主链合计（不含标签读取） | 49.8637 | 48.1065 | 117.0850 | 108.2107 |
| 嵌套子项：四次 marker source.advance | 10.9558 | 10.2176 | 90.4462 | 83.3213 |

主链合计包含前缀、预测、loss/指标及 backward；不含构建/恢复、标签 I/O、
额外诊断。marker 是父阶段中的子项，不能再次加进合计。
本次 marker observed ratio 为 **1.0722 / 1.0855**（耗时减少 **6.74% / 7.88%**）；
主链为 **1.0365 / 1.0820**（减少 **3.52% / 7.58%**）。
未修改 loss/CfC/反向实现，指标与 backward 耗时有明显单次波动；不能把这些
波动归因于本次 CPU 优化，也不能声称 GPU 已跑满。

峰值 Torch 分配显存均为 413615616 bytes（394.45 MiB）；before/after RSS
约 3.44 GiB，没有出现显著内存下降。
机器汇总：`output/native-cpu-sampling-2026-10-09/paired-performance-summary.json`，
原始每组 `paired-*-timing/result.json` 与 `chain/result.json` 保留全部阶段。

旧 default-mode `before-*-timing` 保留，但不与本次 deterministic 运行组成速度比。
额外跨步梯度诊断不计入主链，backward 为 `retain_graph=True`；这不是普通
训练 `retain_graph=False` 的释放成本或显存峰值等价测量。
单次 observed ratio 不代表统计显著，也不保证不同窗口一定加速。

本轮输出约 2.7 GiB，保留完整证据未删除旧产物。

## 复现

输出目录必须不存在。CUDA 命令串行运行；可信冻结模块和 `.pt` 使用本地
可信 pickle 路径，不用于载入未知外部代码。

```bash
.torch-cuda-env/bin/python -m tools.profile_native_coupled_backward \
  --run-index 0 --capture-endpoints --deterministic \
  --reference-coupling-file output/native-cpu-sampling-2026-10-09/native_coupling_before.py \
  --output-dir output/native-cpu-sampling-2026-10-09/before-v400-capture
.torch-cuda-env/bin/python -m tools.profile_native_coupled_backward \
  --run-index 0 --capture-endpoints --deterministic \
  --output-dir output/native-cpu-sampling-2026-10-09/after-v400-capture
.torch-cuda-env/bin/python -m tools.compare_native_cpu_sampling \
  --reference output/native-cpu-sampling-2026-10-09/before-v400-capture \
  --candidate output/native-cpu-sampling-2026-10-09/after-v400-capture \
  --output output/native-cpu-sampling-2026-10-09/v400-exact-comparison.json
```

injInner 将 run-index 改为 1，目录中的 v400 改为 inner。
性能运行去掉 `--capture-endpoints`，输出到独立 timing 目录；保留相同
deterministic 与其它设置。结果 `success` 仅代表比较完成，数值保真验收看
`all_physical_evidence_exact`。

性能目录名为 `paired-before-v400-timing`、`paired-after-v400-timing`、
`paired-before-inner-timing`、`paired-after-inner-timing`；旧实现仍显式加
`--reference-coupling-file`，新实现不加。

## 剩余限制与下一步

- 这次没有修改定位器，不能声称最大热点已解决。下一优先项是同事件的定位
  fallback/迭代几何，采用冻结轨迹、完整身份与精确事件守卫，再评估编译/融合。
- 重复 density solver 构造与不可变 configuration hash 可独立优化，但必须有
  变更失效、错误配置拒绝和回滚测试，不能缓存预测状态跨权重更新。
- 标准 time sampler 保证 owned snapshot；不把它推广成任意可变回调/并发外部
  修改也安全的契约。未开展 CPU workers 流水并行，未新增 GPU marker 后端。
- NumPy source/shard 仍 stop-gradient。优化不补齐端到端可微，不意味着预测更准、
  长时稳定、C0 验收或完整 JOREK 数值求解。
