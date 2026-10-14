# 完整耦合前向与反向性能核查（2026-10-09）

## 范围与证据口径

用户要求先测当前完整反向传播链的卡点，判断 marker 的矩阵求解是否适合迁移 GPU。
本轮只做诊断：不更新优化器、不追加训练、不改变物理步长、不裁剪真实云，
不读取未来真实场作源/shard/平衡回授。保留原始数据、JOREK 与已有工作区修改。

实际训练权重来源为
`output/native-cuda-curriculum-2026-10-09/seed1-run3/epoch-3-parameters-optimizer.pt`。
当前课程路径是两步无梯度完整物理前缀、B=0 记忆重置、两步完整自主耦合预测，
监督四项为完整原生系数、rawD、Gauss 场值与兼容 R/Z 梯度。
这不是四步全反传，也不是正式 burn-in 窗口长度已经确定。

源采用已验证 CPU `batched_v1`，Torch 场/CfC/损失与反向在 CUDA FP64 上运行。
此前 `cuda_fp64_v1` 空间实验出现粒子定位分歧，未获训练资格，本轮不采用。
恢复需要匹配每个 run 的全部 buffer、归一化及其训练支撑、完整初态 digest、
四个控制与配置指纹；不能只匹配参数维度。

计时区分无 cProfile/torch.profiler 的阶段测量与算子插桩测量。
阶段包装仍会引入同步；父子累计时间不能相加。额外跨步 `autograd.grad`
只用于验证梯度连通，不计入实际训练关键路径。
每组无算子插桩测量仅一次，不是多次重复的统计基准；未独占 GPU，初始整卡
查询显示其他占用。因此只用于定位数量级瓶颈，不据此宣称微小速度差异显著。

## marker 中哪些操作解矩阵

| 操作 | 当前实现 | GPU 判断 |
|---|---|---|
| 电荷态推进 | 11 态显式相邻转移、截断和归一化 | 无 11×11 求解；可做向量核，但不是矩阵加速问题 |
| 位置推进 | 当前/前速度的显式向量公式 | 无矩阵求解 |
| 网格定位 | 大量 2×2 Newton 求解，伴随候选查找、阻尼与回退 | 需批量驻留与分支一致性；只搬小 solve 不保证收益 |
| 五通道沉积 | 16 个局部自由度累加 | 适合批量核候选，但需要验证累加精度 |
| 投影 | 16419 自由度稀疏系统，初始化分解并复用，5 个 RHS | 可以评估 GPU 稀疏后端，但先确认占比与传输成本 |

源码：`src/models/marker_state.py:200`、`:215`；
`src/models/native_coupling.py:74`、`:252`、`:292`、`:328`。
投影算子可以包含过滤/边界处理，不能笼统替换为无约束质量矩阵。

旧版 injInner 首事件 cProfile 中，投影累计 0.0212 s（2 次），定位 40.9016 s，
其中标量定位回退 19.1785 s。它在最后一次共享几何优化之前，且存在父子重叠，
只能用于判断候选方向，不能冒充本轮完整反向计时。
原始证据：`output/native-cpu-latency-2026-10-09/after-inner-profile/chain/event-1-profile.json`。

## 梯度边界

`TorchRecomputedSourceEvents` 保留预测 U/rawD/CfC memory 的跨步图。
NumPy 云、出生、shard 与平衡诊断消费当前预测场的 detached 镜像，每步重算，
但这些物理推进本身停止梯度。Torch 的源特征编码及依赖 U 的闭合仍可训练。
marker 投影/shard 误差目前是指标，不是该四项监督的可微 loss。
不能将此次核查表述为全物理链端到端可微。

接口证据：`src/training/base_coupled.py:133`、`:193`、`:213`；
监督证据：`src/training/coupled_metrics.py:88`、`:232`；
前缀证据：`tools/native_memory_burnin.py:21`。

## 本轮实测

两组完整真实窗口均通过，百万 marker 槽位保留，初始活粒子为
v400 11168 / injInner 52604。每组执行四次完整自主耦合事件：前缀两次，
监督预测两次。未来场/投影/shard 标签在预测后才打开。

下表为无 cProfile/torch.profiler 的同步阶段测量，单位秒；冷启动首次场
forward 包含 CUDA 首次执行开销，没有用热身替换真实首事件。

| 不重叠主要阶段 | v400 | injInner |
|---|---:|---:|
| 两步无梯度因果前缀 | 34.3663 | 80.8025 |
| 记忆重置准备＋两步耦合预测 | 13.8009 | 33.2482 |
| 标签读取 | 0.1362 | 0.1396 |
| 四项 loss＋全部逐场/源/shard 指标 | 0.5099 | 0.4609 |
| 两步 loss backward | 0.1185 | 0.0816 |
| 上述合计（不含构建/额外诊断） | 48.9317 | 114.7326 |
| 含构建/恢复/额外诊断的进程总时间 | 58.9817 | 124.5003 |

父阶段内部另有以下计时，不能重复加到上表：

- v400 四次 `source.advance` 共 11.0360 s，四次场 forward 共 1.3310 s
  （第一次 1.2282 s，后续三次共 0.1028 s）；density 5.0013 s、
  shard update 3.0983 s、shard refresh 5.2848 s。
- injInner 四次源推进共 90.3449 s、场 forward 共 0.9561 s
  （首次 0.7177 s，后续三次共 0.2384 s）；density 4.7402 s、
  shard update 2.3143 s、shard refresh 5.9136 s。
- v400 loss 本身 0.1780 s，指标 0.3231 s；injInner loss 0.0863 s，指标
  0.3677 s。指标/D2H 有开销，但远小于完整 CPU 事件等待。

28 个可训练参数张量（3147 参数）两组全部获得有限非零梯度；第二预测节点
的四项 loss 对第一预测节点的 U/rawD/memory 梯度均有限非零。权重/buffer 与
输入初态的完整 digest 前后未变；未创建优化器，更新次数为 0。
本次两组 loss 为 0.2046684205563342 / 1.3075729366752518，与原课程最终
固定节点结果一致到约 1e-11 量级；这不是精确 CUDA 逐位复现或精度验收。

Torch 分配显存峰值两组均为 390766080 bytes（372.66 MiB）；RSS 约 3.27 GiB。
`backward(retain_graph=True)` 为随后独立跨步连通检查保留图，不能将该显存
释放行为等同普通训练。诊断 seed 在构造前固定，不恢复优化器/checkpoint RNG，
不能声称重放原 epoch 的随机轨迹。builder 日志中的 untrained/unit-reference
描述恢复前构造；实际计时在严格恢复当前训练权重与归一化后进行。

原始结果：
`output/native-coupled-backward-2026-10-09/v400-timing/result.json`、
`output/native-coupled-backward-2026-10-09/inner-timing/result.json`。

## 独立热点采样与结论

injInner 独立插桩运行通过，进程总时间 153.3262 s。三份 cProfile
（prefix、forecast_event_1、forecast_event_2）覆盖四次完整事件，合并结果如下。
这不是无插桩基线，父子项重叠，不可求和。

| 操作 | 调用次数 | 累计秒 | 解释 |
|---|---:|---:|---|
| `locate_batch` | 3354 | 64.6636 | 最大 CPU 热点，包含下面标量回退/几何/小 solve |
| 标量 `locate` | 6161 | 31.7007 | 回退仍昂贵，属于定位累计时间的子项 |
| 源 `sample` | 3354 | 27.2963 | 包含 cubic 场/反馈求值与原子表查询 |
| `configuration_digest` | 202 | 14.7861 | 重复配置/状态验证开销，不等于可全部移除 |
| density `reference_solver` | 17 | 14.5992 | 包含构造与配置验证，和上项有重叠 |
| `deepcopy` | 557772（原始调用） | 6.5287 | 原子提交/回滚及状态快照相关 |
| `deposit_batch` | 3354 | 6.4414 | 五通道局部累加 |
| 定位的 `np.linalg.solve` | 102314 | 2.3095 | 87320 标量回退调用＋14994 批量调用，均为 2×2 |
| `charge_step` | 3354 | 0.4761 | 显式电荷态推进 |
| `push_velocity` | 3354 | 0.4012 | 显式位置推进 |
| `project` | 8 | 0.0777 | 含 marker 与独立 density 投影，不仅是 marker |

即使把本次所有 2×2 solve 与投影求解的累计时间完全消除，按插桩口径也只约
2.39 s；定位的其余工作仍然存在。这不是 GPU 实测上限，只是说明
“定位慢”不等于“小矩阵 solve 慢”。主要 CPU 工作是迭代几何、基函数、
候选元素查找与回退，以及场/表采样。

CUDA backward trace 中实际 kernel 共 613 次、累计 20.2822 ms；
13 次设备内拷贝累计 2.1553 ms。最大一组 FP64 GEMM kernel 累计 9.7060 ms。
单次插桩 backward 墙钟 0.07135 s，与无插桩 0.08156 s 不作速度优劣比较。
剩余墙钟包含 CPU autograd 调度、kernel launch、首次模块加载与同步等待。
CfC 不是当前总体热点。`key_averages` 同时有 aten 与 kernel 关联时间，
不能把两者相加；CPU trace 嵌套累计也不能当成墙钟总时间。
完整原始 trace 和算子表在 `inner-profile/chain/` 中。

### 推荐下一步（本轮未实施）

1. 先优化定位/采样：以相同输入、全云身份及历史为守卫，减少标量回退，
   编译/融合 cubic 基函数和几何迭代，减少重复构造/查表。不能放宽定位容差
   或改变粒子子步来获得表面提速。
2. 核查重复 density solver 构造、不可变配置 hash 与 full-state 拷贝。
   缓存必须有变更失效和错误配置拒绝测试，保留事件原子提交/回滚。
3. GPU marker 路径应以“批量定位→场/原子表采样→推进→沉积”的驻留链为目标，
   而非仅迁移小 solve。先定位并修复此前 GPU 空间路径的真实粒子分歧，再对
   两组 full-cloud 冻结权重事件做精度和无插桩延迟验收。
4. 独立窗口 CPU worker/GPU owner 的流水并行可以另做冻结权重基准；
   同窗口下一步必须等待上一预测场，不得跨权重更新复用预测云/源。

GPU 稀疏投影求解不是当前首要动作；本轮没有新增 GPU 矩阵 A/B 基准，
不宣称其一定加速或一定更慢。先前 GPU 定位/求值实验的数值拒绝结果仍然有效。

## 复现与验证

以下 CUDA 命令需在可访问 GPU 的环境中串行运行，输出目录必须不存在。

```bash
.torch-cuda-env/bin/python -m tools.profile_native_coupled_backward \
  --run-index 0 --output-dir output/native-coupled-backward-2026-10-09/v400-timing
.torch-cuda-env/bin/python -m tools.profile_native_coupled_backward \
  --run-index 1 --output-dir output/native-coupled-backward-2026-10-09/inner-timing
.torch-cuda-env/bin/python -m tools.profile_native_coupled_backward \
  --run-index 1 --profile-forward --profile-backward \
  --output-dir output/native-coupled-backward-2026-10-09/inner-profile
.torch-cuda-env/bin/python -m pytest -q \
  tests/test_profile_native_coupled_backward.py tests/test_native_memory_burnin.py \
  tests/test_coupled_metrics.py tests/test_native_production_torch_integration.py \
  tests/test_native_autonomous_pilot.py
```

相关回归 **36 passed, 1 skipped**；工具新增测试 4 项全部通过，包含真实课程
四项 loss 与原 `observe` 的 fixture 精确对照、完整 buffer 恢复、两步前缀
梯度屏障与无优化器更新。沙箱中跳过的 CUDA 专项不能计作通过；本轮三次真实
GPU CLI 运行单独记录，不用 fixture 替代真实完整事件证据。

诊断工具：`tools/profile_native_coupled_backward.py`；本轮不修改生产推进器。

## 方法

使用 diagnosing-bugs skill：复用已有慢链反馈环，按当前真实训练路径构造
无优化更新的阶段计时与跨步梯度检查，再独立进行 CPU/CUDA 算子插桩。
本轮请求是诊断，不据尚未测量的假设修改生产推进算法。
