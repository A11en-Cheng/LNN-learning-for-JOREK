# 出生概率表细化与 debug seed（2026-10-10）

> 后续用户选择：真实 n0 链路及诊断 CLI 默认已改为每最小源宽 **64** 格。
> 下文 128 格实验与成本数据保留为当次测量，不能视为当前默认。
> 64 格的独立概率相对 L2 为 `4.463e-5 / 3.595e-5`；分辨率仍进入配置指纹，
> 旧 128 格检查点不能静默迁移。更多物理窗口见后续窗口统计报告。

本轮不训练、不更新 optimizer，不修改 JOREK、原始数据、roadmap 或 ADR。
提高概率表精度与完整固定场 marker 窗口投影 A/B 分开验证。
数量、总源 clipping、权重、定位和初始化语义沿用
[原生契约修正](native-birth-native-correction-2026-10-10.md)。

## 独立误差实测

机器结果：[refinement-final.json](../output/native-birth-refinement-2026-10-10/refinement-final.json)。
快照为 v400 `jorek004111.h5`、injInner `jorek005681.h5`，保留全部 shard 和原始 `spi_vol_arr`。
比较支持固定为独立的 **96×96 RZ 分区**，不是表自己的格点；积分为圆柱体积测度。

| 每最小源宽格数 | v400 概率相对 L2 | injInner 概率相对 L2 | v400 表内存 | injInner 表内存 |
|---|---:|---:|---:|---:|
| 8 | 2.841e-3 | 2.186e-3 | 0.338 MiB | 0.852 MiB |
| 32 | 1.784e-4 | 1.502e-4 | 5.30 MiB | 13.53 MiB |
| 64 | 4.463e-5 | 3.595e-5 | 21.14 MiB | 54.00 MiB |
| 128 | **1.116e-5** | **9.202e-6** | **84.42 MiB** | **215.70 MiB** |

128 格下 TV 为 `5.678e-6 / 2.790e-6`。这是固定分箱概率的确定性差异，
不是抽样噪声，也不是五通道投影误差上界。v400 略高于严格 `1e-5`：
两组达到约 `1e-5` 量级、低于 `1e-4`，但不能声称两组严格通过 `1e-5`。

v400 有不饱和上界证明，参考使用带 R Jacobian 的 Gaussian 解析积分；
real4 阈值舍入的归一化 TV 上界约 `5.96e-8`。
injInner 参考直接计算原生质量源、逐点 real4 和总源 clip 的复合 Gauss 积分，
不读取新表；RZ 阶数 32→64 的分布相对 L2 变化约 `7.99e-7`。
这不是严格积分误差上界；快照的宽角核 `ns_deltaphi=1000` 使用 8 点角向 Gauss，
与表的 64 点 midpoint 独立。96 分区比上一轮 24 分区揭示更多差异，旧数值不能直接横比。
未认证任意未来源宽、饱和状态或三维核都满足该误差。

## 成本不只是内存

最终运行单线程 BLAS/OMP/MKL，建表 wall time：

| 格数 | v400 | injInner |
|---|---:|---:|
| 8 | 0.022 s | 0.037 s |
| 32 | 0.240 s | 0.464 s |
| 64 | 0.894 s | 1.951 s |
| 128 | 3.547 s | 10.988 s |

这是一次事件快照诊断，不是完整链路加速比；后台投影核查同时运行，不是隔离机器稳定基准。
表内存包含坐标边、节点、概率和 CDF；临时空间基矩阵/角向乘法数组另占内存。
结果中的进程峰值 RSS 累计包含参考积分，不是单表内存。
每次源快照仍重新更新概率；不增加隐式跨事件缓存。n0 成本大致随分辨率平方增长，
未来 periodic_phi 还乘角向格数。抽样仍为 CDF 和闭式条件逆变换；
出生入口保留网格定位/边界重试，不宣称完全没有拒绝。

## 默认和兼容性

真实 `native_real_chain_builder` 的 n0 默认由 8 改为 **128**，最大 `16_000_000` 表单元，
超预算明确失败，不静默降精度。通用 `NativeTabulatedBirthSampler` 默认仍为 8，
避免泛用/三维调用意外大分配。periodic 模式单独配置，不宣称三维闭环认证。
smoke CLI 暴露 `--birth-cells-per-width`、`--birth-max-cells`。
若明确选择 64，两组本次误差仍在 `1e-4` 与 `1e-5` 之间，建表更便宜。
分辨率与 cell budget 进入既有配置指纹；原来 8 格检查点不可默认为 128 格恢复。

## Seed 和续跑

`tools/smoke_native_autonomous.py --seed N` 在整个运行开始时设置 Python/NumPy/Torch，
记录 seed 与初始/最终 RNG digest；非法 seed 在创建目录/读真实数据前拒绝。
事件不重新设置 seed：成功继续消费流，失败与检查点恢复沿用完整 RNG 状态契约。
seed 是运行复现参数，不是物理量，也不是逐步重新采样指令。
出生入口、位置 benchmark、分布比较暴露 `--seed`；ensemble 暴露 `--seeds` 和独立 offsets。
不同 draw 协议的同整数 seed 不构成逐 marker 配对。
细化工具使用独立 `default_rng(seed)`，不消费全局事件 RNG。
测试验证同 seed 完整 newborn 一致、不同 seed 改变位置、连续调用不重复流，
并保留失败 rollback 和跨事件恢复。不是原生 PCG32/MPI 位级重放，
也不承诺仅靠 seed 让所有 CUDA 算子位级确定。

## 复现与验证

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
.torch-cuda-env/bin/python -m tools.validate_native_birth_refinement \
  --output output/native-birth-refinement-2026-10-10/refinement-final.json \
  --resolutions 8 32 64 128 --bins 96 --target 1e-5 --seed 2917
```

需新的输出路径，不能覆盖证据。早期 `refinement.json`、`reference-tight.json` 保留；
当前完整 sweep 和源码哈希以 `refinement-final.json` 为准。
指定 `--target 1e-5` 不会强制通过：v400 的 `empirical_target_met=false` 正确保留。
本地两组回归分别 **42**、**59** 项通过，无真实训练；覆盖出生、独立参考、debug CLI、
真实 builder、投影配置、marker 窗口、检查点和回滚。
投影诊断独立测试另 **11** 项通过，合计 **112** 项（不重复计算早期测试运行）。
投影实测另见 [投影比较](native-birth-projection-comparison-2026-10-10.md)，
公式与误差归因另见 [一手证据](native-birth-projection-primary-evidence-2026-10-10.md)。
