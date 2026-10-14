# 原生元素系数缓存：完整源窗口保真与收益（2026-10-10）

## 结论

**隔离候选已实现，两个真实 phaseA 首个完整 marker 源窗口均通过保真对照；
收益小，暂不接默认后端。** 仅缓存事件内九场/前投影的原生元素系数，没有使用
bilinear 表或近似定位。v400/injInner 的 source 中位耗时观察到减少 2.23%/0.49%，
采样子项减少 3.49%/4.18%。三次配对、两个固定窗口不构成统计显著性或训练吞吐证明。

本轮未训练、未更新 optimizer、未改变九场网络或默认 `native_coupling.py`，
未修改 JOREK、原始数据、roadmap、ADR 或已有用户修改。未进行完整自主耦合链、
真实多事件/BPTT 或训练恢复性能验收，不将 source 恢复测试称为 CfC 恢复认证。

本轮使用 `diagnosing-bugs` 的性能分支：沿用公开源事件反馈循环，固定输入/RNG、
先测基线再做隔离 A/B。未报告新功能缺陷，故不强行做功能故障二分或修改物理公式。
事先提出“重复 gather 成本”“布局改变引入误差”“定位占比压低整体收益”三个
假设。独立子代理只读审查；重测试/性能作业串行，未保证整机其它任务独占。

## 实现与不变部分

- [隔离模块](../src/models/native_coefficient_cache_experiment.py)：
  `NativeCoefficientCachedSource`，只包装原生 CPU `batched_v1` source。
  `configuration` 明确记录实验缓存及 `production_qualified=false`，未接训练入口。
- 每次 `advance` 新建事件缓存，保存只读 node IDs；仅接收原生 source 在事件内
  创建的不可写场/反馈快照。同一静态快照构建一次；动态采样换 snapshot 后重建。
  owner 强引用避免 ndarray 身份复用；不跨事件按用户输入数组身份缓存旧内容。
- 将全元素系数按原生 gather 的 **E,V,F,D 及其 F-input 内层布局**保存，再按粒子
  element 选择，转回求值所需的 E,F,V,D 视图。没有将 E,F,V,D C-pack，也不预乘
  signed sizes、不改变 `einsum` 的乘加顺序，不裁剪完整九场或五通道状态。
- 几何仍调用原生 `_points_geometry`。`locate`/`locate_batch`、接受试探点复用、
  Newton 25 次、六级阻尼、原生容差/fallback、ADAS、电荷步、沉积、投影及历史
  全部沿用默认源实现。代理将定位绑定到原生 mesh，不导致其 type guard 禁用。
- 同一 wrapper 拒绝并发 `advance`；失败与成功都释放事件缓存/锁，只保留小型
  统计字典。缓存是可重建工作空间，不是 marker 物理状态，不写进状态检查点。
  原生 mesh 的并发修改、私有方法覆写或动态改换 source backend 不在诊断契约内。
- 单独系数预算默认 64 MiB，动态换表时检查旧/新系数同时存活的峰值。
  snapshot、粒子 gather 临时数组及整体模型内存由进程预算约束，不把系数预算
  误称整个事件内存预算。新增了相应失败不提交的公开事件测试。

## 基准契约与结果

工具：[benchmark_native_coefficient_cache.py](../tools/benchmark_native_coefficient_cache.py)。
CPU 为 Ryzen 7 9800X3D，WSL/Hyper-V；NumPy 2.5.2，Torch CPU threads=1。
进程启动前 BLAS/OMP/MKL/NUMEXPR threads=1；只读查询已加载 NumPy/SciPy 两个
OpenBLAS 库，实际 threads 均为 1。本轮无 worker 池，也不叠加上轮双线程采样。

| run | 注册初态/启动场 | 初始活粒子 | 共享出生 | 子步 | 末活粒子 |
|---|---|---:|---:|---:|---:|
| v400 | 4110 / 4111 | 11168 | 29 | 40 | 11197 |
| injInner | 5680 / 5681 | 52604 | 56 | 81 | 52658 |

两组均保留 1000000 槽位、3986 元素、4125 节点、全部 H=1 九场四 degree、rawD、
五通道反馈/历史和 marker 持久状态。启动初态/零投影按登记事件读取；真实字段
只作为固定初态源采样输入，不执行预测场或未来真实场回授。出生只计算一次，
候选共享相同出生 realization，并恢复相同 post-birth Python/NumPy/Torch RNG。

outer chunk 固定 8192，不能与旧 chunk=2048/未控 BLAS 的计时直接横比。每模式
各一完整窗口预热，再三次配对，第二轮反序。source 计时包括候选拓扑复制、
系数构建、采样、物理子步及投影；输入克隆、sampler 创建、RNG 恢复、输出
哈希/检查在计时外。baseline 只有采样计时代理，仍调用原生采样和定位。

| run | baseline source | cached source | 耗时减少 | baseline 采样 | cached 采样 |
|---|---:|---:|---:|---:|---:|
| v400 | 5.900852 s | 5.769262 s | 2.23% | 0.748709 s | 0.722612 s |
| injInner | 39.914602 s | 39.717871 s | 0.49% | 7.228790 s | 6.926656 s |

以上均三次中位数；采样/build/particle gather 是嵌套子项，不能与 source 再相加。
v400 整体下降约 0.132 s，而采样下降约 0.026 s，说明其它阶段波动也参与；
不能把全部整体下降归因于缓存。injInner 的差距约 0.197 s，收益小且未建立跨窗口
统计结论。没有自动选择/启用最快模式。

每个静态窗口只构建 field/feedback 各一次。两组实际系数缓冲共 **7142912 bytes
（6.81 MiB）**，node IDs **63776 bytes**；不是数 GiB 的密集空间表。
v400/inner 的窗口 sample calls 为 80/567；warm cache 构建约 0.6–1.9 ms（不含
首次预热）；仍需逐粒子 gather 和原生几何计算，未消除位置反求。
整作业分别 72.29/329.68 s，峰值 RSS 约 1.55/1.55 GiB，包含初始化、完整串行
reference 和候选状态，**不等于缓存内存增量或训练显存**。

预算为 8 GiB/600 s；SIGALRM 中断主线程，底层 native 调用可能延迟响应，不是
独立进程硬 kill。两作业实际均在预算内完成，无额外搜索、粒子裁剪或改子步。

## 保真与测试

两组各八次 source 运行（含两个预热），完整数组 shape/dtype/bytes SHA256、
cloud group、前投影 values/deltas/历史来源、metadata/时钟、末 RNG 都与原生
reference 一致。物理 energy/rad/moments、n_steps、processed count、sampling
诊断也一致；只排除 `execution.elapsed_seconds`/`peak_rss_bytes` 两个资源计数。
初始 fields/rawD/shards/births/marker 输入指纹未变。连续分量有定义的相对误差
为 0，零参考处也无变化；不做单位反换算或 epsilon 放宽。

机器证据：

- [v400.json](../output/native-coefficient-cache-2026-10-10/v400.json)：`success=true`。
- [inner.json](../output/native-coefficient-cache-2026-10-10/inner.json)：`success=true`。

新增 [公开事件测试](../tests/test_native_coefficient_cache.py) **22 项**：静态/动态
rawD、endpoint reference、跨元素/丢失、C/F/负 stride、float32/64、多元素随机
完整 cubic 与混合电荷/非零反馈历史、同数组内容变化、失败/预算不提交及重试。
两事件恢复测试独立执行 native 两事件链，并比较缓存连续运行与物理 runtime
序列化后 fresh wrapper 的完整结果及稳定诊断；不是序列化 callback 或缓存锁。
独立审查发现最初随机用例只覆盖单元素、恢复用例不够独立，已补强后运行。

相关回归 **91 passed in 1.67 s**；新模块/工具/测试编译及 `git diff --check` 通过。
实现与两个实测绑定 SHA256：

- 原生 source 未变：`f6dd533258fe83451c301888297a1154cf0b9d5fcdfb9c7ade23432d90bd551b`。
- 缓存模块：`42cd69fae81b5a2932b4019f2c16e3721a907d32a156b6cd692db94ff9780480`。
- 基准工具：`3f91d1b9cd19c29a41be0029d4416b950f35394831584dfe8bdc195ebba53a9a`。

## 下一步

暂不为 0.49% 的 injInner source 收益增加默认路径和检查点复杂度。保持当前已验收
后端，保留本候选供复核。若继续做性能切片，优先评估**保持默认候选/阻尼/容差/
fallback 的编译批量定位内核**；不能再通过放松定位或跳过物理子步获取“加速”。
本轮未实现该内核。正式切换任何候选仍需完整自主耦合/BPTT与恢复对照、实际
训练吞吐测量，而不是据两个首 source 窗口直接切训练。

复现：使用新的输出路径，两个作业依次运行，不并发计时与重测试。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  .torch-cuda-env/bin/python -m tools.benchmark_native_coefficient_cache \
  --run-index 0 --repeats 3 --output output/native-coefficient-cache-recheck/v400.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  .torch-cuda-env/bin/python -m tools.benchmark_native_coefficient_cache \
  --run-index 1 --repeats 3 --output output/native-coefficient-cache-recheck/inner.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .torch-cuda-env/bin/python -m pytest -q tests/test_native_coefficient_cache.py \
  tests/test_native_marker_sampling.py tests/test_native_locator_events.py \
  tests/test_native_atomic_interpolation.py tests/test_native_cpu_threads.py \
  tests/test_native_lookup_experiment.py
```
