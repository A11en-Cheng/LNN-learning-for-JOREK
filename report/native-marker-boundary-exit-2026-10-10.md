# Marker 外边界末段与完整云复验（2026-10-10）

本轮范围：修复非默认局部细分模式在已知外边界处超限的问题，同时按实际末段时间计算电荷/沉积。**未训练、未更新 optimizer、未改原生默认模式、JOREK、原始数据或训练产物。** 这不是完整 JOREK 时间积分，也不是正式生产资格或预测精度验收。

使用 `diagnosing-bugs` 将完整云红灯缩到真实旧粒子的一次 base 更新；`codebase-design` 将拓扑/完整 cubic 求交核验放在独立 seam；`domain-modeling` 定向扩展 ADR-0023 并区分退出事件与定位失败。复用源码核查子代理补一手证据及独立解析测试；其指出的曲边切触重根、纯环向当前速度反例均由主代理修复并复验。

## 1. JOREK 事实与本轮近似

一手源码/行号/哈希见 [源码报告](native-marker-boundary-primary-evidence-2026-10-10.md)。原生顺序是：完整固定 dt 电荷更新及旧位置五通道沉积 → 完整 push、写 V_prev → nearby 定位 → 下一步跳过 lost。零邻居 lost 为负最后单元号及边界 st，**原生 x 保留 overshoot**；负邻居要求 global search，不等于外边界。

本轮独立 opt-in `local_subcycling_boundary_v2` 保存边界 x/st、负退出单元号与末段前间隔，按冻结变间隔方向轨迹的驻留时间共同更新全部粒子量。它**改变原生末段数值约定**，不能叫原生固定步复刻；旧 `native_fixed` 与 `local_subcycling_v1` 指纹/行为保留，不静默升级旧历史。原先筛选阈值、64最少细步、1024最大细步、原始梯度、密度 floor、速度及 base/gather 时钟均未改变。

## 2. 实现与失败语义

- [marker_boundary_exit.py](../src/models/marker_boundary_exit.py)：使用保存的邻居表；正邻居互反/端点验证，零为已记录外边、负为未知。缓存完整 cubic 边（非顶点直线），与冻结方向推进的 R/Z 二次轨迹求交，检查首次面、几何残差及前后两侧可分辨的 inside→outside 证据。
- 仅为退出核验在当前单元内精化较松的 locator hint；不修改采样 st/x、场数据或更换单元。轨迹已在单元外、反解失败、角点、切向、未知邻居、跨多单元退出均不能冒充确认出界；无证据的细分超限及普通定位失效继续报错，不提交候选。
- [marker_transport.py](../src/models/marker_transport.py) 与 [marker_state.py](../src/models/marker_state.py)：接受末段用同一个 h 更新电荷、OLD P 的辐射/三个矩、OLD 位置沉积、运动及方向历史；电离能按实际电荷差，不是缩放已做完整 dt 的电荷更新。统一投影分母仍是整个 `n×base_dt`。
- [native_coupling.py](../src/models/native_coupling.py)：显式绑定实际拓扑/几何退出 provider 与配置指纹。字段和 ADAS 公式保持原路径；源仍是 NumPy，未扩展梯度边界。
- builder、完整源复验及完整链耗时入口支持 `--marker-transport local_subcycling_boundary_v2`，默认仍 `native_fixed`。完整源复验新增逐臂 started/completed 日志、末段事件及执行计数，恢复标准仍逐位一致。

## 3. 可复核红灯、修复与误差范围

实际命令（环境变量必须进程启动前设置）：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
.torch-cuda-env/bin/python -m tools.diagnose_marker_subcycling_limit \
  --output output/native-marker-boundary-exit-2026-10-10/original-full-red.json

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
.torch-cuda-env/bin/python -m tools.validate_marker_boundary_capture \
  --capture output/native-nearby-motion-2026-10-10/capture-inner.pickle \
  --marker data/EHL-2/phaseA_dataset_1d22_ang15_massloss_neon_v400_injInner/part05680.h5 \
  --output output/native-marker-boundary-exit-2026-10-10/slot973-final.json
```

输出使用 exclusive 新文件；重跑须换文件名。capture 只允许可信本地 pickle，不接受外部输入。该小回放使用完整真实网格、场/rawD、ADAS及原始 slot973 初态，一个 base 区间、不含出生；**单粒子投影不是完整云投影**。

- 红灯再次复现：旧粒子973、i_life=1、element3377，χ=1615.96953082 要求2048细步、超过1024上限。不是新生粒子、未来场或密度 floor 命中。
- 最小化后旧入口实际推进约0.034s报同一异常。v2 两定位路径均在 `t=0.00411356040202463 s`、edge1（0-based s=1）确认退出；最后段 `3.1216897008e-12 s`，小于原拟接受 `4.9259442828e-12 s`，R=0.462647m，st=(1,0.16797512699)。未提高上限，未忽略该粒子。
- 原/nearby 的这个案例输出一致；电荷、原位沉积、末段历史、整个窗平均和后续不再贡献有独立手算测试。真实 Source 输入及原始文件保持不变。
- **保留未收敛项**：min128/max2048 的独立更细对照与 min64/max1024 的单粒子累计驻留相对差为约2.15%，五通道单粒子投影约2.61–2.85%；末段保存的 V_prev 增量相对差约92.4%。V_prev 是最后段首的采样速度，不是壁面瞬时速度；几何求交正确不保证边界奇异输运的临终速度/驻留时间已收敛。这不是统计噪声，不能用整个云的小范数掩盖。

所有相对 L2 按原生存储量 `||candidate-reference||₂/||reference||₂`，不另作单位转换、epsilon 分母或将 R/Z 米与 phi 弧度混为物理距离。完整云速度增量限共同存活、起始已存在且 lifetime 相同支撑，并减各臂自己的本窗起点。

## 4. 完整真实云与对照器修正

两组都是百万槽、完整云和实际独立出生；每组2个给定场源窗口、两条定位路径，各自恢复第二窗，合计6次 Source.advance。**不是神经网络真实自生成场/shard闭环**；真实字段是本轮 transport/source 条件诊断的控制。

中间证据完整保留，不覆盖：

- `inner-full-first.json`：退出核验遇到旧 global locator 较松 hint，明确拒绝；随后只为证书同-cell精化，并增加解析回归。
- `inner-full-hint-refined.json`：六次推进及两臂恢复均完成（524.73s，约2.86GiB）。两窗离散身份/权重/RNG/细分等级与 lifetime 均一致，投影相对 L2 上界4.30e-9，存活速度增量5.66e-7/2.75e-6；但整体标志 false。
- false 的唯一 mandatory 项为 `metadata`：比较器在 runtime 元数据中重复塞入整个 TransportHistory，导致已由数组指标报告的末段浮点 h_prev（最大差约4.93e-16s）又被要求 exact。修正只保留 history 的配置/导入声明为元数据；所有数组继续逐值报告，并新增 level/lifetime 的严格离散 gate。**同臂恢复仍要求完整状态/诊断/RNG逐位相等**。独立反例覆盖浮点时间、等级、身份和配置，未放宽真实离散契约。

最终代码加入曲边切触双根两侧证据及纯环向 current velocity 的二次方向历史路径后，独立完整云复验 **两组均 success=true**：

| 实例/源窗 | 共同持久存活粒子 | 本窗速度增量相对 L2 | 五通道 values 相对 L2 上界 | 五通道 deltas 相对 L2 上界 | 每臂确认退出数 |
|---|---:|---:|---:|---:|---:|
| v400 / 4111 | 11168 | 3.29e-10 | 1.30e-9 | 1.30e-9 | 0 |
| v400 / 4112 | 11197 | 1.44e-9 | 2.59e-9 | 2.00e-8 | 0 |
| injInner / 5681 | 52602 | 5.66e-7 | 4.29e-9 | 4.29e-9 | 2 |
| injInner / 5682 | 52657 | 2.75e-6 | 3.52e-9 | 3.62e-8 | 1 |

表中两臂 **都使用 v2 输运**，只比较旧 Python 全局定位与 nearby；不是 v2 对 JOREK 原生固定步真值。完整机器报告保留旧 debug `x` 数组统计（含 R/Z/phi，不作为物理距离误差）；本表仅使用同一储存单位的速度分量，单粒子回放另列 R/Z。第一窗历史为登记零投影，第二窗 deltas 则是新投影减各臂自身持久前投影。

- [v400 最终机器报告](../output/native-marker-boundary-exit-2026-10-10/v400-full-final.json)：百万槽，初活11168，独立实际出生29/2；六次 Source.advance 完成、两臂各自第二窗恢复完全 exact，36.83s、RSS约2.81GiB；没有触发细分，不能用它证明 stiff 分支覆盖。
- [injInner 最终机器报告](../output/native-marker-boundary-exit-2026-10-10/inner-full-final.json)：百万槽，初活52604，独立实际出生56/11；六次 Source.advance 完成、两臂各自第二窗恢复完全 exact，526.25s、RSS约2.86GiB。身份/元素号/lifetime/原子权重/出生时间/等级/RNG/事件时钟/统一 gather 的严格条件均通过，未放宽 gate、裁剪云、共享对照臂未来出生或补零。
- injInner 首窗确认旧身份973（单元3377）及701199（单元3376）退出，次窗801014（单元3375）退出；都在 s=1 零邻居边，不以初始触发粒子编号白名单判定。完整事件保留实际 h、拟定 h、位置、指标及所需 level，超上限无证据依旧失败。
- injInner 每臂首窗 113314 个 callback组、4376204 个接受粒子步，其中111208为细分/末段；次窗23235组、865446粒子步，其中22819细分/末段。首窗旧/nearby源推进分别约174.07/176.69s，次窗约35.72/35.58s；这些是单次诊断成本（含源采样、ADAS、电荷、沉积、定位、投影，不含整段模型训练），**不是正式配对 benchmark，也没有显示细分模式下 nearby 的速度优势**。整体526s还含数据/网格准备、出生、完整比较、序列化及恢复。
- 两组最终代码/原始文件/调用初态/其他输入哈希与不可变检查均通过；最终末段 provider 指纹为 `saved-cubic-current-cell-exit-v2_side_witness`。

完整真实命令：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
.torch-cuda-env/bin/python -m tools.validate_native_nearby_history \
  --run-index 0 --marker-transport local_subcycling_boundary_v2 \
  --output output/native-marker-boundary-exit-2026-10-10/v400-full-final.json

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
.torch-cuda-env/bin/python -m tools.validate_native_nearby_history \
  --run-index 1 --marker-transport local_subcycling_boundary_v2 \
  --output output/native-marker-boundary-exit-2026-10-10/inner-full-final.json
```

实际回归：27个相关测试文件共 **341 passed**（25.38s；1个已有 Torch profiler warning），其中独立几何41项、退出/权重5项；`tests/test_native_real_chain.py` 独立 **5 passed**（49.47s），合计346项。`test_marker_local_subcycling.py` 的真实 Torch 双事件/跨步反传/检查点恢复同时覆盖旧 v1 与新增 v2；不把 NumPy 源分支说成可微。`compileall` 与 `git diff --check` 通过。复核可沿用前篇完整回归清单，再加入 `tests/test_marker_boundary_geometry.py` 与 `tests/test_marker_boundary_exit.py`；比较器历史反例位于原 `tests/test_benchmark_native_neighbor_locator.py`。

## 5. 后续边界

本轮 numerical provider 仅核验当前单元的首次外边，不承诺多单元大跳的完整几何搜索或所有近切触的可靠求根。未把该模式切成训练默认，未补正式九场训练精度或端到端可微资格。下一步需要分别评估局部细分开销、末段时间收敛及预期误差预算，再决定是否纳入训练配置；不能由定位 A/B 小误差推出 JOREK 真值误差或训练加速。
