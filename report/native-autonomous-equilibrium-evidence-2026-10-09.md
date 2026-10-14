# 当前预测 ψ 的自主 equilibrium 诊断（2026-10-09）

## 交付与范围

新增 `src/models/native_equilibrium.py`、`tests/test_native_equilibrium.py` 及本报告。
采用 codebase-design 的单 callable interface 与 TDD 公共 provider seam；未改公共事件、builder、训练、源或数据模块。
没有完整 source/cloud 运行，没有训练或 optimizer 更新。本报告不是双 run 完整事件 smoke 的结果。

`NativeEquilibriumProvider(fields,time_si)` 每次从当前完整九场的原生 cubic ψ 重算轴和两个 X 点，返回与整份 fields 的 digest、当前 time 绑定的 `NativeFluxEquilibrium`。
其 configuration 包含几何、边界、搜索范围、数值策略及显式 restart-history，适配现有公共诊断刷新与 checkpoint fingerprint。
不保存初态 ψ，不接收未来记录 reader，不替换网络九场，不缓存前一次 ES，不积累隐藏状态。
NumPy 诊断仍是 stop-gradient 路径，不声称源—场端到端可微。

## 可直接接入的初始化接口

```python
from src.models.native_equilibrium import NativeEquilibriumProvider
provider = NativeEquilibriumProvider.from_initial_record(mesh, initial)
equilibrium = provider(predicted_fields, time_si)
```

`initial` 是只读 reader 从唯一初态帧得到的 record，显式指定原 restart index。
工厂只读 `initial.source_path`：校验整个文件 SHA256、index/time、freeboundary，读取 `R_axis_t[-1]`、`Z_axis_t[-1]` 作固定搜索中心。
不读取 `initial.namelist_path`、目录相邻帧或未来路径。
HDF 没有保存 `axis_srch_radius`、`Z_xpoint_limit`、`R_geo`，因此不能假装从 HDF 推导这些参数。
工厂按 **已核验生产 namelist SHA256** 选择显式 profile；未知 SHA256 拒绝，调用者可改用显式 kwargs 构造器（需要自行提供有证据的配置）。

两 target 的 `jorek_namelist_spitest` 一致：130 行 `Z_xpoint_limit=-1,1`；131 行 `xcase=3`；133 行 `freeboundary=.f.`；136 行 `R_geo=1.11`；138 行 `axis_srch_radius=.5`。

|target|namelist SHA256|
|---|---|
|v400|`0fca2b6a5d0cf94e92eb86a4ef11d961fba50a28d06520708e19dfec3d061811`|
|injInner|`bf3de403f17ab713d54f51883eee72699fa9ff3e5dd0549588bfffbd6c6e2c0e`|

等价直接构造为：

```python
provider = NativeEquilibriumProvider(
    mesh, initial.boundary,
    axis_seed=(R_axis_initial, Z_axis_initial),
    axis_search_radius=.5, z_xpoint_limits=(-1., 1.), R_geo=1.11,
    restart_lcfs_lost=initial.equilibrium.lcfs_lost,
    restart_history_provenance=initial.provenance,
    xcase='double', freeboundary=False,
)
```

v400/4111 seed=`(1.2062712395399406, 0.0005574908591449704)`；injInner/5682 seed=`(1.2013364019362007, 0.06935418552994077)`。
工厂额外把初态 SHA256、namelist SHA256、初态 index/time 和 LCFS history length 纳入 configuration。
初态记录只用于构造；后续调用不再打开任何文件。

## 一手源码：实际输入与调用顺序

只读源码根：`/home/ac_desktop/JOREK_code/jorek`，HEAD=`54c675bcc39c33b082c0c35046e28122dc53b3aa`。
该 checkout 有既有 `communication/eqdsk2jorek.f90` 修改及未跟踪 `poincare`；没有修改它们，也不把此 checkout 等价于生产 dirty 二进制。

|源码|SHA256|
|---|---|
|diagnostics/find_axis.f90|`8c9fda30985d2d56408ca033a70b8cdd0615160fcd2527b14297eeaf4b67aa29`|
|diagnostics/find_limiter.f90|`7c5badd8b500abe02455827f1de07dc83d220ca534d9811d47251827763a2a0e`|
|models/equil_info.f90|`02a7cb2c94f94a92e60b1acbe1aba44f0ce1d8b7ec37ef63251bd588c2e3e6b9`|
|core/mod_jorek_timestepping.f90|`f45b62fff5165f2c0f728c7a24d14b919628ff455ed4d1497daafaecceecb5fe`|
|solvers/mnewtax.f90|`515141a0963ffee5e36c69aed4a88d2a6fd1478c0d18a4014af94022ce912fff`|

- `core/mod_jorek_timestepping.f90:190–191`：初始化先 update_equil_state 再复制到 stepper ES。`:353–354` 在流体步开始重算；`:446–447` 更新后在 rank0 再诊断。新 provider 是统一当前场诊断，不模拟历史 rank-stale ES。
- `models/equil_info.f90:177–202`：先 LCFS 历史检查，后当前场轴、轴极性、X 点、limiter。
- `diagnostics/find_axis.f90:63–96`：搜索半径及中心取值；正常续跑优先前一次 ES，restart 可取轴历史。`:102–189` 全部 cubic degree 的 Gauss 物理梯度排名及单元 Newton；`:200–209` 失败可用首次近似点。本实现不采用这个失败 fallback。
- `models/equil_info.f90:446–740`：真实 find_xpoint 在该模块内，而非单独 diagnostics/find_xpoint.f90。`:551–565` 按 Z limits、轴附近区域及装置分支筛选；`:482–484` 定义 `.015 R_geo` 与 4 倍距离；`:685–736` 允许失败近似点并判断 far-axis。
- `models/equil_info.f90:205–260`：两个 far-axis X 点均存在时选 `abs(psi_axis-psi_xpoint)` 较小者（相等取 upper）；只有 freeboundary 才比较 limiter。目标 freeboundary=false，因此本支持分支无需 limiter 磁通。
- `diagnostics/find_limiter.f90:86–145,168–213`：边界 cubic 一维驻点和额外 limiter 点；排除 private 区。**没有在本模块移植**；缺 X 点或 freeboundary 情形拒绝，而不是用固定边界值代替。
- `solvers/mnewtax.f90:18–20,31–77`：Fortran 为局部坐标 Newton、20 次、步长截断和局部区间约束，采用不同终止条件。Python root 并非该迭代逐位移植。

### LCFS 历史是否需要新增运行状态

`models/equil_info.f90:1265–1308` 每次先令 false；仅在 `index_start != 0` 且轴历史存在时，扫描 `i_time=1..index_start`，若轴到边界 **元素中心** 的最近距离 `< R_geo/30` 则 true。
`communication/mod_import_restart.f90:1542` 从保存的 index_now 读入 index_start；普通步只递增 index_now（`core/mod_jorek_timestepping.f90:334`）。
所以固定 restart epoch 内这是固定历史前缀的函数，不是当前 ψ 的拓扑检测，也不是每次追加新轴的动态 hysteresis。
`initial.equilibrium.lcfs_lost` 与 `initial.provenance` 明确持有此前缀摘要；provider 没有隐藏状态，无需扩展现有 callable。
新 restart epoch 必须重建/重新绑定 provider，不得将旧摘要默默移植。
当前场若轴/X 点消失，直接失败，即使固定历史 LCFS=false；不宣称能处理失去闭合磁面的演化。

## 精确复用与近似/范围限制

复用完整 cubic 原生系数、同一 `NativeSourceMesh.point` 的值/物理梯度和固定几何；不使用节点 argmax 或低阶插值。
目标 double-X、fixed-boundary 的磁通选择公式与已验证 stationary 分支一致；时刻和全 fields digest 精确绑定。

以下差异进入 configuration 或本报告，不能称完整生产 ES 等价：

1. 固定初态轴坐标作搜索中心，不读取前一步 ES；预测轴移出半径明确失败。源码正常续跑可移动中心。
2. Gauss 排名后用 SciPy root 解物理梯度，单根最多 80 function evaluations，每分支最多 500 候选元素；每元素只从最优 Gauss seed 尝试。不是完整拓扑遍历。
3. Hessian 分类用局部坐标中心差分（步长 `1e-5`）再变换到物理空间；轴要求同号特征值，鞍点要求异号。Fortran 没有这一显式分类。物理梯度绝对容差 `1e-9`，不逐位复制 Fortran Newton 容差。
4. 单环向模式、double X、两个合法且远离轴的鞍点、非退化磁通间隔为支持域；其他模式、freeboundary、单 null、limiter 转换、搜索失败或间隔 `<1e-6` 均拒绝。没有真值/frozen ES fallback。
5. 统一当前场 ES 不重建生产 MPI rank 缓存时序；不解决前轮 nonlocal 历史 rank-lag 差异。
6. 本模块只是诊断，不证明 learned boundary 为硬物理算子，也不改变 source diffusion、birth 或 density 语义。

## TDD 与实际测试

公共 seam：构造/工厂及 `provider(fields,time)`，没有新增内部测试 seam。

- 首个解析场测试先红（模块缺失），实现后绿。独立 oracle：分片 cubic `psi=(R-2)^2+Z^2-(2/3)|Z|^3`，轴 `(2,0)`/ψ=0，鞍点 `(2,±1)`/ψ=1/3。
- 扰动加 `.1 Z^2+.25`：期望轴 ψ=.25、鞍点 Z=±1.1、ψ_bnd=.25+1.331/3。测试禁用 builtins.open/h5py.File，证实调用不读真值文件；回到原输入得到原输出。
- 几何更改检查先红（未拒绝），加绑定校验后绿。
- 负 ψ 的最大值轴、显式 LCFS=true、重建 provider 与配置指纹、configuration 副本不改变 provider。
- 缺轴/缺鞍点、非有限数据、非法谐波与时间失败；之后原输入可继续重算，无失败缓存污染。
- 工厂先红（缺 classmethod），实现后绿；只允许 named initial HDF，给不存在的 namelist 路径也不读取；未知 namelist SHA256 拒绝。

实际执行：`.torch-cuda-env/bin/python -m pytest -q tests/test_native_equilibrium.py` → **14 passed in 1.02s**。
`git diff --check` 通过。未运行任何含 optimizer 的测试。

## 真实初始化对照（只读、无 source 推进）

|项目|v400 jorek004111|injInner jorek005682|
|---|---:|---:|
|固定 restart index|4110|5681|
|time_si|0.003514583849363869|0.004114048790515563|
|计算 ψ_axis|0.504467466127755|0.5037508301993341|
|计算 ψ_bnd|0.11326831321273491|0.11256607157659754|
|ψ_axis 有符号误差|0|-1.1102230246251565e-16|
|ψ_bnd 有符号误差|-2.7755575615628914e-17|1.3877787807814457e-17|
|X 点 Z 最大绝对误差/m|6.661338147750939e-16|1.5543122344752192e-15|
|LCFS 历史最小距离/m|0.7424498055841912|0.7358615545252232|
|LCFS_lost|false|false|
|provider 调用/s（不含 mesh 建立）|0.012774114962667227|0.01021370803937316|

v400 HDF SHA256=`03e067b742af10048912291b41f8957eedd3037cd1267ae25941f2cae908cc54`；
injInner HDF SHA256=`29e4c447690360a98965a402ca49d590a938563d52cc33769a3dbd064e82f335`。
共同 geometry digest=`90504d17ef927ff4c1a98355858b21ee94fa8cbfe3e72d4d6f166078d143ba39`。

工厂入口也已实际复跑两份初态：v400 factory/call=`0.0948384642 / 0.0144831929 s`，injInner=`0.1058739130 / 0.0105283768 s`，上述 ψ_axis/ψ_bnd 误差保持不变。
工厂 configuration digest 分别为 `6dabe6511ffbd4c450f3783cd2fac3ea234c4b15261a1d12fd2dc31326a9ec7d` 与 `b933d1874f90c15814e113795a6a237d17797ee42a561ea33ea90dd1414168f1`。

另对各自初态 ψ 做 `1.001*psi+1e-4`（仅 dof0 加常数），provider 重算而非使用记录：
v400 ψ_axis/ψ_bnd=`0.5050719335938827 / 0.11348158152594766`；
injInner=`0.5043545810295332 / 0.11277863764817411`。此仿射扰动不改变驻点位置，位置响应由前述解析 Z² 扰动独立覆盖。

复验方式：分别 `read_native_psi_teacher(initial_path,lcfs_restart_index=anchor)`，以 record 的 geometry/units 构建 `NativeSourceMesh(...,projection='none')`，调用上述工厂，只从同一 initial_path 读 values，比较 `provider(values,initial.time_si)` 与 `initial.equilibrium`。
不用任何未来文件。完整事件、恢复、双 run autonomous smoke 与后续训练由主线程集成；本 agent 没有执行，不能把这些单帧证据冒充闭环验收。
