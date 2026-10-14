# JOREK shard / marker 定位与 OLD 场采样：源码证据（2026-10-09）

## 结论与范围

本报告使用 research 技能，只读核查本地第一手源码；仅写本文件。未运行 JOREK/postproc、训练、GPU、性能基准或重型数值任务。已读取 AGENTS、CONTEXT、ADR 索引及相关现行 ADR-0019/0020/0021/0022；OLD 场顺序以 ADR-0020/0021/0022 的覆盖规则为准，不引用 v1 历史 ADR 作实现 gate。

关键结论：

1. **JOREK shard 不是 marker。** shard 先以旧速度生成新位置、用上一烧蚀率缩半径，然后全局 `find_RZ` 定位新位置并采 **OLD 场**，最后计算新烧蚀率及 rocket 对下一步速度的更新。这里的 Newton 是几何逆映射，不是 shard 隐式运动求解。[运动/定位][j-move]、[采场][j-sample]、[rocket][j-rocket]、[流体提交顺序][j-order]。
2. **postproc 没有发现更好的 shard 运行时定位算法。** `jorek2_postproc` 是文件导入及命令诊断程序；`shards` 调用 `pol_pos(R,Z)`，后者还是全局 `find_RZ`。已知 `(element,s,t)` 时它有免逆解入口，但移动后不能把旧 `(element,s,t)` 当成新位置答案。[导入][p-import]、[shards][p-shards]、[位置入口][p-pos]。
3. **可借鉴的已有局部算法来自 marker 的 `find_RZ_nearby`，不是 postproc。** 它使用上一单元/st，Newton 到边界后切邻接单元，失败才全局查找；与 shard 全局查找以及 Python 默认 locator 的容差、初值和候选顺序不同。它是候选算法证据，不是已证明更快或逐位等价的替换。[marker 调用][j-marker]、[nearby 实现][j-nearby]。
4. **静态可缓存的是拓扑、系数布局、几何系数、搜索树、固定查询网格的 basis。** 移动后的 st、Jacobian、basis 求值和新场系数/场样本仍需更新。当前 Python lookup 的静态缓存、同 buffer matmul 刷新及 `native_cubic_confirmed` 已有源码，但本次无真实性能或数值资格结论。[缓存/版本][py-lookup-cache]、[刷新][py-lookup-refresh]、[确认][py-confirm]。

## 1. 版本与可复查边界

核查的 JOREK 根目录为 `/home/ac_desktop/JOREK_code/jorek`，HEAD 为 `54c675bcc39c33b082c0c35046e28122dc53b3aa`（2026-05-19，`Refactor code structure for improved readability and maintainability`）。工作树有不相关的 `communication/eqdsk2jorek.f90` 修改和未跟踪 `poincare`；下表关键 Fortran 文件的 targeted `git status --short` 为空。本地 `jorek2_postproc` 是可执行文件，源码入口为 `diagnostics/postproc/jorek2_postproc.f90`，不是其下一个源码仓库；未证明该二进制与当前 checkout 完全对应，也未执行二进制。[程序入口][p-main]。

Python 仓库 HEAD 为 `0611ce473c632b5d256e4d93c719c3017f01a079`，工作树含主 agent 的未提交/未跟踪实现，因此 Python 证据以实际文件 SHA 为准。核查期间主 agent更新实验文件；本报告采用以下最后读取版本，而不把此前 g17/g33/g65 结果外推到新增分支。

| 文件（相对各自仓库根） | SHA-256 |
|---|---|
| JOREK `models/pellet_module.f90` | `6b9a9c78ccca3d504fcf2aae70792ea12d4b116738b474a36ffa7da3ef7c0e7a` |
| JOREK `grids/grid_utils/find_RZ.f90` | `95835d8183228a5540dcaa6ae1249d8e509163de1f462aaa1e1fe2f3e01578cf` |
| JOREK `grids/grid_utils/mod_find_rz_nearby.f90` | `b1cbbd3b8cbc3250a81132ac9c8f638561d60e832c0113fa13dd984f8999dcf4` |
| JOREK `elements/mod_interp.f90` | `f2a4b352bea5b82b377cc5085ed92023475221aea2a19b02a56d94d829f0a2e3` |
| JOREK `diagnostics/postproc/exec_commands.f90` | `6bef7d2e652a4ad864d540c452c0868153b192d432b408b61209feeb386c55ca` |
| JOREK `diagnostics/new_diag/mod_position.f90` | `7fae80ea4ae1768a0aa9d35d59f4119c04d887c1a40dfcf237409228de12ee18` |
| Python `src/models/native_shard_dynamics.py` | `cb3a40b0063cc1747f95ebdb3033982b236702026015df1baf8c2af8918572a5` |
| Python `src/models/native_source_production.py` | `bda2eb00fdaf1ec3a124ec4a181d7021160598bf7b01ff6430ae9cdfa98d6661` |
| Python `src/models/native_lookup_experiment.py` | `c2cbd13804a7c2e6aa02f0b1e1f3d478dde9e628aec9b8374b5a31499029dd03` |
| Python `src/models/native_coupling.py` | `f6dd533258fe83451c301888297a1154cf0b9d5fcdfb9c7ade23432d90bd551b` |

## 2. JOREK shard：候选新位置之后的实际顺序

### 2.1 不是根据速度直接查一个节点

`type_SPI` 持有 `R/Z/phi`、注入器坐标系速度 `Vel_R/Vel_Z/Vel_RxZ`、半径、烧蚀率、species、源体积、psi/grad_psi 及 drift 状态；该类型没有持久的 `i_elm/st` 字段。[`data_structure.f90:142–164`][j-type]。

`models/pellet_module.f90::update_spi` 的真实调用顺序：

- **:254–255、261–288**：从归一化计算 `V_normalisation/t_norm`；先取移动前的 `spi_delta_phi`，将注入器速度旋到局域柱坐标，用旧 R 计算角速度，显式更新 R、Z、phi，按配置加 toroidal rotation，再包裹 phi。
- **:290–316**：按材料组分计算密度；用上一 `spi_abl` 和更新前半径执行侵蚀，再截断到零。进入本步时半径已为零的 shard 仍先移动，随后跳过其余处理。
- **:325–339**：常数烧蚀模型 0 直接赋 `ns_amplitude`；模型 >=1 在**已经移动的位置**调用全局 `find_RZ`。域外 `ifail=99/999` 时烧蚀率置零并 `cycle`，因此本分支也跳过 rocket。
- **:341–355**：配置 drift 时，另在 `(R+drift_distance,Z)` 调用 `find_RZ`，决定沉积位置是否在域内。
- **:358–398**：在找到的单元/st、当前 shard phi 上调用 `interp_PRZ`；WITH_TiTe + impurities 路径同时取 `[rho,Te,psi,rhoimp]` 的值/局部导数和几何，计算物理 psi 梯度及 shard psi。
- **:425–480、682–734**：单位换算及负值截断；`use_marker` 时纯 Ne Parks3 分支使用字面量 `Z_imp=0`，得到 `ne = n - n_imp`，计算新烧蚀率；非局域分支依当前 OLD equilibrium/非局域表取量并 `max(local,nonlocal)`；可选磁场抑制最后作用到烧蚀率。
- **:772–889**：按 rocket 配置计算加速度，更新注入器坐标系速度。rocket6 使用 `1e5*(Te/2000)^1.26*(ne/1e20)^0.4` 压强、`max(radius,1e-5)` 质量半径下限和实际半径截面积；仍用**移动前**的 delta_phi 做坐标旋转。不是根据新加速度回算本步位置。

上述每步证据分别见 [移动/侵蚀/定位][j-move]、[场求值][j-sample]、[单位与非局域][j-units]、[Parks3][j-parks]、[rocket][j-rocket]。JOREK 支持的材料、烧蚀/rocket/drift/rotation 分支比当前 Python inspected-Ne 子集更广；此处不宣称全分支复刻。

### 2.2 为什么是 OLD 场

`core/mod_jorek_timestepping.f90:393` 先求流体增量；仅在 `step_success` 下，`:413–418` 准备/消费非局域依赖并调用 `update_spi`；`:424` 才 `update_values` 把解加到节点场，`:425–426` 更新原始 deltas 和时间。因此 shard 采的是**未提交新流体解的节点 values**，但空间位置已推进。[调用顺序][j-order]。

`interp_PRZ_1` 的 `deltas` 可选参数默认 false（`elements/mod_interp.f90:142–165`）；`update_spi:360–361` 没传该参数，读取 node `values`，不是 `values+deltas`，也不是旧/新场的时间插值。共享原生 basis 定义场值和导数（`:135、171–187`）。[插值器][j-interp]。

ADAS 读取是初始化工作：`core/mod_startup_teardown.f90:94–96` 调用 `init_imp_adas`；`models/mod_impurity.f90:48–49、66–67` 对 Ne 读 `96_ne` 并构建 coronal。Parks3 `use_marker` 分支置 `Z_imp=0`，不能把它说成每 shard 查 ADAS 全电荷态；非 marker 分支可对已加载的 `imp_cor` 插值。marker 程序也在启动 `marker_spi_source.f90:73` 调用初始化。[启动][j-startup]、[ADAS 初始化][j-adas]、[Parks3][j-parks]。

## 3. 几何定位：global、nearby、postproc 三者不能混称

| 路径 | 初值/候选 | Newton 与终止条件 | 使用对象 |
|---|---|---|---|
| JOREK `find_RZ` / `find_RZ_single` | 全局空间树包含点的单元，依返回顺序首个成功；每单元五个固定起点 | 每起点最多20步，显式2×2逆；L1 RZ残差 `<=1e-15` **或**未裁剪 Newton步的L1 `<=1e-8`；步分量裁到±.25，st裁到[0,1] | shard 运动后、postproc `pol_pos(R,Z)` |
| JOREK `find_RZ_nearby` | 上一单元/st，越边后邻接坐标变换 | 总计最多8轮，当前点几何/Jacobian显式求值；平方RZ残差 `<1e-24`；小det逆限幅；失败全局回退 | marker push 后 |
| Python `NativeSourceMesh.locate[_batch]` | Bernstein保守边界/中心树；scalar按中心距离排序、hint优先；batch先hint单元 | 最多25轮，L2 RZ残差 `<1e-10`；6个固定 damping，trial裁[0,1]，须改善残差且通过Jacobian/R>0 guard；未解决scalar回退 | inspected shard scalar、默认marker scalar/batch |

证据：[global][j-global]、[单元 Newton][j-single]、[nearby][j-nearby]、[Python locator][py-locate]。这些不同契约即使求的是同一个数学映射，也不是逐位等价算法。

### 3.1 全局 find_RZ 的细节

`grids/grid_utils/find_RZ.f90:19–21` 使用 rtree 平面 phi；`find_RZ_general:76` 直接把输入 `ielm_out` 清零，**并不把传入单元当 hint**。`:77–83` 由编译选项选择 no-tree、quadtree 或默认 RTree；`:86–90` 候选逐一尝试，首成功返回。`:134–150` 五初值按 `(.5,.5),(.75,.75),(.75,.25),(.25,.75),(.25,.25)` 顺序；`:155–219` 是几何 Newton。小步成功支路输出的是更新后 st，但 `R_out/Z_out` 仍来自更新前最近一次求值，不能把成功统称为已达到 `1e-15 m` 几何残差。[global][j-global]、[Newton][j-single]。

`elements/mod_interp.f90:1269–1316::interp_RZP_1` 每次取查询 st 的 basis，按 vertex/degree/toroidal 模式从几何系数生成 RZ/Jacobian。任意一般 cubic 单元没有这里可直接替代 Newton 的双线性二次逆公式。[几何求值][j-rzp]。

默认 RTree 的2D构建 `tools/mod_element_rtree.f90:147–176` 预计算元素范围并建树，查询 `:694–729` 只用于候选；`:724–728` 当前实现为计数/填数组两次 C 查询，但本报告不改 JOREK、不推断其成本。`grids/mod_neighbours.f90:241–246` 明确警告网格改变须重新建树，静态树不能隐式支持 moving mesh。[树构建/查询][j-tree]、[重建警告][j-neighbour-cache]。

### 3.2 nearby 确实用旧位置，但不是直接沿用旧定位

`mod_find_rz_nearby.f90:92–99` 从旧单元/st重新求几何（覆盖临时的旧RZ）；`:109–110` 做2×2显式 Newton；`:113–118` 以距离边界比例限步；`:123` 调 `coord_in_neighbour` 变换边界st；`:143/155` 在新查询st求几何；`:157–161` 检查平方残差。实现里没有注释提及的通用 backtracking 循环，也没有实际独立 `element_try_max` 控制，不能按头注释编造算法。[nearby][j-nearby]。

小Jacobian `abs(det)<1e-8` 时 `try_interp:208–211` 用 `sign(1e8,det)`，不是 Python 的正Jacobian拒绝契约。旧单元非法、特殊邻接情况、NaN或8步耗尽可调用全局定位；无邻接侧可标记 lost。边界方向/朝向的st转换见 `mod_neighbours.f90:140–199`。该路径在 `marker_spi_source.f90:655–661` 的 marker 速度求值及 Adams–Bashforth push **之后**调用，不是 shard `update_spi` 的实现。[nearby][j-nearby]、[邻接变换][j-neighbour]、[marker调用][j-marker]。

### 3.3 postproc：诊断采样，不是另一套 shard 动力学

`diagnostics/postproc/jorek2_postproc.f90:63–78` 读命令执行；`exec_commands.f90:327` 导入 restart；`:230–231` 分发 `shards`；`:1996–2004` 对已存 shard R/Z/phi求 `eval_expr` 并写出位置、速度、半径、已有烧蚀率及诊断量。本例程没有移动、侵蚀或 rocket 更新。[入口][p-main]、[导入][p-import]、[诊断输出][p-shards]。

其 `pol_pos(R,Z)` 经 `diagnostics/new_diag/mod_position.f90:230–238` 调同一个 `find_RZ`；若 caller 已合法拥有新位置对应的 `(ielm,s,t)`，`:240–248` 可以不逆解，`:429–434` 正向求几何并填element/nodes。这个“已知新st”的条件不同于“知道上一x及速度”：后者只能给候选物理位置，仍需几何逆解/确认。postproc 不提供这里更快且等价的运行时 shard locator。[位置入口与填充][p-pos]。

## 4. 与 Python 当前实现逐项对应

### 4.1 inspected shard 和生产组合器

`src/models/native_shard_dynamics.py:51–101` 完整校验 shard/九场；`:61–66` 从旧phi/R旋旧速度并移动；`:68–70` 用旧烧蚀率缩半径；`:71` 对每个候选位置单独 `mesh.locate`，**无element/local hint，也不写回持久st**；`:74` `mesh.point(...,old_fields)`；`:77–100` 做 inspected pure-Ne Parks3、非局域/磁抑制及 rocket6。仅纯Ne、无drift/rotation，SI时间/速度，NumPy停止梯度。[shard源码][py-shard]。

`native_source_production.py:42–46` 委托上述更新并提交shard时间；`:48–59` 在显式 postcommit diagnostic事件刷新源体积、非局域依赖和 OLD equilibrium。不能把这些随场/碎片变化的缓存当成永久几何缓存；source density、birth、marker feedback仍是不同离散对象。[组合器][py-production]。

外层 `marker_state.py:675–695::reference_fluid_event` 先保存完整OLD场，求候选新场，再给 `update_shards` OLD场，最后提交新场/原始history；这与上述JOREK调用顺序相符，而不证明 Fortran 数值逐位复现。[事件顺序][py-event]。

### 4.2 marker lookup 不会自动替换 shard

`native_lookup_experiment.py:76–77::__getattr__` 将没有定义的方法转发给原mesh；只重写 `locate_batch` 与 `_marker_points` 等实验方法。shard使用的 scalar `locate/point` 仍落原mesh；`make_lookup_source:315–318` 只浅拷贝marker source并更换该对象mesh、选batch，不修改外部production shard dynamics。若要实验 shard lookup，需要显式新接线/新资格验证，当前报告不做。[实验wrapper][py-wrapper]、[shard调用][py-shard]。

两种当前实验模式：

- `bilinear`：在原生cubic几何/场的固定g×g局部节点取样，微单元双线性逆映射用二次方程两根和近线性分支；先旧microcell hint，再全局最近表节点的4个microcell，未成功才native fallback。几何/value/gradient采用同一双线性表示；不等于cubic表示。[解析逆/搜索][py-inverse]、[采样][py-marker-lookup]。
- `native_cubic_confirmed`：仍以双线性表生成候选，但 `_confirm` 在候选原生单元/st作一次**native cubic Newton校正**，检查校正坐标有限且在[0,1]、正Jacobian/R及原生L2 RZ残差 `<1e-10 m`；拒绝后进入原native fallback。场表改为缓存完整元素cubic系数，现场以cubic basis/Jacobian采场及同源导数。没有“完全无Newton”或“只靠表”的资格。[确认][py-confirm]、[搜索接线][py-inverse]、[完整cubic采样][py-marker-lookup]。

confirmed分支虽确认原几何残差，仍改变默认Newton初值、路径、2×2算式及首成功候选；不保证边界元素ID选择、局部st及后续场值与原算法逐位相同。原生系数重新布局、matmul代替einsum也可改变浮点归约顺序。它应检验elementID/生存状态、native endpoint残差、场值/梯度和整窗口位移/速度/源误差；小残差不能独立证明动力学无误差。[默认轨迹][py-locate]、[实验求值][py-confirm]、[刷新算式][py-lookup-refresh]。

**本报告不执行或评估benchmark**；性能与数值对照结果见 [topology cache 实验报告](native-marker-topology-cache-2026-10-09.md)。新缓存不能沿用旧实验时间表作收益证明；本报告不判断matmul或confirmation的最终快慢，也不登记其它实验的运行状态。

## 5. 哪些能复用，哪些必须刷新

| 对象 | 合法复用范围 | 失效/风险 |
|---|---|---|
| vertex/nodeids/shared DOF索引、邻接side/orientation | 固定拓扑整个mesh绑定期 | 重网格/索引变化必须重建；排序变化可改变边界首成功元素 |
| element size、完整cubic坐标系数、保守边界/树、固定微网格的几何表 | 固定实际几何；3D还须遵守phi依赖 | 节点坐标/size改变必须重建；拓扑相同不代表几何不变 |
| 固定gauss/table查询st上的H/Hs/Ht | 基函数定义/degree及查询st固定 | 移动粒子的新st不是固定查询点，不能复用旧basis数值 |
| 当前粒子/候选在某st的RZ/Jacobian/basis | 同一次求值的多个消费者；或相同accepted trial下一轮 | 跨粒子移动/新st/geometry变更不可沿用；不能省trial几何合法性guard |
| 完整元素OLD场系数或场表、feedback表 | 对应不可变owned snapshot绑定期 | 任何新场snapshot/新time slot/feedback安装必须刷新；identity不能保护原位改写 |
| shard nonlocal/ES、source volume | 显式时钟/provenance和字段依赖绑定的diagnostic事件 | 不是mesh静态数据；禁止用未来场或其它event表替代 |

可复用结构的第一手依据为 [JOREK basis/系数][j-interp]、[geometry tree][j-tree]、[邻接重建警告][j-neighbour-cache]、[Python预计算][py-mesh-cache]、[实验owned缓存][py-lookup-cache]、[source诊断][py-production]。

当前实验实际缓存 `node_ids/local_sizes/basis_matrix/coordinate_coefficients`（`:47–54`），`_tabulate:148–168` 按chunk gather并matmul写入旧buffer；`_marker_points:265–278` 先撤销owner绑定，再刷新两份表，完成后安装新owner，owned输入要求readonly。buffer被原位覆盖，不是历史物理快照；不应由训练history/checkpoint持有其内容作过去场。geometrybytes SHA 在 `validate_fields:87–92` 拒绝实际几何变化，事件内并发几何修改不支持。full9场/feedback合法性不能因只消费部分channel而免检。[缓存/签名][py-lookup-cache]、[刷新][py-lookup-refresh]、[owner安装][py-marker-lookup]、[full-field guard][py-field-guard]。

内存范围亦需区分：本版本 `table_bytes:95–99` 统计指定ndarray，不统计树节点、owner物理状态、临时gather/BLAS或整进程；`estimated_bytes` 不是硬RSS峰值限制。`basis_matrix`若共享basis内存不能重复算峰值。readonly只是缓存协议，不是并发/外部别名篡改的隔离机制。[实际统计/创建][py-lookup-cache]。

## 6. “更快算法”可用性与最小风险结论

下列是代码支持的候选方向，而非性能排序或本轮实现任务：

1. **首先区分两个工作负载。** shard是每base事件每个活shard一次位置采样，当前scalar循环；marker是大量粒子乘多个子步，已有hint batch路径。不能用marker计时证明shard热点，也不能拿postproc批量诊断效率说明动力学效率。[shard loop][py-shard]、[marker locator][py-locate]。
2. **保持表示的低风险候选**：固定geometry/topology gather缓存；同点多场共享basis/geometry；不可变OLD场按事件准备element-local系数；同一accepted-trial几何复用。需要保留dtype/layout、完整输入校验及异常政策；“代数一样”不足以保证bitwise。[现有共享/geometry路径][py-field-guard]、[已有trial复用][py-locate]。
3. **有现成源码但改变搜索契约的候选**：给shard持久化element/st，并借鉴nearby邻接Newton；或批量native采场。只可在新版本显式实验，保持global fallback、边界/lost规则并做对照；原Fortran shard根本不持有这些hint，首成功候选可能改变。事先不能断言更快。[shard类型][j-type]、[nearby][j-nearby]、[Python batch][py-locate]。
4. **已实现的表示/初值实验分支**：双线性解析逆后native cubic确认+原生采场，避免把近似几何直接带入采场；其一次Newton可能成功，也可能增加确认与fallback成本。纯bilinear模式省其内部Newton但引入几何/梯度近似和边界错判；fallback只覆盖被拒绝的点，不能保护错误接受的近似点。本报告仅核查算法证据，不评估benchmark；结果见 [topology cache 实验报告](native-marker-topology-cache-2026-10-09.md)。[实验路径][py-confirm]、[解析逆][py-inverse]。
5. **不要把当前代码没有的算法写成已有能力**：任意cubic映射不能由旧x+v一步直接得到精确新st；冻结旧Jacobian、微网格nearest st直接接受、重新排Horner/FMA、缩减Newton容差/次数均不是默认轨迹精确保留优化。双线性公式只在双线性微单元内解析成立。若需要一般cubic高精度结果，仍需native求值与逆解/确认。[原生geometry][j-rzp]、[默认locator][py-locate]、[实验公式][py-inverse]。

总体判断：**postproc不提供这里更好的runtime shard算法；marker nearby 提供局部搜索的第一手候选；静态拓扑/几何与event-owned完整系数缓存可复用，但移动点的几何以及OLD场因果绑定必须保留。** 新experiment的速度与保真结果见 [topology cache 实验报告](native-marker-topology-cache-2026-10-09.md)；本报告不执行或评估benchmark、不更新性能表、不修改默认/JOREK/roadmap/ADR，源码核查本身不构成数值或性能资格背书。

## 精确源码链接

[j-type]: /home/ac_desktop/JOREK_code/jorek/datatypes/data_structure.f90:142
[j-move]: /home/ac_desktop/JOREK_code/jorek/models/pellet_module.f90:254
[j-sample]: /home/ac_desktop/JOREK_code/jorek/models/pellet_module.f90:358
[j-units]: /home/ac_desktop/JOREK_code/jorek/models/pellet_module.f90:425
[j-parks]: /home/ac_desktop/JOREK_code/jorek/models/pellet_module.f90:682
[j-rocket]: /home/ac_desktop/JOREK_code/jorek/models/pellet_module.f90:772
[j-order]: /home/ac_desktop/JOREK_code/jorek/core/mod_jorek_timestepping.f90:393
[j-interp]: /home/ac_desktop/JOREK_code/jorek/elements/mod_interp.f90:116
[j-rzp]: /home/ac_desktop/JOREK_code/jorek/elements/mod_interp.f90:1269
[j-global]: /home/ac_desktop/JOREK_code/jorek/grids/grid_utils/find_RZ.f90:46
[j-single]: /home/ac_desktop/JOREK_code/jorek/grids/grid_utils/find_RZ.f90:101
[j-nearby]: /home/ac_desktop/JOREK_code/jorek/grids/grid_utils/mod_find_rz_nearby.f90:41
[j-neighbour]: /home/ac_desktop/JOREK_code/jorek/grids/mod_neighbours.f90:140
[j-neighbour-cache]: /home/ac_desktop/JOREK_code/jorek/grids/mod_neighbours.f90:204
[j-tree]: /home/ac_desktop/JOREK_code/jorek/tools/mod_element_rtree.f90:104
[j-marker]: /home/ac_desktop/JOREK_code/jorek/particles/examples/marker_spi_source.f90:655
[j-startup]: /home/ac_desktop/JOREK_code/jorek/core/mod_startup_teardown.f90:94
[j-adas]: /home/ac_desktop/JOREK_code/jorek/models/mod_impurity.f90:7
[p-main]: /home/ac_desktop/JOREK_code/jorek/diagnostics/postproc/jorek2_postproc.f90:63
[p-import]: /home/ac_desktop/JOREK_code/jorek/diagnostics/postproc/exec_commands.f90:327
[p-shards]: /home/ac_desktop/JOREK_code/jorek/diagnostics/postproc/exec_commands.f90:1944
[p-pos]: /home/ac_desktop/JOREK_code/jorek/diagnostics/new_diag/mod_position.f90:200
[py-shard]: /home/ac_desktop/LNN-learning-for-JOREK/src/models/native_shard_dynamics.py:51
[py-production]: /home/ac_desktop/LNN-learning-for-JOREK/src/models/native_source_production.py:42
[py-event]: /home/ac_desktop/LNN-learning-for-JOREK/src/models/marker_state.py:654
[py-mesh-cache]: /home/ac_desktop/LNN-learning-for-JOREK/src/models/native_coupling.py:44
[py-field-guard]: /home/ac_desktop/LNN-learning-for-JOREK/src/models/native_coupling.py:134
[py-locate]: /home/ac_desktop/LNN-learning-for-JOREK/src/models/native_coupling.py:254
[py-lookup-cache]: /home/ac_desktop/LNN-learning-for-JOREK/src/models/native_lookup_experiment.py:27
[py-lookup-refresh]: /home/ac_desktop/LNN-learning-for-JOREK/src/models/native_lookup_experiment.py:148
[py-confirm]: /home/ac_desktop/LNN-learning-for-JOREK/src/models/native_lookup_experiment.py:129
[py-inverse]: /home/ac_desktop/LNN-learning-for-JOREK/src/models/native_lookup_experiment.py:183
[py-marker-lookup]: /home/ac_desktop/LNN-learning-for-JOREK/src/models/native_lookup_experiment.py:262
[py-wrapper]: /home/ac_desktop/LNN-learning-for-JOREK/src/models/native_lookup_experiment.py:309
