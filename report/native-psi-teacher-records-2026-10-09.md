# 原始记录 ψ teacher：reader、全846 shard及出生探针（2026-10-09，B）

本轮无训练、optimizer、搜索、JOREK/原始数据写入或资格授予。仅新增独占 data reader、测试、审计工具和本报告；不改协调文件、A模型或主线程teacher wrapper。codebase-design使复杂的文件/时序/LCFS来源集中在一个reader；tdd通过公开reader逐个红绿，而非测试私有插值助手。当前原始初态九场用于单步机制对照，**不是多步预测的八场真值重置**。

## 最小接口与证据语义

`src.data.native_psi_teacher.read_native_psi_teacher(path, expected_time_si=None, namelist_path=None, gamma=5/3, lcfs_restart_index=None)` 返回冻结 `NativePsiTeacherRecord`：

- `.psi [4,1,N]` 全cubic，保留dtype/每个degree；bytes-backed ndarray，不仅writeable=False，还不能重新setflags(write=True)。
- `.equilibrium` 是保存的ES数值，**仅绑定原始完整九场digest**；不是预测九场ES。主线程混合diagnostic/shard支路必须显式重新绑定并标teacher，不能误称原预测U诊断。
- `.time_si/.geometry_digest/.units/.provenance/.source_sha256`；额外有原始field digest、record index、完整几何/边界及namelist路径/hash。geometry digest与NativeSourceMesh相同键/轴约定。
- 默认只读同目录明确的 `jorek_namelist_spitest`，不猜generic namelist，不读下一输出。HDF `xtime[-1]==t_now`、全部ES历史长度匹配index_now；expected_time_si只允许浮点舍入量，不自动邻帧/插值。
- 原始历史中真实restart存在重复时间（v4002701/2702，inner2703/2704），4110/4111另有约3.3e−11 JU反向舍入。允许64 ULP历史舍入，不容许事件级倒序；这不放宽nonlocal模型的严格feedback未来检查。
- `gamma`为显式调用者单位约定，默认5/3；不声称缺失的HDF gamma或舍入日志精确记录了该值。

LCFS为 **restart-history reconstruction**，默认把该文件作为重启初态。连续teacher节点应传固定原始 `lcfs_restart_index`；不能隐式模拟源码index_start随每个输出更新。一般true结果不认证连续run的真实LCFS flag。两目标全部历史均false，因而对任意真实restart前缀也false，此处连续节点使用该证据安全。

## 一手源码及原始字段

JOREK只读HEAD `54c675bcc39c33b082c0c35046e28122dc53b3aa`，源码机制不等价于生产dirty构建认证，也不因dirty单独否定可实现性。

|来源|含义|
|---|---|
|`communication/mod_import_restart.f90:1542–1544`|保存index_now成为固定index_start；t_now成为t_start|
|`grids/mod_boundary.f90:49–72`|元素相邻两个vertex boundary>0构建边界元素列表|
|`models/equil_info.f90:1280–1302`|历史1:index_start磁轴到边界**元素中心(s=t=.5)**距离<R_geo/30即LCFS丢失；非边缘中点|
|`models/equil_info.f90:206–296,982–1024`|有效X点选择边界ψ、private区域修正、LCFS丢失/退化磁通行为|
|`core/mod_jorek_timestepping.f90:446–469`|新U提交后刷新ES并写axis/bnd/xpoint历史|
|`models/pellet_module.f90:270–315,464–480,699–734,864–887`|旧速度移动→旧rate侵蚀→新位置OLD场→local/nonlocal max→磁减弱→局部rocket6|
|`particles/examples/marker_spi_source.f90:394–437`|出生目标包权重、计数、严格atoms>1e14的至少1包阈值|

两个目标 `jorek_namelist_spitest:129–148` 指定xpoint=true、xcase=3(double)、R_geo=1.11；不是由未来数据推断。`psi_axis_t/psi_bnd_t/psi_xpoint_t/Z_xpoint_t`最后一项同刻读出；没有独立LCFS HDF字段。完整历史×184边界元素重算：4110最小距离0.7424498055841912m；5680为0.7358615545252232m；阈值0.037m，均false。合成反例历史轴距中心.005m、当前轴距.25m，证明实现会返回true且使用历史/元素中心；固定较早restart前缀另证明false，不是默认false。

## 有界实际执行

`tools/audit_native_psi_shards.py` 父进程监督子进程，单进程**60s / 1536MiB**，实际采样提案最多10000（另一次失败探针1000）。逐阶段JSON即使超时/异常也保留。保留全846 shard、真实spi_vol缓存及全百万marker槽；无846×3986体积刷新，无云裁剪。rate相对误差仅对非零真值报告，零值单列绝对误差/coverage，不加epsilon掩盖零范数。

工具用原始初始九场、原始lagged/current projection、同刻teacher ES构造40壳/1000点nonlocal缓存。目标日志确认magnetic_reduction=false、tor_rotation=false，未为通过改开关。缺projection、严格未来检查失败、空壳均不补零、不改n_flux。当前成功案例的40壳无空壳。

|实际原始记录对照（全846）|时间类别|位置R/Z/phi最大绝对差(m/m/rad)|半径最大绝对差(m)|速度最大绝对差(m/s)|rate最大相对差(非零)|
|---|---|---:|---:|---:|---:|
|v4004110→4111|restart重复时间边|0/0/0|0|2.360707185289357e−10|1.6032344223196508e−8|
|inner5680→5681|restart重复时间边|0/0/0|1.0587911840678754e−22|8.799361239653081e−11|3.3145608648808908e−9|
|inner5681→5682|正时间步|0/0/0|5.293955920339377e−23|1.858779796748422e−11|**1.1416139182076568，未匹配**|
|v4004111→4112|正时间步|0/0/0（独立运动）|0（旧rate侵蚀）|未执行|nonlocal输入严格拒绝|

全部846位置均实际变化。前两行不是以零timestamp差推进：采用下一HDF保存的控制dt，分别2.0428876129759165e−7s和4.085775225951833e−7s，明确这是restart边的**shard机制对照**，不是合法连续完整事件。inner正时间步dt为8.171550451903666e−8s。未来HDF仅作控制回放/对照，未将其场/ES/源喂入推进。

inner正时间步rate差异尚未定位；当前重建缓存和生产实际缓存生命周期/已安装aux是否一致仍需独立核查，不能由运动/速度匹配推定rate通过。v4004111 projection比field大4.336808689942018e−19s（1ULP），现有nonlocal严格检查拒绝，原样保留，不把时间截断、向前移动或补零。

## 全槽位真实出生执行（独立5ns机制探针）

出生不依赖nonlocal表，独立使用原始全846率和合法spi_vol。采用生产R_geo=1.11、物理注入量7e21、百万槽位、目标包权重7e15，field_static=True、原始rawD sampler；不改权重/槽数/几何以提高接受率。probe timestep=5ns是显式机制探针，**非真实base/gather回放**。锚点imported particleDt分别5.107219032439791ns和5.044166945619547ns（`output/native-closure-certification-2026-10-08/anchors.json` clock证据），与namelist名义5ns区别声明。工具另输出由imported fluid dt/整数子步数重建的值。

|初始shard/粒子快照|原活粒子→插入后|新生数|实际包权重|样本R,Z,phi|元素|
|---|---|---:|---:|---|---:|
|v4004110 / part04110|11168→11169|1|5017774321565696|1.5969600859497717,−.023438103826623535,3.5431657875347584|1992|
|inner5680 / part05680|52604→52605|1|7699835369029632|.7675807734922137,.008646832497507875,5.9398449153051605|1142|
|inner5681 / 显式旧part05680|52604→52605|1|4845416747106304|.7677577952599364,.008649044852191548,5.9398449153051605|1142|

全部保留1000000槽位、新生为完整11态中性Ne、初速度有限。计数先int再atoms>1e14至少1；第一/第三行截断计数原为0但阈值给1，并非人为扩增。NumPy seed20261009仅保证本实现可重放；MT19937分布不是生产PCG32/MPI逐粒子重现。未推进/裁剪旧云，未授予完整marker反馈事件资格。

v4004110完整审计56.8129647933s、峰值RSS952745984bytes（约908.61MiB）；inner5680为5.4721788364s/934318080bytes；inner5681为5.6981145921s/955019264bytes。另v4004111在1000提案预算耗尽，10.0864s内明确birth_failure，未无限推进。最初工具导入/文件位数/atomic version问题已修，旧失败JSON保留，不覆盖证据。

## 文件、命令与未完成项

修改路径仅新增：`src/data/native_psi_teacher.py`、`tests/test_native_psi_teacher_records.py`、`tools/audit_native_psi_shards.py`、本报告。原有dirty修改保留。

红绿：缺reader→两真实记录测试通过；非因果历史未拒绝→补拒绝（真实重复restart时间保留）；缺固定restart参数→补固定前缀并通过。另有独立中心距离反例、不可变性、未来事件拒绝、缺历史拒绝。专项reader6项；含既有nonlocal/shard机制回归16项；无curriculum或optimizer测试。JUnit `/tmp/native-B-psi-reader-final.xml`、`/tmp/native-B-psi-records-final.xml`；py_compile、diff whitespace检查。

```bash
.torch-cuda-env/bin/python -m pytest -q tests/test_native_psi_teacher_records.py tests/test_native_production_nonlocal.py tests/test_native_shard_dynamics.py --junitxml=/tmp/native-B-psi-records-final.xml
.torch-cuda-env/bin/python tools/audit_native_psi_shards.py --path data/EHL-2/phaseA_dataset_1d22_ang15_massloss_neon_v400/jorek004110.h5 --target data/EHL-2/phaseA_dataset_1d22_ang15_massloss_neon_v400/jorek004111.h5 --max-proposals 10000 --output /tmp/native-B-psi-v400-4110-v3.json
.torch-cuda-env/bin/python tools/audit_native_psi_shards.py --path data/EHL-2/phaseA_dataset_1d22_ang15_massloss_neon_v400_injInner/jorek005680.h5 --target data/EHL-2/phaseA_dataset_1d22_ang15_massloss_neon_v400_injInner/jorek005681.h5 --max-proposals 10000 --output /tmp/native-B-psi-inner-5680-v4.json
```

其余分阶段原始证据 `/tmp/native-B-psi-inner-5681-v3.json`、`/tmp/native-B-psi-v400-4111-v4.json`。最终单位口径重跑为 `/tmp/native-B-psi-v400-units-final.json`、`/tmp/native-B-psi-inner-units-final.json`：位置按R(m)/Z(m)/phi(rad)分别报告max/RMS，禁止跨坐标混用米或整体RMS；速度统一injector坐标m/s。旧JSON混合位置字段仅作历史输出保留，因其所有位置误差精确为0，上表逐坐标零值成立。v400最终短重跑将提案预算设1，只为确认单位输出及显式失败，不替代原10000提案的成功出生证据。后续重跑使用新输出路径，不覆盖已有JSON。

未完成：inner连续步rate差异归因；v400 projection1ULP的共享时间契约处理（本切片未放宽）；生产PCG32逐样本复现；真实imported gather/base完整出生反馈回放；自主预测ψ ES finder；真实两完整耦合事件/恢复。主线程teacher wrapper及九场预测链的专项结果由主线程报告，不冒领。psi教师支路是显式诊断，不是自主闭环资格。

## 因果aux重配补充：原始零文件而非补零（后续定向诊断）

本节补充并纠正前述cache对象选择，不覆盖旧JSON/原始失败记录。用户指出registered startup完成时的int3d使用的是已安装零aux，随后source event写出的同index非零projection尚不属于该缓存。diagnosing-bugs反馈循环重跑原CLI并断言rate<1e−6，得到可重复红信号 `1.1416139182076568`；只更换aux来源，不更改当前完整U、ES、dt、公式、n_flux或时间检查。codebase-design将来源选择保留在诊断工具的显式Interface中；tdd先记录缺CLI选项的红，再实现参数和登记验证。

### 源码顺序与事件表

仍为只读JOREK HEAD `54c675bcc39c33b082c0c35046e28122dc53b3aa`：

- `particles/examples/marker_spi_source.f90:145–165`创建projection，aux指向它的node_list，初始rhs=0；`:175–188`将projection列在fluid event之前，并把其下次start关联fluid；`:311–314`在进入粒子循环前先执行startup events。
- `particles/mod_event.f90:200–209`按数组顺序执行同时到达的事件，不是先fluid后projection。
- `core/mod_jorek_timestepping.f90:334,413–426`先index++，必要时以OLD U初始化nonlocal，然后OLD U更新shard，再提交新U；`:446–469`更新ES/history，`:510`以新U和**this%auxiliary_node_list当前已安装aux**执行int3d。
- `marker_spi_source.f90:673–678,749–753`在粒子窗口采样完成后安装新的RHS，令sim%time=target_time，再按上述数组顺序执行projection→fluid。

|阶段|新U/缓存所在记录|int3d实际aux|随后shard更新|
|---|---|---|---|
|v400 registered startup结束|U4111|P4110（原始零projection）|4111→4112使用cache(U4111,P4110)|
|inner registered startup结束|U5681|P5680（原始零projection）|5681→5682使用cache(U5681,P5680)|
|v400下一普通fluid提交|U4112|P4111（非零，absmax9208.909075027546）|4112→4113使用cache(U4112,P4111)|
|inner下一普通fluid提交|U5682|P5681（非零，absmax91393.41667479296）|5682→5683使用cache(U5682,P5681)|

普通步int3d(U_k)使用P_(k−1)，不是之后source event生成的P_k。这是**cache刷新时已安装aux**与**下次fluid求解消耗的projection**两个对象，不能仅按同号文件合并。v400 1ULP拒绝来自误选较新的P4111，不需要放宽严格检查；换回合法P4110后检查自然通过。时间容差和物理模块零改动。

### 最小工具变更及严格限制

新增 `--feedback-path` 显式选择原始文件；`--feedback-policy registered-startup-zero`只接受登记restart+1的poststartup field index，并验证现有 `output/native-closure-certification-2026-10-08/anchors.json` 中run/index/verified_reconstructed/startup_pending、原始projection路径/hash及全数组确实为零。来源、absmax、文件SHA256、登记文件SHA256均进入JSON/cache provenance。不是由文件为零倒推restart，更不会创建零数组。普通node或未登记的非零/错误文件用该policy明确失败。ordinary非零前projection用默认recorded策略加explicit feedback-path，如实标明显式来源。

增加 `--shards-only` 在846 shard诊断后直接返回；本轮四次重配全部使用它，**不执行birth、不重新加载全云、不重复56s采样**。前轮成功birth JSON不动。

### 实测结果与边界

|846 shard原始单步对照|explicit cache aux|rate最大相对差，非零真值|rate最大绝对差 atoms/s|时间s / 峰值RSS MiB|
|---|---|---:|---:|---|
|v4004111→4112|P4110，零|1.465853195938728e−8|5.087927730176e13|3.5815 / 601.29|
|inner5681→5682|P5680，零|4.898476160541286e−4|8.061459432538112e17|3.4169 / 576.39|
|v4004112→4113|P4111，非零|1.47864042640785e−8|5.3016384241664e13|3.6658 / 621.80|
|inner5682→5683|P5681，非零|8.645688371757467e−3|2.989879628266891e19|3.4165 / 683.67|

四次均为真实正时间步、所有846 shard位置发生变化，R/Z/phi误差分别0m/0m/0rad；零rate真值分别2/122槽，零槽最大绝对差均0，非零覆盖844/724。四次未超60s/1536MiB。inner startup cache重配把原1.1416差异减少约2330倍，但**仍不满足1e−6准确度检查**；下一普通步还有约0.865%最大rate相对差。aux对象误配这一主要原因得到验证，剩余inner rate差异未定位，不声称全部机制或生产构建匹配，不继续改公式/扩展诊断。

新证据（旧产物不覆盖）：

- `/tmp/native-B-causal-aux-v400-4111.json`
- `/tmp/native-B-causal-aux-inner-5681.json`
- `/tmp/native-B-causal-aux-v400-4112-nonzero.json`
- `/tmp/native-B-causal-aux-inner-5682-nonzero.json`

```bash
.torch-cuda-env/bin/python tools/audit_native_psi_shards.py --path data/EHL-2/phaseA_dataset_1d22_ang15_massloss_neon_v400_injInner/jorek005681.h5 --target data/EHL-2/phaseA_dataset_1d22_ang15_massloss_neon_v400_injInner/jorek005682.h5 --feedback-path data/EHL-2/phaseA_dataset_1d22_ang15_massloss_neon_v400_injInner/projections005680.h5 --feedback-policy registered-startup-zero --shards-only --output /tmp/native-B-causal-aux-inner-5681.json
.torch-cuda-env/bin/python tools/audit_native_psi_shards.py --path data/EHL-2/phaseA_dataset_1d22_ang15_massloss_neon_v400/jorek004112.h5 --target data/EHL-2/phaseA_dataset_1d22_ang15_massloss_neon_v400/jorek004113.h5 --feedback-path data/EHL-2/phaseA_dataset_1d22_ang15_massloss_neon_v400/projections004111.h5 --shards-only --output /tmp/native-B-causal-aux-v400-4112-nonzero.json
```

公开CLI/源推进回归保留inner实际rate<1e−6的严格检查，标为**strict xfail**已知未完成项，不把阈值改成宽容值凑绿。两项反例覆盖ordinary node滥用zero policy和poststartup误传非零文件均被拒绝。reader专项最终9 passed/1 xfailed（`/tmp/native-B-causal-aux-final-tests.xml`）；连同既有nonlocal/shard回归19 passed/1 xfailed（`/tmp/native-B-causal-aux-regression.xml`）。本补充仅改审计工具、独占测试、本报告；reader/协调文件/物理公式/时间容差均未改，无训练。
