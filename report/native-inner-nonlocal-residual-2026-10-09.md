# injInner 确定性 nonlocal 消融残差诊断（B，2026-10-09）

结论：固定真实 U/P/ES 的 rate 对照不是出生统计问题。两处可分别复现的一手源码语义解释了当前残差：**积分时 rank 间 ES 生命周期不同；marker aux 在两个同时生效的分支中重复累加**。显式源码兼容缓存重建后，两组全846 shard 最大非零相对差分别降至 `9.723270518963859e-9`、`4.817489288836088e-9`，没有调整公式常数、时间容差或已有 `1e-6` 检查。

这是对原始记录的确定性单步诊断，不是自主预测闭环资格。当前工作树源码不能等价于原生产 dirty 二进制；现存日志的10-rank信息属于后续9290+重启，不能认证5680时的精确构建/进程配置。因此严格结论是：**有原始逐值对照支持的、源码可解释的缓存兼容重建**，不是声称已经审计了完整生产构建。没有改变生产模块默认语义或移除现有 xfail。

## 范围及反馈循环

使用 diagnosing-bugs：先红反馈、两次重复、公开四个可证伪假设后才运行探针；新增诊断通过 tdd 的公开 CLI/实际 reader seam 验证。无训练、optimizer、参数搜索、JOREK/原始数据写入、云裁剪、真值重置、资格授予。出生和RNG均不执行。未来HDF只提供明确回放的dt与独立shard对照，未来场/ES不进入缓存或推进。

实际重复命令（主线程还独立重复一次）：

```bash
.torch-cuda-env/bin/python -m pytest -q tests/test_native_psi_teacher_records.py -k registered_poststartup --runxfail --junitxml=/tmp/native-B-inner-residual-red-1.xml
.torch-cuda-env/bin/python -m pytest -q tests/test_native_psi_teacher_records.py -k registered_poststartup --runxfail --junitxml=/tmp/native-B-inner-residual-red-2.xml
```

两次均 `1 failed, 1 passed`，耗时8.53s/6.86s，inner误差完全相同 `4.898476160541286e-4`。新工具内部再次调用完整 update，positions/radius/velocities/rate 逐位一致。不是把出生seed固定后称确定性，而是rate路径根本没有RNG。

初始假设及实测否证：

|可证伪假设|观察与结果|
|---|---|
|缓存/场采样位置错配会表现为局部场或ψ差异|local主导分支约1e-8；最差nonlocal shard的ψ差仅1e-17。全shard最大ψ差约4.3e-11，不能解释0.865%|
|壳归属/表坐标不一致会改变壳或lookup索引|double实际粗壳边界校正无改变；float32坐标探针亦无改变索引，最大残差原样保留|
|浮点精度/质量常数解释差异|默认REAL4/8反事实不足；省略2/2.01455279245的质量修正后5682仍max 0.00665868、加权L1更差，否证|
|低率少数点放大，释放量加权差可忽略|5682有708/846受影响；加权L1 0.131829%，不是少量极低率点独占。最大10项只占L1 7.8124%|

随后先报告再检验两个更具体假设：rank0提交后ES刷新而其它rank仍用prior ES；双aux累加改变非线性密度correction。没有反复调阈值/拟合常数。

## 基线：完整846，不跨单位聚合、不加epsilon

相对量仅覆盖724个非零truth；122个零truth的绝对rate差均0。以下总量使用全部846：

|对照|sum_pred atoms/s|sum_truth atoms/s|sum_pred/sum_truth−1|sumabsdiff/sum_truth|max非零相对差|>1e-6 shard /846|
|---|---:|---:|---:|---:|---:|---:|
|5681→5682，U5681/P5680零aux|1.030066702881820e24|1.0300520585688022e24|+1.4217061066e-5|1.8259523276e-5|4.8984761605e-4|299（35.3428%）|
|5682→5683，U5682/P5681非零aux|1.5140888556222308e24|1.516058976570937e24|−1.2995015228e-3|1.3182941954e-3|8.6456883718e-3|708（83.6879%）|

非零truth相对误差分位（分母是该shard真实rate，不是全活粒子数）：

|对照|p50|p90|p95|p99|
|---|---:|---:|---:|---:|
|5681|6.3726e-14|3.1369e-4|4.2224e-4|4.6595e-4|
|5682|1.96445e-3|7.61940e-3|8.29142e-3|8.57686e-3|

绝对差分位及全部846 trace保存在JSON。5681最大10绝对误差占L1 26.5151%；5682占7.8124%。纯rate不存在可用出生ensemble或有效样本量来解释这些差异；不应用 `1/sqrt(全活粒子数)`。

|分支（按重建rate的local/nonlocal max选择）|5681数量 / max相对差|5682数量 / max相对差|
|---|---:|---:|
|local|342 / 9.72327e-9|16 / 4.81749e-9|
|nonlocal|382 / 4.89848e-4|708 / 8.64569e-3|

v4004111、4112两对照均844个local主导、0个nonlocal主导。因此v400 rate正常**不能证明nonlocal重建正确**，也不能据此把inner差异归为随机噪声。

最差相对误差shard（明确Fortran编号1-based；JSON shard_id是0-based）：

|对照|Fortran编号 / table index 0-based|预测 / truth atoms/s|local / nonlocal atoms/s|ψ差|
|---|---|---|---|---:|
|5681|548 / 750|3.589799290915518e20 / 3.588041697243820e20|3.473513419202213e20 / 3.589799290915518e20|−2.77556e-17|
|5682|283 / 775|1.6757553970986864e20 / 1.690369807688993e20|4.944502885637457e19 / 1.6757553970986864e20|−8.32667e-17|

`NEW_rate*dt`仅是等窗constant-rate释放量诊断，**不是本次源实际释放量**：本步侵蚀/源预算使用OLD rate。工具/JSON明确标记此区别，不能把诊断积分当真实累计源。

## 源码及因果归因

只读 JOREK HEAD `54c675bcc39c33b082c0c35046e28122dc53b3aa`；引用该源码版本行号，而非认证当前二进制。相关机制文件本轮只读。

### 1. rank ES 不同刻

`core/mod_jorek_timestepping.f90:353` 全rank在求解前刷新OLD ES；`:418–426` OLD场shard update之后提交新U。`:444–469` **只有my_id=0**刷新新ES并记录history；`:510` 所有rank立即调用新U的int3d。此段没有ES广播。`models/equil_info.f90:116–381` update函数本身不广播；独立广播函数从`:1030`开始。

`diagnostics/mod_integrals3D.f90:438–454` 每rank用自己的ES建立粗壳，按 `ceiling(n_elements/n_cpu)` 分元素；`:1740–1765`用该ES归壳；`:2368–2370`归约各rank的vol/number/energy。rank0再用当前ES的40壳坐标插值为1000点（`:2531–2557`）。单一全域当前ES不是这个MPI生命周期的同一缓存对象。

具体partition上界见`:455`，循环使用该区间见`:584–585`。3986元素、候选10 rank：chunk=399；rank r 的1-based区间是 `[399*r+1, min(399*(r+1),3986)]`，rank0为1–399，rank9为3592–3986。oracle依据HDF原始元素顺序，不重新按空间/误差排序。两次ES刷新之间只读 `update_equil_state:116–381` 无隐式广播；独立 `broadcast_equil_state` 在另一个函数中，不被该核心事件调用。

诊断明确采用候选n_cpu=10和同run前一原始record ES，保持当前完整U/P不变，不读未来。5681有11个高斯点壳归属变化，修正后max `9.723270518963859e-9`、加权L1 `4.5460e-11`；5682只有1点变化，不能消除其主残差。该ES候选模型不是“从下一步rate拟合ES”：前一步ES来自原始history/reader，rank分支由源码给出。

具体进程数不能由现有目标HDF读出。工具明确记录同目录 `logfile_spitest:2969` 的10-rank依据及后续9290+重启限界，不把它标成5680精确构建证书。该重建对两组逐值输出的匹配是独立行为证据，后续导入生产兼容策略仍需明确登记配置。

匹配也不能唯一辨认原n_cpu：其它分片若恰好把相关点分配给同类ES，可能给出相同结果。本诊断没有遍历进程数拟合rate；固定候选10只检验源码生命周期。若要求认证exact-replay配置，仍应提供原运行日志/构建指纹或当时缓存，而非从误差反推参数。

### 2. 双aux累加 + 非线性density correction

`diagnostics/mod_integrals3D.f90:671–676` use_marker条件下累加eq_aux；`:681–686` present(aux_node_list)再累加同一aux。目标 use_marker且int3d显式携带aux，因而实际源代码两分支叠加。不是P缺失时补零，也不是改投影文件、权重或粒子数。

`:1103–1108` 先对电荷矩取非负，再把累加后的密度带入 `corr_neg_dens(...,(.1,.1),.001)` 后计算均电荷；`:1131–1146` cap10并形成alpha_e。`models/corr_neg_include.f90:309–329` 密度小于2e-4时为 `1e-4+1e-4*exp((density-2e-4)/1e-4)`。

故不能因“大密度处 numerator/denominator 同乘2会相消”而删掉双分支：在非线性correction区不相消。固定P5681的探针中，有5672个当前LCFS内高斯点的均电荷变化超过1e-12；这改变壳温度/密度与nonlocal率。只按两分支累加就把5682全部显著rate残差消除；零P5680自然不受它影响。

两因素控制结果（max非零相对差），保留全部原公式与真实U/P：

|缓存诊断策略|5681→5682|5682→5683|
|---|---:|---:|
|统一当前ES + 单aux|4.89848e-4|8.64569e-3|
|统一当前ES + literal双aux|4.89848e-4|4.81749e-9|
|候选rank ES + 单aux|9.72327e-9|8.64569e-3|
|候选rank ES + literal双aux|9.72327e-9|4.81749e-9|

因此可保留统一ES surrogate并显式处理aux兼容策略；**不需要因为ordinary步的aux缺口而自动引入rank-stale ES**。原始表 `nonlocal_array.dat` 已由后续诊断覆盖，不能拿它当5681/5682缓存或未来回授。

### 3. rho/rhoimp/Te floor备选检查

`mod_integrals3D.f90:749–758,798–802` 同时保留raw与corrected值；真正壳累计`:1764–1765`是 **raw `(r0+alpha_e*rimp0)`、raw Te0**，不是用于原子/辐射部分的corrected ne或Te。新U后的int3d在export_restart之前，所检查的核心路径没有另一次rho/Te limiter。

真实63776高斯点中，按当前ES的LCFS内统计：

|raw量|5681负点数 / min JU|5682负点数 / min JU|
|---|---|---|
|rho|0 / .6416944|0 / .6416375|
|rhoimp|84 / −3.49195e-7|60 / −3.61244e-7|
|Te|1 / −1.53415e-6|0 / 6.81416e-6|
|aux P4|0 / 0|6769 / −.04710409|
|aux P5|0 / 0|6783 / −.004996002|

独立Fortran oracle保留这些raw rho/rhoimp/Te，不把它们clip成非负，却已匹配rate至约1e-8；因此不需要为“通过”添加不存在的shell rho/Te floor。aux密度correction才是已确认需要复现的非线性操作。local Parks仍按原实现clip其局部ne/Te，与shard OLDfield语义一致。

## 独立oracle、测试和资源

新增 `tests/native_nonlocal_residual_reference.f90` 为小型literal-source算子，不移植JOREK矩阵/求解器。读取完整cubic求值得到的63776点 `psi,Z,rho,rhoimp,Te,P4,P5,volume`，独立实现双分支aux、density correction、private-flux、每rank归壳、全域归约和粗/细表插值。**不调用Python的shell/table助手**。高斯求值和local Parks/update保持现有公开模块；该oracle独立的是nonlocal cache机制，不冒称整个Fortran求解/推进。

编译 `gfortran -O0 -fdefault-real-8 -fdefault-double-8 -ffree-line-length-none`。双精度与只读Makefile.inc:13的当前配置一致，仍不是原生产构建认证；默认REAL4是另一个否证探针，不被静默选为目标参数。

|独立Fortran缓存对照|sum_pred/sum_truth−1|加权L1|max相对差|>1e-6数|
|---|---:|---:|---:|---:|
|5681→5682|4.2903902653e-11|4.5459997387e-11|9.7232705190e-9|0|
|5682→5683|3.2360780722e-12|3.3972916181e-12|4.8174892888e-9|0|

全部846、40壳、1000表点，没有空壳fallback、删shard或改空间宽度。剩余最差均是local分支，JSON保存baseline/每个探针的最差编号、全trace与分位数。

父进程限制每次执行60s/1536MiB，超限/异常保留loaded阶段和stderr，不无限推进。Fortran临时编译10s、执行15s子预算。最终两组分别11.3038s/610.332MiB、11.3630s/609.176MiB worker RSS，无超限/错误。证据在 `output/native-inner-nonlocal-residual-2026-10-09/inner5681-final.json`、`inner5682-final.json`；较早归档均保留不覆盖。RSS口径是被监督的Python worker，不声称包含编译器进程RSS。

新增metrics测试先因缺模块红，完成分母/零值/加权口径后绿。真实public CLI oracle测试先因缺显式 `--fortran-oracle` 红，实现后两真实案例绿；测试同时要求原始重建仍可复现残差、oracle<1e-6及无RNG，不让“只reader通过”冒充机制通过。

```bash
.torch-cuda-env/bin/python -m pytest -q tests/test_native_nonlocal_residual.py
# 3 passed / 21.70s（初次独立oracle）
.torch-cuda-env/bin/python -m pytest -q tests/test_native_nonlocal_residual.py tests/test_native_psi_teacher_records.py tests/test_native_production_nonlocal.py tests/test_native_shard_dynamics.py --junitxml=/tmp/native-B-inner-residual-final-regression.xml
# 22 passed, 1 xfailed / 32.93s；没有curriculum/optimizer测试
```

最终同四文件复跑：`22 passed, 1 xfailed in 34.09s`，JUnit已归档 `output/native-inner-nonlocal-residual-2026-10-09/regression.xml`。py_compile与 `git diff --check`通过。

有界CLI重现示例（输出必须为不存在的新路径，不覆盖旧结果）：

```bash
.torch-cuda-env/bin/python tools/diagnose_native_nonlocal_residual.py \
  --path data/EHL-2/phaseA_dataset_1d22_ang15_massloss_neon_v400_injInner/jorek005682.h5 \
  --target data/EHL-2/phaseA_dataset_1d22_ang15_massloss_neon_v400_injInner/jorek005683.h5 \
  --feedback-path data/EHL-2/phaseA_dataset_1d22_ang15_massloss_neon_v400_injInner/projections005681.h5 \
  --fortran-oracle --seconds 60 --memory-mib 1536 --output /tmp/choose-new-residual-evidence.json
```

## 修改路径和最小后续工作

本切片仅新增/编辑独占文件：`tools/diagnose_native_nonlocal_residual.py`、`tests/test_native_nonlocal_residual.py`、`tests/native_nonlocal_residual_reference.f90`、本报告；生成独占 `output/native-inner-nonlocal-residual-2026-10-09/*.json`。其余dirty修改保留。没有改 `native_source_nonlocal.py`、`native_shard_dynamics.py` 或主线程协调文件。

归因已完成，但**通用生产builder尚未变更**，原registered-poststartup xfail继续保留。协调时需明确两层接口而非偷偷改物理公式：

1. 原始教师记录的兼容缓存重建应显式声明双aux分支、rank0/prior ES状态与经登记的MPI元素分片；记录原始P digest不能变成“修改过P的普通记录”，prior ES也不得假绑定成当前ES。
2. 新预测链若采用统一当前ES、单aux的物理算子，应如实命名其与原历史实现的差异，不要求它逐值复现历史rank-lag/double-aux artefact；该选择由主线程/用户决定，不能据输出误差自行决定纠正JOREK。

若选历史兼容，最小实现范围是nonlocal缓存构建接口与其来源metadata，不需更改 Parks/rocket、出生RNG、JOREK矩阵、时间检查，也不需自主ES finder。需先新增公开缓存接口的红测试，再通知/协调 `native_source_nonlocal.py` 所有权后实施。当前任务未把诊断反事实冒充已完成生产修复、完整两事件/恢复或真实资格。

主线程随后明确：当前完整链继续已有明示近似；**不得自动把rank-stale ES加到生产默认**。本轮按此决定保持所有生产模块/版本指纹不变。两个策略须命名分离：`exact-replay diagnostic` 是源码兼容重建（尚需原构建/partition登记才能认证exact来源）；`unified-ES surrogate` 是当前单一ES的物理策略。不能借数值匹配替用户选择前者，亦不能借后者声称逐值复现历史MPI输出。
