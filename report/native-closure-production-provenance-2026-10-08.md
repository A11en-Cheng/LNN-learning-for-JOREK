# phaseA 生产构建与源码证据边界（2026-10-08）

本项使用 research 组织一手证据；只读 JOREK 源码、Git 对象库、原始 HDF5 与现有日志，没有运行 JOREK、修改导出代码或训练。现行依据为 roadmap_v2 与 ADR-0018～0022（先读 ADR README）；历史 ADR 不作资格门槛。本报告提供生产同构建核验边界，不取代锚点注册事件证明。

## 已确认的版本事实

- 直接读取 v400 `projections004109/004110/004111.h5`、injInner `projections002709/002710/005680/005681.h5` 的 `RCS_version`，全部为 **`v2.25.12.0-1042-gb4165893f-dirty`**。原始路径均在 `data/EHL-2/phaseA_dataset_1d22_ang15_massloss_neon_v400{,_injInner}/`，实际数据根是 `/mnt/d/EHL-2`。
- 当前只读检查源码 `/home/ac_desktop/JOREK_code/jorek` 的 `git rev-parse HEAD` 是 **`54c675bcc39c33b082c0c35046e28122dc53b3aa`**。`git cat-file -t b4165893f` 返回无此对象，不能执行可信的生产提交→当前 HEAD 逐文件差分。未 fetch 外部仓库。
- 当前 `git status --short` 仅有已跟踪 `communication/eqdsk2jorek.f90` 修改（`git diff --stat` 为一行替换）及未跟踪 `poincare`。这只描述当前工作树，不能解释生产构建的 `dirty`。当前目标子系统文件对当前 HEAD 无工作树修改；这也不是生产一致证明。
- [defaults.mk](/home/ac_desktop/JOREK_code/jorek/defaults.mk:354) 用 `git describe --always --dirty --abbrev` 生成 `RCS_VERSION`，并另存标签/时间；它未编码 dirty patch。投影写出直接保存这个字符串：[mod_project_particles.f90](/home/ac_desktop/JOREK_code/jorek/particles/diagnostics/mod_project_particles.f90:562)。相同 RCS 字符串不是内容哈希或可复现构建证明。

## 按子系统的结论

下表“已证”指明究竟是数据事实、当前源码事实还是生产同构建事实。没有把不同 HEAD 推断成算法差异；目前也没有足够证据宣称任一目标子系统生产代码逐字一致。

| 子系统 | 已证一致/可核验事实 | 已发现差异 | 缺证据及资格影响 |
|---|---|---|---|
| 投影更新与输出 | 当前 `mod_uncoupled_projection.f90:258–294` 先由解向量减旧 values 赋 deltas，再提交新 values；`mod_project_particles.f90:538–548` 拷贝 node values，:562–581 保存 RCS/几何元信息。实测目标 HDF 为 float64、无量化且不保存 raw projection deltas（既有 inventory）。 | 记录生产 RCS 与当前 HEAD 不同；没有已证的相关算法差异。 | 生产该两文件及调用路径、输出更新次数/对象身份/重复输出证据缺失；普通段相邻 values 差仍是 `pending_evidence`，而非原则上不可恢复。 |
| 滤波与边界 | 当前 `mod_uncoupled_projection.f90:994–1017` 在 n=0 矩阵含 perpendicular/hyper/parallel 项；:1070、1090–1130 包含物理边界处理。目标配置滤波非零，H=1 不消除滤波。 | 没有生产↔当前已证差异；模型参考的无滤波/无物理约束与生产配置不同属于模型支持缺口。 | 生产矩阵、过滤阶段、边界分支与输出变换无法由 RCS 单独确认；不能把未滤波重投影当生产历史证人。 |
| marker 时钟/gather | 当前 `marker_spi_source.f90:350–355` 以 `sim%time-step_rest_time` 为真实粒子起点，整数子步后保留 residual；:443 用 `n_steps*timesteps`；:670–676 除真实 gather 后清零。后段日志明确打印同类变量。 | 没有已证算法差异。 | 目标普通段 2710 的初始 residual、子步和更新事件未被当前后段日志覆盖，不能默认恒步长或套 restart 清零。 |
| restart | 当前 `marker_spi_source.f90:100–...` 有 restart_particles 分支，:175–180 排投影先于 JOREK 事件，:311–314 初始化 residual=0 并触发启动事件。目标注册 restart 可按已登记规则限定认证。 | 没有生产↔当前已证差异。 | 注册零历史/启动相位是限定事件特例，不由此确认普通段，也不由当前版本缺证推翻已登记 restart。 |
| shard | 当前 `models/pellet_module.f90:464–479` 使用 nonlocal 量，:624–725 选择 abl3，:864–889 选择 rocket6；目标配置明确启用这些分支。 | 模型未实现 rocket6/nonlocal 是模型支持差异，不是原始 shard 缺失。 | 生产 pellet_module、nonlocal 表初始化/场读取阶段与 build 宏需逐项比对；不能将模型支持失败归为状态损坏。 |
| 原子辐射 | 当前 `mod_radiation.f90:52–76` 的 `get_Lz=(PRB+PLT)*n_e`；marker 调用见 `marker_spi_source.f90:547`。 | 模型 PLT-only 与当前所查分支不同；未发现生产代码差异证据。 | 生产该函数、实际 ADAS 文件/校验和及初始化配置需绑定。不能无分支证据额外要求 CX，也不能省略该分支使用的 PRB。 |

## 日志和构建搜查的范围

读取两 run 的 `logfile_spitest`、`jorek.out`、`file.out`、`jorek.err`、`r3.out`、`running_jorek.sh`，并在 run 树、`/home/ac_desktop/JOREK_code`、`/home/ac_desktop/Jorek_env` 搜索 patch/diff/build/compile/version 记录。找到的是运行/后处理日志与第三方依赖补丁，未找到可绑定目标生产二进制的源码差分和构建日志。

- v400 `logfile_spitest:1805–1810` 从 **11710 / 11368.2 JU** restart 开始；injInner `logfile_spitest:1775–1780` 从 **9290 / 11551.8 JU** restart 开始。inner :2034 首写 `projections009290.h5`，:10937–10974 的 marker/gather 记录对应 9291。它们可以核验后段运行输出模式，**不能见证 4110、2710 或替代 5680 的启动时钟与投影链**。
- 两 run 的 `running_jorek.sh` 当前有效命令指向远端 `/thfs1/home/wangshunwen/ZYX/JOREK_code/jorek/marker_spi_source_model502_ntor1`，输入 `jorek_namelist_spitest`、覆盖写 `logfile_spitest`。脚本是路径线索，不能证明每个历史 HDF 由该次命令或当前二进制生成。
- v400 `jorek.out` 是 MPI Invalid count/negative count 失败信息；`file.out` 只有 `00000`。`r3.out` 为 timing 输出，不带目标源码清单或 dirty patch。未把这些后段/失败日志充作初态认证。

## 精确需要补充的现有材料（无需重跑生产）

1. 生产 `b4165893f` 的完整 40 位 commit 与可读 Git 对象/源码归档；该次编译工作树的 **完整 tracked diff 和参与构建的 untracked/generated 文件**。优先 `marker_spi_source.f90`、`mod_uncoupled_projection.f90`、`mod_project_particles.f90`、JOREK timestep/event scheduler、restart import/export、`pellet_module.f90`/nonlocal 初始化、`mod_radiation.f90` 及其被调模块。提供各文件 SHA256 可先限定一致范围，无法仅靠版本字符串替代 dirty 内容。
2. 对应该生产 executable 的 SHA256、构建日期、完整编译命令/宏（model502、ntor1、精度、ADAS/粒子配置）、compiler/linker/library 版本、Makefile.inc/generated revision include；以及二进制与这批 run/segment 的对应记录。现有 Makefile 配置不能证明历史编译使用它。
3. 原始生产日志中覆盖 v400 4109–4112、injInner 2708–2711，以及替代 restart 5679–5682 的段落：导入文件及 index、事件顺序、输出名称、sim/真实粒子时间、particle substep、residual、实际 gather、projection 更新与重复输出/restart。当前 `logfile_spitest` 的后段不覆盖这些索引。
4. 与目标阶段绑定的完整 namelist/运行控制变更记录和 ADAS 数据文件校验和。下一 dt 单独登记为 `external_control` 或明确标记 `recorded_schedule_replay`；不从 namelist 数组恢复自主预测控制。

在这些材料缺失时，**保存状态与注册 restart 特例仍可分别认证**；生产普通投影历史恢复、目标普通 residual/gather 的因果重建保持 `pending_evidence`。这里没有“所有数据不可用”的结论，也没有把未来记录作为初始化输入。
