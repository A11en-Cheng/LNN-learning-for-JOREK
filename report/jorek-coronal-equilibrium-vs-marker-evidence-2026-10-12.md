# JOREK 日冕平衡、marker 与连续云：一手源码及 Hu 论文对照（2026-10-12）

## 核查结论

- 当前 JOREK CE 分支演化总杂质密度，再由当前温密确定平衡荷态；它没有独立推进的主杂质荷态状态。
- 当前 marker 非 CE 分支保留上一荷态概率，按有限反应速率推进；CE_marker 对照则每次查平衡表覆盖概率。
- 连续场表示与平衡假设是两个维度。独立演化的各荷态密度场属于拟议连续非平衡模型，不是旧 CE 分支。
- Hu 的 CE 对照不能用来否定所有连续云表示；论文对照、场装配及真实 SPI 的附加限制见 B–D 节。

## 范围与版本

- 源码根目录：`/home/ac_desktop/JOREK_code/jorek`；下文 `path:line` 均相对此目录，行号指本次读取的工作树。
- Git HEAD：`54c675bcc39c33b082c0c35046e28122dc53b3aa`；提交时间 `2026-05-19T11:22:35+08:00`，题为 “Refactor code structure for improved readability and maintainability”。
- 已核对本文引用的 helper、初始化与单位辅助文件相对 HEAD 无改动；工作树另有 `communication/eqdsk2jorek.f90` 修改和未跟踪 `poincare`，不在核查范围。
- 遵守主仓库 AGENTS.md，已读 `docs/adr/README.md` 及现行文档隔离规则；本报告只提供源码证据，不决定架构或实施模型。
- 范围：CE/ADF11/插值 helper、杂质原子初始化及辐射辅助模块；B–D 节补充 ordinary model502、marker example、选定 Ne 消融分支及 Hu 论文对照。未认证 Hu 原始构建或当前生产二进制。
- 方法：源码与版本信息只读；只新增本 Markdown；未执行 JOREK、测试、训练、benchmark 或安装。

## ADAS 读取与 CE 构造

入口为 [models/mod_impurity.f90:7](/home/ac_desktop/JOREK_code/jorek/models/mod_impurity.f90:7) 的 `init_imp_adas`。
它先释放并重新分配 `imp_adas(n_adas)`、`imp_cor(n_adas)`；只有 `nimp_bg(1)>0 .or. with_impurities .or. use_rcs` 才真正填表，故“数组已分配”不保证“CE 已构造”。证据：`models/mod_impurity.f90:26`、`:36`、`:37`。
物种映射为 C→96_c，H/D→12_h，Ar→89_ar，Ne→96_ne，Fe→93_fe，W→50_w，Be→96_be，N→96_n；D 明确复用 H 数据，未知物种停止。证据：`models/mod_impurity.f90:39`、`:45`、`:58`。
逐物种执行 `read_adf11(my_id,suffix,adas_dir)` → `coronal(imp_adas(i))`，rank 0 写诊断荷态表。证据：`models/mod_impurity.f90:66`、`:67`、`:72`。
启动处调用受 `WITH_Neutrals || WITH_Impurities` 编译条件控制；本次不核查实际二进制的宏配置。证据：`core/mod_startup_teardown.f90:94`、`:96`。

[tools/mod_openadas.f90:28](/home/ac_desktop/JOREK_code/jorek/tools/mod_openadas.f90:28) 的 `ADF11_all` 包含 ACD（复合）、SCD（电离）、CCD（电荷交换）、PLT（激发线辐射）、PRB（复合及轫致辐射）、PRC（中性氢电荷转移辐射）与可选电离能数组。
读取器按 `acd/scd/ccd/plt/prb/prc + suffix + .dat` 尝试加载；目录与文件名直接拼接，不自动补路径分隔符。证据：`tools/mod_openadas.f90:55`、`:71`、`:72`。
缺文件会跳过；ACD、SCD 缺失最终退出，辐射数据没有同样的必需性检查；CCD、PRC 还直接打印“未正确实现”的警告。证据：`tools/mod_openadas.f90:74`、`:90`、`:93`、`:146`。
荷态索引统一成“起始荷态”：ACD/PRB/PRC 在 q=1…Z 读取，SCD/PLT 从原文件终态索引减 1，存于 q=0…Z−1。证据：`tools/mod_openadas.f90:88`、`:116`、`:122`。
未填充项先置 `log10(GRC)=-60`，单位换算后为 −66；已分配数组的这些项返回极小正数，不等同于“缺整个数组时返回 0”。证据：`tools/mod_openadas.f90:111`、`:130`、`:239`。

[tools/mod_coronal.f90:139](/home/ac_desktop/JOREK_code/jorek/tools/mod_coronal.f90:139) 的 CE 构造器固定 10 个密度点、1000 个温度点。
密度轴均匀分布于 `log10(ne[m^-3])=18…21`；温度轴为 1…40000 eV 的对数网格，存储时转换成 `log10(Te[K])`。证据：`tools/mod_coronal.f90:150`、`:151`、`:167`、`:170`。
每个网格点令未归一化 p₀=1，再递推 `p_q=p_(q−1)·SCD_(q−1)/ACD_q`，最后保存 `f_q=p_q/Σp`；构造调用的是 ADAS 样条接口。证据：`tools/mod_coronal.f90:185`、`:189`、`:190`、`:191`、`:194`。
该递推仅使用相邻电离/复合系数，没有 CCD 项、时间步或上一荷态输入；电子密度是查表参数，helper 不在此自洽求解它。证据：`tools/mod_coronal.f90:183`、`:189`、`:190`。
单点 `specific_coronal_equilibrium` 直接从 ADAS 重算同一递推，可选线性或样条系数插值，结果再次除以总和；它不读取预制 CE 网格。证据：`tools/mod_coronal.f90:106`、`:119`、`:122`、`:128`、`:133`。

## 表与输出的实际意义

下表以赋值及输出运算为依据，不能只按字段注释或变量名理解。

| 表／输出 | 意义及单位 | 源码锚点 |
|---|---|---|
| `density`、`temperature` | 分别为 log10(ne/m⁻³)、log10(Te/K) | `tools/mod_coronal.f90:167`、`:170` |
| `Z(:,:,q)`、`p_out(q)` | 荷态概率 f_q；无量纲，总和归一为 1；不是荷态数密度 | `tools/mod_coronal.f90:194`、`:378` |
| `Z_avg_CE`、`z_out/z_avg` | 平均电荷数 〈q〉=Σq f_q，无量纲 | `tools/mod_coronal.f90:196`、`:198`、`:391` |
| `Z_1T`、`p_Te_out` | 每个荷态概率的温度偏导，K⁻¹ | `tools/mod_coronal.f90:94`、`:369`、`:379` |
| `p_Ne_out` | 每个荷态概率的电子密度偏导，m³ | `tools/mod_coronal.f90:370`、`:380` |
| `Z_avg_1T_CE`、`z_Te_out/z_avg_Te` | 〈q〉的温度偏导，K⁻¹ | `tools/mod_coronal.f90:96`、`:399` |
| `Z_avg_2T_CE`、`z_TeTe_out/z_avg_TeTe` | 〈q〉的二阶温度偏导，K⁻²；不是 〈q²〉 | `tools/mod_coronal.f90:97`、`:401` |
| `z_Ne_out` | 〈q〉的电子密度偏导，m³；存在下述可选参数门控问题 | `tools/mod_coronal.f90:386`、`:400`、`:406` |
| `Prad`、`rad_out` | 每个杂质原子的线性辐射功率，W/atom；不是 log10(power) | `tools/mod_coronal.f90:71`、`:199`、`:417` |
| `Prad_1T`、`rad_Te_out` | 上述功率的温度偏导，W/(atom·K) | `tools/mod_coronal.f90:95`、`:414` |
| `rad_Ne_out` | 上述功率的电子密度偏导，W·m³/atom | `tools/mod_coronal.f90:415`、`:419` |
| `radiation_function[_linear]` 的 `Lrad` | `Prad/ne`，辐射系数 W·m³；linear 版可再转 JOREK 单位 | `models/mod_impurity.f90:118`、`:161`、`:185` |
| `ionisation_energy(q)` | q−1→q 的逐级电离能，eV；q=0 为零 | `tools/mod_openadas.f90:36`、`:167`、`:169` |

构造器局部变量 `Z_eff` 与诊断表的 “effective charge” 实际保存 〈q〉；不能当作 〈q²〉或等离子体有效电荷公式。证据：`tools/mod_coronal.f90:196`、`models/mod_impurity.f90:247`、`:254`。
`coronal` 类型没有二阶电荷矩或平均电离能表，也没有这两项输出参数。证据：`tools/mod_coronal.f90:16`、`:31`、`:235`、`:343`。
按已核实的概率／逐级能量语义可数学推导 `〈q²〉=Σq²f_q`、`U_q=Σ_(k=1…q)I_k`、`〈U〉=Σf_qU_q`；其温度导数为对 f_q 的导数作相同加权。
这些是由 `tools/mod_coronal.f90:194` 与 `tools/mod_openadas.f90:36` 推导的定义，不是本范围内已核查的生产装配公式；J/atom 换算乘 `EL_CHG`，常量见 `models/constants.f90:13`。

## 插值、归一化与导数

ADAS 内存中的坐标与系数均有 log10 语义：密度从 cm⁻³ 转 m⁻³加 6，温度从 eV 转 K 加 `log10(EL_CHG/K_BOLTZ)`，系数从 cm³ 转 m³减 6；ACD/SCD 返回 m³/s，PLT/PRB 返回 W·m³。证据：`tools/mod_openadas.f90:15`、`:105`、`:109`、`:130`。
ADAS `interp_linear` 在 log10 系数表上双线性插值后取 10 的幂；`interp` 对 log10 温度作样条、log10 密度作线性插值后取幂。证据：`tools/mod_openadas.f90:240`、`:261`、`:262`。
ADAS `interp_grad_T/n` 返回 log-log 斜率；样条接口则乘 `C/Te` 或 `C/ne` 返回普通偏导，两个 log10 的因子抵消。证据：`tools/mod_openadas.f90:200`、`:220`、`:263`、`:264`。

CE `interp` 默认绑定样条接口；`interp_linear` 绑定表的双线性接口，二者名称不能混用。证据：`tools/mod_coronal.f90:32`、`:34`。
CE 样条处理的是线性概率、平均电荷与功率；样条方向为 log10(Te)，线性方向为 log10(ne)，构建器采用自然样条。证据：`tools/mod_coronal.f90:173`、`:179`、`:222`、`tools/mod_interp_splinear.f90:38`、`:46`。
样条概率逐项截负到 0，然后除以 Σp；偏离归一性超过 10⁻³只打印警告；导数也直接除以 Σp。证据：`tools/mod_coronal.f90:365`、`:373`、`:378`、`:379`。
因此截负后的导数没有同步截断，也未显式执行归一化商法则 `(p_T S−p S_T)/S²`；边界／截负情形不能保证其等于归一化输出的严格导数。证据：`tools/mod_coronal.f90:365`、`:369`、`:379`。
对于线性量 F，普通温度偏导为 `F_T=F_x/(ln(10)·Te)`，二阶为 `F_TT=F_xx/(ln(10)²·Te²)−F_T/Te`，x=log10(Te)。证据：`tools/mod_coronal.f90:369`、`:371`、`:399`、`:401`。
密度偏导同理由 log10(ne) 转成对 ne 的偏导；这些是定另一参数的偏导，均不是时间导数。证据：`tools/mod_coronal.f90:370`、`:400`、`:415`。
仅在请求概率相关输出且温度超出 CE 范围时，样条接口才改从概率加权计算平均电荷；否则查独立平均电荷样条。证据：`tools/mod_coronal.f90:361`、`:382`、`:387`、`:396`。

[tools/mod_interp_splinear.f90:72](/home/ac_desktop/JOREK_code/jorek/tools/mod_interp_splinear.f90:72) 的双线性值接口将两个输入裁剪至网格范围；样条接口在温度越界时使用边缘线性外推，密度方向也允许外推。证据：`tools/mod_interp_splinear.f90:84`、`:120`、`:214`、`:225`、`:228`。
CE `interp_linear` 的概率和导数都作双线性查表；导数表在初始化时由样条接口生成，并非每次对当前双线性值插值器求导。证据：`tools/mod_coronal.f90:91`、`:94`、`:248`、`:249`、`:262`。
`interp_gradients` 则调用双线性梯度 helper：dim=1 对第二轴（温度）求导、dim=2 对第一轴（密度）求导；越界处理也不同于值裁剪。证据：`tools/mod_coronal.f90:301`、`:302`、`tools/mod_interp_splinear.f90:156`、`:160`、`:176`、`:180`。

## 辐射与电离能

`coronal_Prad` 计算 `P=Σf_q[ne·(PRB_q+PLT_q)+n_H·PRC_q]`，返回 W/atom；调用者提供的 fractions 不在此检查或归一化。证据：`tools/mod_coronal.f90:47`、`:56`、`:71`。
未传中性密度时设 log10(n_H)=−99；CE 表构造未传该参数，所以默认表没有可变化的中性密度轴，仅保留这一极小默认项。证据：`tools/mod_coronal.f90:65`、`:68`、`:199`。
`Prad` 字段注释虽写 log10，实际赋值及输出从未对功率取对数；功率体积密度还需乘杂质原子数密度，W/m³ 是依据此单位关系的推导。证据：`tools/mod_coronal.f90:22`、`:71`、`:199`。

两个 `radiation_function` helper 均先取 CE 功率并除 ne；温度导数再按 `Σf'_q(PRB+PLT)+Σf_q(PRB'+PLT')` 单独计算，没有沿用 `rad_Te_out/ne`。证据：`models/mod_impurity.f90:117`、`:128`、`:160`、`:175`。
linear 版的 f′来自预制 CE 导数表，系数导数由 ADAS log-log 斜率转普通导数；故不能假定返回导数严格等于“插值 Prad 再除 ne”的数值导数，极小默认 PRC 项也未加入导数式。证据：`models/mod_impurity.f90:163`、`:164`、`:172`、`:175`。
linear 版可选归一化因子 `C_rad=(GAMMA−1)·MU_ZERO^(3/2)·(central_mass·AMU)^(1/2)·(central_density·10^20)^(5/2)`，令 `Lrad_JU=C_rad·Lrad_SI`。证据：`models/mod_impurity.f90:182`、`:186`。
导数先乘 `C_rad·EL_CHG/K_BOLTZ` 转成每 eV，再除 `EL_CHG·MU_ZERO·central_density·10^20`；非 `with_TiTe` 模式额外除 2。证据：`models/mod_impurity.f90:189`、`:191`、`:192`、`:194`。
此版本将负 Lrad 及其导数置零，遇 NaN 停止；这不是 CE 构造递推的有限性检查。证据：`models/mod_impurity.f90:198`、`:202`、`:207`。

电离能独立尝试 `ion<完整suffix>.dat`、`ion<去掉两位年份suffix>.dat`；读取时要求文件 q+1 等于目标索引 k，找不到时允许不分配能量数组。证据：`tools/mod_openadas.f90:156`、`:158`、`:166`、`:170`、`:181`。
该数组是逐级能量，不由 SCD/ACD 系数积分构造，也不需要温度插值；CE helper 没有计算电离能时间损失的接口。证据：`tools/mod_openadas.f90:36`、`:169`、`tools/mod_coronal.f90:31`、`:343`。

## 辐射辅助、初始表与缓存边界

[particles/projection_functions/mod_radiation.f90:118](/home/ac_desktop/JOREK_code/jorek/particles/projection_functions/mod_radiation.f90:118) 的 `proj_Lz_equil` 临时分配 fractions，调用单点 CE 的线性 ADAS 路径，再调用 `coronal_Prad`；预制 CE 表查功率那一行是注释。证据：`particles/projection_functions/mod_radiation.f90:132`、`:135`、`:136`、`:138`。
同文件的 `get_Lz` 为给定 iz 返回 `ne(PRB+PLT)`，不自动套 CE；虽接收 `n_neutral`，该值未参与返回式。证据：`particles/projection_functions/mod_radiation.f90:52`、`:63`、`:73`、`:75`。
其 `proj_Lz` 使用 log10；`tools/mod_radiation.f90` 同名模块却使用自然 log 后调用要求 log10 的 ADAS 接口，本次未确认构建选择哪个文件。证据：`particles/projection_functions/mod_radiation.f90:35`、`:47`、`tools/mod_radiation.f90:33`、`:45`。
辐射所调用的 `fields_interp_ne_Te` 声明 ne 为 m⁻³、Te 为 K，最低分别限制到 10¹⁶ m⁻³与 1 K，并受 `WITH_TiTe` 宏影响。证据：`particles/mod_ionisation_recombination.f90:68`、`:81`、`:82`、`:83`。

初始化保留 ADAS 原表／样条以及 CE 的概率、均值、功率与温度导数／样条，作为内存查表数据；重新调用初始化会释放旧数组并重建。证据：`models/phys_module.f90:547`、`models/mod_impurity.f90:26`、`:32`、`tools/mod_coronal.f90:153`、`:228`。
rank 0 的 `<species>_charge_distribution.dat` 固定 ne=10²⁰ m⁻³，列为 log10(Te/K)、各 f_q、Σf、〈q〉、Prad/ne；局部变量 `Te_eV` 实际装的是 log10(K)。证据：`models/mod_impurity.f90:235`、`:238`、`:242`、`:247`、`:248`、`:253`。
核查到的初始化只读取 ADAS/ion 文件，没有读取上述诊断表恢复 CE 缓存的路径；不能把诊断输出称作 restart 的独立荷态历史。证据：`models/mod_impurity.f90:66`、`:67`、`:72`、`tools/mod_openadas.f90:80`、`:162`。
CE 类型没有 t、dt、上一 f 或粒子身份，查询为 `intent(in)`；固定数据集与插值选择后，CE 概率、均值、功率及其偏导均由当前 Te/ne 决定，不保存独立荷态演化历史。证据：`tools/mod_coronal.f90:16`、`:31`、`:237`、`:345`。
`coronal_Prad` 本身接受外部 fractions，给定荷态辐射 helper 接受 iz；因此“仅由 Te/ne 决定”只适用于 CE 闭合输出，不能推广到这些通用辅助接口。证据：`tools/mod_coronal.f90:47`、`particles/projection_functions/mod_radiation.f90:55`。

## 静态限制与后续补证边界

- `radiation_function` 在请求 `dLrad_dNe` 时误写 `dLrad_dTe`，没有赋值给密度导数输出；若只提供密度导数参数，还会写未提供的可选参数。证据：`models/mod_impurity.f90:108`、`:129`。
- 样条 CE 的平均电荷分支门控列了 `p_Ne_out`，漏了 `z_Ne_out`；仅请求后者时输出未赋值。证据：`tools/mod_coronal.f90:386`、`:406`。
- 双线性密度梯度分支的 `fx2` 基项使用 `f(ix1,iy2,:)`，与该行差分使用的 ix2 不一致；密度梯度应作为待验证项。证据：`tools/mod_interp_splinear.f90:181`、`:182`、`:183`。
- 单点 CE 的局部 `linear_flag=.false.` 在声明处初始化，按 Fortran 语义隐含 SAVE；传参会覆盖它，后续省略参数不能稳妥理解成每次重置默认。此配置记忆不是物理荷态历史。证据：`tools/mod_coronal.f90:113`、`:117`。
- CE 递推没有显式复合率零值、乘积溢出或总和零值保护；构造后仅检查负概率，不能据此认证所有物种／温度范围的有限性。证据：`tools/mod_coronal.f90:191`、`:194`、`:218`。
- 实际 namelist、编译宏／模块选择、所加载 ADAS 文件版本与范围、跨文件物种一致性及生产运行是否触及上述分支均未核查；源码 suffix 不能替代数据文件 provenance。证据入口：`models/mod_impurity.f90:37`、`:66`、`tools/mod_openadas.f90:96`。
- 装配处的查表密度选择、平均／二阶矩／能量进入方程、marker 状态或 CE 覆盖、论文实验对照见主代理补充；本报告不据 helper 作全运行历史性或架构结论。

上述静态问题没有在本轮执行数值验证；不能据此归因 Hu 文中的差异，也未确认 Hu 原始代码版本包含这些问题。

## B. 主代理补充：ordinary model502 与 marker 的对照

以下是本轮主代理直接核查的源码事实；范围限定于本地普通元素装配与 marker example，不把它当作 Hu 2021 生产构建的逐字节证明。

### B.1 两个维度，不应混成一个选择

| 数值表示 | 强制平衡荷态 | 有限速率的非平衡荷态 |
|---|---|---|
| 网格流体 | 本地 use_marker=false 的主杂质 CE 闭合 | 拟议连续荷态密度 n_0…n_10；尚未实现 |
| 移动物质包 | marker example 的 CE_marker=true 对照分支 | marker example 当前 CE_marker=false 分支 |

“连续化”不蕴含 CE；一个独立演化的平均电荷模型也不自动是 CE，但需要另行证明其闭合。只有每次由当前温密重建平衡荷态、没有独立荷态状态时，才采用了 CE 型闭合。

### B.2 荷态、梯度与密度

- ordinary model502 的非 marker 分支调用 imp_cor%interp_linear，得到 P_imp、dP_imp_dT、Z_imp 与温度导数：[mod_elt_matrix.f90:944](/home/ac_desktop/JOREK_code/jorek/models/model502/mod_elt_matrix.f90:944)。此处 density=20 是 log10(10^20 m^-3)，不是对每个点的电子密度自洽迭代；源码注释也明确这是测试密度估计。
- CE 分支的 P_imp 是当前局域查表结果，不是可随下个场步恢复的独立主杂质荷态状态。rho_imp 仍由第 8 方程及独立注入源推进：[mod_elt_matrix.f90:1717](/home/ac_desktop/JOREK_code/jorek/models/model502/mod_elt_matrix.f90:1717)。
- marker 分支从通道 4/5 得平均电荷，从 3/5 得荷态平方矩；分母有下限，再以商导数分离梯度/历史增量：[mod_elt_matrix.f90:989](/home/ac_desktop/JOREK_code/jorek/models/model502/mod_elt_matrix.f90:989)、[mod_elt_matrix.f90:1005](/home/ac_desktop/JOREK_code/jorek/models/model502/mod_elt_matrix.f90:1005)。
- marker 场装配将 dZ_imp_dT 保持为零，并另消费独立的空间/历史变化：[mod_elt_matrix.f90:988](/home/ac_desktop/JOREK_code/jorek/models/model502/mod_elt_matrix.f90:988)、[mod_elt_matrix.f90:1071](/home/ac_desktop/JOREK_code/jorek/models/model502/mod_elt_matrix.f90:1071)。这是本次场线性化固定投影时的导数约定，不意味着物理电离不依赖温度。
- CE 的 alpha_e_bis=alpha_e+Te*dalpha_e_dT，而 marker 通过独立投影梯度及 delta_alpha_e 修正；场矩阵的容量/压力/历史因此也不同：[mod_elt_matrix.f90:1065](/home/ac_desktop/JOREK_code/jorek/models/model502/mod_elt_matrix.f90:1065)、[mod_elt_matrix.f90:1784](/home/ac_desktop/JOREK_code/jorek/models/model502/mod_elt_matrix.f90:1784)。
- model501 也可找到 use_marker=false 的同类 CE 查表分支，但本轮未逐项认证其完整方程：[model501/mod_elt_matrix.f90:810](/home/ac_desktop/JOREK_code/jorek/models/model501/mod_elt_matrix.f90:810)。

### B.3 出生、有限反应与 CE particle 对照

本地 example 的 CE_marker 在第 98 行硬编码 false；不是读取到某个默认 namelist 参数的证据：[marker_spi_source.f90:98](/home/ac_desktop/JOREK_code/jorek/particles/examples/marker_spi_source.f90:98)。

初始物质包分配全荷态概率，置 P_imp(0)=1：[marker_spi_source.f90:138](/home/ac_desktop/JOREK_code/jorek/particles/examples/marker_spi_source.f90:138)。非 CE 分支用旧 P_imp、SCD/ACD、局域 n_e/Te 和粒子 dt 做相邻荷态转移；截去负分量后归一化：[marker_spi_source.f90:543](/home/ac_desktop/JOREK_code/jorek/particles/examples/marker_spi_source.f90:543)、[marker_spi_source.f90:598](/home/ac_desktop/JOREK_code/jorek/particles/examples/marker_spi_source.f90:598)。

CE_marker=true 则改为当前温密的平衡表，覆盖 P_tmp，辐射同样调用 CE 加权函数：[marker_spi_source.f90:589](/home/ac_desktop/JOREK_code/jorek/particles/examples/marker_spi_source.f90:589)。粒子表示并未因此变为非平衡。

当前子步的电荷矩和非 CE 辐射消费旧 P；电离能使用新旧概率的累计电离势差，再保存新 P：[marker_spi_source.f90:535](/home/ac_desktop/JOREK_code/jorek/particles/examples/marker_spi_source.f90:535)、[marker_spi_source.f90:600](/home/ac_desktop/JOREK_code/jorek/particles/examples/marker_spi_source.f90:600)。时间平均、投影和一步线性化仍属于数值算法，不与 CE 流体逐位等同。

### B.4 辐射、电离能与其它反馈

- CE 主杂质辐射使用当前实际 ne 的 radiation_function_linear，非零条件受温密/杂质下限控制；它与用于 Z/E_ion 的固定测试密度查表不是完全相同调用：[mod_elt_matrix.f90:1179](/home/ac_desktop/JOREK_code/jorek/models/model502/mod_elt_matrix.f90:1179)。
- CE 将平衡概率与逐级电离势求和得到 E_ion(Te)、dE_ion_dT，加入密度/温度历史和输运等联立能量项：[mod_elt_matrix.f90:953](/home/ac_desktop/JOREK_code/jorek/models/model502/mod_elt_matrix.f90:953)、[mod_elt_matrix.f90:1792](/home/ac_desktop/JOREK_code/jorek/models/model502/mod_elt_matrix.f90:1792)。不能概括为“CE 没有电离能损失”。
- marker 把主杂质 E_ion/dE_ion_dT 置零，使用投影的电离/辐射直接项；电子 RHS 为 -(aux_dEion_dt+aux_rad)*dt：[mod_elt_matrix.f90:982](/home/ac_desktop/JOREK_code/jorek/models/model502/mod_elt_matrix.f90:982)、[mod_elt_matrix.f90:1728](/home/ac_desktop/JOREK_code/jorek/models/model502/mod_elt_matrix.f90:1728)。源码中的“Accumulated”注释不能覆盖投影实际 gather 平均与单位定义。
- 非 CE marker 的 get_Lz 按旧荷态加权；低于 1 eV 的辐射门槛与反应温度下限有明确执行顺序：[marker_spi_source.f90:546](/home/ac_desktop/JOREK_code/jorek/particles/examples/marker_spi_source.f90:546)。CE 场分支采用其自身门槛；做数值对照时须登记这些差异。
- 电荷闭合继续作用于电子密度、压力/容量、电阻率、电子—离子交换，不能把荷态差异仅当作一个辐射系数差：[mod_elt_matrix.f90:1081](/home/ac_desktop/JOREK_code/jorek/models/model502/mod_elt_matrix.f90:1081)、[mod_elt_matrix.f90:1164](/home/ac_desktop/JOREK_code/jorek/models/model502/mod_elt_matrix.f90:1164)、[mod_elt_matrix.f90:1292](/home/ac_desktop/JOREK_code/jorek/models/model502/mod_elt_matrix.f90:1292)。影响强弱依赖实际电阻率/温度等配置。
- use_marker 不把所有背景杂质改为非平衡；主杂质之外的背景辐射仍走 CE helper：[mod_elt_matrix.f90:1256](/home/ac_desktop/JOREK_code/jorek/models/model502/mod_elt_matrix.f90:1256)。

### B.5 对真实 SPI 的额外限制

本地 spi_abl_model==3、with_impurities、Ne 分支在 use_marker=false 时另查 CE 平均电荷及温度导数；use_marker=true 则明确写 Z_imp=0 placeholder，再用于局部电子密度/梯度和消融率：[pellet_module.f90:624](/home/ac_desktop/JOREK_code/jorek/models/pellet_module.f90:624)、[pellet_module.f90:684](/home/ac_desktop/JOREK_code/jorek/models/pellet_module.f90:684)、[pellet_module.f90:699](/home/ac_desktop/JOREK_code/jorek/models/pellet_module.f90:699)。

这是当前特定消融分支的实现事实，不应冒充普遍物理规律，也不是 Hu §4 注入对照的原因。未来给 shard 直接传连续云电荷闭合，需要说明是复刻此实现，还是修改该近似；不能静默宣称完全等价。本轮没有认证所有消融模型、nonlocal/rocket 配置或两组 phaseA 的生产构建。

## C. 论文对照与因果解释

文献：[Hu et al., 2021, Collisional-radiative non-equilibrium impurity treatment for JOREK simulations](https://arxiv.org/pdf/2106.12908)，访问日期 2026-10-12。本轮以 v1 的 §4/Figs.2–6 为对照；其中是轴对称 Ar 注入测试，不是我们两组 phaseA Ne SPI 的直接误差认证。

只摘录与本问题有关的结论：Fig.6 强制 CE 的粒子与 CE 流体总体接近；Figs.2–3 中 CE 初始电离冷却可更强，但总体早期冷却偏弱/延迟；Fig.4 的 CE 荷态由同一非平衡运行的温密后处理，不是另一条 CE 运行。上述图像/说明不证明任意网格云与 marker 等价。

基于已核查源码的解释性推断如下：

1. 新生中性物质进入热区，有限速率模型保留逐级电离过程；CE 则直接赋予平衡分布。主要区别是时间闭合而非是否采用空间网格。
2. 在强瞬态中，注入/输运/温度变化时长未必远大于荷态松弛时长，独立荷态不能消去。
3. 辐射为各荷态系数的加权和；在文中工况，保留低荷态阶段提供的早期线辐射改变总冷却。不能推广为“非平衡辐射在所有温度都更强”。
4. 冷却改变 Te，并通过电荷、温度相关电阻率、焦耳热和温度交换形成反馈。由此产生的场/功率差异不是 marker 抽样噪声即可解释。
5. 粒子输运/投影、流体扩散、门槛、密度查表与能量离散仍可贡献数值差异；本轮没有重新运行或量化分离它们。

## D. 对连续云方案的含义（研究建议，不是决定）

平衡闭合的示意条件是 K(Te,ne)f_eq=0、sum f_eq=1；本地 CE helper 的具体过程见前文。

如果每一步从当前 Te/ne 重建荷态场，那么即使该场有值和梯度，也采用了 CE 型闭合。若荷态场从上一状态以有限反应/输运演化，则不是 CE。

推荐候选是独立的各荷态数密度 n_z：

    ∂t n_z + ∇·J_z = Σ_q K_zq(U)n_q + B_z(shard,U).

这是连续非平衡云的拟议关系，不是现有 CE 分支。N、平均电荷、电荷平方矩及梯度由同一状态派生。只保存平均电荷可另研究，但需要反应、辐射与二阶矩闭合，不能默认由当前温度唯一确定。

可以复用 CE helper 作对照或快反应极限诊断，但不将 CE 查表替代强源瞬态的独立荷态推进。未来比较应先固定场、原子表、出生/边界和物理时长，再分离表示、反应与输运误差；本轮不执行该实验、不训练，也不改 JOREK/项目实现/ADR/roadmap。
