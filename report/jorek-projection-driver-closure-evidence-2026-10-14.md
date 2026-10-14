# JOREK：投影作为驱动源、marker 速度与联合状态闭合（2026-10-14）

## 1. 结论与术语

若“投影场”指完整的 Ne 11 荷态密度场，连同流体、shard、控制和必要历史，可构造连续云联合动力学；若指现有 projections*.h5 的五通道，它是合法的流体反馈接口，但不是已证明闭合的云状态。本文区分源码事实与建模推论，不批准 ADR-0024，不修改现行 ADR。

## 2. 现有五通道究竟存什么（源码事实）

marker 子步从完整 P_imp 求两个电荷矩和总原子数，乘权重及子步时长：[marker_spi_source.f90:531](/home/ac_desktop/JOREK_code/jorek/particles/examples/marker_spi_source.f90:531)。各荷态辐射逐项求和，有限速率电离/复合使用相邻荷态及 ADAS SCD/ACD：[marker_spi_source.f90:543](/home/ac_desktop/JOREK_code/jorek/particles/examples/marker_spi_source.f90:543)。电离能变化由更新前后 P_imp 的差乘累计电离势及权重得到：[marker_spi_source.f90:598](/home/ac_desktop/JOREK_code/jorek/particles/examples/marker_spi_source.f90:598)。

| 通道 | 粒子侧生成内容（省略公共 FEM 因子） |
|---|---|
| 1 | 电离能变化，按 t_norm/E_norm 换算 |
| 2 | 子步累计辐射能，按 t_norm/E_norm 换算 |
| 3 | 权重 × Σ z²P_imp(z) × 子步时长 / n_norm |
| 4 | 权重 × Σ zP_imp(z) × 子步时长 / n_norm |
| 5 | 权重 × 子步时长 / n_norm |

每项乘极向原生基函数、degree 对应 size、环向模函数，累加到旧位置单元；随后全部五通道除以实际 gather-time。因此前两项成为窗口功率，后三项成为窗口平均矩，而不是五个瞬时终态变量。[原生沉积](/home/ac_desktop/JOREK_code/jorek/particles/examples/marker_spi_source.f90:614)、[时间平均](/home/ac_desktop/JOREK_code/jorek/particles/examples/marker_spi_source.f90:669)

## 3. 投影后的 values 不是粒子载荷

projection 对载荷装配/MPI 汇总后求解 A x=b，再写入节点系数；构建时允许过滤参数。因此不能将沉积数组直接称为最终投影场，也不能把窗口平均终值恢复成每个 marker 的状态。[求解及节点写入](/home/ac_desktop/JOREK_code/jorek/particles/diagnostics/mod_uncoupled_projection.f90:224)、[投影过滤配置](/home/ac_desktop/JOREK_code/jorek/particles/examples/marker_spi_source.f90:145)

## 4. 流体如何消费五通道（源码事实）

model502 读取 1/2 通道，作为电子能量方程损失；3/4 通道除以带下限的通道 5，形成杂质电荷均值与二阶电荷矩均值，并读取投影增量历史。后者继续参与电子密度、闭合及总有效电荷计算。它们不是统一添加到九场的五个 RHS。[五通道消费及历史](/home/ac_desktop/JOREK_code/jorek/models/model502/mod_elt_matrix.f90:989)、[电子密度](/home/ac_desktop/JOREK_code/jorek/models/model502/mod_elt_matrix.f90:1081)、[有效电荷闭合](/home/ac_desktop/JOREK_code/jorek/models/model502/mod_elt_matrix.f90:1128)、[电子能量项](/home/ac_desktop/JOREK_code/jorek/models/model502/mod_elt_matrix.f90:1728)

另外，total_imp_source 仍独立生成注入源，source_imp 进入杂质密度方程；通道 5 是数密度矩，不是原子出生率或密度注入率。[独立密度源](/home/ac_desktop/JOREK_code/jorek/models/model502/mod_elt_matrix.f90:1214)、[杂质密度 RHS](/home/ac_desktop/JOREK_code/jorek/models/model502/mod_elt_matrix.f90:1717)

## 5. 云也读取流体和反馈（源码事实）

marker 根据当前局域流体温度、背景/杂质密度及投影 4/5 的比值形成电子密度，再调用原子率。代码执行 n_e=n_rho+Z_imp_tmp*n_rho_imp；使用粒子自身平均电荷的替代写法是注释，并非当前执行分支。[局域采样及电子密度](/home/ac_desktop/JOREK_code/jorek/particles/examples/marker_spi_source.f90:492)

所以反馈不是只从云单向流向流体。新的连续表示须明确电子密度闭合是沿用流体 rho_imp 和云平均电荷，还是改变密度表示；不能无说明地同时添加完整云电荷密度和同一物质的流体贡献而重复计算。



## 6. 速度与粒子状态（主代理独立核查）

marker 的速度不是只有平行速度。令 \(\Phi_J\) 为原生流函数 var_u，\(v_{\parallel,J}\) 为原生 var_Vpar，\(t_{\rm norm}\) 为 JOREK 时间换算到秒的因子。源码以圆柱物理分量顺序 (R,Z,φ) 计算：

\[
\mathbf B=R^{-1}(\partial_Z\psi,-\partial_R\psi,F_0),\qquad
\mathbf v_{\rm MHD}=
\frac{R}{t_{\rm norm}}(-\partial_Z\Phi_J,\partial_R\Phi_J,0)
+\frac{v_{\parallel,J}}{t_{\rm norm}}\mathbf B .
\]

第一项来自极向流函数，第二项沿磁场。原生 v_par 是乘磁场的因子，不是未经换算的沿单位磁场方向的 m/s 标量。calc_VBpsiU 的输出 V 虽有“Parallel velocity”注释，但实际包含两项，应以执行代码为准。[完整速度公式](/home/ac_desktop/JOREK_code/jorek/particles/mod_fields.f90:284)、[局域原生场及梯度求值](/home/ac_desktop/JOREK_code/jorek/particles/mod_fields.f90:253)

marker_spi_source 还计算附加速度 V_ext，并以 V_add 传入推进器：

- diff_diffusive_flux 为 false：用流体杂质密度梯度除以带下限的流体密度；
- 为 true：用投影杂质数密度与流体杂质数密度的梯度差，除以带下限的密度差；
- 将所得梯度尺度乘 -get_dperp(psi_norm)/t_norm，得到 SI 附加速度。

因此默认 flag 为 false 不等于关闭扩散；是否为零还取决于扩散参数和局域梯度。这里没有逐 marker 的独立随机热速度或牛顿动力学。[梯度分支](/home/ac_desktop/JOREK_code/jorek/particles/examples/marker_spi_source.f90:496)、[附加速度](/home/ac_desktop/JOREK_code/jorek/particles/examples/marker_spi_source.f90:515)、[实际推进调用](/home/ac_desktop/JOREK_code/jorek/particles/examples/marker_spi_source.f90:655)、[flag 默认值](/home/ac_desktop/JOREK_code/jorek/models/preset_parameters.f90:830)

particle_marker 保存位置、单元及 s/t、权重、身份/出生信息、V_prev(3) 和完整 P_imp(:)。它不同于持有独立平行速度标量 v 的 particle_fieldline 类型。[粒子状态](/home/ac_desktop/JOREK_code/jorek/particles/mod_particle_types.f90:53)

推进器按当前总速度及上一总速度的方向更新 R/Z/φ，并保存 V_prev=V+V_ext。严格按该实现，位移使用
\(|V_{\rm new}|(1.5\hat V_{\rm new}-0.5\hat V_{\rm prev})\Delta t\)，不是一般情况下的标准 \( (1.5V_{\rm new}-0.5V_{\rm prev})\Delta t\)。这里记录实际数值历史，不宣称通用变步长 AB2 保证。[推进器](/home/ac_desktop/JOREK_code/jorek/particles/pushers/mod_fieldline_euler.f90:56)

出生时从局域流体速度初始化 V_prev，全部荷态概率置于中性态；这里不是赋予 shard 的飞行速度。[出生初始化](/home/ac_desktop/JOREK_code/jorek/particles/mod_initialise_particles.f90:530)

n=0 轴对称场也不意味着物理环向速度为零，或粒子没有 R/Z 位移。Hu 等 2021 的 §2.1 同样采用共同流体速度，并包含极向流与平行流；论文当时没有包含额外扩散，不能据此忽略当前本地代码的 V_ext。[论文原文](https://arxiv.org/pdf/2106.12908)

## 7. 两种“投影”必须区分（建模推论）

marker 位置是随流移动的拉格朗日记录。沉积载荷和投影求解后的 FEM 节点系数则属于固定网格表示；五通道不是随某一个 marker 搬动的独立物质对象。若“Lagrange”指粒子随流描述，适用于 marker 而不是整个反馈场；若指有限元插值类别，本路径仍须按原生 cubic 完整自由度定义，不能换成仅节点标量的普通 Lagrange 插值。

共同速度假设为连续云方案提供了依据：将每个荷态的原子数密度 \(c_z(x,t)\) 作为网格状态，在共同速度、局域有限速率反应和空间解析条件下构造：

\[
\partial_t c_z+\nabla\!\cdot(c_z\mathbf v_{\rm adv})
=(K(U,C,\mu)C)_z+b_z(P_{\rm shard},U,\mu,\mathrm{controls}) .
\]

散度应使用真实圆柱几何或等价 FEM 弱形式；速度包括上述所选输运分支，不只是 v_par。若目标是复刻 JOREK 离散映射，还要保留其反馈相位和数值历史。连续模型可以不逐原子维护独立速度；这不意味着有限网格与随机 marker 数值逐位等价，也不意味着原生 V_prev 可在离散复刻中直接删除。

现有五通道反馈则是另一种更强压缩：总数和两个电荷矩不能决定 Ne 的 11 个荷态密度；两个窗口功率摘要也没有一般性地恢复完整荷态、终态空间分布与出生历史。

一个只针对三个电荷矩的解析反例：在 z=0,1,2,3 上，归一化分布
\(a=(0.25,0.25,0.25,0.25)\) 与 \(b=(0.20,0.40,0.10,0.30)\)
具有同样总数 1、平均电荷 1.5、二阶矩 3.5，却具有不同荷态分布，一般电荷分辨反应系数会区别对待它们。这个例子不宣称另外两个功率通道也相同，更不是实际 ADAS 误差试验。五通道完备性仍需独立闭合论证，不能只由通道数或训练收敛断言。

## 8. 对神经网络的结论与建议

**给定合法反馈可以驱动一个场步；只给现有五通道反馈，不能据此保证自主源—场闭合。**

当前 joint_learning 路径已经把投影、投影历史、独立密度源及合法时钟编码进场网络，故不能说旧代码完全没有源驱动。这里的缺口是让可推进源状态进入联合演化，而不只是消费源摘要。[当前接线](/home/ac_desktop/LNN-learning-for-JOREK/src/models/native_coupling.py:871)、[编码契约](/home/ac_desktop/LNN-learning-for-JOREK/src/models/native_source_features.py:120)

建议的状态和接口：

1. 维护完整原生流体 U、解析尺度内的 C=(c0,…,c10)、shard 状态，以及选定时间离散/事件/学习记忆。
2. 从当前候选 U 计算局域温密、速度与 ADAS 系数，从 shard 和场计算出生，共同推进 C、shard 与 U；不能把起点投影冻结为全区间外部驱动。
3. 从同一 C 的荷态矩、反应电离能变化、辐射与窗口累计构造五通道，执行所选 FEM 投影；另保留独立密度注入源。输运/出生导致的全部 C 变化不能被算成电离能变化。
4. 复刻 JOREK 交错映射时保留合法投影窗口与历史；若采用同步联合动力学，明确这是新模型，重新定义内部阶段与记录监督算子。

即：云状态是“能继续演化的材料与荷态”，投影是“该状态/过程对流体的反馈接口”。将 11 荷态场投影到网格作为可推进状态有物理结构依据；将现有五通道投影当作完整云状态没有自动闭合保证。网络可以学习近似闭合，但应标作近似。结构上的因果闭环也不自动证明精度、守恒、任意大步或长期稳定。

## 9. 本轮验证范围

只读源码/论文；新增本证据报告。未修改模型、ADR、JOREK 或训练数据；未训练、运行 JOREK、执行真实云基准或回归测试。三个电荷矩反例仅使用标准库手算复核。

本次环境未找到以前报告引用的两组 phaseA namelist 路径，因而没有新认证两组 run 的具体 diff_diffusive_flux/扩散参数或运行二进制；相关结论按源码条件分支陈述。

JOREK HEAD：54c675bcc39c33b082c0c35046e28122dc53b3aa。本轮读取的下列文件没有显示本地改动；SHA-256：

| 文件（JOREK 根目录下） | SHA-256 |
|---|---|
| particles/mod_fields.f90 | 55f5b60aca9036d4c9c807f488e5c70e8a4887d3c356911333a20071a9fee232 |
| particles/pushers/mod_fieldline_euler.f90 | 95b0635c67e9499c563547fefa74a247d21dc6feec7fd06a6141dfe3d72467a9 |
| particles/examples/marker_spi_source.f90 | 4f6e832b107569d7a8a0155ce5995d04d59cc50771d256cdf99c1e692318991c |
