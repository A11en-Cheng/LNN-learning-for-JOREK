# phaseA 源时间采样、shard 与出生一手证据（2026-10-09）

本报告只读检查 `/home/ac_desktop/JOREK_code/jorek`，未执行 JOREK、训练或参数更新。检查 HEAD 为 `54c675bcc39c33b082c0c35046e28122dc53b3aa`；`git status --short` 为 ` M communication/eqdsk2jorek.f90` 和 `?? poincare`。生产投影 RCS `v2.25.12.0-1042-gb4165893f-dirty` 不同；本报告证明检查版本的机制，不证明生产 dirty 改动或构建等价。下述行号属于检查版本。源码中有被 gitignore 隐藏的文件，复核需显式路径或 `rg --no-ignore`。

## 时间公式及可检验差异

[mod_fields_linear.f90](/home/ac_desktop/JOREK_code/jorek/particles/mod_fields_linear.f90:68) 首先空间评价 U_now，再评价保存的 deltas。结合 [mod_linear.f90](/home/ac_desktop/JOREK_code/jorek/tools/mod_linear.f90:19)，一阶采样为

`U(t) = U_now - rawD * (time_now-t)/(time_now-time_prev)`。

空间一阶导数同样采样，不从另外两个输出文件估计 rawD。门控是 `t_jorek>0`，且 `abs(time_now-time_prev)>1e-10 s`，且非 static；否则值保持 U_now。t_jorek 正时，时间导数仍由 rawD 除以时间跨度或 t_jorek；t_jorek 非正时不插值且导数为零。二阶重载 [164](/home/ac_desktop/JOREK_code/jorek/particles/mod_fields_linear.f90:164) 用有符号跨度 `>1e-10`，不是 abs；不应将所有重载默认视为完全相同。

`previous+(current-previous)*(t-start)/(end-start)` 仅当 current=U_now、previous=U_now−rawD、start=time_prev、end=time_now 且门控允许时等价。检查开始时 `NativeMarkerSource.advance` 和 `NativeSmallShardReference.births` 均直接作端点差，未表达 static/短跨度门控；位置见 `src/models/native_coupling.py:211–214,591`（实现后行号会移动）。历史窗口不得把任意相邻输出差当 rawD。生成事件后的候选增量可以按受控更新规则成为下次 rawD，但需保持求解/边界处理阶段一致。

[marker_spi_source.f90:350](/home/ac_desktop/JOREK_code/jorek/particles/examples/marker_spi_source.f90:350) 规定 `start=sim_time-residual`、`N=int((target-start)/h_particle)`、`gather=N*h_particle`、`new_residual=target-start-gather`；[486](/home/ac_desktop/JOREK_code/jorek/particles/examples/marker_spi_source.f90:486) 在 `start+(k-1)*h_particle` 采样。残余可把采样推到 field time_prev 之前；该公式不 clamp。出生初速度采样却传 `sim%time`，[435](/home/ac_desktop/JOREK_code/jorek/particles/examples/marker_spi_source.f90:435) → [mod_initialise_particles:537](/home/ac_desktop/JOREK_code/jorek/particles/mod_initialise_particles.f90:537)，不应未经解释改成 particle_start_time。

[mod_jorek_timestepping:568](/home/ac_desktop/JOREK_code/jorek/core/mod_jorek_timestepping.f90:568) 设置 time_prev=sim_time、time_now=sim_time+dt。该跨度不自动等同 marker 子步或 rawD 的文件输出间隔。

### live 分支的 static 实证（后续核查追加）

marker 主程序 [76](/home/ac_desktop/JOREK_code/jorek/particles/examples/marker_spi_source.f90:76) 使用 reader i=-1；[do_read:405](/home/ac_desktop/JOREK_code/jorek/particles/mod_fields_linear.f90:405) 将该 reader 的 fields.static 设 true。全 particles/core 对 `%static =` 的只读搜索未发现清为 false 的 live 路径；stepper:568 只更新时端点。两个已有生产日志也在初始化报告 true：`data/EHL-2/phaseA_dataset_1d22_ang15_massloss_neon_v400/logfile_spitest:1811` 和 injInner 对应 `:1781` 均以 `T` 结束。

因此**检查源码的 live 路径为 static=true，不执行上述时间插值**；static 的名称不表示场在 base 事件之间不更新，只表示子步取当前已有 U。日志直接证明的是启动 flag，不单独证明生产 dirty 构建此后没有额外赋值。接口必须显式绑定 static，不能凭“linear”名称把生产配置默认为 false；一般非static分支仍保留并独立测试。

独立解析例：U_now=10、rawD=4、跨度 4 ns，在 time_prev 取 6、跨度中点取 8、time_prev−1 ns 取 5；若 static 或跨度≤1e−10 s，均取10。选一个不同的保存前文件值99，可直接抓出错误端点差。此例是手算测试候选，本研究没有运行 Python 被测实现。

## OLD 场 shard 次序及 rocket6

[core/mod_jorek_timestepping:410](/home/ac_desktop/JOREK_code/jorek/core/mod_jorek_timestepping.f90:410) 中 solver 成功后先 nonlocal 初始化诊断、update_spi，然后 [425](/home/ac_desktop/JOREK_code/jorek/core/mod_jorek_timestepping.f90:425) 才 update_values/update_deltas。因此 update_spi 读取 OLD 场，而非新候选。

[pellet_module:270](/home/ac_desktop/JOREK_code/jorek/models/pellet_module.f90:270) 将注入口固定坐标的旧速度转成当前 shard 局部 R/phi 速度，推进位置；[315](/home/ac_desktop/JOREK_code/jorek/models/pellet_module.f90:315) 由旧 radius 和旧 ablation 缩小半径并截零；随后在新位置 find_RZ，失败域外将新消融设零并 cycle。新位置诊断 density、Te、psi，计算新消融；[864](/home/ac_desktop/JOREK_code/jorek/models/pellet_module.f90:864) rocket6 使用该诊断，最后 [887](/home/ac_desktop/JOREK_code/jorek/models/pellet_module.f90:887) 更新速度。不能先加速再推进位置，或用新消融缩旧半径。

rocket6：`P_R=1e5*(Te_eV/2000)^1.26*(ne_m3/1e20)^0.4 Pa`；`m_eff = species*central_mass/mu_imp + (1-species)*central_mass*spi_rkt_H_m_mod`；`M=(4π/3)*max(r,1e-5)^3*m_eff*m_u*density_1e20*1e20`。注入口方向加速度为 `P_R*π*r²/M*(cos(delta_phi),0,sin(delta_phi))`，新速度加该值乘物理dt。受力面积是实际r²，质量下限才用max(r,1e−5)。不增加自定速度/加速度 clamp。

目标 ablation3 始于 [624](/home/ac_desktop/JOREK_code/jorek/models/pellet_module.f90:624)，不是前面 ablation2 的 Sergeev 公式。Ne 分支 [684](/home/ac_desktop/JOREK_code/jorek/models/pellet_module.f90:684) 在 use_marker 时使用 Z_imp=0 placeholder，所以局部 ne=max(n_SI−n_imp_SI,0)。令 s=species，消融率为 `(27.0837+tan(1.48709*(1-s)/(1+s)))*N_A*(r*100/.2)^(4/3)*(ne/1e20)^(1/3)*(Te/2000)^(5/3)/(20.183*s+2.0141*(1-s))`。nonlocal 用同公式换入表中的 ne/Te，再取最大。最后 [733](/home/ac_desktop/JOREK_code/jorek/models/pellet_module.f90:733) 在配置启用且 B0>2 时乘 `(2/B0)^.843`。诊断 nonlocal 的 ne 使用 aux 投影平均电荷 [1104](/home/ac_desktop/JOREK_code/jorek/diagnostics/mod_integrals3D.f90:1104)，与局部 placeholder 不同。

## nonlocal 是空间诊断，非局部拟合

[mod_integrals3D:1740](/home/ac_desktop/JOREK_code/jorek/diagnostics/mod_integrals3D.f90:1740) 在归一化ψ∈[0,1)内按 n_flux 壳分组 Gauss 体积 `dV=R*Jac*wst*delta_phi`，积分 V、Ne、NeTe。[2530](/home/ac_desktop/JOREK_code/jorek/diagnostics/mod_integrals3D.f90:2530) 得 `Te_shell=∫neTe/∫ne`、`ne_shell=∫ne/∫1`，空壳/非正粒子数失败。按线性 nl_Psi 网格重采样，边缘可外推；不是对每个 shard 查本地 Te/ne。WITH_TiTe + impurities 分支密度用 `r0+alpha_e*rimp0`，alpha_e 的计算仍需依该版本诊断代码，不可简单用 marker 当前电荷替换。ψ坐标依据 [839](/home/ac_desktop/JOREK_code/jorek/diagnostics/mod_integrals3D.f90:839) 受 axisymmetric/s_norm 等分支控制。

[pellet_module:464](/home/ac_desktop/JOREK_code/jorek/models/pellet_module.f90:464) 查表先最近ψ点，若其离轴更远则索引减1，最内侧下溢改1；按此表取温度/密度，再取 `max(local_ablation,nonlocal_ablation)`。表在 int3d_new 诊断更新，core:414 只在首值零时初始化；应保存/重建其真实诊断时序，不默认每 base 重算表。

必要可核验输入：完整 OLD 场/网格及体积权重，ψ轴/边与Z依赖归一化规则，ψ诊断坐标分支，n_flux/n_nonlocal_array，构建 WITH_TiTe/WITH_Impurities，charge/alpha_e 所需数据，表最近诊断时刻或缓存。缺少这些应明确失败；有这些则不应凭 nonlocal 名称声称原则不可恢复。

## 生产出生与独立密度

[marker:392](/home/ac_desktop/JOREK_code/jorek/particles/examples/marker_spi_source.f90:392) 用旧 shard 位置、正杂质组分以及 `max(radius*ns_radius_ratio,ns_radius_min)` 的5倍界定 R/Z出生框；总源率为Σ旧abl*species，MPI平均。初始 nominal weight 为总spi_quantity/n_particles，[96](/home/ac_desktop/JOREK_code/jorek/particles/examples/marker_spi_source.f90:96)。出生数 `int(rate_local*gather/nominal_weight)`，若 rate_local*gather>1e14 则至少1。所有出生先于 marker 子步。

[mod_initialise_particles:499](/home/ac_desktop/JOREK_code/jorek/particles/mod_initialise_particles.f90:499) 用圆柱均匀体积proposal、域内检索和 total_imp_source 拒绝采样；[mod_sampling:74](/home/ac_desktop/JOREK_code/jorek/tools/mod_sampling.f90:74) 的 R 为 `sqrt(u*(Rmax²−Rmin²)+Rmin²)`。接受概率 [marker:809](/home/ac_desktop/JOREK_code/jorek/particles/examples/marker_spi_source.f90:809) 是 `max(source_tmp*dv/P_norm,0)`，dv=.01²*R_geo*(2π/n_plane)，源码没有上限clamp。接受后权重 `real(rate_local*gather/n_assigned,4)`，中性P_imp(0)=1。拒绝后重新抽样，生命周期/空槽必须保留。随机种子由 [417](/home/ac_desktop/JOREK_code/jorek/particles/mod_initialise_particles.f90:417) 运行时种子入口产生并按MPI/线程拆流；缺生产seed、拆流和消费位置不能宣称逐粒子生产轨迹一致，即便同分布机制成立。

total_imp_source 路径 [mod_injection_source:360](/home/ac_desktop/JOREK_code/jorek/models/mod_injection_source.f90:360) 包含 spi_num_vol 数值体积、宽度、toroidal/minor-radius等核设置，不能替换为任意单位积分Gaussian，更不能让独立 density 经过 marker 五通道滤波器。检查开始时小型参考 shard 每活shard固定一粒的出生和固定width密度核均是已声明近似，不能认证生产。

两个实际 `jorek_namelist_spitest`：153行n_flux=40；487–488 nonlocal=.t.，n_nonlocal_array=1000；501 ns_deltaphi=1e3；507–508 ablation3/rocket6；516 ratio1.4；517 v400 min=.05m、injInner=.01m；529 n_particles=1e6。此处主 `jorek_namelist` 含NUL且不是已有审计绑定的配置；应使用实际生产绑定的 spitest 与内部H5有效配置共同核查。

## 复核与边界

```bash
git -C /home/ac_desktop/JOREK_code/jorek rev-parse HEAD
git -C /home/ac_desktop/JOREK_code/jorek status --short
rg -n 'time_prev|time_now|static' /home/ac_desktop/JOREK_code/jorek/particles/mod_fields_linear.f90
rg -n --no-ignore 'nl_avg_Te|nl_avg_ne' /home/ac_desktop/JOREK_code/jorek/diagnostics/mod_integrals3D.f90
```

本研究只完成源码阅读，未执行新模型、机制测试或真实闭环验证；没有训练资格/C0结论。所列解析数值没有数值误差（作为浮点测试时应显式取1e−12级相对容差并按单位另设绝对容差）。读取成本为少量文本搜索，无全云裁剪或生产求解。
