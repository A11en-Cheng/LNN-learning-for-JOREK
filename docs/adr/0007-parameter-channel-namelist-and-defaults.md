# ADR-0007：零维参数通道：namelist 优先、JOREK 默认值填充、派生量同逻辑计算

状态：已接受（2026-09-06）

## 背景

装配 model502 九场方程需要的零维参数大部分不在 `jorek*.h5` 里，而在 run 目录的 namelist
（`jorek_namelist_spitest` / `jorek_namelist`）中，且数值计算中不变。用户在 grilling 中决定：
缺失的参数采用 JOREK 代码默认值和逻辑填充，而不是协议 v1 的"0 + mask=false"。另有若干量
（`time_evol_theta/zeta`）是装配时由 `time_evol_scheme` 与 `tstep` 派生出来的，必须复刻相同计算逻辑。

## 决定

1. 零维参数按优先级读取：`jorek*.h5` 标量 → run 目录 namelist（`jorek_namelist_spitest` 优先，
   回退 `jorek_namelist`）→ JOREK 默认值（以 `preset_parameters.f90` 为准，如
   `visco_par_par=0.0`、`ZK_i_par=ZK_e_par=1.0`、`D_perp/ZK_i_perp/ZK_e_perp=(1e-5,0,0,99,99,0)`、
   `T_0=1e-6, Ti_0=Te_0=5e-7`、`time_evol_scheme='Crank-Nicholson'`）。
2. 缺失即用默认值填充，并给每一项打 `provenance ∈ {h5, namelist, default, derived}`；
   消费方以 provenance 判断真伪，不再以"值是否为 0/1.0"判断。此条修订 ADR-0001 中
   "0 不是物理回退值"的表述：默认值填充必须伴随 provenance，禁止无标记填充。
3. 装配期派生的量必须在通道里用 JOREK 同款逻辑计算：
   - `time_evol_theta/zeta` 由 `time_evol_scheme` 经 `update_time_evol_params.f90` 逻辑派生
     （implicit Euler→(1,0)，Crank-Nicholson→(0.5,0)，Gears→(1,0.5)）；
   - 变 dt 下 zeta 再按 `zeta · 2·tstep/(tstep+tstep_prev)` 重标定（`mod_elt_matrix.f90:225`，
     由 #0006 在序列上下文接线）。
4. 高阶截断与状态依赖闭包（hyper-*、`eta_T_ohm`、`visco_T`、`ZK_*_T` 等）不在通道内
   （ADR-0005/0006），由网络隐式吸收。

## 后果

- `src/data/parameter_channel.py` 是条件向量的唯一出口；parser 直接调用它，不再维护私有参数索引。
- #0004（JOREK 侧 Fortran 补导出）被本通道取代而关闭：缺少的量从 namelist + 默认值获得，
  无需改 JOREK 写出代码。
- 条件向量随 namelist 变化，checkpoint/provenance 必须记录所用 namelist 的 hash。

