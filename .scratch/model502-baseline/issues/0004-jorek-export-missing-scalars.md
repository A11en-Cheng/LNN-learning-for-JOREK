# 0004 JOREK 侧补导出缺失标量（外部依赖）

Blocked by: #0003

交付：在 `mod_export_restart.f90` 为缺失的基底标量（`ZK_i_par/ZK_e_par`、`D_perp` 基准等，清单以 #0003 为准）
各加一行 `HDF5_real_saving`；重编译并重新输出至少一个 2D run 的样例帧。

验收：新 h5 含清单内全部标量且数值与 namelist 一致。标签：ready-for-human（JOREK/Fortran 侧改动）。

状态：superseded（被 #0005 取代）。缺失标量改从 run namelist + JOREK 默认值读取（ADR-0007），
无需修改 JOREK 写出代码；`visco_par_par` 未设定时按 `preset_parameters.f90` 默认 0.0。
