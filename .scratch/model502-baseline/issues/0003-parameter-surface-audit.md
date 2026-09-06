# 0003 mod_elt_matrix 参量面审计

Blocked by: #0001

交付：审计 `/home/ac_desktop/JOREK_code/jorek/models/model502/mod_elt_matrix(_fft).f90` 装配九场方程所需的
全部零维参量与闭包，产出清单，每项标注"已导出（h5 键）/需导出（JOREK 变量名）/由网络吸收（hyper-*）"；
明确高阶截断项（`eta_num/visco_num/ZK_par_num` 及 T 依赖）不在条件集内。

验收：清单落盘为 `report/model502-parameter-surface.md`，与 h5 实际数据集逐一核对一致。

状态：done。清单 = `report/model502-parameter-surface.json`（单一来源）+ 同名 .md（人工视图）；
`tests/test_parameter_surface.py` 用真实 jorek002710.h5 核对全部 exported 键；
高阶截断三项（eta_num/visco_num/ZK_par_num + T 依赖）已定位到行号并标注 absorbed；
#0004 导出清单（ZK_i/e_par、D_perp、ZK_i/e_perp、visco_par_par、theta/zeta、T_0/Ti_0/Te_0）已列明。
