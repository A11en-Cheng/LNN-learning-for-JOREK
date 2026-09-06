# 0003 mod_elt_matrix 参量面审计

Blocked by: #0001

交付：审计 `/home/ac_desktop/JOREK_code/jorek/models/model502/mod_elt_matrix(_fft).f90` 装配九场方程所需的
全部零维参量与闭包，产出清单，每项标注"已导出（h5 键）/需导出（JOREK 变量名）/由网络吸收（hyper-*）"；
明确高阶截断项（`eta_num/visco_num/ZK_par_num` 及 T 依赖）不在条件集内。

验收：清单落盘为 `report/model502-parameter-surface.md`，与 h5 实际数据集逐一核对一致。

