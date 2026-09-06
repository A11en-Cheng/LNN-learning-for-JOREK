# 0005 parser 参数读取通道

Blocked by: #0001

交付：parser 读取全部零维参数（现有 `eta/visco/visco_par/tstep/central_density/central_mass`；#0004 落地后再补
ZK/D_perp）→ condition vector + 同尺寸 mask；缺失=0 且 mask=false；删除 1.0 回退冒充逻辑。

验收：参数解析测试通过；每个参数带"来自 h5 / 缺失"标志；#0004 未落地前缺失项保持 mask=false 而非报错。

状态：done（按 ADR-0007 调整）。`src/data/namelist.py`（Fortran namelist 子集解析）+
`src/data/parameter_channel.py`（h5→namelist→默认值，逐项 provenance，theta/zeta 派生逻辑与 zeta 变 dt
重标定）；parser 集成并把 provenance 编码为 long tensor（h5/namelist/default/derived）。
38 tests 全绿；真实数据验证 ZK_i_par=1.6666e5、ZK_e_par=4.1037e6、visco_par_par=0（default）等。
