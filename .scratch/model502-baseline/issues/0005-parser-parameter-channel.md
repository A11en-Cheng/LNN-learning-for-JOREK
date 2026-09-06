# 0005 parser 参数读取通道

Blocked by: #0001

交付：parser 读取全部零维参数（现有 `eta/visco/visco_par/tstep/central_density/central_mass`；#0004 落地后再补
ZK/D_perp）→ condition vector + 同尺寸 mask；缺失=0 且 mask=false；删除 1.0 回退冒充逻辑。

验收：参数解析测试通过；每个参数带"来自 h5 / 缺失"标志；#0004 未落地前缺失项保持 mask=false 而非报错。

