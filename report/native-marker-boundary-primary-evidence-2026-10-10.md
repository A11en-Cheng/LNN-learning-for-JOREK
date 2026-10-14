# Marker 外边界：JOREK 一手源码证据

范围：只读核查本机 JOREK，未运行训练、粒子推进或大云回放；仅新增本报告。主代理负责几何退出及末段权重实现。本报告不确认 slot973 的具体退出点，也不把新近似视为原生固定步真值。按 `diagnosing-bugs` 的证据分层，以下明确区分确定源码语义与测试建议；复现和修复不在本侧审查范围。

## 1. 原生顺序与最后完整步

- [marker_spi_source.f90:482](/home/ac_desktop/JOREK_code/jorek/particles/examples/marker_spi_source.f90:482)，482–490：每步开始遇到 `i_elm<=0` 才退出，随后保存旧 x/st/element，并在旧位置采样。新位置的失效不会追溯撤销本步。
- [marker_spi_source.f90:535](/home/ac_desktop/JOREK_code/jorek/particles/examples/marker_spi_source.f90:535)，535–537、548、558–581：三个密度/电荷矩和辐射使用完整 `timesteps`，电荷更新也用完整 `timesteps`。矩和辐射使用旧 P；598–603 归一化新 P，并由新旧 P 差计算电离能。
- [marker_spi_source.f90:616](/home/ac_desktop/JOREK_code/jorek/particles/examples/marker_spi_source.f90:616)，616–642：基函数用旧 st，环向因子用旧 phi，RHS 加到 `i_elm_old`；沉积发生在 push 前。655–661 才采速度、push、调用 nearby。调用者不依据 `ifail` 重算时间、退回电荷或缩短沉积。
- [mod_fieldline_euler.f90:67](/home/ac_desktop/JOREK_code/jorek/particles/pushers/mod_fieldline_euler.f90:67)，67–77：完整 dt 的方向 AB push 更新 R/Z/phi，且定位前已写 `V_prev=V+V_ext`。lost 后 locator 不接收、重置或重新采样速度。因此保留的是旧位置采得的本步合速度，不是壁面速度；也没有在此退出链中将宏粒子 weight 或 i_life 改写。
- [marker_spi_source.f90:443](/home/ac_desktop/JOREK_code/jorek/particles/examples/marker_spi_source.f90:443) 和 [675](/home/ac_desktop/JOREK_code/jorek/particles/examples/marker_spi_source.f90:675)：gather 为 `n_steps*timesteps`，最后除统一 gather；不是逐粒子存活时间平均。350–355 的余数调度同样没有单粒子边界事件时间。

结论：一个步初仍活、步末出界的 marker，在 JOREK 此调用链中已完成最后一个完整 dt 的电荷更新和旧位置五通道沉积；下一步才跳过。

## 2. lost 标记并不把 x 截回边界

[mod_find_rz_nearby.f90:50](/home/ac_desktop/JOREK_code/jorek/grids/grid_utils/mod_find_rz_nearby.f90:50)，50–56：目标 R_new/Z_new 为 `intent(in)`，输出只有 s/t、element 和 ifail。117–123 把 Newton 试探限制到参考单元边，然后选择邻居。

[mod_find_rz_nearby.f90:132](/home/ac_desktop/JOREK_code/jorek/grids/grid_utils/mod_find_rz_nearby.f90:132)，132–140：零邻居返回 `i_elm_new=-i_elm_tmp`、边上的 s/t、`ifail=-1`。这里的 `i_elm_tmp` 是跨越过程中最后一个单元，不一定是调用时的旧单元。135 虽插值边界 R/Z，却只覆盖局部 `x_new`，没有写回输入 R_new/Z_new。

因此正常零邻居 lost 的持久状态是：负最后单元号、边界 st、完整 push 后的目标 x/phi、已更新的 V_prev/P。不能断言 lost x 与边界 st 映射一致。[mod_particle_types.f90:60](/home/ac_desktop/JOREK_code/jorek/particles/mod_particle_types.f90:60) 明确负 element 表示该单元边缘的 lost；711–723 将 x/st/element/V_prev 原样放入序列化数组，没有在此统一二者。

其他失败不是同一种边界事件：无效 hint 82–89 直接 global；NaN/八轮耗尽 165–185 global，成功标 ifail=2/3；global 未找到单元可为 element=0/ifail=99（[find_RZ.f90:76](/home/ac_desktop/JOREK_code/jorek/grids/grid_utils/find_RZ.f90:76)，76–98）。DEBUG 几何不连续也可返回 element=0（nearby 144–150）。这些不能单独证明真实外边界或提供可靠边界 st。

## 3. saved neighbours 的边、零、负与角点

- 边号为 1:t=0，2:s=1，3:t=1，4:s=0：[mod_neighbours.f90:125](/home/ac_desktop/JOREK_code/jorek/grids/mod_neighbours.f90:125)，125–138。positive 为直接、1-based 单元索引。
- [mod_neighbours.f90:151](/home/ac_desktop/JOREK_code/jorek/grids/mod_neighbours.f90:151)，151–166：按严格比较选象限，不看本次速度的法向分量；精确角点 `(0,0)/(1,0)/(1,1)/(0,1)` 分别选边 `4/2/3/3`。不是同时试两边或任意挑一条。nearby 的 `fact>=1-1e-12` 也可能让恰落外边的目标进入 lost，不能统一约定“边界端点必为 live”。
- 正邻居须有互反边；朝向由共享 vertex 编号确定，`co` 时沿边参数反转：[mod_neighbours.f90:90](/home/ac_desktop/JOREK_code/jorek/grids/mod_neighbours.f90:90)，90–118、169–199。互反失败会打印错误并把 i_to 置零（170–173），之后 nearby 同样标 lost；故“返回零”不足以区别坏拓扑和物理退出。
- [mod_neighbours.f90:255](/home/ac_desktop/JOREK_code/jorek/grids/mod_neighbours.f90:255)，255–288：初始化未找到的邻居为零；两端 boundary 标记非零的边及重合节点边可保留缺邻居。293–309 随后将退化轴侧、及该单元其他仍为零的侧标成 -1，强制 search。
- 负邻居绝不是直接 lost：[mod_find_rz_nearby.f90:124](/home/ac_desktop/JOREK_code/jorek/grids/grid_utils/mod_find_rz_nearby.f90:124)，124–131 调 global；失败后才置零并进入边界 lost 分支。成功后源码仍继续 nearby，并非干净的 terminal global return。现有 Python 的 terminal-global seam 是已登记的差异，不应把 negative 当几何外边界。
- saved 表直接写/读 element 的邻接数组，而非重新推断：[mod_export_restart.f90:544](/home/ac_desktop/JOREK_code/jorek/communication/mod_export_restart.f90:544)、689–690；[mod_import_restart.f90:1517](/home/ac_desktop/JOREK_code/jorek/communication/mod_import_restart.f90:1517)、1528。

## 4. 末段截到边界：必须登记的数值近似

若新模式接受末段 `h_exit<dt` 并以此更新电荷/沉积，则它改变上述完整末步的原生 residence-time 离散；若将 x/phi 保存为边界事件位置，也改变原生 lost x 保留 overshoot 的约定。即使顺序仍是旧位置 charge/deposit → push → lost，也不能宣称完整 JOREK 固定步复刻。

nearby 的 Newton 边界落点是搜索路径中的参考坐标落点，不是按真实 pusher 时间参数求出的首个物理退出事件。尤其非线性几何及变间隔方向 pusher 下，缩短 h 后的方向系数亦变化；不能不经验证就将完整步的边界距离比例视为退出时间比例。此为测试风险，不是对主实现算法的判断。

χ 超限、global 失败、不互反 positive 邻居或 NaN 都不能替代明确的真实外边几何证据。局部修正应继续保留失败拒绝路径，而非将所有无法细化的 marker 判 lost。现有 local v1 默认、原生公式和 density floor 不应被本次事件近似静默改变。

## 5. 推荐测试审查要点（未在本轮执行）

1. 原生完整步 oracle：平直外边、常速度，步初内部/步末外部；断言五通道已在旧位置完整 dt 累积，P 用完整 dt 更新，lost 为负最后单元，st 在边上，x 仍为 overshoot，V_prev 为本步采样合速度；下一步零贡献。
2. 新末段：解析已知退出时间，三个矩及辐射按 h_exit，电离能按实际 h_exit 的电荷更新得到，不能仅缩放完整 dt 已 clamp/renormalize 后的 q−p。宏粒子 weight 不应被误当时间权重改写。统一 gather 分母 G 保持原窗；常权重的未归一化 number 矩时间平均为 w*h_exit/G，实际 N_imp 投影 RHS 另乘旧位置 HH/size/HZ 并除 n_norm（源代码634行）。
3. 入/出/切向、恰落边界、零时刻接触、角点四种 tie，以及角点微小双向扰动；审查事件规则与原生 tie 差异，避免无穷零长度循环或重复沉积。
4. 内部正邻居、反向共享边、跨多单元再退出：不得在内部边吸收；lost 单元须是最终退出单元。曲边/非凸路径检查首个真实交点，而非包围盒或端点是否在外的替代。
5. 负邻居强制 fallback、退化轴侧、坏 reciprocal positive、NaN/迭代失败：不得误归真实外边界；预算不足且无退出证据仍拒绝提交。
6. 单粒子退出不改变其他粒子的 base/gather 时钟；后续窗不再给 lost 粒子电荷/沉积，序列化 resume 与连续同臂在 x/st/V/P/history/诊断/RNG 上 exact，槽重用后 lifetime 与 history 正确重置。
7. 新事件策略有独立配置指纹；默认 native_fixed/local v1 未选此策略时保持既有行为。边界近似的 checkpoint 不得伪装为旧配置或原生 HDF continuation。

## 一手文件绑定（SHA-256）

JOREK 根：`/home/ac_desktop/JOREK_code/jorek`；本轮只读，源码行号以此副本为准。

| 文件 | SHA-256 |
|---|---|
| particles/examples/marker_spi_source.f90 | `4f6e832b107569d7a8a0155ce5995d04d59cc50771d256cdf99c1e692318991c` |
| particles/pushers/mod_fieldline_euler.f90 | `95b0635c67e9499c563547fefa74a247d21dc6feec7fd06a6141dfe3d72467a9` |
| grids/grid_utils/mod_find_rz_nearby.f90 | `b1cbbd3b8cbc3250a81132ac9c8f638561d60e832c0113fa13dd984f8999dcf4` |
| grids/mod_neighbours.f90 | `9e1be99b70f3319db5677b1e77d0ab25bbffbf3435ff2d9e1c7c743d86b6f203` |
| particles/mod_particle_types.f90 | `14fb42d89cf6e0ed55de5bfda4122aa230da4c67c1f4385abc92e1db9197b421` |
| grids/grid_utils/find_RZ.f90 | `95835d8183228a5540dcaa6ae1249d8e509163de1f462aaa1e1fe2f3e01578cf` |
| communication/mod_export_restart.f90 | `d540c7a240a3b6ec94f30d3a6a4b4d83e4c2964034a4bb6da8eaa2b73e7f1f90` |
| communication/mod_import_restart.f90 | `cded11509b8b9a626f6bcbacfd1d6c42f2a16a7fa596b9a50105420eea454647` |
