# ADR-0004：EHL-2 数据选择与两阶段训练顺序

状态：已接受（2026-09-06）

## 背景

本地 `data/EHL-2/` 已有 12 个 JOREK run（neon/massloss 系列，每 run 约 90–680 帧，共 2864 个 `part0*.h5`）。
用户在 grilling 中确认：只有 `ntor11` 开头的 run 采用 11 个环向分量（n_tor=11），其余全部是 ntor_1 纯 2D；
每个 run 的有效数据从 `part0*.h5` 所引用的最早一个 `jorek0*.h5` 开始；`part0*.h5` 存有粒子/源项信息，
`projections*.h5` 是源项在 JOREK 场上的投影。

## 决定

1. Phase 1/2 的训练集只使用纯 2D（ntor_1）run 的 `jorek*.h5` 九场帧，起始帧以匹配的 `part0*.h5`
   为准；`ntor11` 多环向 run 移出近期训练集，保留为后续 3D 模态研究材料。
2. 执行顺序固定为两步：
   a. 打通参数读取通道（所有影响下一步状态的零维参数：粘性、热导、高阶平滑项等，必须由 H5 输出作为条件输入），
      并完成无源九场单体基线与 rollout 验收（门槛见 ADR-0002）；
   b. 在 model502 与 marker_spi_source 数据接口对齐后，再做二维含源项训练。
3. 数据接口必须以 JOREK 源码的实际写出协议为准（`jorek*.h5`、`part0*.h5`、`projections*.h5`、
   marker_spi_source），不得沿用旧的自定义 run 假设；schema 模块的任何扩展先核对
   `/home/ac_desktop/JOREK_code/jorek/` 后再落地。
4. 既往 5 场短步长 rollout 只作为“网络/训练架构可行”的可行性证据，不作为九场基线；
   五场 checkpoint 不复用（与 ADR-0001 一致）。

## 后果

- parser/schema 的 fixture 要改成真实形状（如 `values (9,4,11,N)`、`x (2,4,1,N)`）并锁定谐波维语义；
  2D 训练只取 n=0 分量。
- 若 JOREK 当前未把输运/平滑参数写进 H5，则“参数读取通道”成为 JOREK 侧的阻塞依赖，
  需要一份 JOREK 写出协议变更（或后处理导出器）ticket，并显式登记单位与时间对齐。
- 源项相关训练被 marker_spi_source 接口对齐阻塞，不排在无源基线之前。

