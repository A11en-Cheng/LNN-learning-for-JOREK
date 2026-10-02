# ADR-0015：Phase 2.3 收口——采用非局域解析骨干，学习修正暂停

状态：已接受（2026-09-29）；细化 ADR-0011 第 2 条（"先做解析-only 对照，再决定修正的必要性与幅度"）。

## 背景

1. 非局域烧蚀支路已可离线计算。`nonlocal_array.dat` 只是调用快照（每次 `int3d_new`
   覆盖），按帧重建（`src/data/nonlocal_table.py`）后，10 个 run 上逐 shard 中位比
   1.00–1.09、log-corr ≥0.98、within-2x 85–96%、总烧蚀率比 0.97–1.04
   （`report/spi-analytic-baseline.md`）。
2. 解析骨干因此不再有结构性偏差。held-out `scan_1d22_ang15_5_massloss_neon_v400` 的
   解析-only gate（`output/fragment_dynamics/v402_analytic/metrics.json`）：
   半径 MAE 1.33e-7 m、烧蚀中位比 1.019、log-corr 0.980、within-2x 90.7%、
   73 个区间的总烧蚀原子数误差 +2.0%、区间中位误差 4.0%、死亡判定一致率 0.9977。
3. 学习修正（乘性烧蚀系数 + 加性加速度）在同一 gate 上让烧蚀/半径/预算指标全面变差
   （中位比 0.52–0.66、总原子量误差 −47%）。原因：监督目标是**下一帧**记录值，输入是
   **当前帧**场与碎片状态；在 50 个流体步的帧粒度下，它学到的是区间漂移而非模型误差。

## 决定

1. Phase 2.3 的交付物 = 非局域解析骨干（含逐帧重建的非局域表）；帧粒度 gate 的登记数字取
   `output/fragment_dynamics/v402_analytic/metrics.json`。
2. `HybridFragmentDynamics` 的修正头保留在代码中作为消融：训练脚本继续同时报告
   baseline/hybrid，`--analytic-only` 用于登记交付物；修正默认不启用。
3. 修正重新启用的条件（满足其一即可）：(a) 拿到逐流体步或区间累计的投影/烧蚀导出
   （P0a/P1）；(b) 监督改为多帧 rollout 的累计量；(c) 解析骨干在同一目标粒度上仍存在
   ≥2× 的系统偏差。
4. 已知缺口（明确**不**由学习修正承担）：速度增量精度（速度 MAE ~3e-2 m/s）、帧内 50 个
   流体步的演化细节、长于一个帧间隔的自回归稳定性——分别交给 Phase 3/4 的 CfC 连续演化
   与更细粒度的目标解决。

## 后果

- Phase 3/4 的 CfC 不再需要"补烧蚀结构误差"，其目标回到场演化与非均匀时间插值。
- 以后重跑 `tests/train_fragment_dynamics.py` 时，要么用 `--analytic-only` 登记交付物，
  要么显式说明重新启用条件已满足，避免把区间漂移当成物理修正学进去。
