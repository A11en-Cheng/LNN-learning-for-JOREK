# Tests layout（2026-10-04 整理）

`pytest` 只收集本目录下的 `test_*.py`（30 个文件，247 个用例，1 个 skip）。
详细隔离规则见 [docs/adr/README.md](../docs/adr/README.md) 与
[report/v2-doc-isolation-and-dirty-data-audit-2026-10-04.md](../report/v2-doc-isolation-and-dirty-data-audit-2026-10-04.md)。

## 目录结构

| 位置 | 内容 | 状态 |
|---|---|---|
| `test_*.py`（本目录） | 现行契约/数据/物理事实/组件测试 | **必须全绿**；新增测试放这里 |
| `legacy/` | v1 GNN 栈 + teacher-source 插值 + v1 physics loss + 被取代的 sampler（20 个文件） | 已归档，**不收集**；仅参考 |
| `train_*.py` / `eval_*.py` / `plot_*.py` / `check_*.py` / `probe_*.py` / `verify_*.py` / `report_*.py` / `build_*.py` / `compare_*.py` / `inspect_*.py` / `estimate_*.py` / `run_*.py` / `make_diagnostics.sh` | 工具脚本（不是 pytest 用例） | 多数是 v1 工具；新工作先看 `roadmap_v2.md` |

## 现行测试分组（保留）

| 领域 | 文件 |
|---|---|
| 数据契约 | `test_model502_schema.py`、`test_normalization_contract.py`、`test_model502_source_parser.py`、`test_marker_projection.py`、`test_namelist.py`、`test_parameter_channel.py`、`test_parameter_surface.py` |
| manifest/窗口 | `test_data_manifest.py`、`test_window_manifest.py` |
| 物理事实/源 | `test_spi_model.py`、`test_nonlocal_table.py`、`test_adas_closure.py`、`test_charge_state_closure.py`、`test_model502_physics.py`、`test_ablation_series.py` |
| 碎片/采样 | `test_fragment_state.py`、`test_fragment_dynamics.py`、`test_field_sampling.py` |
| 组件模型 | `test_ltc_cell.py`、`test_ltc_processor.py` |
| 输出契约/配对（Phase A） | `test_output_contract.py`（ADR-0019 时序断言、restart 登记/剔除）、`test_paired_dataset.py`（`(u_k, part{k}, projections{k}) → u_{k+1}` 加载器） |
| A′ 逐步导出接口（Phase A′） | `test_step_dataset.py`（`export_v3` 布局、`require_single_step`、shard/边界/condition fail-fast、raw 交叉校验、linkage；**原生逐步布局**：`detect_run_layout`、t_now 推导区间、零步长 error/list、去重缺口登记、restart 注册/偏移、part/边界帧）、`test_budget.py`（烧蚀/电离/辐射/全局闭合读取；原生 tail-only 逐 step）、`test_check_export_v3.py`（验收 CLI 端到端：export_v3 + 原生模式） |
| B0 stencil/边界（Phase B） | `test_stencil_operator.py`（FEM 几何/离散关系/守恒/确定性）、`test_stencil_boundary.py`（BC 映射表/多层/NaN）；合成 mesh 必测 + 真实帧 skip-if-absent，fixture 在 `stencil_fixtures.py` |
| B2/B2.1 隐式 stencil（Phase B） | `test_implicit_stencil.py`（逐项度量权重与 `BigR/BB2 ≡ BigR` 等价、17 项精确 λ_max/稳定界、通道-项映射含 `ZK_perp_num`、direct/cg/richardson 三求解器、真解 backward error、**准稳态剖面速率/近零空间投影/field-line 平均守恒**、JOREK 单步稠密对照、确定性与梯度、成本记录、真实帧 λ 表与 rollout smoke） |
| B2/B2.1 九场推进器（Phase B） | `test_field_propagator.py`（浅层 FiLM 恒等初始化/深层 Jacobian=0/分组置换形状不变、物理 dt 多层、LTC 闭式积分、boundary 码 fail-fast 与 namelist flag、**solver 选择/成本记录、鞘层参数与 `gamma_*_stangeby` 推导、corner-pair 跳过、Mach-1 `1/Btot` 关系、全栈准稳态剖面**、参数量报告、真实帧 smoke） |
| parser 契约 | `test_parser_contract.py`（v1 `JorekSequenceDataset` 对齐用例已于 Phase A 由配对加载器测试取代） |

## 已归档的 v1 测试（`tests/legacy/`）

`test_checkpointing`、`test_cfc_interp`、`test_cognition_pipeline`、`test_conditioning`、
`test_data_visualization`、`test_dataset_loader`、`test_encoder_history`、`test_grad_ops`、
`test_interp_dataset`、`test_lnn_processor`、`test_model502_full_loss`、
`test_model502_interfaces`、`test_perception_pipeline`、`test_reconstruction_pipeline`、
`test_segmented_sampler`、`test_source_head`、`test_source_injection`、
`test_step_dynamics`、`test_training_and_vis`、`test_two_snapshot`。

它们依赖 v1 GNN/teacher-source 语义（`source[:, k+1]` 配对，见 ADR-0019），
**不得作为新工作的基线或 Gate**。若其中某个行为仍需要，按 ADR-0018/0019 重写后
放回本目录。

## 运行

```bash
.torch-cuda-env/bin/python -m pytest -q                 # 全量（当前 246 passed, 1 skipped）
.torch-cuda-env/bin/python -m pytest tests/test_output_contract.py tests/test_paired_dataset.py -q
.torch-cuda-env/bin/python -m pytest tests/test_step_dataset.py tests/test_budget.py tests/test_check_export_v3.py -q
.torch-cuda-env/bin/python -m pytest tests/test_ltc_cell.py -q
.torch-cuda-env/bin/python -m pytest tests/test_stencil_operator.py tests/test_stencil_boundary.py -q
```

可复用的组件探针（非 pytest）：`probe_ltc_semigroup.py`（恒等/半群）、
`check_fem_gradients.py`、`check_fem_second_order.py`（FEM 算子正确性）。
