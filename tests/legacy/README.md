# v1 legacy tests（2026-10-04 归档）

本目录是 v1 路线（GNN 栈 + teacher-source 插值器 + v1 physics loss + 被取代的
sampler）的测试归档。**pytest 不收集本目录**（`tests/conftest.py` 的
`collect_ignore` + `pytest.ini` 的 `norecursedirs`）。

## 归档原因

| 文件 | 原因 |
|---|---|
| `test_checkpointing.py`、`test_cognition_pipeline.py`、`test_encoder_history.py`、`test_model502_interfaces.py`、`test_perception_pipeline.py`、`test_reconstruction_pipeline.py`、`test_training_and_vis.py`、`test_conditioning.py`、`test_source_injection.py` | v1 GNN/CfC 栈（ADR-0018 禁用 GNN） |
| `test_dataset_loader.py`、`test_grad_ops.py`、`test_lnn_processor.py` | 依赖旧 `JorekSequenceDataset` / `JorekLNNProcessor` 路径 |
| `test_step_dynamics.py` | v1 trainer/loss 语义 |
| `test_cfc_interp.py`、`test_interp_dataset.py`、`test_source_head.py` | v1 teacher-source 配对（`source[:, k+1]` 一步未来泄漏，ADR-0019） |
| `test_model502_full_loss.py` | v1 插值点物理损失（已暂停） |
| `test_segmented_sampler.py` | 被 `window_manifest` 取代 |
| `test_two_snapshot.py` | v1 两帧辨识评估 |
| `test_data_visualization.py` | v1 parser/可视化脚本（无 pytest 用例） |

## 规则

1. 不得作为新工作的基线、Gate 或参考指标。
2. 不得被现行测试导入；若某个行为仍需要，按 ADR-0018/0019 重写后放回 `tests/`。
3. 需要历史证据时，引用对应 `report/` 条目并标注"（历史）"。
