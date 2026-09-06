# 0001 锁数据 fixture 与 split manifest

Blocked by: （无，可立即开工）

交付：更新 schema/parser fixture 到真实 EHL-2 形状（`values (9,4,1,N)`、`x (2,4,1,N)`），锁定谐波维语义；
以 `part*.h5` 对齐确定每 run 有效帧范围；核对 2D run 清单与跨 run 网格一致性；迁移旧 smoke test 的失效数据路径；
产出序列级 train/val/test manifest（数据版本/hash）与 train-only 统计协议。

验收：`python -m pytest tests/test_model502_schema.py tests/test_dataset_loader.py ... -q` 全绿；
manifest 可由一条命令在固定 seed 下重建。挂点：parser 输出的 `[N,9]` 状态契约。

状态：done。25 tests 全绿；真实数据对齐 2710..8900（620 帧）、10 个 2D run 网格一致（n_grids=1）；
manifest 由 `python -m src.data.data_manifest` 生成，dataset_id 与 seed 可复现。
