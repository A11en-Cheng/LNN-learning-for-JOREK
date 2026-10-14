# ADR-0003：运行环境为 `.torch-cuda-env`（CUDA 13 / torch cu130）

> **文档隔离（2026-10-04）**：v1 路线历史文件。新架构工作只以 ADR-0018/0019 + roadmap_v2 为准；引用须标注"（历史）"，冲突以 0018/0019 为准（索引见 README.md）。

状态：已接受（2026-09-06）

## 背景

仓库原有 venv 指向不存在的 Python 3.11 可执行文件，不可用；系统 Python 没有 pytest/torch。
用户要求按隔壁仓库 `/home/ac_desktop/HyPhy-GLNN_V2/.torch-cuda-env` 的配置镜像一个可跑 CUDA 的 venv。

## 决定

1. 项目 venv 固定为仓库根目录下的 `.torch-cuda-env`，基础解释器 Python 3.12.3。
2. PyTorch 固定从官方 cu130 索引安装：`torch==2.11.0+cu130`、`torchvision==0.26.0+cu130`、`torchaudio==2.11.0+cu130`；
   其余依赖按 `requirements.txt` 从 PyPI 安装。
3. 目标硬件为 RTX 5070 Ti（16GB，driver 591.86，CUDA 13.1）。GPU 与其它进程共享，
   显存预算和峰值显存报告必须记录当时的并发占用。
4. GPU 探测与训练只能在沙箱外执行（沙箱内 NVML 被系统阻断）。

## 后果

- 所有测试、训练、benchmark 一律用 `.torch-cuda-env/bin/python` 运行。
- 显存预算不能按 16GB 满额假设；每次 benchmark 都要记录 `nvidia-smi` 的并发状态。
