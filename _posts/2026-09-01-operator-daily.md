---
layout: post
title: "算子日报 · 2026-09-01"
permalink: /operator-daily/2026-09-01/
date: 2026-09-01 23:59:00 +0800
tags: [算子日报]
---

# 算子日报 · 2026-09-01

> 每日汇总：CUDA 开源仓 Release Notes（算子）、芯片动态、arXiv 算子内核论文。

## TL;DR

- CUTLASS 4.8.0dev 首次加入 **Rubin (SM107)** 支持，新增 FP8/FP4 张量核、稀疏 MMA，TMEM 扩到 576 COL、共享内存上限 328KB。
- cuDNN Frontend 1.27 用 CUTLASS 原语开源了 **FROST 引擎**，并新增门控线性注意力（GDN）算子；NCCL 2.31 在 Blackwell 上默认启用 **TMA 内核**、新增 Compute Fabric 传输。
- 芯片侧 NVIDIA 宣布 **Vera Rubin NVL72 满产**（配合 Groq 3 LPX），主打 agentic 推理的 token 生成与能效。
- arXiv 集中在稀疏矩阵计算、混合精度量化、GPU 批处理优化三条线。

## 一、CUDA 开源仓 Release Notes（算子）

### CUTLASS

- **v4.8.0dev**（08-27）— [链接](https://github.com/NVIDIA/cutlass/releases/tag/v4.8.0dev)：首次加入 Rubin (SM107) 支持，新增 FP8/FP4 张量核、TMEM 从 512 扩到 576 COL、共享内存上限提到 328KB；CuTe DSL 扩展新增异步原子 TMA reduce-store 与稀疏 MMA，补了 grouped GEMM、GQA decode 等示例。需 R615 驱动（随 CUDA 13.4 GA 发布）。
- **v4.7.1**（08-26）— [链接](https://github.com/NVIDIA/cutlass/releases/tag/v4.7.1)：主要是一批 CuTe DSL bug 修复（setmaxnreg 与 warp 特化冲突、JAX 张量别名、TVM-FFI 偏移等），无算子新特性。

### cuDNN Frontend

- **v1.27.0**（08-06）— [链接](https://github.com/NVIDIA/cudnn-frontend/releases/tag/v1.27.0)：新增 Python 原生 `cudnn.pygraph` 图 IR；开源 FROST 引擎（基于 CUTLASS 原语，覆盖 GEMM / grouped MoE matmul / 融合 epilogue / 线性注意力 / SDPA，支持 FP4/FP8/MXFP8）；新增 `cudnn.linear_attention`（GDN 门控线性注意力算子，THD token-packed 布局）。

### NCCL

- **v2.31.2-1**（08-11）— [链接](https://github.com/NVIDIA/nccl/releases/tag/v2.31.2-1)：新增 Compute Fabric Transport (CFT)，支持窗口内存注册与设备侧 Put/Get/Red/NVLS（Blackwell + CUDA 13.3）；Blackwell 上默认启用 TMA 内核并纳入 cost model；新增按 collective 的配置/调优 API；PAT 算法支持节点内 NVLS 分层。

### TensorRT

- **v11.2**（08-04）— [链接](https://github.com/NVIDIA/TensorRT/releases/tag/v11.2)：新增 cuFFT 支撑的 FFTPlugin，支持 ONNX DFT 算子；解析器新增 DFT 与 5D GridSample 支持。

### MatX

- **v1.1.0**（08-07）— [链接](https://github.com/NVIDIA/MatX/releases/tag/v1.1.0)：补齐除 nvCompDx 外全部 MathDx（cuSolverDx / cuBLASDx / cuFFTDx / cuRANDDx）支持；Blackwell+ 新增 256b LD/ST；引入实验性分布式张量。

## 二、芯片动态

- **[Vera Rubin 满产 + Groq 3 LPX](https://blogs.nvidia.com/blog/vera-rubin-lpx-spectrum-x-nvlink-fusion/)**（08-24）：NVIDIA 扩展 Vera Rubin NVL72，配合 Groq 3 LPX 面向 agentic 系统做快速 token 生成。
- **[Vera Rubin NVL72 能效标杆](https://blogs.nvidia.com/blog/vera-rubin-nvl72-efficiency-ai-agents/)**（08-24）：宣称 AI agent 负载每瓦工作量最高 30x，主打高 token 消耗场景。
- **[800V 直流电源架构](https://blogs.nvidia.com/blog/800-vdc-power-architecture-ai-factory/)**（08-11）：面向 AI 工厂的整机供电与功率密度，属基础设施（非算子，供了解）。

## 三、arXiv 论文（算子内核）

- **[Spectral Analysis for Sparse Matrix Computation](https://arxiv.org/abs/2608.29362)**（`2608.29362`，08-29）：首次把稀疏矩阵计算与谱分析联系起来，从稀疏模式角度解释 cache 复用、内存合并与负载均衡。
- **[GPU-Accelerated Blocked Adaptive Randomized Range Finder](https://arxiv.org/abs/2608.28941)**（`2608.28941`，08-28）：基于隐式 Householder QR 的分块自适应随机 range finder，用于低秩近似（如 GaLore 式训练）。
- **[Jacobi 方法并行算法（Part Two）](https://arxiv.org/abs/2608.28952)**（`2608.28952`，08-28）：并行 Jacobi 特征值/SVD，算术成本最优、带宽/延迟可达矩阵乘下界。
- **[B³-PWL: GPU-Batched Branch-and-Bound](https://arxiv.org/abs/2608.28988)**（`2608.28988`，08-28）：GPU 批量 branch-and-bound 求解分段线性优化，替代 CPU 中心化求解器。
- **[Efficient GPU Retrieval for Semantic Search](https://arxiv.org/abs/2608.28968)**（`2608.28968`，08-28）：LinkedIn 语义检索的 GPU 端检索优化（亿级语料）。
- **[CHIPSMORE](https://arxiv.org/abs/2608.30509)**（`2608.30509`，08-31）：计算内互联/内存 chiplet 的多模式多请求 LLM 推理加速器。
- **[ClusterAttention](https://arxiv.org/abs/2608.26965)**（`2608.26965`，08-27）：免训练的双向注意力加速，用快速递归聚类近似注意力。
- **[Q-Strata](https://arxiv.org/abs/2608.30564)**（`2608.30564`，08-31）：MoE LLM 分层比特分配的混合精度量化。
- **[DAMP](https://arxiv.org/abs/2608.27513)**（`2608.27513`，08-27）：GDN/KDA 循环状态的混合精度量化，降低推理显存。
- **[Sparse Fourier Neural Operators](https://arxiv.org/abs/2608.30070)**（`2608.30070`，08-31）：稀疏 FNO 在「选择 / 表示 / 执行」三个层面的稀疏化与开销权衡。

## 数据源

本日报由 open_eye 自动抓取 + Claude 摘要生成（arXiv 宽抓取后人工精选）。
