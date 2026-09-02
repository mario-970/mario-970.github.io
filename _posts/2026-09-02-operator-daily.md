---
layout: post
title: "算子日报 · 2026-09-02"
permalink: /operator-daily/2026-09-02/
date: 2026-09-02 23:59:00 +0800
tags: [算子日报]
---

# 算子日报 · 2026-09-02

> 每日汇总：CUDA 开源仓 Release Notes（算子）、芯片动态、arXiv 算子内核论文。

## TL;DR

- cuDNN Frontend 发布 **v1.28.0**：新增 `cudnn.fla`，把 flash-linear-attention 的 GDN/KDA 算子直接加速到 cuDNN Blackwell (SM100) 内核；GEMM CuTeDSL API 全面支持 JAX。
- NCCL4Py **v0.5.0** 补齐 CFT 逻辑端点寻址与 per-collective 配置 API。
- 芯片动态今日无新增（RSS 未更新，仍为 08-24 的 Vera Rubin 系列）。
- arXiv 集中在 DFT 算子近似、稀疏预处理、量化误差分配三条线。

## 一、CUDA 开源仓 Release Notes（算子）

### cuDNN Frontend

- **v1.28.0**（09-02）— [链接](https://github.com/NVIDIA/cudnn-frontend/releases/tag/v1.28.0)：新增 `cudnn.fla` 加速器，把 flash-linear-attention 的 GatedDeltaNet（`chunk_gated_delta_rule`）与 KDA（`chunk_kda`）算子 monkeypatch 到 cuDNN Blackwell (SM100) 内核，结果与 FLA 一致、不支持则透明回退，另含 `GatedMLP` 适配器；GEMM CuTeDSL API 全面 type-erased 支持 JAX（`cudnn.jax.call` 提供 `jax.jit` 入口，覆盖 blockscaled MXFP8 等）；新增一级 `cudnn.Handle` 对象统一 backend handle/device/stream。

### NCCL（NCCL4Py）

- **nccl4py-v0.5.0**（09-01）— [链接](https://github.com/NVIDIA/nccl/releases/tag/nccl4py-v0.5.0)：新增 CFT 设置与逻辑端点寻址的 host/CuTe DSL API（`team_cft`、`CftLeInfo`、`get_cft_le_info` 等）；新增 per-collective 配置（算法选择、CTA 调优、profiler 标签、vendor 选项）；新增「先编译内核再建资源」的 type-only CuTe DSL 参数（`make_fake_dev_comm` 等）。

## 二、芯片动态

今日无新增条目（NVIDIA 官方 RSS 未更新，最近仍为 08-24 的 Vera Rubin NVL72 系列，已在 09-01 日报覆盖）。

## 三、arXiv 论文（算子内核）

- **[32-point DFT Approximations](https://arxiv.org/abs/2609.01115)**（`2609.01115`，09-01）：免乘法器的 32 点 DFT 近似，用 Frobenius 误差最小化 + 行向对称约束降搜索空间——DFT 算子低复杂度实现（呼应 TensorRT 刚新增的 DFT 算子支持）。
- **[MakoXC: DFT Exchange-Correlation with Matrix-Aligned Sparsity](https://arxiv.org/abs/2609.01025)**（`2609.01025`，09-01）：重做 DFT 交换关联的矩阵对齐 + 知识组织稀疏化，把不规则稀疏负载改造成适合现代 AI 加速器的形式。
- **[ABMC vs Leiden for Parallel ICCG Preconditioning](https://arxiv.org/abs/2609.00561)**（`2609.00561`，09-01）：并行 ICCG 预处理中代数分块多着色 vs Leiden 的对比，缓解前/后代换的串行依赖，兼顾并行度与数据局部性。
- **[The Structure of Quantization Damage in LLMs](https://arxiv.org/abs/2609.01587)**（`2609.01587`，09-01）：用因果混合精度干预研究 PTQ 误差分布，指导把额外精度预算花在全局——量化算子位宽分配的实证依据。
- **[Hardware Acceleration of Block-Diffusion LLM for Edge](https://arxiv.org/abs/2609.01084)**（`2609.01084`，09-01）：面向边缘设备的 block-diffusion LLM 加速，WIFiV-LPDDR 做 precision-tagged 读取、BRQ-KV 规范化 KV 缓存，缓解权重/KV 流量瓶颈。
- **[Projection-based Low-rank Assembly in IgA](https://arxiv.org/abs/2609.01218)**（`2609.01218`，09-01）：面向等几何分析（IgA）的投影低秩装配，降低三维质量/刚度矩阵的装配与存储开销。

## 数据源

本日报由 open_eye 自动抓取 + Claude 摘要生成（arXiv 宽抓取后人工精选，仅收录 09-01 及之后提交的新论文）。
