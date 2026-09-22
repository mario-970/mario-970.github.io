---
layout: post
title: "算子日报 · 2026-09-14"
permalink: /operator-daily/2026-09-14/
date: 2026-09-14 23:59:00 +0800
tags: [算子日报]
---

# 算子日报 · 2026-09-14

> 每日汇总：CUDA 开源仓 Release Notes（算子）、芯片动态、arXiv 算子内核论文。

## TL;DR

- cuDNN Frontend 发布 **v1.29.0** 大版本：HSTU attention 完整内核族、DSA 稀疏 attention 前向闭环、实验性 Flex Attention、`torch.sdpa` 端到端改走 cuDNN Python API。
- 另有两个次要发布：cuda-python **v13.4.1**（补丁）、cuda-quantum **0.16.0**（量子，与算子内核关联较弱）。
- 芯片 RSS 无更新；arXiv 改走 RSS 兜底，收录 7 篇算子内核论文（EqiForge 张量超优化、Hopper 利用率拆解、BLAS 精度分级等）。

## 一、CUDA 开源仓 Release Notes（算子）

### cuDNN Frontend

- **v1.29.0**（09-13）— [链接](https://github.com/NVIDIA/cudnn-frontend/releases/tag/v1.29.0)：面向 Blackwell 的大版本。核心是 **HSTU attention**——完整 CuTe DSL 内核族（打包变长前/后向、FP16/BF16、head 64/128/256、full/causal/local/任意 mask、paged-KV 前向），用 SiLU 替换 softmax 并自动推导块稀疏元数据；**DSA 稀疏 attention 前向**补全 DeepSeek Sparse Attention 的 SM100 forward 路径（H64 D512/D576、H128 D512 small-topk prefill），后向新增确定性 SM100 归约与 BF16 H128/D512 两-CTA 特化（1.10–1.16×）。实验性 **Flex Attention**（`cudnn.flex_attention`，SM90/100/103 前/后向，任意 mask 规划）与 `torch.sdpa` 端到端改由 cuDNN Python API 服务一并上线，另有 LMSD（LayerNorm-Multiply-SiLU-Dropout）BF16 前/后向与 D32 head 支持。

### cuda-python

- **v13.4.1**（09-10）— [链接](https://github.com/NVIDIA/cuda-python/releases/tag/v13.4.1)：cuda-bindings 13.4 补丁版本（bugfix，详见 [release notes](https://nvidia.github.io/cuda-python/cuda-bindings/13.4.1/release/13.4.1-notes.html)）。

### cuda-quantum

- **0.16.0**（09-12）— [链接](https://github.com/NVIDIA/cuda-quantum/releases/tag/0.16.0)：量子计算框架新版本（qubit mapper 支持非连通拓扑、静态旋转门综合、CUDA-Q Logical、Python 3.14 支持等），与 GPU 算子内核关联较弱，供了解。

## 二、芯片动态

今日无新增条目（NVIDIA 官方 RSS 未更新，最近仍为 08-24 的 Vera Rubin NVL72 系列，此前已覆盖）。

## 三、arXiv 论文（算子内核）

> 注：arXiv 导出 API 今日持续 429（服务端 503 兜底），本节改由 arXiv RSS 兜底抓取，仅覆盖 09-14 公告批次（新提交，约 09-11 ~ 09-14）；09-05 ~ 09-10 的补录待 API 恢复后再做。

- **[EqiForge: 用 equality saturation 做张量程序超优化](https://arxiv.org/abs/2609.12330)**（`2609.12330`）：统一 IR 表达高层张量表达式与 tile 化计算，用等式规则组合直接从表达式推导出 FlashAttention 式融合 kernel；attention kernel 在 decode 最高比 FlashAttention 快 1.87×，QK-normalized MLA / mHC 比 torch.compile 分别快 3.16× / 5.84×。
- **[Dissecting GPU Utilization for LLM Inference on Nvidia Hopper](https://arxiv.org/abs/2609.12923)**（`2609.12923`）：用 8 个 NCU 计数器对齐的视图拆解 Hopper 上 vLLM/FlashAttention-3/cuBLASLt 的 SM 利用率——decode 阶段 dense GEMM 退化为小行矩阵乘，bfloat16 GMMA 固定 64 行 fragment，小 batch 只填满一小部分，把利用率缺口映射到 fragment fill、occupancy、stall、wave 量化与 kernel 选择。
- **[Argus: 面向语义区域的跨层 GPU 性能测量编排](https://arxiv.org/abs/2609.12299)**（`2609.12299`）：围绕算子实现/pipeline 阶段等语义区域自动构造 probe 与程序变体、跨 compiler/hardware/system 后端采集并归因性能证据；44 个 persistent-GEMM/attention 配置改善 39 个，把 AlphaEvolve 几何平均加速从 5.4% 提到 8.9%。
- **[Attention Quantization for Tabular Foundation Models](https://arxiv.org/abs/2609.13031)**（`2609.13031`）：对 Q/K/V 做 FP8 量化并用显式 FP8 矩阵乘指令加速 attention，关键是把测试行与训练行的量化误差对齐；Triton kernel 比 16-bit 快 1.7×，TabPFN-v3 / TabICLv2 无损。
- **[How to grade the accuracy of the BLAS](https://arxiv.org/abs/2609.12307)**（`2609.12307`）：为低精度加速器上的高精度矩阵乘实现打「精度分」——A 级达经典浮点误差界、C 级满足 Strassen 类较弱界，并给出一套不可「作弊」的验证测试与对 LU/QR/Cholesky 的精度影响。
- **[Vortex: 桥接极限压缩与高效 LLM 推理](https://arxiv.org/abs/2609.12208)**（`2609.12208`）：面向 systolic-array 加速器的最小硬件开销架构，用 bi-flow 执行 + codebook 级上下文稀疏，把向量量化(VQ)与输入依赖稀疏落地为实际效率；端到端 8.03×–23.7× 加速、5.68×–12.5× 降耗。
- **[RunningTensor: 把线性注意力推广到高阶循环状态](https://arxiv.org/abs/2609.12814)**（`2609.12814`）：把线性注意力/SSM 的二阶（矩阵）循环状态推广到 o 阶张量——rank-1 外积更新、o-1 个向量 query 收缩读出；o=3 即可把工作记忆容量从 O(W²) 提到 O(W³)，仍保持 O(T) 线性。
