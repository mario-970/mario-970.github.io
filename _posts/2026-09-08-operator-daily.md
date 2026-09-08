---
layout: post
title: "算子日报 · 2026-09-08"
permalink: /operator-daily/2026-09-08/
date: 2026-09-08 23:59:00 +0800
tags: [算子日报]
---

# 算子日报 · 2026-09-08

> 每日汇总：CUDA 开源仓 Release Notes（算子）、芯片动态、arXiv 算子内核论文。

## TL;DR

- 本期无新正式 release：09-05 ~ 09-07 只有 cuDNN Frontend 的 nightly CI 构建（`v1.29.0.dev*`，非支持版本），芯片 RSS 也无更新。
- arXiv 补录 09-03 ~ 09-04 的 7 篇算子内核论文，最亮眼的是 **Hardware-Aware FP4 FlashAttention-4**（Blackwell FP4 张量核下 attention 的加速）与 **BF16 仿真 FP32/FP64 GEMM**（Intel AMX）。

## 一、CUDA 开源仓 Release Notes（算子）

本期无新正式发布。09-05 ~ 09-07 仅 cuDNN Frontend 的自动化 nightly 构建标签（`v1.29.0.dev*`，标注「Not a supported release」），不收录。

## 二、芯片动态

今日无新增条目（NVIDIA 官方 RSS 未更新，最近仍为 08-24 / 08-11 的 Vera Rubin 与供电系列，此前已覆盖）。

## 三、arXiv 论文（算子内核）

> arXiv 数据最新至 09-04；09-05 ~ 09-08 无新算子内核论文。以下为 09-03 / 09-04 提交。

- **[Hardware-Aware FP4 FlashAttention-4](https://arxiv.org/abs/2609.04105)**（`2609.04105`，09-03）：指出 Blackwell FP4 张量核不会自动让 attention 变快——矩阵乘缩小后 softmax 转换与片上依赖占主导；提出 Direct-P（非因果推理）与「前向量化直通反传」的因果路径。
- **[BF16 Component-Product Emulation of FP32/FP64 GEMM on Intel AMX](https://arxiv.org/abs/2609.04663)**（`2609.04663`，09-04）：用低精度 BF16 矩阵乘做「分量积」来仿真高精度 FP32/FP64 GEMM，在 Intel AMX 上给科学计算做算法桥接。
- **[Accelerating Atom Simulations with Variable-Block Sparse Matrix Library](https://arxiv.org/abs/2609.04397)**（`2609.04397`，09-03）：面向量子算子的变块稀疏矩阵库，保留块结构（块形状随化学物种/基组变化）以启用高效块算法，而非把块打散成标量稀疏。
- **[Improving the Hyperpower Method for Inverse Matrices](https://arxiv.org/abs/2609.04433)**（`2609.04433`，09-03）：首次用标量范数型加速器提升矩阵求逆迭代法的收敛阶（+50% 到六阶），同时减少矩阵-矩阵乘次数。
- **[Scale-QLoRA: Code-Invariant Adapter Merging for Native 4-bit Microscaling LLMs](https://arxiv.org/abs/2609.04526)**（`2609.04526`，09-03）：在原生 4-bit microscaling（NVFP4 / MXFP4）checkpoint 上合并 LoRA，需经量化器回写重导出 scale，解决合并步骤不再「免费」的问题。
- **[Fast Gauss Sums via Flash Attention](https://arxiv.org/abs/2609.04910)**（`2609.04910`，09-04）：复用 flash attention 的硬件感知工程，把高斯核求和（MMD / SVGD 等核方法核心）映射到 attention 原语上加速。
- **[FlexPosit: Tunable Fractional Precision for LLM Inference Accelerators](https://arxiv.org/abs/2609.04724)**（`2609.04724`，09-04）：可调分数精度的 posit 数制用于 LLM 推理加速器，在精度与硬件效率间提供更细的权衡。

## 数据源

本日报由 open_eye 自动抓取 + Claude 摘要生成（arXiv 宽抓取后人工精选，仅收录上期之后的新条目）。
