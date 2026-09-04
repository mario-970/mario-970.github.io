---
layout: post
title: "算子日报 · 2026-09-04"
permalink: /operator-daily/2026-09-04/
date: 2026-09-04 23:59:00 +0800
tags: [算子日报]
---

# 算子日报 · 2026-09-04

> 每日汇总：CUDA 开源仓 Release Notes（算子）、芯片动态、arXiv 算子内核论文。

## TL;DR

- 偏轻的一天：Release 仅 cuda-python 两个发布（**cuda-core 1.2.0**、**cuda-pathfinder 1.8.1**），芯片 RSS 无更新。
- arXiv 补录 4 篇 09-02 才被索引到的低比特量化/kernel 论文：FP4 block scaling、2-bit Leech 格解码、1.58-bit 三值化、B-spline 激活优化；09-03 当天无强算子内核论文。

## 一、CUDA 开源仓 Release Notes（算子）

### cuda-python

- **cuda-core v1.2.0**（09-03）— [链接](https://github.com/NVIDIA/cuda-python/releases/tag/cuda-core-v1.2.0)：CUDA Python 核心绑定新增 `utils.CopyOptions`（buffer 拷贝的访问顺序/位置提示/overlap 模式）；新增 kernel 启动属性 `programmatic_stream_serialization`（同流内 kernel 可重叠执行）；`ProgramOptions.use_bundled_headers`（NVRTC 13.3+ 使用捆绑头文件）；新增图节点 `update()` 与 `graph[node]` 视图，可原地更新可执行图的 kernel/memcpy/memset 节点而无需重实例化。
- **cuda-pathfinder v1.8.1**（09-02）— [链接](https://github.com/NVIDIA/cuda-python/releases/tag/cuda-pathfinder-v1.8.1)：CUDA 工具链组件定位库（定位 headers / 动态库 / bitcode / 静态库 / 二进制工具），本次把 `ptxas` 加入可定位的二进制工具列表——服务于 JIT 编译工具链在运行时定位 ptxas/nvdisasm/头文件。

## 二、芯片动态

今日无新增条目（NVIDIA 官方 RSS 未更新，最近仍为 08-24 / 08-11 的 Vera Rubin 与供电系列，此前已覆盖）。

## 三、arXiv 论文（算子内核）

> 本节为 09-02 提交、因 arXiv 收录延迟到今天才可见的论文；09-03 当天无强算子内核相关新论文。

- **[UE5M3 FP4 Block Scaling for Stable LM Pretraining](https://arxiv.org/abs/2609.02846)**（`2609.02846`）：针对 E2M1 FP4 动态范围窄导致预训练不稳的问题，用 E2M1 与 UE5M3 块缩放配对实现稳定 FP4 预训练，把工作留在 FP4 矩阵乘内部（对照 TE 的 current-tensor scaling + RHT 方案）。
- **[Unfolding the Leech Lattice: Fused Multi-Shell Decoding and VRAM Layouts for 2-Bit LLM Weights](https://arxiv.org/abs/2609.02652)**（`2609.02652`）：为 Leech 格 2-bit 向量量化补齐多壳解码 kernel，并量化 batch-1 decode 阶段 GEMV 的 serving 成本与 VRAM 布局。
- **[Post-Training Ternarization of Qwen3-4B](https://arxiv.org/abs/2609.01962)**（`2609.01962`）：用 KOTMS 旋转 + E2M-ATQ 三值化 + GPTQ 式误差校正，端到端把 Qwen3-4B 转成 1.58-bit，评估有效位宽、存储压缩与部署。
- **[FlashKAN: B-Spline KANs via Truncated Power Form](https://arxiv.org/abs/2609.01956)**（`2609.01956`）：用截断幂形式替换 KAN B-spline 激活的 Cox–de Boor 递归（原递归占前向 >90% 时间），降低激活原语的求值开销。

## 数据源

本日报由 open_eye 自动抓取 + Claude 摘要生成（arXiv 宽抓取后人工精选，仅收录上期之后的新条目）。
