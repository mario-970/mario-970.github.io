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
- 芯片 RSS 无更新；arXiv 抓取今日受接口限流（HTTP 429）暂缺，待补录。

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

> 本节今日受 arXiv API 持续限流（HTTP 429，重试仍失败）暂缺，待接口恢复后补录 09-05 ~ 09-13 提交的算子内核论文。

## 数据源

本日报由 open_eye 自动抓取 + Claude 摘要生成（arXiv 宽抓取后人工精选，仅收录上期之后的新条目）。
