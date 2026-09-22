---
layout: post
title: "算子日报 · 2026-09-22"
permalink: /operator-daily/2026-09-22/
date: 2026-09-22 23:59:00 +0800
tags: [算子日报]
---

> 每日汇总：CUDA 开源仓 Release Notes（算子）、芯片动态、arXiv 算子内核论文。

## TL;DR

- 今日 CUDA 开源仓与芯片动态均无新条目：抓到的 release 全是往期已收录的老版本，RSS 最近两条 Vera Rubin 报道也已见于 09-21 日报，内容集中在 arXiv。
- arXiv 收录 4 篇算子内核相关论文：PyTorch **MPS 后端在 2³² 元素边界上 batched matmul 静默返回错误结果**（CUDA 的 `bmm` 正确，但 `arange` 同样越界出错）、全流水线 **FP8 RL 训练不稳定**的根因与 Calibrated Clipping 修复、稀疏 attention 的「高稀疏陷阱」与 FP8 fused kernel 加速、JEPA 训练的 mask-aware 执行重构。

## 一、CUDA 开源仓 Release Notes（算子）

> 今日无新 release。抓取窗口内的 CUTLASS `v4.8.0dev`/`v4.7.1`、NCCL `v2.32.3-1`、CCCL `python-1.2.0`、cuda-python `v13.4.2`、nccl4py `v0.5.0`、cuda-quantum `0.16.0` 均为往期日报已收录条目；cudnn-frontend 仍为 nightly dev 构建（`v1.30.0.dev*`，非正式 release），不做收录。

## 二、芯片动态

> 今日无新条目。RSS 最近两条（Vera Rubin NVL72 MLPerf Inference v6.1 首秀、AI Infra Summit 的 Rubin/DSX 能效）已在 09-21 日报收录。

## 三、arXiv 论文（算子内核）

- **[Silent Failures at the $2^{32}$ Boundary: A Technical Report on Large-Tensor Matrix Multiplication in PyTorch's Apple MPS Backend](https://arxiv.org/abs/2609.22991)**（`2609.22991`）：PyTorch 的 Apple MPS 后端在输出超过 2³² 元素时，`torch.bmm`（连带 `matmul` 与 eager attention）**静默返回错误结果**——相对误差大于 1，既不抛异常也不告警，2.4.1 至 2.14.0 全部版本复现。三条规则可解释全部结果：输出超 2³² 且某操作数是转置 view 时，整个输出等价于忽略 stride 的计算；连续输入超 2³² 时只有越界之后的 batch 错乱（索引在 2³² 回绕）；≥2³¹ 元素的 view 反而抛异常，即**问题更大时显式报错会退化成静默失败**。CUDA 上 `bmm` 结果正确，但 `torch.arange` 超 2³² 元素时同样静默出错——对算子开发者而言，这是 int32 索引溢出在「大统一内存桌面卡」上变得日常化的直接警示，论文附了扫描 harness 与越界防护。

- **[Towards Full Pipeline FP8 Reinforcement Learning for LLMs](https://arxiv.org/abs/2609.22870)**（`2609.22870`）：全流水线 FP8 RL 即使引入 TIS 之类的修正仍会训练失稳（训练中期熵值异常飙升、输出乱码）。作者定位到此前被忽略的原因：**复合 FP8 量化噪声扭曲 importance ratio**，把负 advantage 的 token 不成比例地推出信任域、梯度被错误置零，病态输出得不到惩罚并逐步累积。为此提出 **Calibrated Clipping**：动态对齐 FP8 clipping 边界与 BF16 分布（匹配下界裁剪分位数并相应重平衡上界）。在 GRPO/DAPO、8B–32B 规模与多种 FP8 scaling granularity 下消除了熵尖峰，性能恢复到 BF16 基线水平——直接关系到 FP8 GEMM 的 scale/clipping 策略设计。

- **[SparkDiffusion: Mitigating the High-Sparsity Trap — A Unified Framework for up to $265\times$ Single-GPU Acceleration of Visual Generation](https://arxiv.org/abs/2609.23153)**（`2609.23153`）：指出视频扩散 Transformer 的**「高稀疏陷阱」**：注意力稀疏度压到极端后，step-local 训练 loss 继续下降但最终生成质量停滞甚至退化——主导误差来自高噪声的「结构生成」阶段，只有 terminal-aligned 训练才能纠正。据此给出分阶段方案（先把稀疏结构适配成粗先验，再纠正终端分布），实例化为短稀疏 warm-up + 少步轨迹混合蒸馏 + **FP8 量化与 fused kernel**；在 Wan2.1/2.2 上以 97% 稀疏度保持视觉质量，Wan2.1-T2V-14B-720P 单卡 RTX 5090 端到端 **265×**（H100 上 220×），1.3B-480P 视频 1.3 秒完成去噪。

- **[Mask-Aware Execution for Efficient JEPA Training](https://arxiv.org/abs/2609.22674)**（`2609.22674`）：JEPA 训练管线当前效率低下的三点：每个输入要过多个 mask 专用分支、目标侧存在冗余计算、token routing 受带宽限制，且开销随 mask 数增长。M-JEPA **不改学习目标，只重构执行**：把 mask 无关计算与 mask 相关路由分离，共享 context encoder 执行、融合（带 backward 的）token routing/slicing、对目标 token 并集做稀疏 target encoder 执行、masked patch embedding。在 A100 上覆盖 5 种 JEPA 变体，2–10 个 mask 时端到端最高 **1.7×** 加速，高稀疏度下 patch embedding 单项 4.75×——典型例证：kernel 融合与稀疏执行重构本身即是主要收益来源。
