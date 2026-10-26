---
layout: post
title: "算子日报 · 2026-10-26"
permalink: /operator-daily/2026-10-26/
date: 2026-10-26 23:59:00 +0800
tags: [算子日报]
---

> 每日汇总：CUDA 开源仓 Release Notes（算子）、芯片动态、arXiv 算子内核论文。

## TL;DR

- **FP4 量化的低秩补偿可以不用 tensor core 算**：BiLoCo 把低秩修正的左右因子存成 ±1 二值向量，每 rank 存储省 **16×**，修正项退化为加减法，可在 CUDA core 上与 FP4 tensor-core GEMM **同一 kernel 并发执行**；B300 上 batch=1、rank≤512 时融合 kernel 仅比纯 FP4 GEMM 慢 **11%** 以内。
- **稀疏 attention 的 selection 不该物化 score 表**：HyScale 的 HyTopK 在飞行中求精确 top-k、不回写整张分数表，selection replay 让重计算复用前向选出的索引；GLM-5.1 的 1M-token 训练在 512 张 H20 上比 Megatron-LM 快 **2.07×**。
- **解压可以做成 GPU kernel 的一等公民**：编译式 GPU 查询引擎把 tile 解压融进生成的 kernel、在 shared memory 内与合作式查询执行合流，避免解压列落 HBM；驻留内存最多降 **4×**、带宽最多降 **6×**。

## 一、CUDA 开源仓 Release Notes（算子）

> 本期无新 release。上期收录 cuda-pathfinder v1.8.4 后，窗口内新增的只有 `cudnn-frontend` 的三条 nightly：`v1.32.0.dev76372792`（10-24）、`dev76547535`（10-25）、`dev76742898`（10-26），正文均自述 *Automated nightly build from NVIDIA-internal CI. Not a supported release.*，按既有口径不计入；30 天窗口内的其余正式 release 均已在往期各期收录。

## 二、芯片动态

> 本期无新条目。NVIDIA Developer Blog 源继续返回 0 条；NVIDIA Blog 在 7 天窗口内唯一命中的仍是 10-21 的 Google Cloud × Vera Rubin GA，已收录于 2026-10-21 期。本节连续为空属**数据源侧**状况，不是抓取故障。

## 三、arXiv 论文（算子内核）

> 本期增量窗口覆盖 10-23 至 10-26 共 **46 条**（未触及 `max_results` 上限）。46 条全部为 10-23 提交：10-23 是周五，这批投稿到本次抓取时才进入 API；10-24、10-25 为周末，10-26 当天提交尚未入库。逐 id 与往期各期文章比对，无一重合；按算子开发相关性精选 3 篇。

### 低精度与 GEMM

- **[BiLoCo: Binary Low-Rank Corrections for Post-Training Quantization](http://arxiv.org/abs/2610.32243v1)**（`2610.32243`，cs.LG，10-23）：FP4 已成为降低 LLM 推理成本的关键手段，Blackwell 也提供了原生 FP4 tensor core，但直接做 FP4 PTQ 仍有明显精度缺口。SVDQuant 用高精度低秩修正补误差，代价是每个 rank 占很多比特，而且修正项和主 FP4 GEMM 抢同一批 tensor core。BiLoCo 把左右因子从 BF16 换成 **±1 符号向量**，每个 rank-one 分量存储省 **16×**；修正项因此只剩加/减，可交给 **CUDA core** 与 FP4 tensor-core GEMM **并发**执行。4B–72B 模型上，BiLoCo 在同等内存预算下追平或超过 SVDQuant。作者在 **B300** 上实现为单个融合 kernel：FP4 GEMM 走 tensor core，bit-packed 修正走同一 launch 的 CUDA core，两者在 **GEMM epilogue** 合并；batch=1 时，rank 到 512 为止最多比纯 FP4 GEMM 慢 **11%**，rank=128 时 batch 到 8 为止最多慢 **20%**。对 kernel 侧的启示是：低精度主算子的误差补偿未必要占用 tensor core，**epilogue 融合 + 异构计算单元并发**可以把额外精度几乎藏进主 GEMM 的尾部。

### 稀疏 attention 与长上下文训练

- **[HyScale: Scaling Long-Context Training for Learned Sparse Attention](http://arxiv.org/abs/2610.32057v1)**（`2610.32057`，cs.AI/cs.DC，10-23）：learned sparse attention 已进入生产（DeepSeek-V3.2），但训练时的 **selection** 成了瓶颈：每个 query 都要给所有历史 token 打分，而真正读取只有 k 个；现有系统还会物化完整 score 表，并在重计算时重复 selection。HyScale 的关键观察是——selection 的输出是**索引集合**，不是 activation。据此设计 **HyTopK**：不建 score 表，边扫描边维护每个 query 的精确 top-k；**selection replay** 则在反向重计算时复用前向选出的索引，不再重跑打分。配合针对端到端训练的系统优化，GLM-5.1 的 1M-token 序列训练在 512 张 H20 上比 Megatron-LM 快 **2.07×**；DeepSeek-V4-Flash 的 CSA 层、Qwen3.8-Flash-Next 的 QSA 层在 32 张 H800、1M token 下分别快 **2.01× / 1.57×**。对写 sparse-attention kernel 的人是明确信号：**别物化 score 表、让 selection 只留下索引、把重计算设计成可回放**，收益来自访存和重复计算，而不只是 FLOPs。

### GPU kernel 工程

- **[Tile-Based Decompression in Compiled GPU Engines](http://arxiv.org/abs/2610.32695v1)**（`2610.32695`，cs.DB/cs.DC，10-23）：GPU 要跑出峰值分析性能，工作集得装进 HBM；但 GPU 分区/共享、工作集增长、HBM 每美元容量停滞，使这个预算越来越紧。轻量压缩能同时降占用和带宽，现有 GPU 压缩执行方案却要么绑定特定执行模型，要么先把解压列物化到 HBM 再查询。本文把解压当作查询引擎的**一等物理算子**：压缩相关逻辑收在窄 backend 接口后，新编码不必改下游算子；实现在 Proteus 中，并系统比较解压值应放在 GPU 存储层级的哪一层。最终采用**编译器集成的 shared-memory 策略**，把协作式 tile 解压与 tuple-at-a-time 查询执行融进**同一个生成的 GPU kernel**，完全避免解压列落 HBM。与 HeavyDB、Sirius 在 GPU 常驻数据上相比，查询时间有竞争力，同时驻留内存最多降 **4×**、内存带宽消耗最多降 **6×**。这是一份把「压缩/解压」从数据准备阶段挪进 kernel 内部、用 shared memory 换 HBM 带宽的完整工程样本。

> 其余新入库条目多为应用/训练方法向（医疗影像、EEG、机器人、视频生成、遥感、推荐与安全攻击等）或纯理论工作，未收录。
