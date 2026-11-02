---
layout: post
title: "算子日报 · 2026-11-02"
permalink: /operator-daily/2026-11-02/
date: 2026-11-02 23:59:00 +0800
tags: [算子日报]
---

> 每日汇总：CUDA 开源仓 Release Notes（算子）、芯片动态、arXiv 算子内核论文。

## TL;DR

- **线性注意力 kernel 的稀疏化**：SparseGDN 平均只保留约 **29%** 的 recurrent events，配合 block-adjacent Delta-solve 与 compact state snapshot，在 Qwen3.5-4B/9B 的 4K–128K RULER 上把 GDN-core prefill 相对原生 FLA kernel 加速 **1.50–1.94×**（含在线选择开销）。
- **低精度侧今天有新东西**：SACQ 把 VLA 权重压到 **1.5 bit** 仍有 **93.85%** 的 LIBERO 成功率、显存降 **4.83×**，叠加 INT8 激活与低精度 attention 后比 BF16 快 **1.51×**；TVQ 用 straight-through 训练把 LLaMA2-70B 做到 **2-bit** 仅掉约 **1%**；KV-cache 诊断则指出 value 占 Fisher 质量的 **94%**，但实测 KL 由 key 的量化误差主导。
- **硬件选型不能只看代际**：B200/B300 的功率分解基准显示 B300 在卷积推理上最多慢 **5.6×**、3D ViT 训练慢 **8×**，原因是这些 workload 只吃到 1100W 功耗上限的 **36–53%**；在输入常驻显存的测试下，差异可定位到**逐 kernel 成本**。

## 一、CUDA 开源仓 Release Notes（算子）

> 本期无新的正式 release。cuDNN Frontend 在 10-31 / 11-01 / 11-02 各有一个 `1.33.0.dev*` nightly（正文自述 *Automated nightly build from NVIDIA-internal CI. Not a supported release.*），按既有口径不计入；30 天窗口内其余正式 release（CUTLASS v4.8.1、CCCL python-1.3.0 / v3.5.0、TensorRT 11.4、cuda-pathfinder v1.8.4 / v1.8.5、cuda-core v1.2.2）均已在往期各期收录。

## 二、芯片动态

> 本期无新条目。NVIDIA Developer Blog 源继续返回 0 条；NVIDIA Blog 在 7 天窗口内无新的芯片/架构类命中。本节连续为空属**数据源侧**状况，不是抓取故障。

## 三、arXiv 论文（算子内核）

> 本期增量窗口自 10-30 起（`fetch_state` = 2026-11-01），返回 **30 条**，全部为 **10-30** 提交——10-30 的论文在上一期（10-30 23:59）尚未被 API 索引，本期补录。10-31 / 11-01 为周末，窗口内没有新的提交日期。逐 id 与往期比对：30 条均为新条目。另外用不带日期过滤的全量探测复核了窗口左端：10-29 比上期多出 **6 条晚到**（联邦学习量化、基因组可解释性、LiDAR token 选择、病理 SAE、分布学习、TensorSSM），均属应用/理论方向，未收录。按算子开发相关性精选 **5 篇**。

### 线性注意力与序列 kernel

- **[SparseGDN: Chunk Sparse in Linear Attention](http://arxiv.org/abs/2610.41420v1)**（`2610.41420`，cs.LG/cs.AI/cs.CL，10-30）：GDN 的 long-context prefill 受限于 dense intra-chunk 计算、recurrent-state 更新和中间 state 存储。SparseGDN 用 block-adjacent solver 降低局部 Delta-solve 计算，并用 input-dependent 动态选择器在 head pair 间分配 recurrent-update 预算；GPU 实现只为被选中的 event 物化更新，再用 compact state snapshot 与 group-local 交互重建每个 token 的输出，减少计算与访存。Qwen3.5-4B/9B 上平均只保留 **29%** 的 recurrent events，4K–128K RULER 的 GDN-core prefill 相对原生 FLA kernel 加速 **1.50–1.94×**（已含在线选择开销）。

### 低精度量化与 KV cache

- **[SACQ: Structure-Aware Codebook Quantization for Ultra-Low-Bit Vision-Language-Action Models](http://arxiv.org/abs/2610.41408v1)**（`2610.41408`，cs.CV/cs.LG/cs.RO，10-30）：把 LLM 的 PTQ 配方直接搬到 VLA 上，超低比特会明显掉点。SACQ 做 weight-only 的 vector-codebook 量化：按层的角色用二次误差度量拟合 codebook，按「压缩该层后最终 action 误差的增量」分配混合位宽，再用两阶段后量化细化（重排 codeword index 做二阶误差补偿、针对解码后的 action 校准 codebook 与 row-group scale）。GR00T N1.7 上 **1.5 bit** 的语言/action 权重在四个 LIBERO suite 平均 **93.85%** 成功率、显存降 **4.83×**；叠加 INT8 激活与低精度 attention 后仍有 **93.55%**，比 BF16 快 **1.51×**。
- **[Trainable Vector Quantization for Large Language Models](http://arxiv.org/abs/2610.41822v1)**（`2610.41822`，cs.LG，10-30）：VQ 权重压缩的瓶颈是离散参数没法用梯度直接优化，现有方法只能做逐层重建。TVQ 用 straight-through estimation 让 VQ 的 latent weight 可训练，从而直接最小化任意目标，并放进 block-wise knowledge distillation 里平衡质量与训练成本。LLaMA2-70B 上做到 **2-bit** 近无损，常见 QA 基准只掉约 **1%**；实现是直接的 PyTorch，训练预算不大。对算子侧的含义是：2-bit VQ 的 decode 需要高效的 codebook gather/反量化 kernel，精度上已经不再是硬限制。
- **[Where KV-Cache Error Lives in Looped Language Models: A Diagnostic Study of Ouro and Huginn](http://arxiv.org/abs/2610.41825v1)**（`2610.41825`，cs.DC，10-30）：looped LM 每个 loop 都保留一份 KV，量化误差会沿 loop 累积。作者用 Ouro-1.4B（4 loops）与 Huginn-0125（32 loops）做 dense/模拟量化：按单位误差计，value 占对角 Fisher 质量的 **94%**，但实测 KL 主要由量化误差更大的 key 主导；cell-level allocator 能在 **25–48%** 的 BF16 字节数下降低 one-step KL。对 KV-cache 量化 kernel 的直接启示是别只按 Fisher 质量分配比特，key 的误差方向与 loop 深度同样要进代价模型；Ouro-2.6B 上 measured-cost 分配相对 Fisher 的优势在 **31%** 预算处被反证。

### 硬件与性能

- **[Matching Workloads to Blackwell: Power-Resolved Benchmarking of NVIDIA B200 and B300 across HPC and AI Workloads](http://arxiv.org/abs/2610.41919v1)**（`2610.41919`，cs.DC/cs.AR，10-30）：B300 不是 B200 的均匀升级。两套独立 8-GPU 集群上，B300 在卷积推理上最多慢 **5.6×**、3D ViT 训练慢 **8×**、70B LLM 吞吐低 **24–27%**、双精度 Quantum ESPRESSO 慢 **16×**。作者用 per-GPU board power 解释卷积差距：受影响 workload 上 B300 只吃到 **1100W** 上限的 **36–53%**，而它赢的 ViT-Large 上两颗芯片都到 **95–97% TDP**；测试把输入常驻显存、排除 host 数据管线，因此差异可定位到**逐 kernel 成本**。同一硅片的两台 B300 仅因 vLLM 开发版不同，TTFT 就差 **2.1×**——对 kernel 选型和 benchmark 复现都是提醒。

> 其余 25 条多为 3D/视频/医学影像、时间序列、推荐系统与优化理论等应用向工作，或与 GPU 算子无直接关系，未收录。
