---
layout: post
title: "算子日报 · 2026-10-19"
permalink: /operator-daily/2026-10-19/
date: 2026-10-19 23:59:00 +0800
tags: [算子日报]
---

> 每日汇总：CUDA 开源仓 Release Notes（算子）、芯片动态、arXiv 算子内核论文。

## TL;DR

- **本期增量只来自 arXiv**：10-16 提交的 37 条候选中精选 7 篇；CUDA 开源仓自 10-17 起无新 release，两个芯片 RSS 源也没有命中关键词的新帖。
- **注意力稀疏化与低精度量化是主线**：Sink-Sparse 用可复用的静态 sink mask 加 pooled-KV 分支，给 DiT 的稠密自注意力做免训练替换，端到端加速 1.54–1.57×；BS-Spec 用 BFP 位共享从目标模型直接派生 draft，配 LUT 变精度数据通路；SpikeQ 则针对 SNN 芯片的 **pre-fire 膜电位**做量化。
- **其余三篇偏向映射与训练系统**：qGAMMA 做混合精度加速器的算子映射搜索；LeMA 在多 GPU 上把精确 LMA 做到十亿参数（8 卡 7.97×）；SETAREH 用 Haar 小波稀疏化近似内积，减少重复向量运算的存储与传输开销。

## 一、CUDA 开源仓 Release Notes（算子）

> 本期无新 release。窗口（≥ 2026-10-17）内 10 个仓库均无新发布；10-16 的 cuda-core v1.2.2 与 cudnn-frontend nightly 已在上期说明，不重复收录。

## 二、芯片动态

> 本期无新条目。Developer Blog 源侧继续返回 0 条；NVIDIA Blog 在 RSS 窗口内没有标题命中芯片关键词的新帖。

## 三、arXiv 论文（算子内核）

> 本期 37 条候选全部提交于 10-16（arXiv 对周末提交的索引有滞后，尚看不到 10-17 之后的批次），逐 id 与往期各期比对无重合；按算子开发相关性精选 7 篇。

### 注意力与推理

- **[Sink-Sparse Hybrid Attention for Efficient Inference in T2I/T2V Diffusion Transformers](http://arxiv.org/abs/2610.22975v1)**（`2610.22975`，cs.CV，10-16）：DiT 推理中重复调用的稠密自注意力是主要瓶颈，作者发现预训练 DiT 的注意力里存在**可按层/头识别、且能跨生成复用的高权重结构**。据此提出免训练替换：用离线校准集得到的**静态 sink mask** 跳过大部分计算，再补一条轻量的 **pooled-KV 分支**保住全局信息，权重与采样调度都不改。Wan2.1-T2V-1.3B 上端到端 **1.54×**，Z-Image-Turbo 上 **1.57×** 且 GenEval2 表现相当——对算子侧的意义是稀疏 attention 的 mask 可以**离线算好、跨请求复用**，不必每步动态搜索。

### 量化与低精度

- **[BS-Spec: BFP-Based Bit-Sharing for Self-Speculative LLM Decoding with LUT-Driven Mixed-Precision Acceleration](http://arxiv.org/abs/2610.22877v1)**（`2610.22877`，cs.AR，10-16）：自推测解码的痛点是 draft 配置对负载不敏感、混合精度执行效率低。BS-Spec 用**块浮点（BFP）位共享**从同一个目标模型**部分加载权重与 KV cache 的高位**来派生多个 draft 表示、保留共享指数，**无需单独存一份 draft 模型**；离线校准按服务负载选出权重/KV 精度组合，运行时用 **LUT** 切换。配套的混合精度 BFP 加速器为 draft 与 target 都设计了 LUT 驱动的变精度数据通路。相对常规自回归解码最高 **1.9×**，相对所对比的位共享自推测基线平均再快 **67.2%**。
- **[SpikeQ: Spike-Guided Post-Training Quantization for Spiking Neural Networks](http://arxiv.org/abs/2610.23046v1)**（`2610.23046`，cs.NE/cs.AI/cs.CV，10-16）：SNN 的量化误差**只通过被翻转的脉冲影响后层**，因此 MSE 相同的误差可能造成完全不同的精度损失，而**翻转脉冲比例相同则损失相同**。SpikeQ 据此改写量化目标：权重逐级选择时**模拟下游神经元的脉冲误差**；膜电位则针对芯片实际比较的 **pre-fire 值**（而非 reset 后的状态）建 reset-aware 网格，把阈值放在舍入边界上，使当前时间步的舍入不翻转脉冲。2-bit pre-fire 膜电位下现有网格全部崩溃，SpikeQ 仍保留大部分全精度精度；权重与膜电位都 4-bit 时距全精度 2.59 个百分点以内，内存缩 **7.8–7.9×**、能耗降 **6.4–6.9×**。
- **[Reusing Calibration Gains across Model Representations](http://arxiv.org/abs/2610.22771v1)**（`2610.22771`，cs.LG/cs.DB，10-16）：研究 scale calibration 学到的更新能否**跨同一预训练模型的不同表示复用**。迁移流程只提取正的 group-wise gain，直接装进接收方的权重或**原生量化 scale**，接收方不做任何优化；在量化模型上**保持整数码、零点与存储布局不变**。Qwen3-1.7B 的浮点/GPTQ INT3 一对表示之间双向迁移，保留了接收方单独校准所得 text-loss 改善的 **76.9–90.5%**；把同一组 gain 在投影内打乱则双方都变差，说明**学到的 group 归属在迁移后仍然重要**；Llama-3.2-3B 对照保留 **98.6–99.3%**。对量化算子部署的直接含义是：**校准结果可以作为一种与表示解耦的增量复用**，不必为每种权重格式重跑校准。

### 加速器映射与训练系统

- **[qGAMMA: A Framework for Optimal DNN Mapping on Multi-Precision Accelerators](http://arxiv.org/abs/2610.23222v1)**（`2610.23222`，cs.AR/cs.AI，10-16）：面向同时支持多种定点/浮点精度的通用空间加速器，搜索**混合精度 DNN 的最优映射与加速器配置**。方法用领域定制的遗传算法，配一个**基于硬件综合结果标定的解析性能模型**来评估候选，目标是最小化指定的性能/能效/资源指标。摘要只给了框架与评估结论，未给出具体加速比数字——适合用来理解混合精度算子在不同 PE 阵列上的**映射空间与精度-能效权衡**，但暂时拿不到可复现的性能结论。
- **[Scalable Levenberg-Marquardt Training Framework for Deep Learning](http://arxiv.org/abs/2610.22925v1)**（`2610.22925`，cs.LG/cs.DC，10-16）：LMA 收敛快，但 Jacobian 的存储与计算随「样本数 × 参数数」膨胀，一直难以上规模。LeMA 给出 LMA 的**精确对偶形式**，把内存依赖从参数维数上移除；再用 **Jacobian 分片**在多个 GPU 上**即时并行计算**各分片，避免物化完整 Jacobian，使内存从「参数数平方」降为「batch 大小平方」，十亿参数模型也能做**精确 LMA 步**而不必用近似 PCG。一个 **10 亿参数 CNN** 在 8 卡节点上相对单卡 **7.97×**，强扩展效率 **99.7%**——其分片即时计算与通信模式对需要构造 Jacobian/Hessian 这类稠密算子的训练系统有参考价值。
- **[SETAREH Transform: A Sorting-Assisted Haar-Wavelet Sparsification Method for Approximate Inner-Product Computation](http://arxiv.org/abs/2610.23338v1)**（`2610.23338`，cs.LG，10-16）：针对「固定系数向量 × 连续到达的输入向量」这类重复内积，SETAREH 先对两侧做**同一序列的置换与 Haar 小波变换**，在截断前保持内积**精确相等**，然后只保留一部分变换后的系数-输入对，以可控误差换取表示与传输成本下降。合成高斯向量和 APPA-REAL 上的 CNN 年龄估计都显示可用**远少于原维度的配对**近似原内积。需要注意：论文**没有 GPU kernel 实现**，收益取决于可接受的近似精度以及置换元数据的存储/传输成本，更像一种可压缩表示而非现成算子优化。

> 未收录条目多为应用/训练方法向（视频几何校正、扩散模型设计空间、EEG/医学/自动驾驶、LoRA 与优化器理论、联邦学习等），与算子内核无直接关系。
