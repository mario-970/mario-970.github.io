---
layout: post
title: "算子日报 · 2026-11-26"
permalink: /operator-daily/2026-11-26/
date: 2026-11-26 23:59:00 +0800
tags: [算子日报]
---

> 每日汇总：CUDA 开源仓 Release Notes（算子）、芯片动态、arXiv 算子内核论文。

## TL;DR

- **NVFP4 注意力开始拼 kernel 内的寄存器与流水细节**：JetDiffusion 用可训练的稀疏 + 量化注意力（VSQA）配原生 NVFP4 自定义 kernel，90% 稀疏下注意力比 FlashAttention 快 **8–29×**，Wan2.1-14B 在 RTX 5090 上端到端快 **273×**；本期 NVIDIA 博文则给出 SM120 上 SageAttention3 的寄存器回收调度，把 N128 分数 tile 拆成两个 N64 槽位复用，注意力 kernel 延迟最高降 **1.64×**，优化已进 FlashInfer 主干。
- **低位宽的表示层继续细化**：Any-Rate 用「编码位平面」在 2–4 bit 区间做半比特步进的多精度嵌套量化，2 bit 保住 FP16 精度的 **69%**；BReD 用块级减法抖动稳住 4-bit EMA 优化器状态；ISTS 则从位切片里挖低阶位的 Hamming 聚簇复用，GEMM 平均快 **1.27–3.09×**。
- **线性注意力与分布式稀疏注意力同时补「稳定性」和「并行度」**：Stable Gated DeltaNet-2 把擦除算子对称化，给出 O(√T) 的状态增长界且保留 chunkwise 并行；SWGDN 以滑窗注意力替换短卷积、不改状态更新与 chunkwise 算法；L2C2 在 chiplet 上做分布式稀疏注意力，prefill / decode 分别快 **1.45× / 1.53×**。

## 一、CUDA 开源仓 Release Notes（算子）

> 本期无新条目。窗口内唯一的新发布是 cuDNN Frontend 的 nightly 自构建 `v1.34.0.dev85638137`（11-26 13:56 UTC），正文自述 *Automated nightly build from NVIDIA-internal CI. Not a supported release.*，按既有口径不计入（前一个 nightly `dev85344621` 已在 11-25 期注记）。CUTLASS `v4.9.0dev`（11-23）、NCCL `v2.32.7-1`（11-24），以及 CCCL python-1.3.0、TensorRT v11.5、cuda-python v12.9.11、cuda-core-v1.3.1、cuda-pathfinder-v1.9.1、cuda-quantum 0.17.0 均为往期已报道。

## 二、芯片动态

- **[How to Optimize NVFP4 Attention on NVIDIA RTX Blackwell Through Register Recycling](https://developer.nvidia.com/blog/how-to-optimize-nvfp4-attention-on-nvidia-rtx-blackwell-through-register-recycling/)**（11-25）：SM120（RTX PRO 6000 Blackwell）上 BF16 注意力的瓶颈在 MMA 而非 softmax，NVFP4 的 Tensor Core 吞吐更高，但 SageAttention3 原设计对每个 N128 分数 tile 串行执行 QK → softmax → NVFP4 量化 → PV，且寄存器已占 SM 的 **95.3%**（producer 3,072 + consumer 59,392 = 62,464），放不下第二块完整分数缓冲，softmax 与量化期间 Tensor Core 空转。文章把每个 N128 分数 tile 拆成两个 N64 寄存器槽，一半完成指数与 NVFP4 量化后立即复用其寄存器计算下一个 tile 的 QK，让相邻 tile 的 QK 与当前 tile 剩余的 softmax/PV 交错；在线 softmax 状态仍在整块 N128 上统一更新，因此不改变数值语义。RTX PRO 6000 Blackwell 上注意力 kernel 延迟最高降 **1.64×**（文中表格：4,096 causal 1.32×、4,096 noncausal 1.25×、7,040 causal 1.19×），Reactor 接入生产后在 LongLive、Helios、OmniDreams、Cosmos 3 Nano 上端到端延迟最多降 **15.34%**；核心优化已合入 FlashInfer 主干。

## 三、arXiv 论文（算子内核）

> 本期增量窗口返回 **150 条**，正好顶到 `max_results` 上限，但按提交时间倒序、最新一条为 11-25 18:59 UTC（恰好是 arXiv 当日 14:00 ET 公告截止），因此 11-25 与 11-24 的条目是完整的；被截断的是窗口最左端的 11-23 旧条目（当前快照里 11-23 只剩 2 条，均已在昨日快照中审阅过）。与昨日快照逐 id 比对，真正新增 **77 条**：11-25 提交的 **61 条**，加上 11-24 晚到索引的 **16 条**（昨日抓取时尚未进入 API）。**11-26 的投稿尚未进入索引**，留待下期补录。按算子开发相关性精选 9 篇。

- **[JetDiffusion: Pushing the Speed Limits of Video Generation with NVFP4 Sparse Attention](http://arxiv.org/abs/2611.33268v1)**（`2611.33268`，cs.CV/cs.LG，11-24 晚到）：视频生成的注意力是长序列瓶颈，稀疏化、低比特量化与步数蒸馏各自都能提速，但三者在激进 NVFP4 下联合适配极易掉质量。JetDiffusion 以可训练的 Video Sparse Quantized Attention（VSQA）为核心，配原生 NVFP4 计算 + kernel fusion 的自定义推理 kernel：90% 稀疏度下注意力相对 FlashAttention 在不同硬件上快 **8–29×**。作者给出分阶段训练配方，使模型稳定适配「量化 + 稀疏 + 少步去噪」，Wan2.1-14B 在 RTX 5090 上端到端生成快 **273×**，画质相当。
- **[Any-Rate: Half-Bit Multi-Precision LLM Quantization with Coded Bit-Planes](http://arxiv.org/abs/2611.34402v1)**（`2611.34402`，cs.LG/cs.AR/cs.CL，11-25）：嵌套多精度量化只读每个权重的前缀比特，就能从一个 checkpoint 导出多种精度，适合端侧按内存/能耗预算切换，但现有方法只支持整数比特，2 bit 附近精度塌陷。Any-Rate 在 2–4 bit 区间以 **半比特**步进、整体覆盖 2–8 bit：把部分位平面限制为 8 权重块上的扩展 Hamming 码字（0.5 bit/权重），每行跨精度共享一张 LUT，并对各精度分别设编码目标以摊薄的共享损失。四个 LLM 上 2/3 bit 达多精度方法 SOTA，2 bit 保留 FP16 精度的 **69%**、4 bit 与最好结果差 1 个点内；解码速度与现有 kernel 相当，LPDDR 能耗模型在同等精度下最多省 **27%** 能量。代码：`github.com/hoonki2002/Any-Rate`。
- **[BReD: Block Replay Dithering for Stable Low-Bit EMA Optimizer States](http://arxiv.org/abs/2611.33951v1)**（`2611.33951`，cs.LG/stat.ML，11-25）：低位宽存优化器状态能显著降低训练显存，但状态每一步都要反量化—更新—再量化，EMA 类状态的量化误差会被递归喂回下一步：偏差累积、方差被放大，衰减因子接近 1 时尤甚。BReD 把经典减法抖动搬到优化器状态量化上，用**确定性种子重放 + 块内共享抖动值**，既不额外存随机张量也不逐元素生成抖动。5 个优化器 120M、AdamW/Muon 1.1B/3.4B 预训练与 7B SFT 上，4-bit BReD 与全精度状态基本持平（预训练 PPL 偏移 ≤0.5、SFT ≤0.2、下游平均变化 ≤0.7 点），3-bit 也保持稳定收敛。代码：`github.com/ZhanHeshen/BReD`。
- **[A Differential Computing Accelerator for LLM Inference Leveraging Bit-Slice Transition Sparsity](http://arxiv.org/abs/2611.34630v1)**（`2611.34630`，cs.AR，11-25）：位切片架构沿比特维拆分量化的操作数，以利用切片级稀疏，但现有设计主要挖高阶切片的稀疏，低阶稠密切片里的复用机会被浪费。作者观察到 bit-slice vector（BSV）在 Hamming 空间高度聚簇、离少数代表模式只差几个比特翻转，于是 ISTS 用这些代表 BSV 作共享 **Steiner 点**，从 Steiner 点的部分和只用少量加减重建目标 BSV 的部分和；算法侧用 SATC 选 Steiner 点、FlipPrune 剪翻转以扩大可复用集合，硬件侧同时覆盖静态线性层权重与运行时生成的 attention K/V。相对 ANT、BBS、LUT Tensor Core、Transitive Array、MCBP-BRCR 平均快 **3.09× / 2.41× / 2.18× / 1.27× / 1.41×**。
- **[MixMoMo: Accelerating Extra-Large MoE Inference with Mixed-Precision GPU-NPU Execution on Mobile Devices](http://arxiv.org/abs/2611.33981v1)**（`2611.33981`，cs.AR，11-25）：端侧 MoE 用混合精度量化平衡质量与效率，但离线指标与 routing 分数反映不出专家对量化的敏感度，不同位宽又难在 GPU/NPU 间均衡分配，tile 级反量化还吃不满 NPU 的向量并行。MixMoMo 把离线量化影响画像与基于激活的运行时风险评估结合，做有质量约束的专家位宽分配；执行上按精度结构化调度以均衡 GPU–NPU 分工，并用 HVX 并行反量化提高 NPU 向量利用率。两个移动平台、三个 MoE 模型、五个基准上，解码相对统一 Q4_0 平均快 **3.72×**，相对最强混合精度 MoE 基线快 **1.53×**。
- **[L2C2: Enhancing Large-scale Chiplets for Agent Long-Context LLM Inference with Sparse and Distributed Attention](http://arxiv.org/abs/2611.33691v1)**（`2611.33691`，cs.AR，11-25）：长上下文推理把稀疏注意力铺到分布式 chiplet 上时需要上下文并行，随之而来的是复杂的分布式注意力。L2C2 在 partial-prefill 阶段提出 **Split-Comm Attention**，借 chiplet 的高带宽互连用通信换更少的内存访问，并与 Ring Attention 组合成按序列长度自适应的 Bi-Mode；解码阶段分析访存不均衡的成因，用静态交错 token 分配，再加动态 agile cache balancing 适配稀疏性偏斜。大规模 chiplet 上 partial-prefill 与解码分别快 **1.45×** 与 **1.53×**。
- **[Stable Gated DeltaNet-2: Understanding and Mitigating Recurrent State Instability](http://arxiv.org/abs/2611.34051v1)**（`2611.34051`，cs.CL/cs.LG，11-25）：Gated DeltaNet-2（GDN-2）用逐通道独立的擦除与写入增强状态更新表达力，但长上下文训练会发散。作者定位到擦除时读取方向与修改方向不对称，产生非正规擦除算子，方向增益逐 token 累积，即使没有新写入状态也会指数增长。Stable GDN-2 用**对称擦除**保证状态传播非扩张，并用衰减调节的残差写入保留写入灵活性；理论上给出 **O(√T)** 的状态增长上界，同时保留 chunkwise 并行计算。约 1.6B 参数模型上训练稳定：循环设置下 128K RULER 平均分 **+6.80**，混合设置下九个下游任务平均 **+0.62**。
- **[One Token, Many Lessons: Sliding-Window Attention within Gated Delta Networks](http://arxiv.org/abs/2611.33257v1)**（`2611.33257`，cs.LG/cs.AI，11-24 晚到）：从 test-time regression 的视角看，线性注意力的状态矩阵是在线学习的回归模型，其效果不只看更新规则，也看用于更新的 key-value 对如何构造。SWGDN 把 Delta-Rule 线性注意力里的短卷积换成**滑窗注意力**，对投影后的 Q/K/V 做内容相关的局部聚合，使连续更新所用的 KV 对更相似（在 GDN2 与 KDA 上都成立）；该局部混合可嵌入任意 Delta-Rule 线性注意力，**不改状态更新方程与 chunkwise 并行算法**。多个基准上优于原 Delta-Rule 基线。
- **[CAMPS: Composable Adapters for Mixed-Precision Solvers](http://arxiv.org/abs/2611.33703v1)**（`2611.33703`，cs.MS/math.NA，11-25）：混合精度求解器通常给每个组件固定精度，低精度求解器难以在更高精度求解器内部按深度复用。CAMPS 用迭代精化把不同精度的求解器串起来：adapter 在父精度算残差与更新，把修正求解交给更低精度的子求解器（直接法、迭代法或另一个 adapter），并对每个修正做残差检查，不增大才保留。作者证明 adapter 可放在任意兼容的精度边界而不破坏接口，成功输出满足根精度与停止容差决定的**后向误差界**，并给出每个边界的收敛充分条件（从叶子向根计算）。FP64/FP32/FP16 的稠密与稀疏系统上，稠密良态系统单次求解（含分解）比 FP64 直接法快至多 **9.6×**；与嵌套求解器 F3R 速度相当而峰值内存约为其 **1/3.4**；病态时 FP32 修正失败会被检测并回退 FP64。
