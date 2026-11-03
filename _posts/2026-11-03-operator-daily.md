---
layout: post
title: "算子日报 · 2026-11-03"
permalink: /operator-daily/2026-11-03/
date: 2026-11-03 23:59:00 +0800
tags: [算子日报]
---

> 每日汇总：CUDA 开源仓 Release Notes（算子）、芯片动态、arXiv 算子内核论文。

## TL;DR

- **Blackwell 低精度 attention 三连**：FLAMIA 用 MXFP8 与混合 NVFP4/FP8 训练路径，把含量化的 fwd+bwd attention 调用加速 **1.10–1.42×**；Jet-QSparse 用 NVFP4 KV + 原生 FP4 QK，在 64 block 预算下让 Qwen3.8-27B 的 128K decode 端到端快 **1.94×**；Integer-Exact 则把 MXFP4 权重行化成「一个 INT8 向量 × 2 的幂」，在 M5 的 INT8 矩阵单元上跑出与量化参考逐位一致的 W4A8。
- **非 NVIDIA 平台 kernel 继续卷**：FlyWheel 在 TPU v6e 上把 softmax attention 做到 **81%** BF16 峰值（4.14× RPA）；Ascend-SpMM 用 Cube/Vector 分路把 SpMM 相对 NPU 基线提升 **12.70–20.55×**；ReWY 重写 delta-rule attention 反向，GPU 上 **1.15–1.81×**、Ascend 950DT 上 **1.09–2.89×**。
- **kernel 工程方法论**：Megakernel 研究指出 CTA residency 会双向误判 overlap 机会，PDL 可把 ThunderKittens decode 步降 **5.7%**；KernelSeek 用「在决策点提问」的 coding agent 把 FlashAttention-3 从 **0.50×** 做到 **1.13×**，并产出不依赖 CUTLASS 的单文件 Hopper/Blackwell kernel。

## 一、CUDA 开源仓 Release Notes（算子）

> 本期无新的正式 release。cuDNN Frontend 在 11-01 / 11-02 / 11-03 各有一个 `1.33.0.dev*` nightly（正文自述 *Automated nightly build from NVIDIA-internal CI. Not a supported release.*），按既有口径不计入；30 天窗口内其余正式 release（CUTLASS v4.8.1、CCCL python-1.3.0 / v3.5.0、TensorRT 11.4、cuda-pathfinder v1.8.4 / v1.8.5、cuda-core v1.2.2）均已在往期各期收录。

## 二、芯片动态

> 本期无新条目。NVIDIA Developer Blog 源返回 0 条；NVIDIA Blog 在 7 天窗口内无新的芯片/架构类命中。本节连续为空属**数据源侧**状况，不是抓取故障。

## 三、arXiv 论文（算子内核）

> 本期增量窗口覆盖 10-31 至 11-03。首次抓取命中 `max_results=150` 上限，单独补抓窗口左端 10-31 后共 **164 条**（10-31 51 条、11-01 52 条、11-02 61 条）；11-03 提交尚未被 arXiv API 索引（最新 `published` 为 11-02 18:49 UTC）。逐 id 与往期各期比对，164 条均未被收录。另对 10-30 做追补探测，发现上期抓取后新公布 **17 条**，其中 MXCode、TileWeave、CLiS 与算子直接相关，按往期「晚到补录」惯例收于本期。按算子开发相关性精选 **16 篇**。

### 低精度与量化 kernel

- **[Integer-Exact Microscaling: Running MXFP4 Language Models on INT8 Matrix Units](http://arxiv.org/abs/2611.00486v1)**（`2611.00486`，cs.LG，10-31）：在 INT8×INT8 是最快矩阵通路的硬件（如 Apple M5 的 Neural Accelerator）上，MXFP4 权重必须先变成 INT8 操作数；现有 kernel 把每个 32 元素 block scale 都卷进 reduction。作者指出，只要一行的 block 指数跨度 ≤3，该行就等价于「一个 INT8 向量 × 一个 2 的幂 scale」，整个 linear 退化成一次 INT8 GEMM + 每输出一次 rescale；强制该条件只给受影响 block 增加至多 1/16 的最大舍入误差，七个模型中 **97.9–99.97%** 的行天然满足（Qwen3.8-Flash-Next 的 9500 万行中 **99.93%**）。W4A8 在 M5 上与量化参考逐位一致，dense 模型 prefill 比 MLX 快 **1.34–1.74×**，GSM8K **94.8%** vs MLX MXFP4 **91.0%**。
- **[FLAMIA: FLAsh MIcroscaling Attention for Fast and Causally Safe Training on Blackwell](http://arxiv.org/abs/2611.03344v1)**（`2611.03344`，cs.LG，11-02）：Blackwell 的 MXFP8 / 混合 NVFP4-FP8 矩阵吞吐要进 attention，得同时解决量化开销、共享 scale 导致的 causal 泄漏、以及前后向数值不一致。FLAMIA 用 token-block scaling 让量化操作数在矩阵转置方向复用，对 scale 泄漏加定向保护，统一前后向计算，并把量化融进 kernel；document-local scaling 与 workload-aware 调度让 packed 变长文档无需重排。MXFP8 路径把含量化的 fwd+bwd attention 调用相对最强基线加速 **1.10–1.40×**（B200）/ **1.06–1.42×**（B300），并在 Llama 类预训练上验证了稳定性。
- **[Jet-QSparse: Accelerating Agentic LLM Inference with 4-Bit Sparse Attention](http://arxiv.org/abs/2611.01022v1)**（`2611.01022`，cs.LG，10-31）：decode attention 的瓶颈是 KV 容量与吞吐。Jet-QSparse 把 KV cache 压成 NVFP4，query-key 直接在 Blackwell tensor core 上做原生 FP4 计算；dense scan 选 128-token block，再用 FP16 做稀疏 value 聚合、FP32 累加，兼顾精度与连续 tile 读取。128 block/KV head 在四个 hybrid LLM 上未检出精度损失；64 block 时 Qwen3.8-27B 在 128K 上下文、batch 16、单张 GB300 上端到端 decode 比 dense BF16 快 **1.94×**，KV 池 token 容量是 BF16 的 **2.58–2.76×**。
- **[SHq: Learnable Grids for LLM Weight Quantization](http://arxiv.org/abs/2611.02966v1)**（`2611.02966`，cs.LG，11-02）：block-scaled LUT 格式（NF4、BOF4，以及 Rubin 的 3-bit LUT 指令）把权重存成小网格的索引，但现有实现都固定网格。作者扫了 2–6 bit 发现没有单一固定网格最优（2 bit 是 BOFS、4/5 bit 是 NF、6 bit 是 INT），提出只含两个自由度的闭式网格族 SHq：companding 强度 α 控制间距从均匀到浮点，偏移 c 引入非对称；再用 FP8 block scale 的符号位在网格与镜像间选择，单个 block 的可用值最多翻倍。它只改 LUT 内容、可直接替换，5 个 3B–35B 模型上 3/4 bit 全部取得最低 KL，4.5 bit 时把 NVFP4 的 KL 降 **41–67%**。
- **[BitLLM: Efficient Low-Bit, Conversion-Hiding LLM Decoding on Mobile GPUs](http://arxiv.org/abs/2611.00547v1)**（`2611.00547`，cs.AR/cs.LG/cs.OS/cs.PF，10-31）：移动 GPU 上低比特推理的瓶颈不是算力，而是量化值与浮点之间的频繁类型转换，以及自回归 decode 时 KV cache 的动态访存。BitLLM 给 KV cache 做自适应布局与打包，保证整段 decode 的访问模式稳定；执行管线做成 integer-native，把类型转换推迟/重叠到计算热路径之外，关键 MatMul 始终留在整数域。相对 llama.cpp、MNN-LLM、LiteRT-LM、MLC-LLM 平均加速 **1.78× / 1.73× / 1.58× / 1.47×**，长上下文下每 token 能耗最多降 **88.9%**。
- **[MXCode: Vector Quantization for MX-Native Inference at Scale](http://arxiv.org/abs/2611.00403v1)**（`2611.00403`，cs.LG，10-30 晚到补录）：向量量化能把权重压到约 2 bit，但学到的 codebook 通常和原生 FP4 计算不兼容。MXCode 给出三条路线：PostScale 把学到的 codebook 量化到 MXFP4；PreScale 在 codebook 学习前先抽 MX scale；PreScale+ 再用 VQ 压这些 scale。三者都能在不物化高精度中间权重的前提下重建 MX-native W4A4 操作数。在 Qwen3-235B 上，GB300 的模型加载显存比 Native FP4 低约 **50%**、serving 吞吐接近 Native FP4；显存受限的 prefix 复用下 P90 TTFT 最多改善 **94.5%**。

### Attention 与序列 kernel

- **[FlyWheel: Attention Kernels Built for the TPU](http://arxiv.org/abs/2611.01426v1)**（`2611.01426`，cs.CL/cs.LG，11-01）：TPU 的 MXU 常因重复加载矩阵操作数、过多向量计算和依赖停顿而闲置。FlyWheel 重写 softmax 与三角求逆以减少向量工作，协调数据布局提高矩阵操作数复用，并重叠独立计算隐藏延迟；同时覆盖 softmax attention 与 Gated DeltaNet。TPU v6e 上 softmax attention 比 SOTA RPA 快 **4.14×**、比 Splash Attention 快 **1.80×**，达到 **81%** BF16 峰值；Gated DeltaNet prefill 最多 **3.61×**；接入 vLLM 服务 Qwen3.8-27B，prefill 吞吐最多 **1.56×**。
- **[ReWY: Faster Gradient Computation for Delta-Rule Attention Family](http://arxiv.org/abs/2611.01940v1)**（`2611.01940`，cs.LG/cs.DC，11-01）：DeltaNet / GDN / KDA / GDN2 的 chunkwise 反向会计算多余的伴随量。ReWY 把 chunk correction 的微分写成一个三角残差系统，直接消掉 inverse adjoint，把 7 个局部 WY 矩阵乘降到 3 个；再用 centered matrix contraction 替代逐对 vector-gate 处理，并让 scalar-gate reduction 复用 key-gradient 乘积、避免构造 key Gram 矩阵。变换在精确算术下保持递推与梯度不变。NVIDIA GPU 上完整低层反向几何平均加速 **1.15–1.81×**（H100/H200 上四个算子 1.18–1.70×），Ascend 950DT 含梯度转换的 public-autograd 路径 **1.09–2.89×**。
- **[Fast and Exact Relative-Position Attention via Structure-Aware Kernels](http://arxiv.org/abs/2611.01182v1)**（`2611.01182`，cs.LG/cs.DC，10-31）：相对位置 attention 的加性 bias 需要按 token 对做偏移相关计算，难以塞进 fused attention kernel。作者把它精确融进 FlashAttention 式 tiled 计算：offset-aware tiling 避免重复索引与访存，只算必要的 key/query-relation 乘积并在前后向复用，反向用 offset-based gradient sharding 消除 atomic 竞争。统一 Triton 实现覆盖多种相对位置机制；8K 序列上 fwd 加速 **1.8–6.4×**、bwd **2.0–21×**，峰值内存分别降 **2.9–128×** / **1.4–54×**，端到端转录快 1.8×、省 2.4× 内存。
- **[CISA: Retrofitting Dense Attention Models with Chunk-Indexed Sparse Attention](http://arxiv.org/abs/2611.03200v1)**（`2611.03200`，cs.LG，11-02）：CISA 不改权重、不做稀疏训练，用现有 key 的紧凑摘要选固定大小 KV chunk，避免 token 级索引和额外全量 index cache；每个 KV group 只发一个 indexing query，组内 query 共享选择结果，prefill 时 QShare 让相邻 query 复用选择。Nemotron 3 Ultra 的 BF16 单层测试里，1M prompt 末端 query 的 core attention prefill 快 **11.7×**、decode 快 **13.0×**；把 chunk scoring/selection 等全部 GPU 工作算进去仍有 **8.6× / 3.0×**。NVFP4、4096 chunk 预算下 MRCR/LongBench v2/GraphWalks 总分与 dense 差距 ≤1.5 个百分点；4 张 B300 服务 1M prompt、32 并发、4K 输出时端到端吞吐是 dense 的 **2.29×**。
- **[TileWeave: Heterogeneous Tiling for Sparse Prefill on Mobile SoCs](http://arxiv.org/abs/2611.00408v1)**（`2611.00408`，cs.LG，10-30 晚到补录）：block-sparse attention 的选择不规则，和移动 NPU 偏好大而规整的 tile 冲突。TileWeave 按跨 query 复用度分流：高共享 key block 走粗粒度 NPU tile，低复用 block 只对真正选中它的 query block 走 GPU；配套 NPU 原生 pooled-key selector，以及按 NPU 消费顺序发布 GPU partial 并在 NPU 现有 output store 里合并的 runtime。Snapdragon SM8750/SM8850 上 1k–16k token 的全模型 prefill 比 XAttention 快 **1.35–3.23×**、比 dense 最多快 **1.58×**，RULER/LongBench 精度与 XAttention 持平。
- **[CLiS: Plug-and-Play Attention Acceleration for Video Diffusion on TPUs via Certified Stabilization](http://arxiv.org/abs/2611.00444v1)**（`2611.00444`，cs.LG，10-30 晚到补录）：长上下文 TPU attention 里，online softmax 反复做 max reduction 与 accumulator rescaling 是瓶颈。CLiS 在进入 KV 循环前固定每行的 stabilizer，并逐 head 认证数值安全性，认证失败就退回 online softmax；它直接插进现有 fused-attention kernel，不改权重与架构。Wan 2.2 / LTX-2 在 TPU v6e/v7x 上把 DiT 单步延迟相对调优后的 online softmax 降最多 **19.6%** 且无可测质量损失；初步的 causal LLM 实验里 continued pretraining / SFT 吞吐分别高 **13.5% / 7.7%**。

### GPU 架构与 kernel 工程

- **[Understanding GPU Megakernels: Why Residency Can Mislead](http://arxiv.org/abs/2611.00495v1)**（`2611.00495`，cs.DC，10-31）：作者把 ThunderKittens 与 VDCores 两个 LLM megakernel 在 H100 上拆回独立 kernel 做对照，发现 CTA residency 会在两个方向上误判「有用 overlap」：驻留并不等于有活可干，也不等于瓶颈资源有空闲。用 programmatic dependent launch（PDL）把启动和操作数加载提前，TK 的 decode step 延迟降 **5.7%**；VDCores 在满足内存序时性能相当。但在整个算子依赖下，TK 的 persistent 执行虽能从 prefetch 获益，runtime 与完成信号的排序开销会吃掉收益；为通信预留 SM 可能多出 GEMM wave，而内存带宽饱和后填满空闲 SM 也几乎无益。
- **[Ask, Don't Equip: Question-Driven Coding Agents for Deep Kernel Optimization](http://arxiv.org/abs/2611.03517v1)**（`2611.03517`，cs.LG，11-02）：现有 GPU kernel 优化 agent 要么脚本化、要么完全开放，前者换样本效率、后者换搜索次数，共同问题是缺少「剩余差距在哪里」的方向。KernelSeek 用通用 coding agent + 最小 harness，在决策点让 agent 回答一个简短问题（先由人给，后来自问题库），不限制下一步动作。同任务/评测器/硬件/周内，工具丰富的自主 agent 在 133 步后停在 **0.50×** FlashAttention-3，KernelSeek 做到 **1.13×**；随后产出 Hopper/Blackwell 的单文件 FlashAttention forward kernel，不依赖 CUTLASS/CuTe/cuBLAS/cuDNN，每种架构 182 个场景全对，几何平均 **1.16×** FA3、**1.46×** FA4。
- **[Ascend-SpMM: Accelerating Sparse Matrix-Dense Matrix Multiplication on Ascend NPUs](http://arxiv.org/abs/2611.01936v1)**（`2611.01936`，cs.DC/cs.DB，11-01）：Ascend NPU 的 Cube/Vector 物理分离和单线程标量派发让 SpMM 很难直接套 GPU 方案。Ascend-SpMM 建了一个以访存为中心的成本模型来量化硬件亲和性，再用「冲突有界、密度感知」的任务映射把稠密区域送 Cube、稀疏残差送 Vector，并给出跨处理器冲突的理论上界；子 kernel 用层次化流水和基于 round-robin ring buffer 的隐式同步掩盖访存延迟。相对各矩阵的最强 NPU 基线几何平均快 **12.70–20.55×**，端到端 GNN 训练最多 **6.8×**，性能与 SOTA GPU 实现相当。
- **[A comparison of parallel iterative methods for ILU triangular solves](http://arxiv.org/abs/2611.00552v1)**（`2611.00552`，math.NA，10-31）：精确三角求解的并行度受 level set 大小和数量限制，GPU 上通常用并行迭代法近似 ILU 的三角求解。作者比较 Jacobi、ISAI、GMRES 多项式与 AIRG（基于 GMRES 多项式的 reduction multigrid，并提出能约束 fine-fine 块对角占优比的并行粗细分）在 ILU(0)/ILU(1) 上的表现。AIRG 在 PDE 矩阵上最稳健、迭代数几乎不随加密增长且无需逐问题调参，但 setup 与内存开销最大；ISAI 作为内层求解时相对 hipSPARSE 精确三角求解的中位加速最高（ILU(0) **4.5×**、ILU(1) **6.8×**），不过主要由小问题贡献；最大的 ILU(1) 问题上四种方法中位加速都在 **2.1–2.6×**。

> 其余与算子相关但本期未展开的还有：AttentionIndex（倒排索引加速 learned sparse attention 选择）、ShardFly（融合布局转换/量化/通信的 persistent 权重更新 kernel）、k²-tree 上的 GPU SpMV、LoReS（recurrent state 量化）、MatMoE、GateTrim、SparseLift 等；其余大批条目属应用/理论方向，未收录。
