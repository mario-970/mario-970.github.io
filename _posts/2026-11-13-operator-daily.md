---
layout: post
title: "算子日报 · 2026-11-13"
permalink: /operator-daily/2026-11-13/
date: 2026-11-13 23:59:00 +0800
tags: [算子日报]
---

> 每日汇总：CUDA 开源仓 Release Notes（算子）、芯片动态、arXiv 算子内核论文。

## TL;DR

- **消费级 Blackwell 上的 FP8×FP4 MoE 算子**：GC-MoE 针对 SM120 只有 warp 级 `mma.sync.aligned.kind::mxf8f6f4`、没有 TMEM / 双 CTA MMA / cluster multicast 的约束，把 token dispatch 与 block-scaled MMA 重新做成一个持久化 kernel；8×RTX PRO 5000 上对 FlashInfer W4A8 MegaMoE 快 **1.522×**、对 Cake MegaMoE 快 **1.387×**，18 个 decode 点全胜。
- **Kernel 工程的两种减法**：MegaOps 不融合 kernel，改用独立 CUDA stream + 中间 tensor 的 tile 级依赖，拿到 ThunderKittens megakernel 的吞吐却保留 Triton/cuBLAS/cuDNN 的模块化；cuVein 从优化后的二进制重建指令级访存复用结构，指导下最多消掉 **90.18%** 的访存指令、kernel 最高快 **2.84×**。
- **低精度与稀疏的边界继续被推**：SoScale 用 Hessian 感知的 scale-code 搜索，在 MXFP4 / NVFP4 / HiF4 上按 layer 输出误差选 scale；TQIR 把 trellis 量化约束到可精确重建的 INT8/INT4 码字，让 edge LLM 走整数 GEMM；稀疏 3×3 卷积的算术强度从约 **96** 降到约 **10 FLOP/byte**，直接稀疏卷积在 A100 上最高快 **1.65×**。

## 一、CUDA 开源仓 Release Notes（算子）

> 本期无新的受支持 release。窗口内新增的只有 cuDNN Frontend 11-12 / 11-13 的两个 `1.33.0.dev*` nightly 自构建（自述 *Not a supported release*），按既有口径不计入；其余正式版（CUTLASS v4.8.1、CCCL python-1.3.0、cuDNN Frontend v1.32.3、NCCL v2.32.6-1、cuda-core v1.3.0、cuda-python v13.5.0、cuda-pathfinder v1.8.6）均已收录于往期。

## 二、芯片动态

> 本期无新条目。窗口内 RSS 仍只返回 11-09 的 [NVIDIA 与韩国 MSIT 扩大主权 AI 合作](https://blogs.nvidia.com/blog/korea-sovereign-ai-collaboration/)，已收录于 11-09 期。

## 三、arXiv 论文（算子内核）

> 增量窗口（提交日 11-10 起）返回 **117 条**，实际提交日集中在 11-10～11-12。其中 CAFIN、KernelMosaic、cuVein 三条 11-10 论文是往期抓取窗口里没有、本期才被 arXiv 索引出的迟到条目；逐 id 与往期各期比对无重复，按算子开发相关性精选 11 篇。

### MoE 与 kernel 融合

- **[GC-MoE on SM120: Overlapping Expert-Parallel Token Dispatch with Warp-Level FP8xFP4 Block-Scaled MMA on Consumer Blackwell](http://arxiv.org/abs/2611.16219v1)**（`2611.16219`，cs.DC/cs.LG，11-12）：Expert-parallel MoE decode 的每一层都在「token 交换 → grouped GEMM → combine」上串行，而每卡在飞的 token 很少；此前的 GC-MoE 已在 Hopper（SM90）和数据中心 Blackwell（SM100）上把三段融进一个 persistent、Green-Context 分区的 CUDA Graph。这篇把它重做到 SM120（compute capability 12.0，如 RTX PRO 5000）：该架构只通过 warp 级 `mma.sync.aligned.kind::mxf8f6f4` 暴露 block-scaled tensor core，没有 TMEM、双 CTA MMA 和 cluster multicast，于是 kernel 改为 swap-AB 的 warp 级 block-scaled MMA——FP4 权重放寄存器 A 操作数、FP8 激活放 B，并补上通过 MMA 立即数选 scale byte、compact-weights 路径、自适应 token-tile 几何、按 expert 粒度的流式 dispatch（批量 bulk-pull 引擎 + 64-bit 去重 pull cache）、position-major arena 上由常驻 relay copier 排空的 BF16 返回，以及把 shared expert 折进计算 role。8×RTX PRO 5000 over PCIe 上跨 18 个 decode 点对 FlashInfer CuTe DSL W4A8 MegaMoE 快 **1.522×**、对 Cake MegaMoE 快 **1.387×**（各 18/18 全胜），融合 shared expert 后对 Cake 快 **1.388×**；94 个数值用例和 752 个 per-rank 检查全部通过，跨算子 cosine 相似度 ≥ **0.9999997**。
- **[MegaOps: A Kernel Overlapping Framework with PyTorch-like Modularity and Megakernel Performance](http://arxiv.org/abs/2611.16248v1)**（`2611.16248`，cs.DC，11-12）：kernel-per-operator 框架的依赖以整个 kernel 为粒度，kernel 之间形成 barrier，导致 SM 空转、计算与通信串行、后一个 kernel 也无法在前一个执行时预取权重；megakernel 能拆掉 barrier，代价是放弃模块化。MegaOps 的关键观察是「拆 barrier 不必融合 kernel」：把有依赖的 kernel 放到不同 CUDA stream 上，再把依赖下沉到中间 tensor 的单个 tile——生产者和消费者只按数据 tile 同步、不按 thread block 互相等待，因此任何采用相同 tiling 的两个 kernel 都能在不改代码的情况下组合。在单卡 H100/A100/L40 与 4×A100 / 8×H100 上跑 dense Llama3、Qwen3：吞吐追平手调 ThunderKittens megakernel，decode 达到最好基线的 **1.09–1.10×**、prefill 为 **0.98–1.10×**，且复用的是现成的 Triton、cuBLAS、cuDNN kernel。
- **[CAFIN: CUDA-accelerated Adaptive Fisher Information Matrix Network Optimization via GPU-Native Kernel Fusion](http://arxiv.org/abs/2611.14589v1)**（`2611.14589`，cs.LG/cs.AI，11-10）：二阶优化器 AdaFisher 用对角 Kronecker 结构和指数滑动平均压低 Fisher 信息矩阵近似成本，但 PyTorch 实现每步要串行发多个 CUDA kernel、分配中间 tensor、反复读写显存，执行开销盖过了算法收益。CAFIN 不改优化公式，把 FIM preconditioner 构造、EMA 更新、梯度 preconditioning、动量累加、weight decay 和参数更新逐级融合进 CUDA kernel，并用共享内存复用、寄存器驻留计算和 in-place 操作减少访存与中间分配。ResNet-18 在 CIFAR-10/100、Tiny ImageNet 上，GTX 1660 Ti 与 RTX 3060 Laptop 实测 FIM 构造延迟最多降 **78.4%**、optimizer step 延迟降 **2.1–26.9%**，每步少分配约 **18.91 MB** 中间显存；数值在预设 FP32 容差内一致，收敛精度与随机种子下的表现可比。
- **[Reconstructing Fine-grained Memory Reuse Structure in GPU Kernels](http://arxiv.org/abs/2611.14741v1)**（`2611.14741`，cs.DC/cs.PF/cs.PL，11-10）：优化 GPU 数据搬运需要知道哪些指令访问同一份数据、复用发生在 thread / warp / block / grid 哪一层，但现有 profiler 只把访存行为归到单条指令，aliasing、动态索引和运行时参数又让静态分析一般性地失效；cycle-level 模拟能重建关系但成本不适合生产负载。作者提出两个表示：**Topological Distance** 给每条动态访存打上「包含其最近一次复用的最小执行 scope」标签，**PCDepGraph** 再把这些标签组织成 PC 级图；由于定义在逻辑执行 scope 而非瞬时硬件状态上，表示跨运行、跨 GPU 稳定。工具 cuVein 直接从优化后的二进制重建 PCDepGraph，不采样、开销一般 **<10×**。在 RTX 4090 与 H100 的 HPC / AI 负载上，按其指导的优化最多消除 **90.18%** 的访存指令、kernel 最高加速 **2.84×**，且 reuse-scope 分布在平台间保持一致，说明捕获的是程序结构而非硬件行为。

### Kernel 生成与性能建模

- **[KernelMosaic: An Open Dataset and Agentic Framework for CUDA Kernel Generation](http://arxiv.org/abs/2611.14604v1)**（`2611.14604`，cs.LG/cs.AI，11-10）：训练 LLM 写高效 CUDA kernel 需要覆盖多样计算需求的监督，而这类监督难造——规格异构，可用实现又来自反复测试与改进。KernelMosaic 把 CUDA 任务与经校验的解法、轨迹配对：任务以自然语言描述、参考程序、数学规格及三者交集表达，经人工评审与多样性筛选；agentic 框架带强 LLM 走实现、正确性修复、性能优化，并提供可复用的 CUDA 指导与执行反馈。用该数据 SFT 一个小开源模型后，KernelBench L1/L2/L3 正确率从 **4% / 0% / 0%** 升到 **88% / 82% / 16%**，每一级「比 PyTorch eager 快」的任务占比也上升；消融显示精选轨迹的监督显著优于直接回答，并促使模型持续测试与修改。
- **[Instruction-Aware Transfer of GPU Implementation Crossovers: a Tensor-Product Discontinuous Galerkin Operator on GB200, H100, and A100](http://arxiv.org/abs/2611.15032v1)**（`2611.15032`，cs.PF/cs.DC，11-11）：论文固定数值、输入和代码，在 GB200、H100、A100 上比较同一个 FP64 tensor-product DG 算子的最多六种实现（p=7–1023），看最快的路径如何随机器切换。p=511 时，fused-epilogue GEMM 路径在 GB200 上比 fused Tensor Core kernel 快 **5.0%**，在 A100 上却慢 **6.9%**；按标称峰值缩放能复现 A100 的排序，但在 H100 上失效——因为 profiler 定义的 FP64 预算里，每条指令的实际成本依指令类型而变。作者改为按架构相关成本给每条发射指令定价，用校准侧的 counter profile 和每个 kernel 一个校准因子，且不对目标机器拟合。留一 GPU 的回顾性评估中，p≥511 的路径时间在常见 MMA 形状上迁移误差中位数 **1.9%**，并成功预测 H100 形状上 crossover 的方向；不做任何拟合、只限制一张 A100 的 SM 数，也能复现 A100→GB200 的排序反转。结论是：对生产相关阶数，该模型还不足以挑出最快的融合路径，但能指出 crossover 的机制。
- **[Read Once, Answer Many: Parallel Prompting for Shared-Context LLM Inference](http://arxiv.org/abs/2611.15130v1)**（`2611.15130`，cs.CL/cs.AI/cs.DC/cs.LG，11-11）：文档 QA、数据标注、LLM 评测常对同一篇长文档提许多独立问题。prefix caching 去掉了重复的 prefill，但每个 decode step 仍要为每个问题各读一遍文档的 KV cache，decode 因此受显存带宽限制并随问题数增长。Parallel Prompting 把 P 个问题作为单个请求里的 P 条独立 stream 一次解码：每一步只为所有 stream 读一次共享文档，stream-id attention mask 在 attention kernel 内求值，保证每个答案只条件于文档和它自己的问题。实现只往 SGLang 三个文件加了约 80 行并加一个小 engine 包；mask 不需要稠密 tensor，每步形状固定可直接 CUDA Graph 捕获。CodeLlama-7B、8k–16k token 文档、每篇 64–256 个问题上，比 stock SGLang + FlashInfer 在 A100 快 **3.7×**、在 RTX A6000 快 **5.9–6.0×**，跨两个 SGLang release；在 2 GPU × 2 release × 6 workload × 6 input set 共 144 次计时中全部更快。代码已开源（[GitHub](https://github.com/jlab-nlp/parallel-prompting)）。
- **[TraceKV: Policy-Preserving Low-Rank KV Cache Compression for Reasoning Vision-Language Models](http://arxiv.org/abs/2611.15569v1)**（`2611.15569`，cs.CV/cs.AI/cs.LG，11-11）：推理型 VLM 的 KV cache 同时被长多模态上下文和不断增长的 reasoning trace 撑大；低秩压缩把 KV 存成低维 latent、还能叠加量化，但现有方法在压缩 reasoning KV 时会让模型预测策略逐渐偏离 dense 模型，导致推理失败。作者定位到两个原因：静态上下文与 on-policy reasoning 状态之间的 calibration 失配，以及不同 KV 状态的误差敏感度不均。TraceKV 是 training-free 框架，把 on-policy calibration、敏感度感知 SVD 和全局 rank 分配统一在「保持策略」的原则下，显著降低 reasoning 期间的 policy-KL。多个 VLM 与视觉推理基准上，与量化结合可做到 **5.64×** KV cache 压缩且推理精度接近无损；配套优化的低秩 attention kernel 带来最高 **1.83×** 端到端 decode 加速。代码将开源于 [GitHub](https://github.com/AK-DREAM/TraceKV)。

### 低精度量化与稀疏算子

- **[SoScale: Second-Order Scale Search for FP4 Quantization](http://arxiv.org/abs/2611.16426v1)**（`2611.16426`，cs.LG/cs.AI，11-12）：MXFP4、NVFP4、HiF4 这类实用 FP4 格式把 4-bit payload 与 block 级或分层 scale code 配对，精度对每个局部分布如何缩放非常敏感。默认的 max-abs scaling 保住动态范围，却可能把稀缺的 FP4 level 花在极少占据的 outlier 上，密集取值反而分辨不足；现有 scale 搜索最小化局部逐元素重建误差，优化的是 tensor 保真度而非 layer 输出保真度。SoScale 做的是分层 FP4 的权重 scale-code 二阶搜索：在 Hessian 感知的 layer 重建目标下搜索硬件合法的 scale code，并与 GPTQ 式误差恢复耦合，按对 layer 输出的影响选 scale。Qwen3 系列上，在 MXFP4、NVFP4、HiF4 三种格式下都稳定降低下游量化退化。
- **[TQIR: Trellis Quantization with Integer Reconstruction for Bandwidth-Efficient Edge LLM Acceleration](http://arxiv.org/abs/2611.15231v1)**（`2611.15231`，cs.AR，11-11）：自回归 LLM 推理常受外部显存带宽限制，而现有超低比特 trellis 量化在运行时还要把权重重建回浮点，并承担 Hadamard 变换与 decoder 的开销。TQIR 做算法-硬件协同：把存储精度 S 与解码权重 W、激活 A 的计算精度解耦；离线用浮点候选做 trellis matching，但把每个候选约束成 INT8 或 INT4 码字，运行时能精确重建并直接走整数 GEMM。为压低变换成本又保住整数权重路径，它把逆 Hadamard 变换挪到激活侧，采用 H128 子空间 Hadamard 变换，并用子空间对齐的 128×2 trellis tile 保持原 256 权重序列长度。TSMC 16nm 综合的 decoder 在 0.0509 mm² 内做到 **307.2 Gweights/s**，decoder 面积效率分别是 Ecco 与 Codeword Decomposition 的 **13.93×** 和 **1.52×**，说明固定码率 trellis 压缩能在匹配访存与 PE 速率的整数 GEMM 数据通路上降低权重流量。
- **[Bit-Sparsity Aware Acceleration with Compact CSD Code on Generic Matrix Multiplication](http://arxiv.org/abs/2611.15136v1)**（`2611.15136`，cs.AR，11-11）：AI 与通用计算对矩阵乘的需求同时覆盖浮点和量化整数，而 bit-sparsity 加速此前主要靠剪枝、量化等软件手段在神经网络里生效，通用计算场景不一定可用。Bit-Cigma 提出以硬件为中心的 bit-sparsity 架构：CCSD 编码在 canonical signed digit 表示上做片上稀疏化，把 CSD 编码成本减半同时达到最优 bit-sparsity；分段点积用多级 exponent matching 处理长 FP 向量；同一数据通路兼顾 FP 与 QINT。65nm、1GHz 实现下，处理 FP32 时相对 SOTA 的 Bitlet 达到 **3.2×** 性能、**6.1×** 面积效率和 **15.3×** 能效，且计算误差为零。
- **[Theoretical and Empirical Analysis of Sparse Convolution](http://arxiv.org/abs/2611.17212v1)**（`2611.17212`，cs.PF/cs.DC/cs.LG，11-12）：边缘设备上的 CNN 经训练期剪枝后卷积核高度稀疏，但运行时利用稀疏性会被不规则访存和 processing element 间的负载不均拖住。论文用 roofline 分析稀疏权重卷积：3×3 稀疏卷积的算术强度约 **10 FLOP/byte**，而 dense 版本约 **96 FLOP/byte**——稀疏化把计算降下来，同时也把它推向 memory-bound。作者给出面向高稀疏度的稀疏卷积实现技术，初步结果在 A100 上相对 dense 卷积最高 **1.65×**、在 Jetson Nano（Tegra X1）上最高 **1.52×**，支持把直接卷积作为现代加速器/边缘设备上稀疏推理的实用路径继续挖。
