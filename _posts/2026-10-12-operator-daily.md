---
layout: post
title: "算子日报 · 2026-10-12"
permalink: /operator-daily/2026-10-12/
date: 2026-10-12 23:59:00 +0800
tags: [算子日报]
---

> 每日汇总：CUDA 开源仓 Release Notes（算子）、芯片动态、arXiv 算子内核论文。

## TL;DR

- **arXiv 积压窗口落地**：10-09 / 10-10 提交的算子论文这次入库了——增量抓取 45 条**全部为 10-09 提交**，与 10-09 / 10-10 两期逐 id 比对**零重合**（此前两期窗口为空，本窗口正是补录）。本期精选 10 篇：量化/低精度 3 篇（DualQuant W4A4、DuoRate MoE 精度分配、S³R 稀疏推测解码）、attention 与 KV 3 篇（CompileMask mask 编译器、Adaptive DHiSparse 分层稀疏 KV、Structured Block Skipping）、算子结构与访存 4 篇（ReplaySSM、Monarch-MoE、APA matmul 稳定性、CacheFlex L2-as-SPM）。
- **CUDA 仓本期无新正式 release**：≥ 10-10 只有 `cudnn-frontend` 三条自构建 nightly，按既有口径不计入。
- **芯片动态一条**：NVIDIA 在 OCP Global Summit 宣布**开放 Vera Rubin 架构**（机架/生态层面，无 kernel 级细节）。

## 一、CUDA 开源仓 Release Notes（算子）

> 本期无新条目。窗口（≥ 10-10）内只有 `cudnn-frontend` 的三条自构建 nightly——`v1.32.0.dev73250492`（10-12）、`dev73063366`（10-11）、`dev72889881`（10-10），正文均自述 *Automated nightly build from NVIDIA-internal CI. Not a supported release.*，延续既有口径不计入。30 天窗口内其余正式 release（CUTLASS v4.8.0、CCCL v3.5.0 / v3.4.3 / python-1.2.1、cuDNN Frontend v1.31.0、TensorRT v11.4 / v11.3、NCCL v2.32.3-1 与 nccl4py v0.6.0、cuda-python v13.4.3 / cuda-core v1.2.1 / cuda-pathfinder v1.8.3、cuda-quantum 0.16.0）均已在各期收录。

## 二、芯片动态

- **[OCP Global Summit: Amid Ecosystem Adoption, NVIDIA Opens Vera Rubin Architecture to Help Scale AI Factories Worldwide](https://blogs.nvidia.com/blog/ocp-global-summit-vera-rubin-scale-ai-factories/)**（10-12）：NVIDIA 在 OCP 全球峰会上宣布开放 **Vera Rubin 架构**并公布一批面向 AI 工厂规模化的合作与贡献——属于**机架/开放标准与生态层面**的架构开放，通稿未给出 kernel 级、精度格式或算子 API 细节，对算子开发者的直接可操作信息有限；列为架构动向备查。

## 三、arXiv 论文（算子内核）

- **[Fast and Stable Arbitrary Precision Approximate Matrix Multiplication](http://arxiv.org/abs/2610.13905v1)**（`2610.13905`，math.NA，10-09）：给 APA（任意精度近似）矩阵乘法——即**在精确算术下也只输出近似结果**的快速 matmul 算法——补上三块理论：一个自动生成编码矩阵的搜索算法（改进了 $\langle 2,4,5 \rangle$ 矩阵乘法的 border rank 上界）、用替代基把加法次数与主项系数压下来、以及证明 Bini 1978 那类"用改良 APA 算精确乘积"的方案具备与快速精确矩阵乘法同类的**数值稳定性**。对算子的意义：APA 是"用非精确算法换更低渐进复杂度"的路子，稳定性证明是它能进库的前提。
- **[ReplaySSM: Cache SSM Inputs, Not State](http://arxiv.org/abs/2610.13778v1)**（`2610.13778`，cs.LG，10-09）：把 SSM 递推展开成「检查点 + 近期输入缓冲区」，用**重放输入**代替每步读写 state——只在少数步更新检查点，把解码期占总流量大头的 state 访存砍掉约一半。同一机制让推测解码回滚退化为指针操作，还能让低精度检查点只服务输出、高精度检查点服务递推更新，**量化误差不进后续 state**。已在 vLLM 的 Mamba-2 与 Gated DeltaNet 上实现，batch 256 下解码加速 1.20–1.47×，120B 级模型上吞吐约为 vLLM 推测解码的 1.8×。
- **[DualQuant: Dual-scale Calibration-free Weight-and-Activation Quantization for LLMs](http://arxiv.org/abs/2610.13767v1)**（`2610.13767`，cs.LG，10-09）：**免校准的 W4A4 PTQ**——关键观察是「最小化权重重建 MSE 解出的缩放因子可以直接当激活缩放用」，于是激活量化不再需要校准数据。用变量分裂 + 块坐标下降求得闭式更新，**首次正式支持 MX 家族与 NVFP4 块格式**；Llama-3.1-8B 在 RTX 4090 上跑通真实 packed W4A4，相对 BF16 最高 2.61× 显存下降、2.46× 吞吐提升，精度基本对齐校准式 SmoothQuant。
- **[Adaptive DHiSparse: Speculative Decoding over a Load-Adaptive KV Hierarchy](http://arxiv.org/abs/2610.13708v1)**（`2610.13708`，cs.AI/cs.CL/cs.DC，10-09）：长上下文 serving 里 KV 显存先耗尽、算力还没打满——把历史 KV 摊在 host 内存、只取被选中的行，并用**融合回取（fused retrieval）**保证精确性，同时按负载自适应调节常驻比例（空闲时多留历史，吃紧时释放）。100K 上下文 / 并发 128 下比 HiSparse 快 1.45×，DeepSeek-V4.1-Flash 500K 上下文下是常驻方案的 1.85×，低负载时 host 取行占比从 14.2% 降到 1.4%。
- **[CompileMask: Bridging the Gap Between Expressiveness and Efficiency in Attention Masking](http://arxiv.org/abs/2610.13474v1)**（`2610.13474`，cs.DC，10-09）：现有 attention kernel 只原生支持少数 mask 模式，自定义 mask 要么走谓词（表达力够、开销大）要么走区间（快但受 interval 数限制）。CompileMask 把**可见性规则离线编译成 mask 专用的预处理 kernel**，在线只读 batch metadata 生成紧凑 Execution Descriptor，让 attention kernel **用元素索引比对预计算端点**来判可见性；配一套区域式 DSL（MaskSpec），相对 SOTA 最高 3.8× 加速。
- **[DuoRate: Two-Rate Per-Expert Precision Allocation for Mixture-of-Experts Serving](http://arxiv.org/abs/2610.13328v1)**（`2610.13328`，cs.LG，10-09）：指出 MoE serving 存在两套不同的「精度费率」——常驻显存按 router 能碰到的**所有**专家付，decode 每步只按被选中的专家付（即 **store-read gap**：BitsMoE 检查点上 2.54 bpw 的存储会变成 7.50 bpw 的解码流量），而现有 per-expert 分配器只约束前者。方案是离线存一份**前缀可读**的权重流 + 畸变张量，在线用 workload 自己的 prefill 无标签地解一个面向 decode 泛函的 Lagrangian 分配（路由频率可证明从每个专家决策中消掉）。Qwen3-30B-A3B 上 2.9 bpw 即比肩读取多 1.4–1.6 bpw 的 4-bit 格式。
- **[Monarch-MoE: Structured Experts for Multimodal Continual Instruction Tuning](http://arxiv.org/abs/2610.13215v1)**（`2610.13215`，cs.LG，10-09）：把 LoRA 专家里的**稠密低秩因子换成块对角因子 + 固定置换**（Monarch 结构），实现块级梯度隔离——单个块对更新只影响一个输出块，同时所有输入/输出块仍连通。为让紧凑专家跑在受限显存上，配套做了 batched 计算、**融合前向算子**与优化后的反向；适配器参数降约 75% / 50%，单张 24GB RTX 4090 可训可推。
- **[How Much Computation Does Every Forecast Need? Structured Block Skipping in Two Time-Series Foundation Models](http://arxiv.org/abs/2610.13208v1)**（`2610.13208`，cs.LG，10-09）：在 Chronos-Bolt 与 TimesFM 2.5 上检验**推理期整块绕过 Transformer block** 的可行性：mask 只在校准窗口上选，全开时严格退化为稠密模型；中等稀疏度（7/11、16/20 块）下 A100 batch-1 中位延迟降 11.2% / 13.4%。但更激进地跳过会显著伤害 Traffic / Weather，且数据集专属 mask 迁移性差、无法可靠路由——结论是**跨架构的结构冗余确实存在**，但还不是可部署的动态推理方案。
- **[S$^3$R: Fast and Faithful RL Rollouts via Self-Sparse Speculative Decoding](http://arxiv.org/abs/2610.13202v1)**（`2610.13202`，cs.LG/cs.CL，10-09）：profile 显示 RLVR 长上下文 rollout 中 **attention 占自回归解码 kernel 时间的 72–76%**。做法是用稀疏 attention 提议 token、再用**全 attention 验证 + 精确推测采样**产出响应，draft 与 verify 共用同一份 policy 参数（不需要额外训练 drafter），使 response group 仍服从稠密策略分布；配 KV-aware 调度避免 KV 反复换入换出。长上下文数学推理上端到端训练吞吐 1.39×，且避开了直接稀疏采样 + 训练侧修正的不稳定。
- **[CacheFlex: Direct Software-Managed Access to Higher-Level Cache for Scalable Vector Support](http://arxiv.org/abs/2610.12940v1)**（`2610.12940`，cs.AR，10-09）：ISA 与微架构协同设计，允许运行时把**私有 L2 的一部分转成软件管理的 scratchpad（SPM）**，由软件显式控制数据放置与生命周期，并解决跨域一致性与 OS 上下文切换，面积开销仅 L2 的 2.02%。基于 Armv8-A SVE 在 gem5 上以 GEMM、FlashAttention 与整层 workload 评测：向量宽度 VL 4→16 时 kernel 级几何平均加速 1.12→1.44×、内存子系统能耗最多降 40%，整层叠加 FlashAttention 最高 1.78×（BERT-Base，T=4096）。对 kernel 开发者的意义：**收益随向量宽度增长**，是软件管理片上存储这条路线在宽 SIMD 下的量化证据。
