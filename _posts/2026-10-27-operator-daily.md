---
layout: post
title: "算子日报 · 2026-10-27"
permalink: /operator-daily/2026-10-27/
date: 2026-10-27 23:59:00 +0800
tags: [算子日报]
---

> 每日汇总：CUDA 开源仓 Release Notes（算子）、芯片动态、arXiv 算子内核论文。

## TL;DR

- **cuDNN Frontend v1.32.0 是本期最重的更新**：Mamba-2 进入 FROST 线性注意力族（SM100/103/107）；Qwen 稀疏 attention 训练提供显式 forward/backward，128K token、selection width 2052 时索引元数据从 16 GiB 降到 **256.5 MiB**；fully fused MXFP8 projection 可直接接 MXFP4 权重；并新增 CuTe DSL 的 LayerNorm/RMSNorm/GroupNorm 等归一化 kernel。
- **MXFP8 端到端 RL 的收益有明确前提**：sampler 与 learner 同量化网格时一致性最好；MoE step time 降 **10.7%**、生成降 **24%**，但短序列 dense 反而慢 **5%**，且不量化 KV cache 时 BF16 的 train-rollout 一致性仍高 **2.5–10×**；dense 场景要超过约 **4.5k token/micro-batch** 才回本。
- **kernel 侧三份实用结论**：TMA 在 Hopper 上的取舍被整理成按 operand 的 dispatch 规则并外推到 B200；Triage 用单个 fused kernel 在 RTX Blackwell FP4 tensor core 上执行 skip/NVFP4/残差三档 attention，最高 **2.32×**；FuseLSE 把 9 个 `torch.logsumexp` 子 kernel 合成 1 个，小负载快 **10–11×**。

## 一、CUDA 开源仓 Release Notes（算子）

### cuDNN Frontend

- **v1.32.0**（10-27）— [链接](https://github.com/NVIDIA/cudnn-frontend/releases/tag/v1.32.0)：一次覆盖线性注意力、稀疏 attention 训练、低精度 serving 与归一化 kernel 的大版本。
  - **Mamba-2 进入 FROST**：`cudnn.linear_attention.ops.mamba2` 提供 forward/backward/state-summary，支持 SM100/SM103/SM107、BF16/FP16、packed sequence、初始/最终 recurrent state、skip connection、输出 gating，state dim 128；checkpoint 可用 FP32，间隔须为 128 token 的倍数。
  - **Qwen 稀疏 attention 训练**：`qsa_forward`/`qsa_backward` 面向 Blackwell SM100/103 与 Rubin SM107，BF16 D256、12Q/1KV 或 24Q/2KV、4-token block。indexed 路径在 128K token、selection width 2052 下把选择元数据从 bitmap 的 16 GiB 压到 **256.5 MiB**，并避开 4 GiB selection bitmap；`selection_mode="causal_topk"` 提供有界 workspace 的两遍 backward。
  - **低精度 serving**：Rubin SM107 gated attention block 增加 paged **EXTEND/DECODE**，把 projection、可选 QK norm、RoPE、cache write、attention、gating、output projection 合成 block API；unfused MXFP8 forward 与 per-tensor FP8 backward 支持 packed sequence，fully fused MXFP8 projection 可直接接 **MXFP4 权重**；SM120 增加 MXFP8 prefill 专用 kernel 与 FP16/BF16 小 packed-query decode tile。
  - **SDPA 与归一化**：SM100/103 新增 native FP16/BF16 **Q/K576、V512** attention，支持 absorbed-MLA cache layout、paged KV、split-KV 与 sinks，并增加 paged per-tensor FP8 D256；CuTe DSL 新增 LayerNorm、RMSNorm、GroupNorm、InstanceNorm、BatchNorm 的 forward/backward，FP32/FP16/BF16 I/O，GroupNorm/InstanceNorm/BatchNorm 有原生 channels-last 路径。
  - **工程与兼容性**：FROST plan 可序列化（记录 engine、public knobs、graph IR，可选编译后 kernel）；JAX grouped op 扩展到 MXFP8/NVFP4/MXFP4/BF16，覆盖 GLU/dGLU、weight gradient、quantization、Hadamard fusion；Python binding 从 pybind11 迁到 nanobind，依赖 `nvidia-cutlass-dsl[cu13]>=4.8.0`，并移除一批 experimental alias。

> 本期其余更新不计入：`cudnn-frontend` 同日还有 nightly `v1.33.0.dev77004905`（10-27），正文自述 *Automated nightly build from NVIDIA-internal CI. Not a supported release.*，按既有口径不计入；30 天窗口内的其他正式 release（CUTLASS v4.8.1、CCCL v3.5.0 / v3.4.3 / python-1.2.1、TensorRT v11.4、cuda-pathfinder v1.8.4、cuda-core v1.2.2、cuda-python v13.4.4）均已在往期各期收录。

## 二、芯片动态

> 本期无新条目。NVIDIA Developer Blog 源继续返回 0 条；NVIDIA Blog 在 7 天窗口内唯一命中的仍是 10-21 的 Google Cloud × Vera Rubin GA，已收录于 2026-10-21 期。本节连续为空属**数据源侧**状况，不是抓取故障。

## 三、arXiv 论文（算子内核）

> 本期增量窗口覆盖 10-24 至 10-26 共 **98 条**（未触及 `max_results` 上限）：10-24 提交 20 条、10-25 26 条、10-26 52 条。上一期窗口里 10-24/10-25 为周末、10-26 尚未入库，因此这批条目此前均未出现在日报中；逐 id 与往期各期文章比对无一重合。按算子开发相关性精选 9 篇。

### 低精度与 GEMM

- **[End-to-End MXFP8 Reinforcement Learning on Blackwell: What Matching Buys, When It Pays Off, and How It Fails](http://arxiv.org/abs/2610.34915v1)**（`2610.34915`，cs.LG，10-26）：在 verl 里把训练侧（Megatron/Transformer Engine）与 rollout 侧（vLLM、SGLang）都切到 MXFP8，B200 上跑 Qwen3-8B 与 Qwen3-30B-A3B。核心结论是 **sampler 必须和 learner 用同一量化网格**：固定权重下，一半 linear 层同网格可把 train-rollout KL 降 **11%**，全同网格降 **16%**，同步后的权重与 learner 的 FP8 forward 权重逐字节一致；MoE 上换网格的 FP8 pipeline 从第一步起 KL 就翻倍。速度不是无条件收益——MXFP8 每 micro-batch 有固定开销、每 token 才省，dense 场景要超过约 **4.5k token/micro-batch** 才回本；端到端 MoE step time 降 **10.7%**、生成降 **24%**，短序列 dense 反而慢 **5%**，trace 显示新增时间主要是 GPU 等 host。作者还列出权重同步必须保持的 7 个条件并给出 live-engine 检查；7 次硬件故障中 5 次是静默的。注意实验均为单 seed、≤60 step，不构成最终模型质量结论。
- **[Accelerating SO(2)-Equivariant Materials Models through Low-Precision Emulation](http://arxiv.org/abs/2610.35164v1)**（`2610.35164`，cs.LG/cs.DC，10-26）：SO(2) 等变 GNN 的瓶颈是训练/推理里的高精度张量运算，直接换 FP8 会掉精度。FlashSO(2) 用 **Ozaki 整数分解**在低精度 MMU 上「模拟」高精度矩阵乘，在吞吐、精度、显存三方面同时超过 TF32。SO(2) 层上达到 MALOQ / UMA 参考实现的 **40× / 7×** 吞吐，在 GH200 上维持 **236 / 232 effective TFLOP/s**（约 FP32 峰值的 3.5×）；模型级 MALOQ 训练快 **3.9×**、峰值显存降 **58%**，UMA 分子动力学快 **3.3×**、显存降 **51%**，均无精度损失。对算子开发的启示是：Ozaki 这类「用整数 MMU 拼出高精度」的 emulation，可能比等待原生低精度格式更适合对精度敏感的科学计算算子。

### 稀疏 attention 与 KV cache

- **[Triage: Skip, Skim, or Refine Attention Tiles in Video Diffusion Transformers](http://arxiv.org/abs/2610.35706v1)**（`2610.35706`，cs.LG/cs.CV，10-26）：视频 DiT 的 attention 是数万 visual token 的瓶颈；现有稀疏/低精度方案对所有保留 tile 用同一精度，但作者观察到不同 tile 的近似误差差异很大。Triage 给每个 tile 三档：直接跳过（用 block mean 代替）、用 **NVFP4** 粗算、用 NVFP4 + 残差修正精算；一个轻量 router 从 block statistics 预测各档误差，只在误差收益值得额外计算时升级。三档由**一个 fused kernel** 在 RTX Blackwell 的 FP4 tensor core 上单 pass 执行。Wan2.2、HunyuanVideo、LTX-2.5 上端到端最高 **2.32×** 于 dense FlashAttention，五个设置里都快于所有 baseline，且 PSNR 比最高 PSNR baseline 高 **0.3–3.0 dB**。这给 attention kernel 的信号是：**tile 级异构精度 + 单 kernel 多档 dispatch** 可以同时拿到稀疏和低精度的收益。
- **[INTRA: Interleaved Non-contiguous Token spaRse Attention](http://arxiv.org/abs/2610.34056v1)**（`2610.34056`，cs.LG，10-25）：稀疏 attention 要在保持长程通信的同时映射到 GPU。INTRA 从 relay-communication 视角构造**互补稀疏组**：Scatter 层在局部邻域广采样形成 relay group，Gather 层在后续层重连，从而在 CQS 约束下用多层路径覆盖全局；配套的 CQS-aware FlashAttention-style kernel 在运行时计算结构化索引、加载非连续 token 并紧密打包。A100 上 2K FLUX.1-dev 生成 **1.53×**，LLaMA-3.1-8B 长上下文 prefill 效率提升；attention-only 基准相对 MoBA / density-constrained NSA 最高 **8.4× / 10.4×**。对写 sparse attention kernel 的人来说，重点是**索引运行时生成 + 非连续 token 打包**这条路径已经能落在真实 GPU 上。
- **[ZipFusion: Fusing Lossless KV Decompression with Attention for Efficient LLM Decoding](http://arxiv.org/abs/2610.33719v1)**（`2610.33719`，cs.AR/cs.DC，10-24）：现有 lossless KV 压缩在 attention 前先把 KV 解压回 GPU 显存，省了传输却省不掉重复的 KV 读。ZipFusion 让 attention **直接消费压缩 KV**：压缩格式与 attention kernel 协同设计，kernel 在寄存器里用简单位运算重建原值，省掉独立解压 pass；同时把压缩省出的显存按层重新分配给额外 KV，memory manager 给正在生成的请求预留空间。实现基于 FlashInfer 并接入 SGLang；H100 上 decode attention 比 FlashInfer 快约 **1.25×**，8K context 下同等显存预算的 decode 吞吐比未压缩 SGLang 高 **26.3%**，各自最大 batch 下达到最佳 lossless baseline 的 **2.76×**。这是一条明确的「**压缩格式 + kernel 共同设计、解压融进寄存器**」路线。
- **[RAQ-MLA: Role-Aware Quantization for Multi-Head Latent Attention Caches](http://arxiv.org/abs/2610.33625v1)**（`2610.33625`，cs.AI/cs.LG，10-24）：MLA 缓存的是压缩后的 KV latent 加一条窄 positional key，而现有 KV 量化方法按标准 attention 的 K/V 设计，低比特下精度损失明显。作者从激活分布和量化误差如何传播到 attention 输出两个角度分析，指出两个缓存张量角色不同：**latent 主要供 value，positional key 主导 attention score**。RAQ-MLA 因此对每个张量拟合可逆变换，分别最小化「attention 加权后的 value 误差」和「score 扰动」；无需训练即可把完整 MLA cache 压到约 **2.5 bit/element**，理论 KV 容量提升 **6.5×**，在四个 MLA 模型的推理/代码任务上接近无损。对 MLA attention kernel 的直接含义是：**latent 与 positional key 应分开量化、分开定误差目标**。

### GPU kernel 工程

- **[When and Why to Use TMA: A Workload-Grounded Characterization of Tensor Memory Acceleration on NVIDIA Hopper GPUs](http://arxiv.org/abs/2610.34907v1)**（`2610.34907`，cs.DC/cs.AR，10-26）：TMA 在生产 kernel 里用得很广但很挑 operand——CUTLASS、FA3、FlashInfer 都只对部分 tensor 用 TMA，其余退回 cp.async、直接 global load 或 DSM。本文在 H100 上做 workload-grounded 表征：梳理 FA3 forward 里 TMA/multicast/DSM/cp.async 的选择，用 microbenchmark 隔离 transfer-size crossover、multicast scaling 等判别特征，给出按 operand 选 load 机制的规则；还比较 TMA multicast 与 DSM 作为 cluster-broadcast 原语，并在 page size、datatype、model shape、执行阶段上拆解 FA3 paged-KV 的 TMA vs cp.async dispatch。最后用 B200 上的 CUTLASS GEMM 验证定性规则可跨代沿用，但绝对 crossover 随 tensor core 与带宽 scaling 移动。
- **[One Megakernel is Not Enough: Predicting Where to Cut for Streaming Inference](http://arxiv.org/abs/2610.35336v1)**（`2610.35336`，cs.PF/cs.DC/cs.LG，10-26）：问题不是「要不要 megakernel」，而是**融合应该延伸到哪里、哪些边界值得保留**。作者发现最优配置通常是多个 megakernel 的组合，或完全 conventional graph，在测试的 13 个 model×batch 组合里没有一个是单 megakernel。框架从 conventional-graph profile 与 standalone kernel benchmark 预测候选设计，模拟 tile/event graph 估计 time-to-first-output，无需端到端跑候选；并把模型分为 megakernel-stable / receding / advancing / averse 四类。H100 NVL 上 NH-TTS、batch 32 时，保留 3 个专用 megakernel 比单 megakernel 的 mean batch-p95 TTFA 低 **21.4%**、吞吐高 **28%**；相对 TensorRT graph，TTFA p95 低 **4.05×**、吞吐高 **19.7%**。对 kernel fusion 的启示：**边界本身有成本，但没有必要为了「一个大 kernel」牺牲专用化**。
- **[Speeding up Log-Sum-Exp: Kernel Fusion at the Memory Wall, Integer Arithmetic at the Compute Wall](http://arxiv.org/abs/2610.33647v1)**（`2610.33647`，cs.DC，10-24）：`torch.logsumexp` 在 GPU 上每次调用派发 **9 个子 kernel**，带来冗余 HBM 流量和 launch 开销；CPU 上 exp/log 则是算力瓶颈。FuseLSE 把 LSE 合成**单 pass fused kernel**，在 GPU 的 SFU 上算 exp/log；IntLSE 进一步用近似整数算术替换 CPU 上的 LSE。GPU 上小负载快 **10–11×**、大负载快约 **6×**，收益来自减少 launch 和 HBM 流量；CPU 上 IntLSE 比 FuseLSE 再快 **3–6×**。这是一个小而完整的案例：**基础归约原语也值得单独写 kernel**，尤其当框架实现是多个 elementwise/reduction kernel 串联时。

> 其余新入库条目多为应用/训练方法向（医疗影像、EEG、机器人、视频生成、遥感、推荐、安全攻击、科学计算应用等）或与 GPU 算子无直接关系的理论工作，未收录。
