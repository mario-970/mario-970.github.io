---
layout: post
title: "算子日报 · 2026-11-05"
permalink: /operator-daily/2026-11-05/
date: 2026-11-05 23:59:00 +0800
tags: [算子日报]
---

> 每日汇总：CUDA 开源仓 Release Notes（算子）、芯片动态、arXiv 算子内核论文。

## TL;DR

- **生产 serving 把 kernel 融合推到模型级**：Runway 用 ring cache + 写入时 FP8 量化，并把 TP all-reduce 融进 FP8 GEMM（交换机内归约），57B 视频 DiT 在 4×H100 上把单 chunk 从 240.9 ms 压到 147.4 ms（**1.63×**，FP8 峰值 43%）；SLA 把 TPU 上的 gated delta attention 重排成单个 fused kernel，TPU v7x 单层比 Tokamax KDA 快 **1.6–2.5×**，2048 芯片预训练快 **11–24%**。
- **长上下文与 KV/状态压缩继续走「低秩 + 稀疏」**：SEED 发现循环状态能量集中在 rank-one sink，按 sink+低秩因子存储，Qwen3.8-Flash-Next 在 GB300 上每 GPU 吞吐提升 **1.2–1.4×**；NanoKV 对自回归视频 KV 做 cluster 中心 + 低比特联合残差，HY-WorldPlay 有效压缩 **26.09×**；免训练的混合稀疏 attention 让 256K decode 最高快 **2.23×**。
- **kernel 工具链与低精度可移植性**：TauLang 把时间作为一等公民，在静态调度加速器上联合求解 tensor 放置与周期调度，十个 kernel 平均 **1.40×** 延迟收益；Easy Multi-Precision 展示高层向量化代码在 double/single/half 的矩阵乘、复数矩阵乘与 FFT 卷积上仍能贴近硬件峰值。

## 一、CUDA 开源仓 Release Notes（算子）

### cuda-python

- **v13.5.0** — [链接](https://github.com/NVIDIA/cuda-python/releases/tag/v13.5.0)（2026-11-05）：跟进 CTK 13.5 新增 `cuTensorMapSetAllocationClass`、`CUtensorMapAllocClass` 枚举与 `CU_DEVICE_ATTRIBUTE_TENSOR_MAP_LARGE_TENSOR_MIN_SIZE`，TMA descriptor 的分配类和大 tensor 阈值可在 driver 层控制；加入 `cuFabricCpyBatchAsync`、`CUDA_FABRIC_COPY_BATCH_OP` / `CUDA_FABRIC_CPY_BATCH_NODE_PARAMS_V1`，以及 `CU_GRAPH_NODE_TYPE_MEMCPY_BATCH`、`EXTERNAL_ENGINE`、`FABRIC_CPY_BATCH` 等 graph 节点类型。`runtime` 增加 `cudaMemcpyBatchNodeParams`，`nvvm` 文档化 `-nvvm-latest` 编译选项；Python 3.10 支持移除、新增 3.15，`cudla` / `nvml` 枚举去前缀改名但保留旧别名。对写 TMA / graph 的算子代码，新的 tensor map allocation class 与 large-tensor 属性值得跟进。

### cudnn-frontend

- **v1.32.2** — [链接](https://github.com/NVIDIA/cudnn-frontend/releases/tag/v1.32.2)（2026-11-04 20:14 UTC，上一期出刊后发布）：修复 PyTorch SDPA 的变长 attention override 未遵循所选后端的问题，避免参数错误与意外的 cuDNN dispatch（[#2216](https://github.com/NVIDIA/cudnn-frontend/pull/2216)）。对显式指定 backends 的 vLLM / SDPA 调用方，varlen attention 不再被静默改道。

> 说明：cuDNN Frontend 11-05 另有一个 `1.33.0.dev79508903` nightly（正文自述 *Automated nightly build from NVIDIA-internal CI. Not a supported release.*），按既有口径不计入；30 天窗口内其余正式 release（CUTLASS v4.8.1、CCCL python-1.3.0 / v3.5.0、TensorRT 11.4、cuda-pathfinder v1.8.4 / v1.8.5）均已在往期各期收录。

## 二、芯片动态

> 本期无新条目。两个 RSS 源在 7 天窗口内均无芯片/架构类新命中。本节连续为空属**数据源侧**状况，不是抓取故障。

## 三、arXiv 论文（算子内核）

> 本期新增条目为 11-03 晚到补录的 9 条与 11-04 首次被 arXiv API 索引的 40 条；11-05 提交尚未被索引。11-03 晚到的 Runway Characters 与 Easy Multi-Precision 按往期「晚到补录」惯例收入。按算子开发相关性精选 **8 篇**。

### 低精度与量化 kernel

- **[Runway Characters: Serving a 57B Video Diffusion Transformer for Real-Time Conversation](http://arxiv.org/abs/2611.05642v1)**（`2611.05642`，cs.DC，11-03 晚到补录）：对话视频要求单用户、batch=1 下实时生成 704×1088、24 fps，57B 自回归视频 DiT 每个 chunk 四次去噪，4×H100 上每 chunk 只有 166.7 ms 预算。关键观察是每个 chunk 的 240 次 attention 都读 5,788 槽 KV cache，而新 token 只有 757 个，原 step 的 **29%** 是每次调用都重复的整 cache 工作；于是把 RoPE 与 FP8 量化挪到写入时一次完成，用 ring cache 和位置重置时的 delta rotation 维护历史，并把 tensor-parallel all-reduce 融进 FP8 GEMM 做交换机内归约。不启用融合 kernel 时 step 从 240.9 降到 159.3 ms（**1.51×**），启用后 147.4 ms（**1.63×**，FP8 峰值的 **43%**）；线上 99.5% 的会话平均高于 23.5 fps。对算子开发的启示是 decode 阶段的「重复整 cache 计算」应在写入时预计算，集合通信也可以直接折进 GEMM 的 epilogue / reduction。
- **[NanoKV: High-Ratio KV Cache Compression for Autoregressive Video Generation](http://arxiv.org/abs/2611.06377v1)**（`2611.06377`，cs.CV，11-04）：自回归视频生成的 KV cache 随 rollout 线性增长，现有压缩要么掉质量，要么还原全维 KV 后 attention 计算量不变。NanoKV 冻结主干，把跨 token / head 的共享内容存成 cluster 中心，再把 head-channel 联合残差编码成低比特系数；K 与 V 用不同拟合目标——K decoder 保下游 block 输出，V encoder / decoder 保部署量化路径上的重建。NanoKV-Fast 复用同一份系数做 mixed-dimensional attention，不改主干即可省计算。三个视频主干上均改善质量-存储折中；HY-WorldPlay 上有效压缩 **26.09×**（约 KVTC 的两倍）且 PSNR 高 **1.0 dB**，NanoKV-Fast 保留 **20.53×** 压缩并比 KVTC 端到端快约 **1.15×**。

### Attention 与稀疏 kernel

- **[DecoupAttn: Decoupled Sparse Attention for Training-Free Acceleration of Video Diffusion](http://arxiv.org/abs/2611.05917v1)**（`2611.05917`，cs.CV，11-04）：少步视频扩散的稀疏 attention 每个去噪步都要重新找关键 block，预算分配多按均匀或局部指标，跨步复用 mask 还会积累误差。DecoupAttn 观察到只要每个 key 都被补偿，被跳过 block 的误差会相互抵消：用语义 cluster 对全部 key 做补偿，再让固定位置的少数 block 做精确 attention 修正；Distortion Budget 以 rate-distortion 目标把精确计算分配到最能降低最终 latent 失真的 timestep / layer / head，Reassign Cache 每步把漂移的 key 重新分到缓存质心并刷新补偿，而已预算的 block 位置保持不变。推理只需标准 block-sparse kernel 加一个 fused compensation kernel，且免训练——离线用少量校准片段编译计划。
- **[Accelerating Long-Context Decoding in Large-Scale Hybrid LLMs Without Retraining](http://arxiv.org/abs/2611.06133v1)**（`2611.06133`，cs.CL/cs.PF，11-04）：混合模型里少数 full-attention 层在长上下文 decode 时仍逐 token 变贵。该工作不改训练、不动权重，把每层 attention 换成组合稀疏：最近 token 的滑窗、固定的 pinned prefix，再加固定预算的最相关 token；窗口大小与预算按层用多目标搜索确定。六个混合模型（4B–397B、三个家族）在 21 个长上下文数据集上保持基线精度，64K/256K decode 上 Qwen3.5-4B 最高快 **1.26× / 2.22×**，Qwen3.5-122B-A10B 为 **1.20× / 2.23×**，Ornith-1.0-397B 为 **1.16× / 2.06×**。
- **[Systolic Linear Attention: Co-Designing Tile-Grouped Gated Delta Attention for TPUs](http://arxiv.org/abs/2611.06828v1)**（`2611.06828`，cs.LG，11-04）：GDN、KDA、GDN-2 这类 gated-delta 线性 attention 的 tile 形状是按 GPU 的小 tensor core 与向量寄存器调的；搬到 TPU 的 256×256 脉动阵列后，每块矩阵一半空着，向量单元又成瓶颈。SLA 把 key 维度加宽到 256 并用 grouped value heads，把 decay gate 分成四个 64 通道 tile，chunk solve 全部用块矩阵乘在单个 fused kernel 里完成。TPU v7x 上完整 SLA-R4 层比开源 Tokamax KDA 快 **1.6–2.5×**（精确 chunk inversion 时 **1.4–2.4×**），收益主要来自 head sharing 与 kernel fusion；2048 芯片、6.5B–63.1B 稀疏 MoE 预训练比 KDA 快 **11–24%**，小模型匹配种子下困惑度还略低。

### 长上下文与状态压缩

- **[Shallow Encoding Emits Deep Memories](http://arxiv.org/abs/2611.06084v1)**（`2611.06084`，cs.LG/cs.DC，11-04）：混合模型每个 softmax / linear attention 层都带一份 KV 或循环状态，prefill 要跑满所有层，decode 每步还要读写整个 recurrent state。SEED 有两个观察：写深层记忆比读深层记忆容易，于是训练深层写路径的副本（emitter），从较浅 anchor layer 的 residual stream 写深层记忆，除最后一个 prompt token 外只跑 **63%** 的层；recurrent state 的能量集中在每个 head 固定方向的 rank-one sink，剩余部分是低秩工作记忆，因此把状态存成 sink + 低秩因子，Gated DeltaNet / Kimi Delta Attention / Mamba-2 的更新可精确扩展，周期压缩的开环误差有与压缩次数无关的界。权重冻结、不压缩 attention cache、每个生成 token 仍跑所有层；五个生产混合模型（30B 到 300B+）的六个 checkpoint 在 11 个官方考试上近乎无损，SGLang + GB300 上 Qwen3.8-Flash-Next 每 GPU 吞吐提升 **1.2–1.4×**。

### 数值与 kernel 工程

- **[TauLang: A Language and Compiler for Space-Time Programming on Statically Scheduled Accelerators](http://arxiv.org/abs/2611.06079v1)**（`2611.06079`，cs.PL/cs.AR，11-04）：Groq TSP 这类静态调度加速器没有 cache、仲裁器和硬件调度器，编译器必须定死每个操作的 slice 与发射周期；现有 kernel 语言靠硬件解决时序、把调度隐式编码在代码结构里，无法直接目标。TauLang 把时间做成一等公民：程序员用参数化显式时间调度操作、循环和组合 kernel，TauIR 代数校验时间合法性，再用 ILP 联合求解 tensor 放置与周期调度，用显式时间把相连操作聚成 timing cluster 剪枝搜索空间。在 TSP 类机器的周期精确模型上，联合搜索相对 space-first 流程在十个 kernel 上平均 **1.40×** 延迟收益；少量时间 pin 可把延迟保持在未定时调度一个周期内，并把联合搜索扩展到 **1,636** 个 GEMM block，而未定时搜索在 32 个 block 时就找不到调度。
- **[Easy Multi-Precision Acceleration via Vectorization](http://arxiv.org/abs/2611.05558v1)**（`2611.05558`，cs.PF/cs.DC，11-03 晚到补录）：该工作用高层向量化代码（把操作写成矩阵数学）在跨越近十年的多种加速器上考察精度与性能可移植性，覆盖内存带宽、实数/复数矩阵乘（double/single/half）以及用 FFT 实现的 2D 卷积。结论是这些高层环境仍能贴近硬件峰值，且「降低计算强度、换更高内存强度」的算法很有潜力。对算子库的意义是：多精度 kernel 不必为每代硬件手写，向量化抽象可以同时覆盖精度维与硬件代际；前提是数据布局与算法选择先把带宽 / 计算强度权衡做对。

> 其余与算子相关但本期未展开的还有：Phantom（相邻去噪步只重算 outlier token 的 temporal caching 加速器，平均 **4.0×** 加速、**5.5×** 节能）、PTQ4SAM3（W4A8 图像编码器，语义保持 PTQ 把 RefCOCO Pr@0.5 从 **42.1** 提到 **72.5**，但不新增推理算子）、Enabling AWS Cloud FPGA Development Using High-Level Vitis HLS（FPGA 高层综合流程）。
