---
layout: post
title: "算子日报 · 2026-11-16"
permalink: /operator-daily/2026-11-16/
date: 2026-11-16 23:59:00 +0800
tags: [算子日报]
---

> 每日汇总：CUDA 开源仓 Release Notes（算子）、芯片动态、arXiv 算子内核论文。

## TL;DR

- **稀疏 attention 的「按 query 选、按 tile 算」**：Token2Tile 在低秩空间里做 per-query 打分与残差，再把选中的 key 打包成 GPU tile；MiniMax-H3 / Wan2.1-14B 上比 dense FlashAttention-3 端到端快 **1.48–1.54×**，VBench 掉分在 0.5 以内。
- **把通信从 SM 上拿开**：Zero-SM NCCL 用 stream memory operations + copy engine + CPU proxy 驱动 RDMA，GB200 NVL72 上 AllGather 带宽最高 **1.36×**，overlapped GEMM 的中位减速从 **33% 降到 4%**；代价是小消息延迟更高。
- **其余**：cuda-quantum 0.17.0 发布（量子框架，与 GPU 算子关联较弱）；Poincaré 卷积用 patch 统计量分解把训练更新加速 1.5–1.8×、峰值显存降 38–55%；另有 BPU 脉冲编码 matmul 与 Gated DeltaNet 的 FPGA 分页 state tile。

## 一、CUDA 开源仓 Release Notes（算子）

### cuda-quantum

- **0.17.0**（11-16 22:28 CST）— [链接](https://github.com/NVIDIA/cuda-quantum/releases/tag/0.17.0)：量子计算框架，与 GPU 算子内核关联较弱，供了解。新增独立的 `cudaq-core` 与 [EDEM](https://pypi.org/project/cudaq-edem/) 两个 PyPI 包，便于构建/分发编译器扩展与大规模 QEC 的 detector error model；编译器配置能力重做，新增 `optimization_level` 编译选项、target YAML schema v2-alpha 和更快的 `custabilizer` stabilizer 模拟后端；CUDA-Q Realtime 的 transport / dispatch API 大改，允许插入自定义 dispatcher。性能项里对算子侧稍有参考的是把 cuquantum 升到 26.09.0、cutensor 升到 2.8.1，以及 dynamics 路径改用 fabric-qualified 内存分配并复用（MNNVL）。

> 窗口内其余更新不计入：cuDNN Frontend 11-15 / 11-16 的两个 `1.33.0.dev*` nightly 自构建（自述 *Not a supported release*），按既有口径不计入；30 天窗口内其余正式 release（CUTLASS v4.8.1、CCCL python-1.3.0、NCCL v2.32.6-1、cuda-pathfinder v1.9.0、cuda-core v1.3.0、cuda-python v13.5.0）均已收录于往期。

## 二、芯片动态

> 本期无新条目。两个 RSS 源在 7 天窗口内均未返回命中关键词的新内容，窗口内没有芯片 / 架构类更新。

## 三、arXiv 论文（算子内核）

> 增量窗口（提交日 11-13 起）返回 **35 条**，全部提交于 11-13——这是 arXiv 索引本期才放出的迟到条目，逐 id 与往期抓取结果比对无重复；11-14 起的投稿尚未进入索引。按算子开发相关性精选 5 篇。

- **[Token2Tile: Token-Wise Sparse Attention with Tile Execution through a Low-Rank Space for Video DiTs](http://arxiv.org/abs/2611.17930v1)**（`2611.17930`，cs.CV，11-13）：视频 DiT 的 attention 二次成本是推理瓶颈，稀疏化的难点在粒度错位——每个 query 需要自己的 key 集合，GPU kernel 却按 tile 执行。Token2Tile 是 training-free 的稀疏 attention 算子：在由 query/key 在线算出的低秩空间里完成聚类、per-query 打分和残差近似，以保留 attention logits；再把 query cluster 选中的 key 按 tile 打包，做到「按 query 分配精确工作量、按 tile 执行」。MiniMax-H3 与 Wan2.1-14B 上相对 dense FlashAttention-3 端到端加速 **1.48–1.54×**，VBench 与 dense 差距在 0.5 分内；同速下在 H3 上比 SOL / SubBlock 的 PSNR 高 **5.5–6.8 dB**，在 Wan 上比 SOL 高 **4.6–4.8 dB** 且更快。
- **[Zero-SM NCCL: A Systematic Guide to Designing and Deploying SM-Free GPU Communication](http://arxiv.org/abs/2611.18075v1)**（`2611.18075`，cs.DC，11-13）：NCCL 用 SM 搬运数据和同步，会挤占同时运行的 GEMM，互连带宽越高代价越大。Zero-SM NCCL 完全不发 SM kernel：用 CUDA stream memory operations 同步，scale-up 域内用 copy engine 异步搬数据，跨节点网络 RDMA 由 stream-ordered 的 CPU proxy 驱动；支持 AllGather / AlltoAll 以及单边 `ncclPutSignal` / `ncclWaitSignal`。GB200 NVL72 与 B200 NVL8 上 AllGather 带宽最高达传统 SM kernel 的 **1.36×**，overlapped GEMM 的中位减速从 **33% 降到 4%**（单次操作延迟更高）；Megatron-LM 中 FSDP 最多快 **3.5%**、单边 put 让流水并行最多快 **15%**。作者也给出边界：大传输、GPU-bound 负载和深流水收益明显，小消息与 host-bound 场景会变慢。
- **[Efficient Poincaré Convolutions through Patch-Statistics Factorization](http://arxiv.org/abs/2611.18125v1)**（`2611.18125`，cs.LG/cs.NE，11-13）：HNN++ 的 Poincaré 卷积要把每个重叠 image patch 映射到 Poincaré ball 再算 score，显式构造 ball-valued patch 很贵。作者证明截断后的 score 只依赖每个 patch 的两个统计量——平方范数，以及与各 filter 的内积——这两者都能由普通卷积加一个单通道 box filter 直接得到，因此分解后的层**精确**等价于原 score，却不需要构造 ball 值 patch；单精度输出/梯度与双精度的接近程度不劣于显式参考实现。RTX A6000 + JAX 上 Poincaré U-Net / ResNet-20 的完整训练更新比两种显式实现中较快者快 **1.5–1.8×**，峰值显存降 **38–55%**；替换官方 PyTorch Hyperbolic U-Net 的卷积后训练 step 快 **1.4×**、内存减半。
- **[BPU-SpikeMM: Exact Matrix Multiplication via Spatio-Temporal Spike Encoding on Real Neuromorphic Hardware](http://arxiv.org/abs/2611.18275v1)**（`2611.18275`，cs.DC，11-13）：脉冲计算与矩阵乘在数学上等价，不等于能在真实神经形态硬件上直接读出结果——后者可能还需要数据恢复、bit-weight 重建或状态读出。BPU-SpikeMM 给出 BPU 执行范式下时空脉冲编码 matmul 的精确等价性证明，区分「数学等价」与「native direct-readout 等价」，并实现支持多种编码方案的 offloading pipeline；在真实 BPU 上按正确性、时间/空间资源开销和事件驱动指标评估编码设计空间，所有被测编码都通过 direct-readout 等价验证。对算子开发的意义主要是另一种硬件上的 matmul 映射与编码代价模型，而非 GPU kernel 直接可用的技术。
- **[The State Is Not the Bottleneck: A Paged-State Gated-DeltaNet Tile on a Tri-Mode Multicast Mesh, Bit-Exact from RTL to Silicon](http://arxiv.org/abs/2611.18422v1)**（`2611.18422`，cs.RO/cs.AR/cs.LG，11-13）：线性 attention 用固定大小的 recurrent state 取代增长的 KV cache，但边缘 FPGA 的片上 memory 装不下 state，必须分页到外部存储。作者在 Agilex 5 FPGA 上实现一个完整 Gated DeltaNet 层的 tile：模型 state 需 6.2 MiB，器件只有 0.87 MiB block memory，分页带来每 token 26.0 MB 流量、最多占总流量 11.2%；75 MHz 下该层每 token 11.15 ms，其中权重矩阵向量乘占约 62%、含分页的 state 阶段只占 13%。结论是对这类模型和存储接口，解码瓶颈在权重搬运而不是 state 分页；tile 作为 tri-mode 处理单元 multicast mesh 的一个节点，RTL 到硅片 bit-exact 通过 18 层仿真与板上多轮 64/1024 token 测试。
