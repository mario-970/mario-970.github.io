---
layout: post
title: "算子日报 · 2026-11-20"
permalink: /operator-daily/2026-11-20/
date: 2026-11-20 23:59:00 +0800
tags: [算子日报]
---

> 每日汇总：CUDA 开源仓 Release Notes（算子）、芯片动态、arXiv 算子内核论文。

## TL;DR

- **长上下文稀疏 attention 的瓶颈正从「选哪些块」转向「选择本身的开销」**：Twindex 发现主 KV 下沉到主机内存后，indexer cache 反成 HBM 瓶颈（1M token 时占单请求 HBM 的 63% / 75%），HBM 只留压缩副本、候选集回主机内存用原始行复选，1M 上下文下并发数提升 1.62–3.03×、decode 吞吐最高 2.54×；DTS 用 delta anchor 降低打分成本，128K 时打分流水线延迟比 XAttention 低 38%，512K 上 4×H100 prefill 达 3.3×。
- **MoBA / MLA 的 kernel 级改造继续吃满新精度**：LightningMoBA 用动态路由、KV-centric ownership 与 Hopper 运行时流水线，把端到端生成提升 1.20–1.60× 且 RULER 反升；AscMLA 用非对称 K/V 缩放让 MLA 直接处理单份 FP4 payload，比 FP8 cache 省 38.2% 显存，比 TRT-LLM 的 FP8 kernel 快 1.35×。
- **稀疏与量化往更细的维度走**：HiSA 把稀疏性放到 RoPE 的 hidden 维，只算约 1/3 的 QK 乘积还写进 FlashAttention-3；TRQ 用按视频在线拟合的 2-bit 平面旋转把 2-bit KV cache 的 VBench 一致性从 90.38 提到 92.02，并配了 fused Triton kernel。硬件侧 Splice-Core 的低位拼接 GEMM 把 tile 能耗降 77%、Kameleo 在 bit-serial 加速器上按 GEMM 在线选激活精度，MXU prefill 周期降至约 43–47%。

## 一、CUDA 开源仓 Release Notes（算子）

> 本期无新条目。与 11-19 期相比，窗口内新增的只有 cuDNN Frontend 的 nightly 自构建 `v1.34.0.dev83981319`（11-20），正文自述 *Automated nightly build from NVIDIA-internal CI. Not a supported release.*，按既有口径不计入（前一版 `v1.34.0.dev83610449` 已在 11-19 期注记）；其余正式 release（CUTLASS v4.8.1、CCCL python-1.3.0、cuDNN Frontend v1.33.0、TensorRT v11.5、NCCL v2.32.6-1、cuda-python 各包、cuda-quantum 0.17.0 等）均已收录于往期。

## 二、芯片动态

> 本期无新条目。两个 RSS 源在 7 天窗口内仍只返回 11-16 发布的 4 条：Rubin HPC、MLPerf Training/Endpoints 榜单、CUDA-X 科学计算库（cuPhoton / cuPotential / cuSpa）与 CUDA 20 周年回顾，均已在 11-17 至 11-19 期注记并判为非算子内核更新；NVIDIA Developer Blog 仍返回 0 条。

## 三、arXiv 论文（算子内核）

> 本期增量窗口（提交日 11-17 起）返回 **150 条**，触及抓取上限。逐 id 与 11-19 期快照比对，新进索引 **70 条**：11-18 追补 14 条、11-19 提交 56 条，且均未被往期文章收录。左端 11-17 因上限被截掉 37 条，单独核对后确认全部在 11-19 期快照内（已审阅范围）；11-20 当日投稿尚未进入 arXiv 索引，下期补录。按算子开发相关性精选 8 篇。

- **[Twindex: Index Cache Offloading with Coarse-to-Fine Selection for Long-Context Sparse-Attention Serving](http://arxiv.org/abs/2611.25078v1)**（`2611.25078`，cs.DC，11-18）：稀疏 attention 服务把主 KV 移到主机内存后，每步都要对全部历史打分的 indexer cache 反而成了 HBM 新瓶颈——1M token 时占 Qwen3.8-Flash-Next 单请求 HBM 的 **63%**、DeepSeek V4.1-Flash 的 **75%**。Twindex 在 HBM 只留一份「共享分量 + 压缩残差」的紧凑副本（离线学一次），每步先用它选出更大的候选集，再由模型自带 indexer 在主机内存的原始行里挑 top-k；不改权重、可跑在 CUDA graph 内。1M token 下可容纳的请求数是纯 KV offload 的 **1.78× / 1.62× / 3.03×**，主机内存不设限时 decode 吞吐为 **1.44× / 1.25× / 2.54×**，benchmark 均分最多低 3.1%。
- **[AscMLA: Unlocking FP4 Efficiency for Long-Context MLA Decoding through Asymmetric Scaling](http://arxiv.org/abs/2611.26106v1)**（`2611.26106`，cs.DC，11-19）：FP4 能把 KV cache 的存储和带宽一起砍掉，但块缩放下 K/V 的动态范围需求不同，而 MLA 的 K、V 共享同一份 latent，已有方案只能靠反量化或分开编码绕开。AscMLA 把**非对称 K/V 缩放**和流水化的 scale 下发结合起来，让 kernel 直接消费单份 FP4 payload，避免额外执行与存储。cache 占用比 FP8 少 **38.2%**，kernel 几何平均比 TRT-LLM 的 FP8 MLA 快 **1.35×**、比 CAKE 的 NVFP4 MLA 快 **1.26×**。
- **[Accelerating Mixture of Block Attention via Dynamic Routing, KV-Centric Execution and Runtime Pipelining](http://arxiv.org/abs/2611.26268v1)**（`2611.26268`，cs.DC，11-19）：MoBA 按 query 选 KV 块，FlashMoBA 却用固定 Top-k 预算和 query-centric 调度——固定预算无法适配不同分数分布，query range 归属又让多个 range 选中同一列时无法复用 K/V。LightningMoBA 三项协同：动态路由对每 query 的块分数设阈值、直接产出变长稀疏元数据；KV-centric ownership 按 (query head, KV block) 列聚合选中行、拆开高扇出列并复用已加载的 KV tile；Hopper 上让规则 K/V 传输、不规则 query gather 与 Tensor Core 计算流水重叠。匹配 Top-16 保留率下 5 个 checkpoint 端到端生成快 **1.20–1.60×**，RULER 均值从 74.42 升到 75.28。
- **[DTS: Delta-Based Tile Scoring for Efficient Long-Context Sparse Attention](http://arxiv.org/abs/2611.25938v1)**（`2611.25938`，cs.LG，11-19）：块稀疏 attention 省的是矩阵乘，但准确给块打分本身可能更贵。DTS 是 training-free 的内容自适应 tile 打分：按局部向量变化选 query / key anchor，冗余区域用更少代表点、异质区域保留更多细节；并用 parallel delta packing 把序列切成 tile 对齐的 chunk、并发做有界局部扫描来加速 anchor 选择，选中的 tile 仍用原始 Q/K/V 计算。LLaMA-3.1-8B-Instruct 上 128K 的 RULER / LongBench 与全 attention 差距 ≤0.10 个百分点，打分流水线延迟比 XAttention 低 **38%**；512K 时 4×H100 的 prefill 比 PyTorch + FlashAttention-2 快 **3.3×**。
- **[HiSA: Sparsifying the Hidden Dimension of RoPE Attention in LLMs](http://arxiv.org/abs/2611.26363v1)**（`2611.26363`，cs.LG/cs.CL，11-19）：几乎所有的稀疏 attention 都在 token 维稀疏，HiSA 换到 **hidden（特征）维**：Q/K 投影不变，但只按 token 位置的静态、无参数规则算部分点积。它利用 RoPE 的结构把 query / key 分解成可省略的部分，从头训练时只算约 **1/3** 的 QK 乘积，性能不降反升；推理时解码更快、显存更小，并已扩到 **FlashAttention-3** 的训练与推理 kernel。作者还显示 dense attention 本来就学得会抑制被 HiSA 丢掉的那部分分数（代码：github.com/ambroise-b/hisa）。
- **[Online Temporal Rotations for 2 Bit KV-Cache Quantization in Video Diffusion Transformers](http://arxiv.org/abs/2611.26334v1)**（`2611.26334`，cs.CV/cs.AI，11-19）：视频 DiT 跨去噪步复用 KV cache，量化时间上相距很远的 cache 向量会带来大范围和高误差，但相邻帧的 cache 存在逐通道的稳定模式——且这个模式随视频变化。TRQ 先对时间相邻的 cache 块拟合一个正交变换，把逐通道均值压到接近常数的方向，再用一棵 **2-bit 的 2×2 平面旋转树** + 全局共享 Hadamard 来参数化，保持量化格式轻量；旋转按视频在线拟合、不需要梯度也不改权重，贪婪解对「均值压平」目标全局最优，并配了 fused Triton kernel。2-bit 下 VBench 主体一致性从 QVG 的 90.38 提到 **92.02**（LongCat-Video-13B）、86.33 → 87.61（HY-WorldPlay-8B）。
- **[Splice-Core: Low-Order-Bit-Splicing GEMM for Highly Energy-Efficient AI Inference Accelerators](http://arxiv.org/abs/2611.25617v1)**（`2611.25617`，cs.AR/cs.AI，11-19）：Splice-Core 是 temporal-unary/binary 混合 GEMM 单元，目标是压低低位累加的延迟，同时让高位激活保持按值执行。两种 PE：Splice-BP 用 bit plane 组织，Splice-LUT 单周期选出共享的权重倍数。45nm 下 26 个 INT4 CNN/LLM 负载上，两者几何平均 tile 能耗分别降 **77.4% / 76.4%**；Splice-LUT 另把 tile 吞吐提 **44.4%**、EDP 降 **6.13×**；W8A8 卷积与 expert projection 的 tile 能耗降 77.3–88.1%，post-P&R 面积比二值脉动阵列小 14% / 9%。
- **[Kameleo: In-Situ Activation-Precision Selection for Bit-Serial LLM Accelerators](http://arxiv.org/abs/2611.25971v1)**（`2611.25971`，cs.AR，11-19）：weight-only 量化把推理成本推到激活算术上，bit-serial 变长加速器只算每层需要的尾数位，但位长一直靠离线校准（全模型 perplexity 搜索），换模型/换数据就要重搜、且会静默失效。Kameleo 改成**运行时在加速器上选**：用「同一 GEMM 在降精度与全精度下的输出之差」作局部代理替代全局 perplexity——bit-serial 数据通路让降精度结果本来就是全精度计算的中间态，这个比较几乎免费；用工作负载的前若干 token 跑两遍即可定下每个 GEMM 的精度。4 个 LLM、7 个数据集上在 1% perplexity 预算内把 MXU prefill 周期降至原来的 **1/2.11–1/2.34**，接近用评测数据离线校准的上限，而离线配置迁移到其他数据常超预算；选择逻辑只增 **3.7%** 面积。
