---
layout: post
title: "算子日报 · 2026-11-17"
permalink: /operator-daily/2026-11-17/
date: 2026-11-17 23:59:00 +0800
tags: [算子日报]
---

> 每日汇总：CUDA 开源仓 Release Notes（算子）、芯片动态、arXiv 算子内核论文。

## TL;DR

- **MXFP4 训练可以逐位复现**：CertiSum 在量化时收集统计量、认证原生 FP32 累加是否已经精确，只对未通过认证的输出用精确 kernel 重算；AMD MI350X 与 NVIDIA GB200、不同并行布局下训练状态逐位一致，代价是大输入形状下步延迟增加 11.8–60.5%，但仍比 BF16 快最多 1.24×。
- **低精度要把累加也一起压**：MIAO 按脉动阵列的流水级分级收窄 MX 累加器，MXFP8/6/4 下 PE 阵列面积最多降 38.7%、能耗降 36.1%；MiSTiQ 把 KV cache 压到约 1.25 bit/元素并让 attention 直接在 NVFP4 下算；Adaptive Precision Cholesky 让 FP8/Ozaki INT8 的 GEMM/TRSM 保持误差界，GH200 上比 cuBLAS ADP 快 69%。
- **推理侧继续向 kernel 要吞吐**：Nemotron 3.5 Transcribe 按 pass 在 persistent megakernel 与融合 kernel 链间切换，H100 上比 vLLM 峰值高 1.39×；PoCoAttn 相对 dense FlashAttention-3 把视频 DiT 的 attention 加速 4.1×；RoShift 把 SiLU 激活变换成稀疏表示，RTX 5090 解码快 1.72×；RescuCore 给 Mamba-3 在 SM 内加专用执行单元，prefill 快 1.4–9.4×。

## 一、CUDA 开源仓 Release Notes（算子）

> 本期无新条目。窗口内只有 cuDNN Frontend 的三个 nightly 自构建——`1.33.0.dev82913247`（11-17）、`1.33.0.dev82596365`（11-16）、`1.33.0.dev82384066`（11-15），正文自述 *Not a supported release*，按既有口径不计入；cuda-quantum 0.17.0 已收录于 11-16 期；30 天窗口内其余正式 release（CUTLASS v4.8.1、CCCL python-1.3.0、NCCL v2.32.6-1、cuda-pathfinder v1.9.0、cuda-core v1.3.0、cuda-python v13.5.0）均已收录于往期。

## 二、芯片动态

- **[NVIDIA Announces Rubin HPC — A New GPU to Accelerate Agents for Science, Engineering and Analytics](https://blogs.nvidia.com/blog/rubin-hpc/)**（11-16 16:00 UTC）：SC26 上发布，Vera Rubin 平台新增面向科学计算、工程与分析的 HPC GPU，定位是让 agent 驱动的仿真与数据处理共用同一平台；官方博客暂未给出 HBM、FP64 等具体规格。

> 同期 RSS 另有 MLPerf Training/Endpoints 榜单、CUDA 20 周年回顾和 CUDA-X 科学计算库（cuPhoton / cuPotential / cuSpa）发布三条，均非算子内核更新，不计入。

## 三、arXiv 论文（算子内核）

> 本期增量窗口（提交日 11-14 起）返回 **98 条**：11-14 共 20 条、11-15 共 27 条、11-16 共 51 条，逐 id 与往期文章比对均未收录（上一期只看到 11-13 的迟到条目，11-14 起的投稿本期才进入索引）。按算子开发相关性精选 8 篇。

- **[CertiSum: Bitwise-Identical MXFP4 Training Across Platforms and Parallelism Layouts](http://arxiv.org/abs/2611.21584v1)**（`2611.21584`，cs.LG/cs.DC，11-16）：MXFP4 训练的求和顺序会随并行布局和 GPU 厂商变化，相同输入得不到逐位一致的结果。CertiSum 的关键观察是 MXFP4 乘积的 FP32 原生累加对很多求和顺序本来就是精确的：量化时顺带收集统计量，对原生 matmul 结果做认证，未通过认证的输出再用精确 kernel 重算，并在 attention 与 weight-gradient 中保持量化块跨布局不变、合并部分和不引入舍入。Qwen3-8B-Base 200 步实验中，AMD MI350X 与 NVIDIA GB200 之间、以及 MI350X 上 data/tensor/context 并行变化时训练状态均逐位一致；大输入形状下步延迟比非精确 MXFP4 增加 11.8–60.5%，但仍比 BF16 快最多 **1.24×**、比确定性 BF16 快 **1.70×**。
- **[MIAO: Microscaling Intermediate Accumulation Optimization](http://arxiv.org/abs/2611.19722v1)**（`2611.19722`，cs.AR，11-15）：MX 量化通常只压操作数，累加仍保留高精度，这部分硬件成本没有被量化本身解决。MIAO 面向 TPU / Trainium 这类脉动阵列，联合选择乘积精度与逐级累加器位宽：先用 MX block scale 校准压低中间值幅度，再让前级用窄累加器、后级逐步加宽，并用逻辑综合成本估计加校准过的精度预测器搜索配置。MXFP8/6/4 下 PE 阵列面积最多降 **38.7%**、能耗降 **36.1%**，相对纯 MX 量化困惑度最多只多退化 1.60%。
- **[MiSTiQ: MicroScaling Trellis Quantization for Extreme KV Cache Compression](http://arxiv.org/abs/2611.21216v1)**（`2611.21216`，cs.LG，11-16）：KV cache 的 codebook 矢量量化能压到很低 bit，但 codebook 大小受硬件限制，且多数方法仍要反量化回高精度再算 attention。MiSTiQ 把 trellis 编码与 MX 格式结合：用 beam search 在不需要大 codebook 的情况下完成编码，并推导 MX-Hash 直接在目标 MX 元素格式下生成重建值，使低精度 attention 计算真正可用。Llama3 与 Qwen3 上约 **1.25 effective bits/element** 时 WikiText 困惑度和 GSM8K 准确率优于基线，NVFP4 下两个模型的 LongBench 平均分均为最高。
- **[Adaptive Precision Cholesky with Error Guarantees](http://arxiv.org/abs/2611.19687v1)**（`2611.19687`，math.NA，11-15）：低精度格式要进 BLAS，缺的是误差保证。作者证明 GEMM 与 TRSM 都用定点 Ozaki 1 方案、或用低精度 MicroScaling GEMM 的 Cholesky，仍满足与标准浮点 Cholesky 相同的 backward error bound，并据此构造自适应精度 Cholesky：在给定 element-wise 误差界内逐面板选择精度。GH200 上自适应 Ozaki 1 比固定 7-slice 的 right-looking Cholesky 快 **29%**、比 cuBLAS Automatic Dynamic Precision 快 **69%**，且保持同样的误差保证；FP8 分解（breakdown 时加对角扰动）在 13 个 SuiteSparse 矩阵上作 PCG 预条件，迭代次数相对同一矩阵对角缩放后的 CG 最多减少 **12×**。
- **[High-Throughput Serving of Nemotron 3.5 Transcribe with Regime-Aware Megakernels and Kernel Fusion](http://arxiv.org/abs/2611.20095v1)**（`2611.20095`，cs.PF/cs.DC/cs.LG/eess.AS，11-15）：连续 batching 让每轮迭代的算子组合和最优融合粒度都在变，固定用 persistent megakernel 或固定用融合 kernel 链都不是最优。该服务引擎为每个 pass 按离线测得的交叉点在两者间选择，并可选择性跳过 MoE 层。H100 SXM 上吞吐达 **4,345× 实时**、为调优后 vLLM 峰值的 **1.39×**（低并发最多快 4.39×），B200 上 6,367×、比 vLLM 高 1.64×；WER 1.91–1.93% 与 vLLM 持平，400 ms p95 预算内可服务 8× 的并发请求。按 pass 选择本身也有收益：128 并发时比一直用 megakernel 高 23%，8 并发时比一直用融合 kernel 高 9%。
- **[PoCoAttn: Position-Content Collaborative Sparse Attention for Fast and High-Fidelity Video Generation](http://arxiv.org/abs/2611.20307v1)**（`2611.20307`，cs.CV/cs.LG，11-15）：视频 DiT 的 3D full attention 是推理延迟主项；静态稀疏 mask 抓不到输入相关的 attention，动态稀疏又要反复在线重建 mask、重复发现跨输入基本稳定的位置结构。PoCoAttn 把 attention 拆成输入稳定的位置先验（离线标定的相对 offset 表）与内容相关残差（在线按 head 自适应比例选 key），并对丢弃的 key 做补偿，另按去噪步建模早期稀疏误差的放大来分配每步预算。四个视频扩散模型、三类任务上 PSNR 比基线高 **1.9–5.0 dB**，相对 dense FlashAttention-3 attention 最多快 **4.1×**、去噪端到端快 **2.0×**。
- **[RoShift: Lazy Shift and Weight-aware Rotation for Activation Sparsity in LLM Inference](http://arxiv.org/abs/2611.20549v1)**（`2611.20549`，cs.LG，11-16）：SiLU 类 LLM 的激活天然稀疏度低，硬拉高稀疏率会明显掉精度。RoShift 是 training-free 的激活变换：lazy shift 把 SiLU 激活分布重新居中到 0 附近、把修正推迟到 FFN 末尾，再用同时考虑激活和权重的 rotation 降低稀疏化误差，并把 shift 与 rotation 所需的修正融合进 kernel 换成解码加速。50% / 60% 稀疏率下平均准确率比 TEAL 高 **1.60 / 3.29 个百分点**；RTX 5090 上 50% 稀疏时解码比 dense 模型快最多 **1.72×**。
- **[Integrated GPU Core for Mamba-3](http://arxiv.org/abs/2611.20409v1)**（`2611.20409`，cs.AR/cs.LG，11-16）：Mamba-3 的 MIMO、指数-梯形离散化和 RoPE 带来大量 element-wise 与串行操作，即使做了 kernel fusion，常规 GPU 仍受 kernel launch 多、数据搬运大、寄存器压力高拖累。RescuCore 在 SM 子分区内与 CUDA Core / Tensor Core 并列增加一个专用执行单元，共享寄存器堆、shared memory 和 warp scheduler，把离散化、归一化、RoPE、MIMO、recurrent state 更新与 output gating 流水起来，dense projection 仍走常规 GPU 资源。cycle-accurate 仿真 + RTL 综合下，prefill 快 **1.4–9.4×**、能耗降 1.4–5.0×，自回归解码快 1.3–2.8×、能耗降 1.2–2.6×；采用 BF16、state size 32 / MIMO rank 4 时面积开销约为 A100 die 的 5.9%。
