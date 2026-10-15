---
layout: post
title: "算子日报 · 2026-10-15"
permalink: /operator-daily/2026-10-15/
date: 2026-10-15 23:59:00 +0800
tags: [算子日报]
---

> 每日汇总：CUDA 开源仓 Release Notes（算子）、芯片动态、arXiv 算子内核论文。

## TL;DR

- **本期增量日 = 10-14**（arXiv 尚未索引 10-15 提交）。除 10-14 的 55 条外，另补 **10-13 一批 14 条**——上一期跑完后 arXiv 才公布这批，上期窗口（150 条上限、当时 10-13 只见到 59 条）里根本不存在，故非重复；与历史各期逐 id 比对无重合。本期精选 8 篇。
- **kernel 与数据布局 4 篇**：Birdseed（首个并行稀疏输出 tensor compiler）、增量 SpMM 的调度复用实证（**tiling 在 837 个负载里 779 个反而输给直接 CSC**）、TOMOS（一份 FP32 存储同时给出 32/16-bit 视图的复合精度格式）、TrimSO(2)（把 edge tensor 物化消进 dataflow，HBM 流量降 3.81–9.88×）。
- **低精度硬件 1 篇**：MFSA 多格式脉动阵列原生吃 MX / NVFP4，Online Block-Scale Rebasing 把 block-scale 的增量开销砍掉 **64.2% 面积、67.5% 能耗**。
- **量化 3 篇**：PN-Mix（用 query–key 对范数决定键缓存通道对精度，1.5 bit 才明显掉点）、SpectrumQuant（协方差最大化 → 闭式正交旋转，显存不到 SpinQuant 的 1/3）、NatScale（MoE 混合精度按「对比校准」而非单纯排序分配位宽）。
- **CUDA 仓本期新增 cuda-python v13.4.4**（仅补 Python 3.15 wheel，其余与 13.4.3 相同）；**芯片动态本期无新条目**（命中的仍是 10-12 已收录的 Vera Rubin 一文）。

## 一、CUDA 开源仓 Release Notes（算子）

### cuda-python

- **v13.4.4**（10-15）— [链接](https://github.com/NVIDIA/cuda-python/releases/tag/v13.4.4)：**纯打包发布**，release note 仅一句「now ships wheels for Python 3.15. It is otherwise identical to the 13.4.3 release.」——对写算子的用户没有 API 或精度层面变化，升级与否取决于是否要用 Python 3.15。

> 窗口（≥ 10-14）内其余更新仍为 `cudnn-frontend` 的自构建 nightly（`v1.32.0.dev74082381` 10-15、`v1.32.0.dev73793365` 10-14），正文自述 *Automated nightly build from NVIDIA-internal CI. Not a supported release.*，按既有口径不计入。30 天窗口内的其他正式 release（CUTLASS v4.8.0、CCCL v3.5.0 / v3.4.3 / python-1.2.1、cuDNN Frontend v1.31.0、TensorRT v11.4 / v11.3、NCCL v2.32.3-1 与 nccl4py v0.6.0、cuda-python v13.4.3 / cuda-core v1.2.1 / cuda-pathfinder v1.8.3、cuda-quantum 0.16.0）均已在各期收录。

## 二、芯片动态

> 本期无新条目。RSS 窗口（7 天）内唯一命中仍是 10-12 已收录的 [NVIDIA 开放 Vera Rubin 架构](https://blogs.nvidia.com/blog/ocp-global-summit-vera-rubin-scale-ai-factories/)，按「不跨天重复」不再收录。

## 三、arXiv 论文（算子内核）

> 本期为 **10-14 增量 + 10-13 补录**：10-13 那 14 篇是在上一期日报抓取**之后**才被 arXiv 公布的（上期 150 条窗口里 10-13 只见 59 条，本期为 73 条），上一期无从看到；10-14 的 55 条为全新一天。

### kernel 与数据布局

- **[Parallel Sparse Outputs for Tensor Compilers](http://arxiv.org/abs/2610.19308v1)**（`2610.19308`，cs.DC，10-13）：稀疏 tensor compiler 长期只能串行、或对访问模式与输出格式做假设。Birdseed 是**首个支持任意标准稀疏格式、任意 loop ordering 下并行**的 tensor compiler：实现**并行稀疏 scatter**，直接写入 fibertree 格式的 list / bytemap / hash / dense 各层，在内存与时间上给出渐近改进，并在 histogram 构造、SpMSpV 这类难并行 kernel 上有实测加速。
- **[How the Dirty Set Moves: Characterizing Schedule Reuse in Incremental Sparse Matrix Multiplication](http://arxiv.org/abs/2610.19489v1)**（`2610.19489`，cs.PF/cs.DC/cs.MS，10-14）：很多场景矩阵固定、每步只有少数输入行变化（dirty set），据此做**增量 SpMM**。本文给出一个反直觉的实证结论：**保留 tiled schedule 复用极少胜过直接 CSC**——21 个 SuiteSparse 矩阵、837 个负载中 779 个是直接按列增量更新更快，尽管增量执行本身仍高度划算（较 8 个全量重算实现含 oneMKL / PyTorch 快 2.1–8.2×）。拆开看，收益来自 **per-tile preparation** 而非缓存已装配的 schedule，加缓存很少有用（tile 集在一组别的 tile 介入后基本不再回来）；只有当 tile 能把多条目合并进一次输出写时 tiling 才快。结论：**决定复用是否划算的是矩阵结构与 dirty set 的移动方式，而不是调度的精巧程度**。
- **[TOMOS: Eliminating Data Duplication in Mixed-Precision Matrix Computations](http://arxiv.org/abs/2610.19047v1)**（`2610.19047`，cs.MS，10-13）：混合精度算法要按不同精度用同一个矩阵，通常得**重复存储、反复转换或靠算子模拟**。TOMOS 提出复合精度矩阵格式，**从单一份表示同时提供隐式 32-bit 与 16-bit 访问**：每个 FP32 元素拆成 16-bit 主字（低精度计算用）加一个互补字（用于精确重建 FP32），并用次正规表示改善低精度视图的数值质量。存储仍占 FP32 大小，**数据搬运量随计算精度缩放**。优化的 GEMM / GEMV kernel 在 A100 上吞吐与厂商库相当，混合精度最小二乘求解器较 FP32 最高 1.6× 且无冗余矩阵分配。
- **[TrimSO(2): Fast and Memory-Efficient Training of Equivariant Interatomic Potentials](http://arxiv.org/abs/2610.20029v1)**（`2610.20029`，cs.DC，10-14）：SO(2) 卷积类等变势模型训练慢、显存重，根因是现有实现会在稠密矩阵乘**前后物化巨大的 edge tensor** 做表示变换，既吃内存又搬数据。TrimSO(2) 把这些变换**融进模型已有的 dataflow**（gather 时旋转、与激活一起重组、reduction 时逆转），不产生单独中间张量，并支持训练所需的高阶微分。前向 / 反向 / 二阶反向算子级加速 **2.71–16.73×**，模型级 HBM 流量降 **3.81–9.88×**；A100 / H100 上官方 eSEN、UMA、EquiformerV3 负载训练加速 2.07–2.94×、峰值训练显存降 1.53–3.16×。

### 低精度硬件

- **[MFSA: A Multi-Format Systolic Array with Native Micro Scaling Support](http://arxiv.org/abs/2610.19645v1)**（`2610.19645`，cs.AR，10-14）：模型内混用 BF16/FP16、多种 FP8、FP4、MX、INT 已成常态，硬件要柔性高效地都吃下来。MFSA 用两个架构选择省面积：PE 级把开源 trans-precision FPU 扩成 **dot-product-accumulator PE**，以共享数据通路支持 FP32/FP16/BF16/CFP8/FP4/INT16/INT8/INT4 并**按精度成比例提升吞吐**；阵列级提出 **Online Block-Scale Rebasing（OBSR）**，在 block 边界把 block scale 折进运行中的 FP32 部分和，**不额外加跨 block 累加器**。TSMC 28nm / 1GHz 下，同吞吐的 BF16/FP8/FP4 级联 FMA-PE 基线分别比单格式 DPA-PE 大 1.5×/2.5×/4.5×；16×16 MFSA 峰值 0.5/1/2/4 TOPS（FP32/BF16/FP8/FP4），OBSR 把 MX 增量开销较朴素实现砍 **64.2% 面积、67.5% 能耗**且不损真实 LLM trace 上的数值稳定性；在 BF16+CFP8 阵列上补齐其余格式仅多 25.7% 面积，OBSR 还能把 E4M3 scale 的小数部分折进 MFPE 点积以同时支持 NVFP4，较朴素实现面积只多 11.8%。

### 量化与低精度

- **[PN-Mix: Pair-Norm Guided Mixed-Precision Key Cache Quantization for Efficient Long-Output LLM Inference](http://arxiv.org/abs/2610.19605v1)**（`2610.19605`，cs.LG/cs.CL，10-14）：长输出推理下 KV cache 是显存与带宽瓶颈，而 2-bit 量化掉点严重；已有混合精度键缓存方案只用 **key 自身统计量**估通道重要性。本文在 **RoPE 通道对（channel-pair）层面**做 attention logit 扰动分析，指出**query–key 对范数乘积大**的通道对上量化误差对 logit 影响最大。PN-Mix 据此分配精度，且显著通道对**只在 prefill 阶段判定一次、整个 decode 复用**，不引入在线统计开销。推理、代码生成、AIME 基准上，**2.26 bit 平均精度即追平/超过用 2.32–2.39 bit 的既有混合精度方法**；Pareto 扫描中可平滑退到 1.5 bit，而基线在 2.25 bit 以下掉点更陡。
- **[SpectrumQuant: Spectrum Alignment for LLM Quantization](http://arxiv.org/abs/2610.20303v1)**（`2610.20303`，cs.LG/stat.ML，10-14）：正交变换能在不改模型函数的前提下重塑激活分布，因此成为低比特量化的常用手段；现有做法要么用 Hadamard 这类**固定、与数据无关**的旋转，要么在标定时**优化学习到的旋转**。本文从第一性原理研究旋转选择，证明**最大化协方差即导向谱对齐（spectral alignment）**，从而给出构造「量化友好」正交变换的判据；SpectrumQuant 据此用标定数据估激活协方差、**闭式**构造协方差感知正交变换，并刻画该类旋转为「对齐 + 混合」两部分之积，Hadamard 矩阵是其中一个特例（一类 butterfly 矩阵也属此类）。Llama / Mistral 上媲美 SOTA 旋转策略，而**显存不到 SpinQuant / DartQuant 这类学习式旋转的 1/3**，耗时也显著更低。
- **[Beyond Expert Rankings: Contrast Calibration for Mixed-Precision MoE Quantization](http://arxiv.org/abs/2610.20353v1)**（`2610.20353`，cs.LG/stat.ML，10-14）：MoE 每 token 只激活少数专家、但**全部专家权重都要存**，混合精度量化因此按敏感度给专家分配位宽——可是**光有专家排序并不能决定该以多大力度加权其误差**。NatScale 用模型相关的幂变换，把基于梯度的专家权重**校准到一个信息论「对比」目标**上（同时保持排序不变），并**联合评估每个专家 gate / up / down 的配置组合**来估输出误差，再结合权重谱、方差与精度衰减的先验，在存储预算下选配置。三个模型、九个设置上全部取得最低的 native-to-quantized KL 散度与最高的 top-token 一致率（对 5 个分配基线），困惑度在 7 个设置上最低；结论是**精度分配必须目标感知，不能只看排序**。

