---
layout: post
title: "算子日报 · 2026-11-23"
permalink: /operator-daily/2026-11-23/
date: 2026-11-23 23:59:00 +0800
tags: [算子日报]
---

> 每日汇总：CUDA 开源仓 Release Notes（算子）、芯片动态、arXiv 算子内核论文。

## TL;DR

- **CUTLASS 4.9.0dev 把 Rubin 的低延迟与 FP6 路径铺开**：TMA load 带上 validity reporting（C++ 与 CuTe DSL 同步），FP4 MMA 支持 2:4 结构化稀疏，新增 K=64 的 dense FP6 MMA，tcgen05 commit 变体可提前释放 A 操作数；EFC 把 SwiGLU/GEGLU/ReGLU 融进单次 MMA，并补齐 Blackwell MXFP8 与 Rubin FP8 的 block-scaled dense GEMM。
- **MoE megakernel 的调度细化到 warp 级**：Weave 按每卡 routing 统计在 SM 与 warp 两级分配通信/计算资源，并在层内动态切换 warp 角色，EP8 下比 DeepGEMM-MegaMoE 的 MoE 层平均快 **1.30×**（SM103 最高 1.97×）。
- **arXiv 补录 11-20 批次**：增量窗口返回 45 篇，全部是 11-20（周五）提交、上期尚未公告的批次；精选 4 篇——Weave、THOT 稀疏 attention 内核、PLoQ 4-bit 激活量化、PennyGS 4K 3DGS 训练 kernel。芯片动态无新增。

## 一、CUDA 开源仓 Release Notes（算子）

### CUTLASS

- **v4.9.0dev**（2026-11-23 07:22 UTC，本地上 15:22）— [链接](https://github.com/NVIDIA/cutlass/releases/tag/v4.9.0dev)：Rubin 支持扩到低延迟 kernel 与 FP6 吞吐——C++ 新增带 validity reporting 的 TMA load；CuTe DSL 扩展与 Primitives 同步获得 TMA validity load、FP4 MMA 的 2:4 结构化稀疏、shared-memory ring buffer、A 操作数提前释放，以及 K=64 的 densely packed FP6 MMA；CuTeDSL pipeline 支持新的 Rubin tcgen05 commit 变体。DSL 扩展 API 现在可直接用于 `@cute.jit` / `@cute.kernel` / `cute.compile`，无需 opt-in（不想用时可用 `CUTE_DSL_USE_EXTENSION_COMPILER=0` 或 `cute.compile[cute.DisableCuteExtCompile](...)` 回退旧编译器路径）；Blackwell/Rubin GEMM 示例扩大了 preferred/fallback cluster shape 覆盖，新增 `print_tmem_tensor` 检查 TMEM 张量（暂只支持 32-bit dtype）。JAX 侧支持 `jax.vmap` + `cutlass_call`、自动 provision workspace 的扩展 kernel、动态 host 参数与类型转换，并降低 kernel launch 的 CPU 开销。Epilogue Fusion Configuration（EFC）可用 `accum_split` 把累加器拆成 N 份，在单次 MMA 上融合 SwiGLU/GEGLU/ReGLU 等 gated activation（输出沿 N 变窄后经 TMA 写回）；block-scaled 路径新增 Blackwell MXFP8（E8M0，每 32 个 K 元素一个 scale factor，FP32 累加，N tile 64/128/192/256）与 Rubin `DenseGemmEFCSm107`（FP8 K=64 MMA、B collector 复用）。示例新增 distributed JAX+NCCL4Py、用 TMA validity reporting 做跨 kernel 细粒度数据驱动同步的 Rubin dense GEMM、Rubin FP4/FP8 GEMM 与 Blackwell FP16/BF16/TF32/FP8 隐式 GEMM Fprop Conv。迁移注意：`%` / `//` 正转向 Python 语义，当前默认仍是 legacy 并给出告警，可用 `--divmod-semantics=python` 提前切换。

> 该 tag 在 GitHub 上未标记 prerelease，与 08-27 收录的 `v4.8.0dev` 同属 CUTLASS 开发线快照，按往期口径计入。窗口内 cuDNN Frontend 另有 nightly 自构建 `v1.34.0.dev84492328`（11-22）与 `v1.34.0.dev84733400`（11-23），正文自述 *Automated nightly build from NVIDIA-internal CI. Not a supported release.*，按既有口径不计入；其余正式 release（CUTLASS v4.8.1、CCCL python-1.3.0、TensorRT v11.5、NCCL v2.32.6-1、cuda-python v12.9.11、cuda-core v1.3.1、cuda-pathfinder v1.9.1、cuda-quantum 0.17.0 等）均已收录于往期。

## 二、芯片动态

> 本期无新条目。两个 RSS 源在 7 天窗口内仍只返回 11-16 发布的 4 条——Rubin HPC、MLPerf Training/Endpoints 榜单、CUDA-X 科学计算库与 CUDA 20 周年回顾，均已在 11-17～11-19 期注记，且不涉及算子内核或架构层面的开发更新；NVIDIA Developer Blog 源仍返回 0 条。

## 三、arXiv 论文（算子内核）

> 本期增量窗口（提交日 ≥ 11-20）返回 **45 条**，全部为 11-20（周五）提交——上期指出的「11-20 及之后投稿尚未进入索引」的批次现已公告；与往期文章逐 id 比对均未收录，11-23（周一）当日投稿尚未公告。按算子开发相关性精选 4 篇。

- **[Weave: Routing-Adaptive Runtime Warp Scheduling for MoE Megakernels](http://arxiv.org/abs/2611.27765v1)**（`2611.27765`，cs.DC，11-20）：MoE + expert parallelism 下，现有 megakernel 要么在编译期固定通信/计算的资源划分（MegaMoE），要么只在 SM 粒度做运行时适配（mKernel）——分给通信的整块 SM 的 Tensor Core 随之闲置。Weave 在 SM 与 warp 两级按每卡 routing 统计分配资源，并随层内执行进度切换 warp 角色：层首尾把所有 warp 投入计算，output scatter 交给空闲的通信 warp；融合 cost model 在 launch 时按运行时 routing 统计选调度组合。EP8 下比 DeepGEMM-MegaMoE 的 MoE 层平均快 **1.30×**（SM103 最高 **1.97×**），SM90 平均 1.24×（最高 1.72×）；EP16 的 CDNA3 上比最快的 RCCL+AITER 平均快 1.44×，比 vLLM 快 1.67×。
- **[Topology Hub Optimal Transport: Eliminating Long Context Dilution at Flash Speed](http://arxiv.org/abs/2611.27951v1)**（`2611.27951`，cs.LG，11-20）：长上下文稀疏 attention 的检索会被随长度扩散的归一化质量稀释。THOT 用 local window + K 个共享 hub 的线性规模图替代 L×L 密集矩阵：simplex sparsemax（`O(L log L)`）或免排序的 token-local thresholded ReLU（`O(L)`）做容量门控，把 distractor 质量压到零；三趟 1+2 退火的 row-closed Sinkhorn 抑制二部残差。核心算子实现为 batched split-factor Triton kernel（BF16 Tensor Core、FP32 potentials），给出 exact rank-(D+2R+1) adjoint 与 chunked exclusive-prefix causal 变体（验证零未来泄漏）。L4 上 H=8、D=64、L=262K 的 forward+VJP 为 **100.1 ms**，比 latent softmax 快 2.2×、编译器临时量小 4.0×；L=4096 时 **1.34 ms**，对比 cuDNN FlashAttention 的 4.37 ms。注意多层 LM 跑的是未融合路径。
- **[PLoQ: A Prompt-Specific Low-Rank Basis for 4-bit Quantization of LLM Activations](http://arxiv.org/abs/2611.27092v1)**（`2611.27092`，cs.LG/cs.CL/cs.DC/cs.PF，11-20）：4-bit 激活量化的瓶颈是少数 dominant directions 决定量化范围；现有方法把它们放进高精度子空间并离线折叠进权重，子空间对所有 prompt 固定，且没有权重可吸收变换时无法使用。PLoQ 改为按 prompt 估一个低秩基：量化前抽走 dominant 分量单独量化，再在该层 GEMM 内恢复，不改权重；基在 prefill 阶段估计、decode 期间复用。在与 SOTA 混合精度量化相同位宽（W/A/KV 均 4-bit、1/8 通道 8-bit，平均约 4.2 bit/激活）下，6 个 Llama/Qwen 模型（1B–70B）到 8k token（其中两个到 16k）全部更优，最多找回 SOTA 仍损失的 46% perplexity；Llama-3.1-8B/8k 上 prefill 只增 4%、decode 增 8%。
- **[PennyGS: Full Resolution 3DGS Training in Minutes on Consumer GPUs](http://arxiv.org/abs/2611.27525v1)**（`2611.27525`，cs.GR/cs.PF，11-20）：4K 全分辨率 3D Gaussian Splatting 训练此前需要研究级 GPU。PennyGS 逐个 stage 改训练 kernel：更精简的 splat（flat RGB + directional cone opacity）、融合 projection 与 key emission、前向 rasterization 的 strided pixel sampling、近似 tile SSIM loss、warp-per-splat 反向 rasterization，以及融合 backward projection 与 sparse Adam step。30k step 预算下，在 4 块消费级 GPU 上于 Mip-NeRF 360 / Tanks&Temples 4K 均快于其余 9 种方法至少 **3.9×**，峰值显存 <2 GB；RTX 3060 上 6m25s 训完全分辨率 Mip-NeRF 360 场景，快过其他方法在 RTX 4090 上的成绩。
