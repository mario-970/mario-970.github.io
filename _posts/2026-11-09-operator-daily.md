---
layout: post
title: "算子日报 · 2026-11-09"
permalink: /operator-daily/2026-11-09/
date: 2026-11-09 23:59:00 +0800
tags: [算子日报]
---

> 每日汇总：CUDA 开源仓 Release Notes（算子）、芯片动态、arXiv 算子内核论文。

## TL;DR

- **把 softmax 从向量单元挪进矩阵核**：AMD CDNA4 的 Fused-Softmax 把 softmax 主要运算折进 attention GEMM，MXFP8/6/4 kernel 在 MI355X 上分别达到约 3.0/4.0/4.6 PFLOP/s，相对 BF16 attention 快 **2.1×/2.8×/3.2×**；kernel 已随 AITER 开源，并支撑 AMD 的 MLPerf Inference v6.1 Wan2.2 提交（同硬件较上一轮 **1.7×**）。
- **FP4 KV cache 进入原生解码**：BitDecoding-FP4 用双残差 query 量化把 query logit 误差从 60% 压到 1%，配合 underflow-aware 概率缩放与 outlier-aware key 量化，B300 上 Qwen3-4B / Qwen3-30B-A3B 的 serving 峰值吞吐达 BF16 KV 的 **2.4–3.0×**，NVFP4 下精度差距在 0.5 分以内。
- **访存与混合精度各有新解**：MINE 把 dtANS 熵编码做成 GPU kernel，2.1-bit 权重下 RTX 4060 Ti / RTX 5090 单流解码快 **2.52×/1.48×**；WENO 研究发现 FP32 与 FP64 单元并发只在 A100 有明显收益，GH200/GB200/Blackwell 基本不重叠，而 double-float 仿真在 RTX 6000 Blackwell 上比原生 FP64 快 **2.51–3.94×**。

## 一、CUDA 开源仓 Release Notes（算子）

> 本期无新条目。30 天窗口内最新正式 release 仍是 11-05 的 `cuda-python v13.5.0` 与 `cuda-pathfinder v1.8.6`（11-06 期已收录）；11-06 的 cuDNN Frontend `1.33.0.dev79840341` 属 NVIDIA 内部 CI 的 nightly 自构建（自述 *Not a supported release*），按既有口径不计入。

## 二、芯片动态

- **[From Chips to Skills, NVIDIA and Korea Expand Sovereign AI Collaboration](https://blogs.nvidia.com/blog/korea-sovereign-ai-collaboration/)**（11-09）：NVIDIA 与韩国科学技术信息通信部（MSIT）扩大主权 AI 合作，范围覆盖 AI 半导体、数据中心基础设施、人才、模型与安全；合作将为韩国 NPU/XPU 公司探索加入 NVIDIA NVLink Fusion 生态铺路。属芯片生态/合作动态，不涉及具体架构发布。

## 三、arXiv 论文（算子内核）

> 本期增量为 **11-06 提交的 47 条**：这是 11-06 期「11-06 提交尚未被索引」之后的首个公告批次（周末 arXiv 不发公告，11-07/11-08 两期均按空窗口跳过）。与往期逐 id 去重后，按算子开发相关性精选 5 篇。

### 低精度 attention 与 KV cache

- **[Fused-Softmax for efficient quantized attention](http://arxiv.org/abs/2611.09225v1)**（`2611.09225`，cs.AI/cs.LG/cs.PF，11-06）：低精度矩阵吞吐增长远快于向量单元，softmax 已成为 attention 的流水线瓶颈。Fused-Softmax 不训练、直接重写并近似 softmax，把其主要运算折进 attention GEMM，让工作从向量单元转到矩阵核；AMD CDNA4 上的 MXFP8/MXFP6/MXFP4 kernel 在 MI355X 达到约 3.0/4.0/4.6 PFLOP/s，相对 BF16 attention 快 **2.1×/2.8×/3.2×**，而额外误差仍低于量化 GEMM 操作数本身的误差底。kernel 已随 AMD 开源 AITER 发布，并支撑 AMD 的 MLPerf Inference v6.1 Wan2.2 提交（同硬件较上一轮 **1.7×**）；Wan2.2-T2V-A14B 上 MXFP6/4 混合调度端到端最高 **2.05×**。
- **[BitDecoding-FP4: Native FP4 KV Cache Decoding via Algorithm-System Co-Design](http://arxiv.org/abs/2611.09825v1)**（`2611.09825`，cs.AI/cs.AR/cs.CL/cs.PF，11-06）：现有 FP4 KV 系统要么用朴素量化导致精度骤降，要么依赖向量码本查表拖慢 decode。BitDecoding-FP4 从三处误差入手：双残差 query 量化把 query logit 误差从 **60% 降到 1%**，underflow-aware 概率缩放防止 E4M3 scale 把长上下文概率舍成 0，outlier-aware key 量化用静态平滑 + Hadamard 旋转 + scale 搜索压 key 离群值；系统侧用转置 softmax + fused 概率量化 + paged KV 上的堆叠 query MMA 消除 softmax 瓶颈。B300 上 Qwen3-4B 与 Qwen3-30B-A3B 的端到端 serving 峰值吞吐为 BF16 KV 的 **2.4–3.0×**，NVFP4 下精度与 BF16 KV 差距在 **0.5 分**以内（MXFP4 为 1 分）。

### 推理访存与熵编码

- **[MINE: Entropy-coded Memory-efficient Inference for Large Language Models](http://arxiv.org/abs/2611.09424v1)**（`2611.09424`，cs.LG/cs.DC/cs.PF，11-06）：单流 decode 强内存受限，每个 token 都要把全部权重从显存读一遍，而此前无损压缩只在 BF16 等特定数据类型上有效。MINE 的核心 dtANS-LLM 是适配 LLM 推理的 GPU 熵编码方案，利用量化后仍未消失的权重非均匀分布，暴露压缩率/解码开销的折中参数并按 tensor、GPU 自动调优，最终以 vLLM 插件形式按内存预算选最优 kernel 配置。2.1-bit EntQuant 权重的 Llama-3.1-8B-Instruct 在 RTX 4060 Ti / RTX 5090 上分别比最快基线快 **2.52×/1.48×**，等效位宽 2.24 / 3.43 bit。

### 混合精度与硬件协同

- **[Concurrent Use of FP32 and FP64 Computing Units on GPUs and Mixed-Precision Placement in WENO Reconstruction](http://arxiv.org/abs/2611.09651v1)**（`2611.09651`，cs.DC，11-06）：科学计算离不开 FP64，但 NVIDIA GPU 的 FP64 吞吐有限；double-float（DF）仿真用两个 FP32 表示约 48 位有效数，可在 FP64 单元受限时反超原生 FP64。作者在 A100、GH200、GB200、RTX PRO 6000 Blackwell Max-Q 上考察 FP32 与 FP64 单元能否并发：微基准显示两者共存只在 A100 上超过单用较快 kernel，GH200/GB200 不行，RTX 6000 Blackwell 对 FFMA 可以、对 DF 仿真则要降低 DF kernel 发射密度才行。落到 WENO-Z 重构，A100 有可观重叠，GH200/GB200/Blackwell 几乎没有；且最快放置不一定保持设计精度——W32P64 在 WENO7/9 上达不到设计阶，而全重构 DF 仿真在 RTX 6000 Blackwell 上比 FP64 快 **2.51–3.94×**。
- **[Atik: A Shared-Mesh Transformer Accelerator for RISC-V](http://arxiv.org/abs/2611.09522v1)**（`2611.09522`，cs.AI/cs.AR，11-06）：Transformer 推理反复调用稠密矩阵乘与 attention（含数据相关的 softmax 归一化）。Atik 是面向 RocketChip 的开源 RoCC 加速器，用一套共享定点 MAC mesh 同时执行矩阵乘与稠密 attention，收发 BF16 tensor，采用流式 online-softmax 调度、从不物化完整 score 矩阵，并通过小型描述符 ABI 调用。FireSim 逐周期评测三种 mesh：8×8 配置相对标量 Rocket BF16 参考的峰值 matmul 加速 **31.5×**、attention **225.8×**；ViT / TinyBERT / GPT-2 prefill 的混合投影聚合加速 **44.2×/77.1×/32.3×**（非全模型硬件回放）。
