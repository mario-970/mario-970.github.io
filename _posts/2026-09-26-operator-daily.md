---
layout: post
title: "算子日报 · 2026-09-26"
permalink: /operator-daily/2026-09-26/
date: 2026-09-26 23:59:00 +0800
tags: [算子日报]
---

> 每日汇总：CUDA 开源仓 Release Notes（算子）、芯片动态、arXiv 算子内核论文。

## TL;DR

- **Xtrace** 把 kernel 探针插到**已编译的二进制**上（而非编译前），因此不干扰编译器优化、追踪的正是 GPU 真正执行的那份 binary：H100 / B300 / MI300X 上保留 94–98% 原始指令（NVIDIA 的 Neutrino / IKET 只有 8–48%），额外开销 0.9–2.8%；它让 coding agent 用 **3.9× 更少迭代**达到 FlashAttention-3 同等性能，还能追闭源 cuDNN kernel 并据此把开源 FlashAttention-4 吞吐提升 **5.2–13.3%**。
- **MicroQonv** 给卷积层做 microscaling：每个张量只量化一次，并把激活量化**挪到 im2col 之前**（改用 channel-batch-first im2col），权重量化代价降 **2×**、激活最高降 **9×**，激活访存最高降至 **1/7.53**。
- 另一篇把 KV cache 当四阶张量做谱分析，结论清晰可落地：token / feature 模态低秩可压（key 尤其），head / layer 模态近乎满秩、任何压缩比都压不动——Tucker 因能「不动满秩模态」在 2×–5× 全区间误差最低。Release Notes 与芯片动态本期均无新条目。

## 一、CUDA 开源仓 Release Notes（算子）

> 本期无新条目。窗口内（≥ 09-25）只有 cudnn-frontend 的 nightly dev 构建（`v1.31.0.dev*`），非正式 release；其余 09-23 及更早的正式发布（cuda-python v13.4.3 / v12.9.9 / cuda-core v1.2.1 等）已在上一期或更早收录。

## 二、芯片动态

> 本期无新条目。NVIDIA Developer Blog 与 NVIDIA Blog 两个源在 7 天窗口内无命中关键词的新文章。

## 三、arXiv 论文（算子内核）

- **[Xtrace: High-Fidelity GPU Intra-Kernel Tracing via Binary-Level Instruction Splicing](https://arxiv.org/abs/2609.28769)**（`2609.28769`，09-23）：现代 GPU kernel 越融越复杂，intra-kernel tracing 成了主流 profiling 手段——往 kernel 里插探针记录运行时状态，而**追踪保真度直接决定优化效率**。问题在于现有工具都在**编译前**插探针：既干扰编译器优化（追的是另一份 binary，不是 GPU 真正执行的），运行时开销也大。Xtrace 改为把探针直接插入**编译后的 kernel binary**，只用插入点处已死值占用的寄存器，并借编译器自带的 hazard table 消解所有冒险，再对指令顺序、寄存器分配与控制位做调度以压低探针开销；支持 19 种 NVIDIA 与 AMD GPU 架构，已在 https://g-watch.github.io 公开。在 H100、B300、MI300X 上对生产级 LLM kernel 评测，Xtrace 保留 kernel **94–98%** 的指令，而 NVIDIA 的 SOTA 工具 Neutrino / IKET 仅 8–48%；Xtrace 额外开销 **0.9–2.8%**，后者 3.8–75.6%。效果上，它引导 coding agent 以**少 3.9× 的迭代次数**达到同样的 FlashAttention-3 性能；得益于 binary 级插桩，Xtrace 还能追**闭源 cuDNN kernel**，用其轨迹把开源 FlashAttention-4 吞吐再提升 **5.2–13.3%**。

- **[MicroQonv: Reshaping Convolution Tensors for Efficient Microscaling in Training and Inference](https://arxiv.org/abs/2609.28358)**（`2609.28358`，09-23）：microscaling（MX）量化能把参数压到 8 bit 以下还保住精度，但落到卷积层并不顺：朴素做法要把全精度权重和激活搬到计算单元上、每个张量量化两次，访存远超预期；更糟的是激活在量化前先经过 im2col 展开，张量体积被撑大。MicroQonv 的思路是**每个张量只量化一次**，并把激活量化提到 im2col **之前**——改用其提出的 **channel-batch-first im2col**，让展开后的布局天然适配分块量化。结果是权重量化代价降 **2×**、梯度降 **2×**、激活最高降 **9×**（精度损失可忽略），访存与存储相对全精度最高降 **7.53×**；对 YOLOv8nano / YOLOv26nano，microscaling 量化后的激活访存分别再降 **3.5×** 与 **2.2×**。此外它让**4-bit microscaling** 能用于端侧持续学习的 quantized latent replay，精度反而提升 **+5.7% ~ +11%**。

- **[Tensor Decomposition of Transformer Key-Value Caches: Spectral Structure and Format Comparison](https://arxiv.org/abs/2609.28029)**（`2609.28029`，09-23）：把自回归 Transformer 的 KV cache 看作横跨 attention head、token、feature、分组 layer 的**四阶张量**，在 Mistral-7B-v0.3 与 LLaMA-2-13B 上实测四个模态展开的奇异值谱，并在**等存储预算**下对比 Tucker、CP、tensor train、t-SVD 四种分解。谱结构把四个轴干净地分成两类：**token 与 feature 模态呈低秩**（key 尤其明显），**head 与 layer 模态近乎满秩**，在任何实际误差水平下都压不动。因此四种分解里 **Tucker 在 2×–5× 每个压缩比上重构误差最低**——原因正是它可以把满秩模态原样留着；作者还给出一条 mode-pinning 定理，仅凭实测谱即可证明满秩模态被保留。二维展开基线与四阶分解的对比则显示 key 与 value 偏好不同表示：**二维方法在 key 上误差更低，四阶 Tucker 在 value 上更低**。另有两个影响可压模态的谱性质：value 的误差下限在每个压缩比上都高于 key；**post-RoPE 的 key 丢失了 pre-RoPE 41%–64% 的可压缩性**（两个模型均如此）。
