---
layout: post
title: "算子日报 · 2026-10-22"
permalink: /operator-daily/2026-10-22/
date: 2026-10-22 23:59:00 +0800
tags: [算子日报]
---

> 每日汇总：CUDA 开源仓 Release Notes（算子）、芯片动态、arXiv 算子内核论文。

## TL;DR

- **低精度与 GEMM 的取舍有了更硬的判据**：MXFP4 点积在归约前还是归约后应用 scale，实数上等价、binary32 下却会因溢出/下溢不等价；作者用整数状态机把 rewrite 合法性精确编码成 QF_LIA，cvc5 秒级判定，而浮点编码在 k=2 时就撞 1 小时上限。另一篇在等比特下比较 2:4 稀疏与降精度：只要还有更低精度可用，加 2:4 既更不准也更慢；降到硬件最低精度后 2:4 才让线性层快 **1.3–1.7×**。
- **稀疏 attention 的优化重心继续从「选哪些块」转向「怎么分配、怎么寻址」**：ALLOT 只替换 layer-step 的密度分配、保留现有 selection 与 kernel，等时或更短 wall-clock 下 HunyuanVideo PSNR 提升 **11.8 dB**；UlaCache 用统一逻辑地址管理稀疏 KV，B300 上 decode 峰值 **1.73×**（DP）/ **2.51×**（TP）。
- **kernel 工程侧有两份实测**：GPU 量子纠错解码器用 CUDA Graphs、合并稀疏访存、warp 级集合通信与 INT8 量化，RTX 5080 吞吐达 GH200 的 **91%**、中位提交到纠错延迟短 **8.9×**；FlashNext-RL 针对 MoE、稀疏 attention 与 gated residual 的 kernel 优化，把单节点 8×B200 的训练吞吐从 395 提到 **4,438 tokens/s/GPU**。

## 一、CUDA 开源仓 Release Notes（算子）

> 本期无新 release。上期收录 CUTLASS v4.8.1（10-21）后，窗口内 10 个仓库只有 `cudnn-frontend` 的 nightly `v1.32.0.dev75836140`（10-22，正文自述 *Automated nightly build from NVIDIA-internal CI. Not a supported release.*），按既有口径不计入；30 天窗口内其余正式 release 均已在往期各期收录。

## 二、芯片动态

> 本期无新条目。NVIDIA Developer Blog 源继续返回 0 条；NVIDIA Blog 在 7 天窗口内只有 10-21 的 Google Cloud × Vera Rubin GA 一条，已在上期收录。本节连续为空属**数据源侧**状况，不是抓取故障。

## 三、arXiv 论文（算子内核）

> 本期抓取窗口覆盖 10-19 至 10-21 共 **150 条**（触及 `max_results` 上限；左端 10-19 与上期 128 条充分重叠，10-19 新增仅 1 条）。其中 **56 条**为本期新入库：10-21 提交 42 条、10-20 追补 13 条、10-19 追补 1 条；逐 id 与往期各期共 260 个已收录 id 比对，无一重合。按算子开发相关性从新入库条目中精选 6 篇。

### 低精度与 GEMM

- **[Deciding Scale Placement in MXFP4 Dot Products with Integer Arithmetic](http://arxiv.org/abs/2610.29445v1)**（`2610.29445`，cs.LO/cs.AR/cs.PL，10-21）：MXFP4 把 32 个 4-bit 元素配一个 2 的幂共享 scale。编译器 lowering 点积归约时，可以**在归约前应用 scale**，也可以**推迟到归约后**；两者在实数上等价，但 binary32 下会因溢出/下溢产生不同结果，且等价性还取决于归约树形状。作者利用元素字母表有限、scale 为 2 的幂这两个性质构造整数状态机，覆盖溢出、NaN 与次正规舍入，在 scale code 与整数系数有界时证明其忠实性，从而把等价性查询**精确编码为 QF_LIA**。对固定归约树给出 scale 指数和的闭式充分上界，并取决于最大真子树规模；在 k∈{2,3,4,8,16,32} 的四种树形与 k=64 的两种树形上，上界加一个指数步长后 SMT 即找到反例。cvc5 秒级证明整数版本，而标准 QF_FP 编码在 k=2 时就撞上 1 CPU 小时上限。配套 MLIR lowering 原型用这些认证条件做静态/运行时守卫来选择方案——对 blockscaled FP4 算子意味着可以做**可验证的 rewrite**，而不是一味保守拒绝。
- **[Is 2:4 Sparsity Still Worth It? Semi-Structured Sparsity versus Quantization in LLM Inference](http://arxiv.org/abs/2610.28947v1)**（`2610.28947`，cs.LG/cs.PF，10-20）：Ampere 起 NVIDIA tensor core 就支持 2:4 半结构化稀疏，且可与量化叠加；实际问题是**给定一个精度，下一步该加 2:4 还是降到下一个更低精度**。作者在等比特（计入稀疏元数据）与等 tensor-core 档位两种口径下，覆盖 transformer 层全部 6 个 GEMM operand，在三代 GPU 上实测 kernel，并把在线剪枝与量化开销算进去。结论是只要还有更低精度可用，加 2:4 都是更差的一步——既更不准也更慢，因为 2:4 沿硬件强制轴每组四个丢弃的能量不随比特下降，而量化误差会降。等量化触到硬件支持的最低精度后，2:4 仍能把线性层加速 **1.3–1.7×**；H100 上一个重训过的 2:4 模型以 FP8 跑 vLLM，prefill 吞吐比稠密 FP8 高 **27%**，GSM8K 持平。代价是质量要靠重训夺回，且重训先恢复 perplexity、后恢复推理；Blackwell 的 sparse FP4 用 pairwise 4:8 pattern，掉点更大，作者的训练预算内没能恢复。

### 稀疏 attention 与推理系统

- **[Sparsity Should Be ALLOTted by Output: Training-Free Allocation of Block-Sparse Attention for Video Diffusion Transformers](http://arxiv.org/abs/2610.29441v1)**（`2610.29441`，cs.CV，10-21）：视频 DiT 的全 3D attention 开销大，block-sparse attention 只算选中的块，但现有方法要么忽略**稀疏度分配**，要么用启发式规则或局部指标来分配。作者观察到不同层、不同去噪 step 的稀疏化对最终视频保真度影响差异很大，而**最敏感 layer-step 组合的身份在不同输入间几乎不变**。ALLOT 把每个 layer-step 对当作一个 cell，通过 cell probing 与 trajectory calibration 估计该 cell 对最终视频误差的贡献，再在实测延迟预算下联合优化各 cell 密度。它**保留现有 selection 策略与 kernel，只替换分配组件**；在等时或更短 wall-clock 下，配 XAttention 全面优于对比的分配策略，HunyuanVideo 上 PSNR 提升 **11.8 dB**，Wan2.1、Wan2.2、HunyuanVideo 上四个额外 pipeline 的九个设置中均不劣于默认。
- **[UlaCache: Unified Logical Addressing for Sparse LLM Serving](http://arxiv.org/abs/2610.29820v1)**（`2610.29820`，cs.DC，10-21）：长上下文放大 KV cache 内存开销，稀疏 attention 虽能减少计算与访存，但被选中的 KV 必须持续可用。UlaCache 围绕**统一逻辑地址空间**管理 KV，覆盖 GPU KV pool 与可选 CPU 内存；稳定的逻辑引用让推理引擎的分配与选择独立于物理放置，并用统一的分配/稀疏访问接口同时支持 DP 与 TP attention。分布式存储时，本地 GPU buffer 跨 decode step 保留已选 KV，只传输缺失项；启用 CPU 存储时，Parallel Bucketed LRU（PB-LRU）管理跨请求共享的 GPU pool，并行更新 recency 元数据与选择驱逐对象，先传 KV 再更新地址映射。在 CPU/GPU 内存预算匹配且开启投机解码时，H200 上 decode 峰值加速 **1.16×**（DP）/ **1.70×**（TP），B300 上 **1.73×**（DP）/ **2.51×**（TP）。

### GPU kernel 与训练系统

- **[Engineering Low-Latency, High-Throughput Quantum Decoders on NVIDIA GPUs](http://arxiv.org/abs/2610.28867v1)**（`2610.28867`，cs.DC/quant-ph，10-20）：面向量子纠错的 GPU 解码器，组合 Relay belief propagation 与 order-zero ordered statistics 后处理。工程上做了几件对通用稀疏 kernel 也有参考价值的事：用 **CUDA Graphs** 控制执行、对已收敛样本停止更新、把未解决样本入队后处理；通过**合并消息访存、预取与寄存器复用**优化稀疏遍历；二值后处理用 **warp 级集合通信**加速；混合 FP16-FP32 与 **INT8 量化**压缩消息存储与流量，同时用更高精度累加器保精度。在 surface、hypergraph product 与 bivariate bicycle code 上测 A100、GH200 与消费级 RTX 5080：bivariate bicycle 码在错误率 0.003 时，INT8 把均摊解码时间相对独立调优的 FP32 减半、逻辑错误率几乎不变；把独立 syndrome batch 连续投给并发解码器实例时，RTX 5080 吞吐达 GH200 最高值的 **91%**，中位提交到纠错延迟短 **8.9×**。
- **[FlashNext-RL: A System Optimization Study for Long-Horizon Agentic RL of Qwen3.8-Flash-Next](http://arxiv.org/abs/2610.28998v1)**（`2610.28998`，cs.LG/cs.CL，10-20）：长程 agentic RL 既要高效 rollout 又要训练长序列，而 Qwen3.8-Flash-Next 的显存需求与模型专属算子让现有开源实现难以支撑。FlashNext-RL 基于 prime-rl、用 vLLM 做 SWE rollout，在单节点 8×B200 上以 BF16 训练完整 48 层模型（前两层、n-gram table、sparse-attention indexer 与多 token 预测头冻结），优化了 host offload、梯度累积，以及 **MoE、稀疏 attention 与 gated residual stream 的 kernel**。全模型研究里，非 padding 训练吞吐从 32K 配置的约 **395** 提升到优化后 131K 配置的 **4,438 tokens/s/GPU**，硬件 matmul 吞吐（不计 attention 组件）从约 **20** 提升到 **213 TFLOPs/GPU**；8 层模型上的累积消融量化了各项优化的贡献。配合三个推理节点，系统支撑目标 1,024 条可训练 rollout、131,072 token 上下文的在线 SWE RL，step 5–15 平均每轮 22.4 分钟，entropy 与 trainer-sampler mismatch KL 均有界。

> 其余新入库条目多为应用/训练方法向（医疗影像、EEG、机器人、3D 高斯、遥感与推荐等）或与 GPU 算子无直接关系的理论工作，与算子内核无直接关系，未收录。
