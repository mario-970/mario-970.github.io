---
layout: post
title: "算子日报 · 2026-09-28"
permalink: /operator-daily/2026-09-28/
date: 2026-09-28 23:59:00 +0800
tags: [算子日报]
---

> 每日汇总：CUDA 开源仓 Release Notes（算子）、芯片动态、arXiv 算子内核论文。

## TL;DR

- **PISA** 攻的是块稀疏 attention 里最贵的一步——**选块**：传统做法要对全部 query-block 对打分，复杂度仍是 O(N²)。它用金字塔 Top-K 把 key 组织成 coarse-to-fine 层级，每层只在有限候选集上用 **LogSumExp** 打分筛出下一层候选，层数 O(log N)，整体降到 **O(N log N)**；训练与推理各配一套**硬件感知 Triton kernel**，把层级路由与 LogSumExp 打分融合、**不物化 QK 打分矩阵**，常识推理持平基线、检索任务更好。
- **《The KV Cache Is the New Memory Wall》**是篇 SoK，把长上下文推理的瓶颈算清楚了：推导出 **arithmetic intensity 随上下文长度衰减的闭式表达**，按 **H100 / B200 / MI300X** 的硬件拓扑（含 per-die 带宽切分）参数化，给出 **KV 流量反超权重流量的 crossover 长度**（Llama-3-70B BF16 权重 140 GB 已超单卡 80 GB HBM，一条 128k token 序列再加 42 GB KV）。统一协议下每类方法取一代表在 128k 上评测，结论是**三段式**：crossover 之前压 KV 几乎不提速，之后才轮到「拿精度换带宽」；**paging / prefix sharing 无损但不省带宽**，量化与 eviction 才直接砍带宽（4-bit 以下衰减加速，eviction 在位置敏感任务上不连续），tiering 则把带宽墙变成受 PCIe/NVLink 约束的互连问题。
- 另有三篇量化工作：**output head 的 softmax 重参数化**在量化前先做**功能等价**的行均值平移（一维系数按 validation KL 搜），Phi-4-mini 上 W4 的 AW-MSE KL 从 **0.936 降到 0.256**，且 shift-compatible 头不增任何推理算子、保住 packed W4 执行，Phi 输出头量到 W4 后 batch-1 生成延迟降 **10.8%**；**混合模型的循环状态**用可观测 Gramian 推失真权重做无校准混合精度分配，4-bit 平均负载把超额 NLL 降 **3.3–27.9×**；**looped transformer 的 PTQ** 被拆出两种失效模式（feedback exposure / calibration blindness，一步 GPTQ 在 7 个架构 9 个 checkpoint 里有 5 个不如 RTN），跨递归步累积 Hessian 后 **9/9 全胜**并追回 bf16 精度。Release Notes 与芯片动态本期均无新条目。

## 一、CUDA 开源仓 Release Notes（算子）

> 本期无新条目。窗口内（≥ 09-27）只有 cudnn-frontend 的 nightly 自动构建（`v1.31.0.dev70065177` / `v1.31.0.dev70079617` / `v1.31.0.dev70215909`），release note 自述「Automated nightly build from NVIDIA-internal CI. Not a supported release.」且只保留最近 14 个 tag，非正式发布；其余正式发布（CUTLASS v4.8.0、TensorRT v11.3、NCCL v2.32.3-1 / nccl4py v0.6.0、cuda-python v13.4.3 / v12.9.9 / cuda-core v1.2.1、CCCL python-1.2.0、cuda-quantum 0.16.0）均已在 09-21 ~ 09-27 各期收录。

## 二、芯片动态

> 本期无新条目。NVIDIA Developer Blog 与 NVIDIA Blog 两个源在 7 天窗口内无命中关键词的新文章。

## 三、arXiv 论文（算子内核）

- **[Block Sparse Attention with Log-Linear Complexity](https://arxiv.org/abs/2609.31093)**（`2609.31093`，09-25）：块稀疏 attention 是长上下文省算力的常规手段，但**选哪些块**本身就是瓶颈——传统块选择要对所有 query-block 对打分，序列一长又退回二次复杂度。PISA 改用**金字塔 Top-K 选择**：把 key 通过 pooling 组织成 **O(log N)** 层的 coarse-to-fine 层级，从最粗一层开始选，每层在一个**有界候选集**上用 **LogSumExp** 打分、筛出进入下一层更细粒度的候选，直到最细层；因为每层候选集有界、层数只有 log N，整体复杂度降到 **O(N log N)**。为落地，作者写了训练与推理两套**硬件感知 Triton kernel**，把层级路由与 LogSumExp 打分**融合在同一个 kernel 内**，全程**不物化 query-key 打分矩阵**（这既是省显存的关键，也是避免二次访存的关键）。在语言建模任务上，常识推理等 benchmark 与基线**性能相当**，检索类任务上更好。

- **[The KV Cache Is the New Memory Wall](https://arxiv.org/abs/2609.30854)**（`2609.30854`，09-25）：长上下文自回归推理的瓶颈是**显存带宽而非算术吞吐**，且随序列变长，约束资源从模型权重转到 **KV cache**——Llama-3-70B 在 BF16 下权重 140 GB，已超过单个加速器的 80 GB HBM，而**一条 128k token 的序列还要再加 42 GB KV**。围绕压缩、驱逐、分页、共享、卸载 KV 的方法已经很多，但各自的工作负载、硬件与质量指标不统一，跨论文无法比较。这篇 SoK 把领域**解析化地统一**：用一套严格区分「推导结论」与「原论文报告值」的协议，推导出 **arithmetic intensity 随上下文长度衰减的闭式表达**，并按硬件拓扑参数化——覆盖 **NVIDIA H100、NVIDIA B200、AMD MI300X**，包含 **per-die 带宽切分**，给出 **KV 流量反超权重流量的 crossover 长度**；随后把文献分为**量化、token 驱逐、KV 分页、前缀缓存、异构分层**五个域，每个域挑一个代表方法，在**统一的 128k 上下文协议**下实测。核心结论是一个**三区间结构**：在硬件相关的 crossover 之下权重流量占主导，此时压 KV 几乎带不来加速；越过 crossover 后 KV 流量占主导，各域都在**拿质量换带宽**、收益逼近 roofline 上限。分域看，**分页与前缀共享是无损的，但解决的是容量而非带宽**；**量化与驱逐直接砍带宽**，其中量化在 **4-bit 以下精度衰减明显加速**，驱逐在位置敏感任务上衰减呈**不连续**跳变；**异构分层把带宽墙转化成了互连问题**，上界由 PCIe / NVLink 而非 HBM 决定。文末给出按硬件、上下文长度、质量预算选择压缩域的设计准则。

- **[Softmax Reparameterization for Output-Head Quantization](https://arxiv.org/abs/2609.31291)**（`2609.31291`，09-25）：大词表让 output head 成为小模型推理的显著开销。这篇提出 **softmax 重参数化**：在量化**之前**先选出一个**功能等价**的 output head——把**词表行均值的一个标量倍数**从每一行 output row 里减掉，系数按 validation KL 单独为 RTN、activation-weighted MSE、full-Hessian GPTQ 各搜一次。这个一维搜索的搜索空间同时包含「原始头」和「固定 mean-centering」，**保持全精度 softmax 分布不变**，也**不改动已训练好的 decoder 权重**；对 soft-capping 这类**非线性 logit 路径**再用一个 rank-one 修正兜住。效果上，七个 head 的 W4 增益集中在「基线量化把预测扭曲得厉害」的地方：Phi-4-mini 上 AW-MSE 的 KL 从 **0.936 降到 0.256**；换更强的 GPTQ 校准依然成立，且与 per-channel scaling、affine 量化**互补**。系数在 WikiText 上选定后**冻结**，迁移到 C4 与 OpenWebMath：在冻结系数 ≠ 1 的 **18 组对比中全部优于 mean-centering**，其余 6 组持平。W2 压力测试下收益扩展到几乎整个「模型 × 量化器」矩阵。matched residual 分析显示，保真度提升可以伴随**更大的 logit 重构误差**同时**降低残差的 Fisher 加权代价**。工程侧最实用的一点：对 **shift-compatible** 的 head，重参数化**不引入任何推理算子**、且**保住 packed W4 执行**；在 decoder 保持 BF16 的前提下，把 Phi 的 output head 量到 W4 使 **batch-one 生成延迟相对 BF16 head 基线降 10.8%**。

- **[Low-Bit Recurrent States in Hybrid Language Models](https://arxiv.org/abs/2609.30950)**（`2609.30950`，09-25）：混合语言模型靠**定长循环状态**换掉部分 attention，但现有量化器基本都在 8 bit 以上。这里的观察是：**量化误差的留存程度取决于各通道的衰减率**——衰减慢的通道误差会一直留在状态里。作者从**可观测 Gramian（observability Gramian）**导出失真权重，与**归一化后的状态取值范围**一起用于**混合精度的比特分配**，全程**不需要校准数据、不做 rotation、不训练**；衰减率本身也按**对数刻度**量化。实验上，按 token 逐次量化状态时，**4-bit 平均负载**把超额负对数似然相对**七种基线中的最优者**再降 **3.3–27.9×**（三个混合模型，元数据开销另计）；**6 bit 时 NLL 与 FP32 状态基线相差不到 0.005 nats**。消融把增益拆到可变位宽、衰减加权、范围归一化三项上。另需注意边界：**写回频率降低后增益会缩小**，且依赖具体模型与比特预算——这直接影响 kernel 侧的更新粒度选择。

- **[Quantizing Looped Transformers: Feedback Exposure and Calibration Blindness](https://arxiv.org/abs/2609.30820)**（`2609.30820`，09-25）：looped transformer **跨递归步复用权重**，理论上特别适合低比特量化。这篇指出标准 PTQ 在它上面有**两种不同的失效模式**。其一是 **feedback exposure**：在 Huginn-3.5B 上，per-channel INT4 主要**死在非残差的 loop-entry adapter** 上，而量化残差主干反而没那么伤——原因是被量化的层扰动了循环状态，却**没有 identity 路径**把扰动原样旁路掉，误差于是被**喂回后续递归步**累积；受控实验在线性滤波器与 Mamba 状态空间模型上复现了同样的现象，说明这不是 transformer 独有的。其二是 **calibration blindness**：grouped INT4 下，一步式 GPTQ 基线的 Hessian 只用 **step-0 的激活**构建，导致**递归后期才用到的输入方向几乎没有被加权**。后果很直接：在来自 **7 个 looped 架构的 9 个 checkpoint** 上，一步式 GPTQ 有 **5 个在主任务指标上不如朴素 RTN**。修法是把 **GPTQ 的 Hessian 跨递归步累积**，这样在**全部 9 个 checkpoint 上都胜过一步式 GPTQ 与 RTN**，并在 Huginn 上**恢复到 bf16 精度水平**。文章的意义在于把 looped 模型的 PTQ 拆成两个可分别回答的问题：**误差从哪儿进入循环**、**校准看到了哪些状态**。
