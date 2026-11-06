---
layout: post
title: "算子日报 · 2026-11-06"
permalink: /operator-daily/2026-11-06/
date: 2026-11-06 23:59:00 +0800
tags: [算子日报]
---

> 每日汇总：CUDA 开源仓 Release Notes（算子）、芯片动态、arXiv 算子内核论文。

## TL;DR

- **AMD CDNA4 的 kernel 空间被集中开采**：ZYFA 用直接 FP8 概率计算 + ping-pong 调度 + 二次幂缩放，把 MI355X 上的 FP8 FlashAttention prefill 做到 AITER 的 **1.16–1.76×**（32K、head dim 128 为 1.26–1.37×，DeepSeek 式 MLA 达 1.99×），数值误差还低 62–78%；KOSMOS 用自调度 persistent kernel 把 TP+SP collective 融进 GEMM、整个专家层一次 launch，MI350X/MI355X 上 DeepSeek-V3 / Qwen3-235B 代理训练比 DeepEP 快 **1.29–1.83×**；AMD capability-gain 分析则指出，ISA 级 FP8 收益会被「未融合的量化」在 LLM 里吃成逐层减速。
- **稀疏 attention 转向「层级选择 + 每 tile 精度分级」**：RecSA 在 KV cache 上建树，把每个 query 的选择与读取都压到 O(log L)，单张 H200、524K 上下文下 prefill/decode 分别快 **3.2×/3.8×**，1M RULER 得 **60.5**；TriAttention 把 QK tile 分到跳过 / 4-bit / 8-bit 三档，prefill fused kernel 合并两档贡献，相对 BF16 基线最高 **2.65×**（对 Torch attention **7.88×**）。
- **低比特推理把「精度分级」下沉到 tile 与内存层级**：TileQuant 以硬件对齐的 2D weight tile 为 PTQ 分配单位，生成吞吐达原模型 **2.66×**；StrataFlux 把 host RAM/SSD 当额外精度来源做选择性流式取回，2-bit Llama-3.1-8B 的 WikiText-2 PPL 从 **9.46 降到 8.40**，Qwen3-8B 在 2-bit 权重 + KV 下 AIME 2024/2025 从 **11.0% 升到 47.1%**。

## 一、CUDA 开源仓 Release Notes（算子）

### cuda-python

- **cuda-pathfinder v1.8.6** — [链接](https://github.com/NVIDIA/cuda-python/releases/tag/cuda-pathfinder-v1.8.6)（2026-11-05 21:23 UTC）：CUDA 工具链组件定位库新增 Windows CTK 13.5 的 CUPTI DLL 名 `cupti64_13.dll`，`load_nvidia_dynamic_lib("cupti")` 可以识别新版 CUPTI（[PR #3368](https://github.com/NVIDIA/cuda-python/pull/3368)）。对算子的落点是 Windows 上的 profiling / 工具链探测不再因 DLL 改名失败；发布正文只有文档/PyPI/conda 链接，内容取自仓库内 `1.8.6-notes.rst`。

> 说明：cuDNN Frontend 11-06 的 `1.33.0.dev79840341` 仍是 NVIDIA 内部 CI 的 nightly 自构建（自述 *Not a supported release*），按既有口径不计入；30 天窗口内其余正式 release（CUTLASS v4.8.1、CCCL python-1.3.0 / v3.5.0、TensorRT 11.4、cuda-python v13.5.0、cuda-pathfinder v1.8.5）均已在往期各期收录。

## 二、芯片动态

> 本期无新条目。两个 RSS 源在 7 天窗口内均无芯片/架构类新命中。本节连续为空属**数据源侧**状况，不是抓取故障。

## 三、arXiv 论文（算子内核）

> 本期增量为 **11-05 首次被 arXiv API 索引的 52 条**，以及 11-04 晚间补录的 15 条（RecSA、TileQuant、LowRAM、AMD capability-gain 均为此类）；11-06 提交尚未被索引。按算子开发相关性精选 **10 篇**。

### 低精度与量化 kernel

- **[TileQuant: Tile as the Precision-Allocation Unit for LLM Mixed-Precision Post-Training Quantization](http://arxiv.org/abs/2611.07217v1)**（`2611.07217`，cs.LG/cs.AI，11-04 晚到补录）：PTQ 的敏感度在列内也不均匀，整列保留会连带保护不敏感权重，运行时 outlier 检测又给 decode 加开销。TileQuant 把精度分配给硬件对齐的 2D weight tile：离线用输入激活统计做低成本打分、再用梯度/损失敏感度补充，每个投影在保留预算内把最高分 tile 留在 16-bit、其余 8-bit，kernel 直接消费紧凑权重缓冲做原生整型与 16-bit 计算并用局部累加减少访存。生成吞吐达到原模型 **2.66×**，在语言建模、推理、长上下文检索与 QA 上保持质量。
- **[StrataFlux: Memory-Hierarchy-Native Low-Bit LLM Inference](http://arxiv.org/abs/2611.08363v1)**（`2611.08363`，cs.LG/cs.DC，11-05）：低比特权重/KV 的精度损失可以用 host RAM 与 SSD 里的高精度数据补，但朴素 offload 会把生成卡在传输上。StrataFlux 把精化表示、选择与投递一起设计：只流式取回当前需要的精度、与计算重叠、并在多个 decode step 间复用；2-bit Llama-3.1-8B-Instruct 在 RTX 5090 上把 WikiText-2 PPL 从 **9.46 降到 8.40** 且保留量化基座 **95.4%** 的吞吐，2-bit 权重 + KV 让 Qwen3-8B 的 AIME 2024/2025 从 **11.0% 升到 47.1%**（GPU 内存不变）。128K 上下文下 3-bit 权重 + 2-bit KV 达到 vLLM 4-bit AWQ + 8-bit KV 相当的 PPL 与速度，GPU 内存约减半。

### Attention 与稀疏 kernel

- **[ZYFA: Speed and Accuracy of FP8 FlashAttention on AMD CDNA4](http://arxiv.org/abs/2611.07496v1)**（`2611.07496`，cs.DC/cs.LG，11-05）：FP8 只加速矩阵乘，attention 两次矩阵乘之间的 softmax 仍在向量单元上跑——MI355X 上 AITER 在 32K token 只到 FP8 峰值的 **39%**（BF16 为 49%），且 CDNA4 不像 Hopper/Blackwell 那样在硬件里跟踪矩阵指令完成，softmax 与矩阵乘的重叠要靠指令顺序。ZYFA 用三条技巧做 prefill kernel：直接从 score 用乘加 + 整型转换算出 FP8 概率（跳过指数）、一组线程算 softmax 时另一组发矩阵指令的 ping-pong 调度、按 2 的幂缩放概率块后再舍入并按舍入和归一化。三个模型的真实 attention 输入上吞吐为 AITER 的 **1.16–1.76×**，32K、head dim 128 的高斯输入上 **1.26–1.37×**，DeepSeek 式 MLA 达 **1.99×**；数值误差比 AITER 低 **62–78%**。
- **[TriAttention: Unified Sparse Mixed-Precision Attention for Long-Context LLM Inference](http://arxiv.org/abs/2611.07256v1)**（`2611.07256`，cs.LG/cs.AI，11-05）：不同 QK tile 对「跳过」和「降精度」的容忍度不同，TriAttention 把每个 head 的 tile 分成跳过 / 4-bit / 8-bit 三档，用两条对角距离边界压缩策略存储，使策略大小与上下文长度无关。离线校准按归一化投影输出误差与整 tile 执行成本选边界，模型级分配器在共享计算预算下给每个 head 定边界；prefill 的 fused kernel 按精度分组遍历、跳过被省略 tile 并把两档 QK 贡献合并成一次归一化输出，prefill 准备的量化 K/V 直接复用为压缩 KV cache 供 decode。相对 Torch attention 最高 **7.88×**、相对 BF16 基线 **2.65×**，长上下文与通用任务质量基本持平。
- **[Recursive Sparse Attention: Million-Token Length Extrapolation at Logarithmic Cost](http://arxiv.org/abs/2611.07232v1)**（`2611.07232`，cs.CL/cs.LG，11-04 晚到补录）：多数稀疏 attention 要为每个 query 给全部历史 block 打分，单 query 选择成本 O(L)；层级选择能降到 O(log L)，但此前可训练方法只在训练长度内报告精度。RecSA 在 KV cache 上建树，每个节点用一个 key 与 mean-pooled value 概括子树，每层最多选 k 个候选，未选候选的摘要与精确 token 共享一次 softmax；每个条目的 RoPE 位置取它在当层候选中的序号，任意长度下相对距离都落在训练范围。以 Qwen3-4B 初始化、在 32K 长度训练 0.62B token 后，RULER 131K–2M 全面领先，1M 上下文得 **60.5**（最好基线 Dense+DCA 为 40.0）；优化 kernel 让 64K–2M 的 prefill/decode 都快于 dense，单张 H200、524K 上下文下 prefill **3.2×**、decode **3.8×**。
- **[Muir: Linear Attention with Sensitivity-Allocated Gating and Matched-Filter Readout](http://arxiv.org/abs/2611.08376v1)**（`2611.08376`，cs.LG，11-05）：GDN-2 用 channel-wise decay 与解耦的全秩 erase/write gate，但分析发现状态更新对 erase 的敏感度远低于 write——key 激活与单位归一化把能量集中到稀疏通道，value write 则作用于所有通道，因此 erase gate 可以用低秩参数化；慢衰减的 state 行范数远大于快行，会主导无权 readout。Muir 把所有 gate 融进敏感度分配的低秩投影，用 matched-filter readout gate 重新加权 state 行，把省下的参数投到更宽的 FFN；针对大 chunk 下标准 chunkwise 算法数值不稳，设计了稳定的并行更新。166M–1.34B 参数上 PPL 比 GDN-2 低 **1.8–2.0%**、比 Transformer 低 **2.5–3.6%**，吞吐与 GDN-2 持平，32K 训练比 Transformer 快 **2.7×**，4K 训练长度可外推到 10M token 流。

### KV cache 压缩

- **[LowRAM: Low-Rank Keys and Amortised Values for Composable KV Cache Compression at Test-Time](http://arxiv.org/abs/2611.07025v1)**（`2611.07025`，cs.LG/cs.CL，11-04 晚到补录）：同一 attention head 内 value 很大程度上可由对应的 key 预测。LowRAM 在测试时给每个 head 拟合轻量预测器，只为重建最差的 token 存残差，得到 **4×** 近无损 value 压缩；key 侧用跨层低秩压缩，两者组成非对称管线。每个 decode step 只重建选中的 KV 子集以压低开销，4× 压缩下峰值 decode 吞吐最高 **5×**；与量化、eviction 正交叠加后长上下文可达 **26.9×** 压缩并保留基线 **94.5%** 性能。
- **[TACT-KV: Tri-Axis Cosine Transform for Compressing Volumetric KV Caches in Medical VLMs](http://arxiv.org/abs/2611.07593v1)**（`2611.07593`，cs.CV/cs.LG，11-05）：一次 CT/MRI 体积产生数万 vision token，KV cache 随 batch 与上下文线性增长，已有方法只看特征/token 统计或 1D/2D 变换，没利用三个解剖轴的空间相关性。TACT-KV 对 KV cache 做可分离 3D DCT，按层用 reverse water-filling 在频带间分配比特、每个频带用固定标量码本量化，且免校准。六个医学 VLM、两个 3D CT 基准里 11/12 组取得最低 1-bit prediction-flip rate，方法也迁移到四个通用 VLM 的视频任务；同等 HBM 下单 GPU decode batch 最高 **8×**。

### Kernel 调度与硬件

- **[KOSMOS: Self-scheduling Persistent Kernels for Communication-Compute Overlap](http://arxiv.org/abs/2611.07477v1)**（`2611.07477`，cs.DC/cs.AI/cs.LG/cs.PF，11-05）：TP+SP 与 EP 的集合通信卡在 transformer 层的关键路径上，现有 fused kernel 用固定角色/调度，留出大片空闲窗口，预先把通信-计算切分也不能跟随数据就绪。KOSMOS 用按消费者需求排序的队列做自调度 persistent kernel：workgroup 按数据就绪在通信与计算角色间切换，不需要调度器、逐调用成本模型或调好的切分；它把一层的所有 TP+SP collective 与 GEMM 融合，EP 整个专家层（router 到 fold）每个方向一次 launch。MI350X/MI355X 上 Megatron-LM 的 TP+SP 端到端 BF16 快 **1.10–1.28×**、MXFP8 快 **1.10–1.25×**，DeepSeek-V3 与 Qwen3-235B 代理模型训练比 DeepEP 快 **1.29–1.83×**。
- **[Capability Gain from ISA to Shared Execution on AMD MI300A and MI250X](http://arxiv.org/abs/2611.07198v1)**（`2611.07198`，cs.DC，11-04 晚到补录）：新指令的峰值加速有多少能穿过库、应用、共享设备这些层保留下来，此前很少被量化。作者定义 capability gain，从 ISA 微基准一路追到 hipBLASLt、hipSPARSELt、LLM 推理与混合精度 HPL，并测到最多 8 路并发流的共享设备：FP8 在 8192³ GEMM 上相对 MI250X FP16 仍保留大部分加速，但**未融合的量化会让它在 LLM 里变成逐层减速**；2:4 稀疏在 hipSPARSELt 里最多保留约一半加速；队列共享时 FP8 的 GEMM capability gain 可以 >1，但共享 LLM serving 拿不到 FP8 收益。对算子开发的直接结论是量化必须融进 kernel，否则 ISA 级 FP8 收益会被额外的格式转换吃掉。

> 其余与算子相关但本期未展开的还有：ProQuant（面向边缘 MLLM 的 W4A8 QAT，Qwen3-VL-2B 平均精度 +6.6%、约 1% 性能损失）、ACSeL（把量化配方纳入硬件约束下的配置搜索，用算子级 profile 校准延迟）、DictLLM（共享字典 + 层稀疏系数的 5.1× 后训练压缩）、DMuon（矩阵优化器在 FSDP/HSDP 上的拓扑感知 owner 与通信调度，256×H100 中位吞吐为 FSDP2+AdamW 的 1.05×）。
