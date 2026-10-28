---
layout: post
title: "算子日报 · 2026-10-28"
permalink: /operator-daily/2026-10-28/
date: 2026-10-28 23:59:00 +0800
tags: [算子日报]
---

> 每日汇总：CUDA 开源仓 Release Notes（算子）、芯片动态、arXiv 算子内核论文。

## TL;DR

- **今日最实用的三份 kernel 工程结果**：PipeLink 把独立编译的 kernel 组件按 contract 链接进同一个 kernel，51 组 Hopper/Blackwell 配置与单体编译差距 ≤ **0.2%**，CuTe attention 嵌进 ThunderKittens decode megakernel 后 decode 快 **2.16×**；H100 的 DSMEM 会因 floorsweeping 吃掉 cluster 可用资源，wave-aware clustering 让 Hopper FA4 吞吐最多 **+5.30%**；SpGEMM 用固定结果大小的 morsel 拆解多维 skew，在偏斜负载上稳定优于 SOTA。
- **Agentic kernel 生成开始补「可靠性」这一课**：83 个可批处理产物里查出 **19 个 batch-invariance 违规**，加入静态依赖分析与动态逐位校验后，BI 通过率从 **67.6% 升到 91.9%**，仍保留 91.8% 的基线加速；JustGoal 只给优化目标、不带 kernel 专用基础设施，在 B200 上 **8/10 算子**超过开源库（1.14–2.34×），提示专用知识与 workflow 的增益要拿同模型 goal-only baseline 来对照。
- **低精度、KV 与 MoE 侧的三条结论**：S3R 用「每 token 共享一个随机阈值 + 结构化误差修正」绕开 vector unit 瓶颈，INT8 训练在 TPU v5p 上比 BF16 快 **1.24–1.32×**；DCKV 端到端训练 KV codec，Qwen3.5-2B 上比 bf16 少 **55×** 字节仅掉 ≤1.8 分；MoE channel selector 改用免校准的 top-k，配合 ternary sketch reader 在 **17.8%** 专家读取下保住 **98.1%** 精度，Triton kernel 在 batch 1 比 vLLM 快 **1.60–1.90×**。

## 一、CUDA 开源仓 Release Notes（算子）

### CCCL

- **python-1.3.0**（10-28）— [链接](https://github.com/NVIDIA/cccl/releases/tag/python-1.3.0)：维护版本，无新公开 Python API，但对 JIT 算子开发有几个直接相关的修复。
  - **stateful operator 缓存不再按数组长度重编译**：闭包捕获的 device array 长度此前被烘进 LLVM IR 常量，构建缓存必须按 `(dtype, shape)` 建键；现在长度与指针一起放进 state buffer、调用时加载，缓存键只依赖 dtype，长度变化不再触发全量重编译。
  - **`gpu_struct` 字段映射改用 `dtype.names`**：带 field title（别名）的结构化 NumPy dtype 不再被误判出多余成员。
  - **`cuda.compute` reduce/scan 降低 per-call 开销**：把运行时求值的 `cast(...)` 换成带引号的 no-op 注解，省掉每次调用的 union 对象分配，代表 benchmark 的主机侧开销从 **4.32→3.70 µs**（reduce）、**3.06→2.55 µs**（exclusive scan）。
  - 其余：`numba-cuda-mlir` 版本 pin 放宽到允许 0.6.0（`>=0.5.2,!=0.5.3,<0.7`）；修复 Windows + CTK 12.4 下 `__half`/`__nv_bfloat16` 重复定义的 NVVM 链接错误；合并排序的 `make_*` 与一次性 API 现在共享缓存键。

> 30 天窗口内其余 release（CUTLASS v4.8.1、cuDNN Frontend v1.32.0 及其 nightly、CCCL v3.5.0 / v3.4.3 / python-1.2.1、TensorRT v11.4、cuda-python 系列）均已在往期各期收录，本期不重复。

## 二、芯片动态

> 本期无新条目。NVIDIA Developer Blog 源继续返回 0 条；NVIDIA Blog 在 7 天窗口内无新的芯片/架构类命中。本节连续为空属**数据源侧**状况，不是抓取故障。

## 三、arXiv 论文（算子内核）

> 本期按 `fetch_state` 从 10-27 起抓增量，实际返回 **150 条**（触及 `max_results` 上限，按提交时间倒序）：10-27 全部 **54 条**、10-26 共 72 条、10-25 被截断后 24 条。10-28 提交的论文尚未被 arXiv API 索引（最新一条为 10-27 17:59 UTC），留待下一期回补。与被截断的 10-25 无关，10-27 的新条目完整；与上一期抓取逐 id 比对，另有 10-26 晚到 20 条、10-25 晚到 1 条，其中 GateRQ 与算子相关，补录于本期，其余晚到条目属应用/理论方向。按算子开发相关性精选 11 篇。

### 低精度与量化

- **[Efficient Quantization Training with Stochastic Rounding via Structured Subspace Randomness](http://arxiv.org/abs/2610.36587v1)**（`2610.36587`，cs.LG，10-27）：标准 SR 为每个梯度元素抽独立随机阈值，但加速器的随机数单元吞吐远低于矩阵单元，逐元素抽签会把 INT8 训练拖到比 BF16 还慢。S3R 改为**每个 token 在 hidden 维共享一个阈值**，并量化一次同时供 activation/weight 两个梯度矩阵乘使用；S3R-EC 再从 activation 与 weight 梯度中减掉结构化平均舍入误差。LLM 预训练里未修正的阈值共享会明显伤害收敛，S3R-EC 追平独立 SR 的验证 loss，并把 TPU v5p 上 dense LLM 的完整 INT8 step 加速到 BF16 的 **1.24–1.32×**、独立 SR 的 **1.11×**。
- **[Let the Gate Decide: Per-Lookup Precision for Quantized Conditional Memory in LLMs](http://arxiv.org/abs/2610.35985v1)**（`2610.35985`，cs.LG/cs.AI，10-26 晚到补录）：conditional memory 层给 LLM 挂大容量 hashed n-gram 表，低比特量化会先丢长尾事实；而按访问频率给高频行更多比特反而比随机分配更差——真正决定量化误差权重的是**缩放该行的 gate**，它对高频 n-gram 小、对长尾实体大，且可从 1-bit code 算出。GateRQ 把每行存成 **1-bit 残差码阶梯**，第 1 层常驻 GPU 显存用于算 gate，只有 gate 排名高的 lookup 才读更深层。无需重训即可把 DeepSeek-V4.1-Flash 的 GPU 表显存压 **7.8×**，vLLM 吞吐比 FP8 表高 **11%–31%**；每次 lookup 读 1.50 bit 时 PopQA 约掉 1 分，读 2.10 bit 时两个模型仅掉 0.2 分。

### Attention、KV cache 与 MoE

- **[DCKV: Learned Transform Coding of the Prompt KV Cache in Hybrid Language Models](http://arxiv.org/abs/2610.36950v1)**（`2610.36950`，cs.LG/cs.AI，10-27）：混合模型只在少数 full-attention 层保留 KV cache，但这部分仍随 prompt 线性增长、需要存储或搬运。DCKV 是首个**端到端训练**的 KV cache codec：变换、量化器与熵模型都以 prompt token 和已解码的相邻位置为条件，并同时以码长和冻结模型输出变化为优化目标。Qwen3.5-2B 上比 bf16 少 **55× 以上**字节时三个长上下文基准最多掉 1.8 分，少 **100× 以上**字节时最多掉 3.6 分。
- **[Channel Selection for Mixture-of-Experts Decoding Needs No Calibration Data](http://arxiv.org/abs/2610.37386v1)**（`2610.37386`，cs.LG/cs.AI，10-27）：MoE 小 batch decode 的瓶颈是反复从显存读 routed expert 权重，channel selector 本应只读重要通道。作者指出按离线数据校准的阈值既不能保证读多少通道（按 18% 预算校准的 Prox 实际平均只读 **12.9%**），也不能稳定带来精度收益，改用**免校准 top-k** 后在 43 组对比中 40 组不劣于阈值版。配套 reader 用模型自身权重构造三元 sketch 筛通道、再用低秩映射恢复被跳过的通道；三个 post-trained MoE 上以 **17.8%** 的 routed-expert 读取保住 **98.1%** 全模型精度，Triton kernel 在 batch 1 比优化过的 vLLM 快 **1.60–1.90×**、更大 batch 最高 **2.69×**。
- **[Balanced State Subspaces for Mamba-2 State Compression](http://arxiv.org/abs/2610.36617v1)**（`2610.36617`，cs.LG，10-27）：Mamba-2 把所有上下文压进固定大小的 recurrent state，decode 的显存与带宽主要花在这里。作者发现训练后的 state 能量几乎 rank-one，但输出读的并不是能量方向——能量集中在 recurrence 累积的 bias-like 方向，内容却位于与坐标不对齐的低维子空间。方法用**可控/可观 Gramian 的 balanced truncation**压缩 write–read 子空间，并借 Mamba-2 的 scalar-identity transition 精确施加旋转；HARP 再按 Hankel spectrum 逐层分配 rank。单次 forward 即可校准，state 维度减半时 WikiText-2 perplexity 平均只升 **0.3 分（1.8%）**，GHOST 同条件升 **1.6 分（9.3%）**。

### GPU kernel 工程

- **[Kernel Performance Engineering under Physical Non-Idealities](http://arxiv.org/abs/2610.37466v1)**（`2610.37466`，cs.AR，10-27）：H100 的 distributed shared memory（DSMEM）能减少跨 SM 数据搬运，但 IC 物理非理想性导致的 **floorsweeping** 会限制 cluster 可用资源，可能把收益抵掉。作者据此做 **wave-aware clustering**，在 global-load-and-broadcast microbenchmark、GEMM 与 Hopper 版 FlashAttention 4 上验证；FA4 吞吐相对基线最多 **+5.30%**。进一步发现 TDP 受限时，减少数据搬运可通过 DVFS 抬高 SM 频率，解释了一些「不该快却快了」的现象；同样的功耗约束也限制 Tenstorrent Wormhole 的吞吐扩展。
- **[PipeLink: Contract-Guided Linking of GPU Kernel Components](http://arxiv.org/abs/2610.37200v1)**（`2610.37200`，cs.DC，10-27）：高性能 kernel 由 loader、mainloop、epilogue、通信阶段等组件并发共享同一 SM 的片上内存、barrier、warp 与寄存器，但组件把自身位置硬编码，无法独立编译和替换。PipeLink 让每个组件编译成携带**资源 contract** 的 LTO object，检查 contract 后用一次联合求解完成 SM 资源重定位并生成组件间搬运，由 vendor LTO backend 折成常量。51 组 Hopper/Blackwell 配置与同一份源码单体编译差距 **≤0.2%**；把 CuTe attention 算子嵌进 ThunderKittens decode megakernel 后 decode 快 **2.16×**，四 owner 的 MoE 训练 megakernel 追平原单体，GEMM+SwiGLU 组合在 Blackwell 上比 cuBLAS 带编译 activation 快 **1.2×**。
- **[Taming Multi-Dimensional Skew in Sparse Matrix Multiplication with Contention-Aware Scheduling [Technical Report]](http://arxiv.org/abs/2610.36822v1)**（`2610.36822`，cs.DC/cs.DB，10-27）：SpGEMM 的真实负载同时存在**多维 skew**（计算量、输出密度、数据局部性无法用单一指标平衡）与**cache contention**（稠密子结构造成片上干扰）。AMS 用固定结果大小的 morsel 把 skew 维度解耦，配合 contention-aware load balancing、按需激活 SMT 的 SMTact，以及逐 morsel 选择累加器的 HYB。在矩阵平方、关系查询、图分析等负载上，偏斜场景显著优于 SOTA，非偏斜场景保持可比性能，带宽利用率也更高。
- **[God Does Not Play Dice: Enforcing Batch Invariance in Agentic Kernel Generation](http://arxiv.org/abs/2610.36623v1)**（`2610.36623`，cs.AI/cs.LG/cs.SE，10-27）：LLM serving 动态组 batch，若 kernel 缺少 **batch invariance（BI）**，同一请求换个 batch 上下文就会得到不同数值结果。作者在 83 个可批处理产物中确认了 **19 个 BI 违规**，并提出 contract-guided 静态依赖分析 + 动态逐位一致性校验，作为 kernel agent 优化循环里的验收门，先拒掉结构风险与动态不 invariant 的候选，再进入性能 benchmark 并回传反馈。三个 agent 框架上 BI 通过率从 **67.6% 提升到 91.9%**，同时保留 **91.8%** 的基线几何平均加速。
- **[JustGoal: With Only a Goal, the AI Baseline Is Above the Expert Kernels](http://arxiv.org/abs/2610.36434v1)**（`2610.36434`，cs.DC，10-27）：作者提出只给**优化目标与约束**、不给 kernel 专用基础设施/知识库/预设 workflow 的 goal-only baseline，由通用 agent 自建评测循环。B200 上它在 **8/10 算子**上超过开源库，逐算子几何平均加速 **1.14–2.34×**，在 9 个可比算子中 5 个超过公开最快的专用 agent kernel。同模型同范围对照里，KDA-0.5 harness 只到 JustGoal 的 **0.95×**，单机制消融也看不到领域知识或预设 workflow 的一致收益——专用机制需要拿 goal-only baseline 证明自己。
- **[Less is more: learning what to profile for GPU kernel optimization](http://arxiv.org/abs/2610.36576v1)**（`2610.36576`，cs.LG/cs.AI，10-27）：给 kernel 优化 LLM 的 profiling 信息不是越多越好：指标数量增加并不单调提升优化效果，且最有用的信息随 kernel 变化。AutoProfiler 从优化反馈中学习一棵确定性的 **profiling tree**，用 REWRITE / SPLIT 迭代增删节点、由 LLM 生成候选、用实测决定接受与否；树建好后对任意 kernel 快速采集所需信息，LLM rewriter 只需调用一次。KernelBench 上在受限 profiling 预算内优于固定 profiling 配置。

> 其余新入库条目多为应用/训练方法向（视觉生成、医疗影像、遥感、机器人、推荐、科学计算应用等）或与 GPU 算子无直接关系的理论工作，未收录。
