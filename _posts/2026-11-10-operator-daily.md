---
layout: post
title: "算子日报 · 2026-11-10"
permalink: /operator-daily/2026-11-10/
date: 2026-11-10 23:59:00 +0800
tags: [算子日报]
---

> 每日汇总：CUDA 开源仓 Release Notes（算子）、芯片动态、arXiv 算子内核论文。

## TL;DR

- **NCCL v2.32.6-1 补全 Vera Rubin 支持**：为 Ring/Tree/NVLS/NVLSTree 加入带 VR 调优数据的流水线 cost model，并补齐 symmetric AllGather / ReduceScatter / AllReduce 的 Rubin 模型；32–40 卡 VR200 的大 NVLink 域拓扑搜索更快。Device API 新增 TMA 版 LSA 广播 `ncclLsaCopyTma`，以及实验性的 Flow 流水化集合 kernel 与低延迟 `AlltoAll_LL`。
- **异构硬件的稀疏与定制算子**：AscSpMM 用 Vector-CSR 把索引解码和矩阵计算拆给 Ascend 的 Vector/Cube 引擎，910B3 上几何平均比 GEMM 快 **2.36×**、比已有 SpMM 快 **19.92×**、比 A100 cuSPARSE 快 **5.82×**；Trainium 侧把 C++ 自定义算子编进 GPSIMD，DeepSeek-V4.1-Flash 的 Engram n-gram 哈希快 **1.79×**、表检索最高 **4.95×**。
- **Kernel 工程的三条硬结论**：7B 代码模型替换 PyTorch 算子的瓶颈是正确性而非速度（正确的 kernel 平均也只有 eager 的 0.97–0.99×），profiling 引导提示最强；vLLM 的 atomic Split-K 在 **97.61%** 的同输入对比里给出不同结果，确定性 Triton 版只多 1.02–2.62% 延迟；CVSA 用 softmax 覆盖度做闭环校验，128K 上下文 decode 中位加速 **6.21×**。

## 一、CUDA 开源仓 Release Notes（算子）

### NCCL

- **v2.32.6-1**（11-09 19:28 UTC，本地上 11-10 03:28）— [链接](https://github.com/NVIDIA/nccl/releases/tag/v2.32.6-1)：Vera Rubin 的性能模型补全版。为 Ring、Tree、NVLS、NVLSTree 加入带 VR 调优数据的流水线 cost model，补齐 symmetric AllGather / ReduceScatter / AllReduce 的 Rubin 模型，并加速 32–40 卡 VR200 这类大 NVLink 域的拓扑图搜索（缩短 communicator 初始化）；tuner 插件接口升到 v7 并支持 Rubin。算子侧新增 TMA 版 LSA 广播 `ncclLsaCopyTma`（Blackwell 及以后），以及实验性的 Flow device API——用 Simple / LL / LL128 协议在 LSA 与 GIN 上搭流水化集合 kernel（签名尚不稳定）；`ncclGetPeerPointer` 扩展为可按 world rank 寻址所有可 load/store 的本地 peer。基于 Flow 的 General Kernels（GENK）默认不编译，需 `SYM_GENK=1` 与 `NCCL_SYM_GENK_ENABLE=1` 开启；另新增面向小/中等 per-peer 消息的低延迟 symmetric `AlltoAll_LL`（`NCCL_SYM_KERNEL=AlltoAll_LL`）。诊断侧 profiler 插件接口升到 v8，可在每个事件上带 NCCL 采样的 GPU/CPU 时间戳。

> 说明：cuDNN Frontend 11-10 的 `v1.32.3-rc.80792049` 是 1.32.3 的 RC 构建（wheel 版本即 1.32.3、待 promote 到 PyPI，当前尚无正式 1.32.3 tag），按 10-30 期「RC 不计、只收正式版」的口径暂不收录；11-05 / 11-06 的两条 `1.33.0.dev*` 仍是内部 CI 的 nightly 自构建（自述 *Not a supported release*），同样不计入。

## 二、芯片动态

> 本期无新条目。窗口内 RSS 只返回 11-09 的 [NVIDIA 与韩国 MSIT 扩大主权 AI 合作](https://blogs.nvidia.com/blog/korea-sovereign-ai-collaboration/)，已收录于 11-09 期。

## 三、arXiv 论文（算子内核）

> 本期增量为 **11-07～11-09 提交的 113 条**：周末投稿在周一/周二的首个公告批次放出，此前 11-07/11-08 两期均按空窗口跳过。逐 id 与往期各期比对无重复，按算子开发相关性精选 7 篇。

### 稀疏计算与异构算子

- **[AscSpMM: Efficient SpMM on Ascend NPUs via Pipelined Vector-Cube Collaboration](http://arxiv.org/abs/2611.11823v1)**（`2611.11823`，cs.AR，11-09）：SpMM 需要索引驱动的操作数选取、间接寻址和分散访存，与 NPU 矩阵引擎偏好的固定形状、连续布局冲突，GPU Tensor Core 的 SpMM 方案也无法直接搬到向量/矩阵引擎解耦的 NPU。AscSpMM 用参数化的 **Vector-CSR** 表示把不规则的操作数准备与规整矩阵计算分开：Vector 引擎解码索引并 gather/重排，Cube 引擎做矩阵乘累加，双槽环形缓冲区实现依赖感知的异步流水，配合操作数复用与运行时自适应 tiling 减少跨引擎停顿。Ascend 910B3 上几何平均相对 GEMM 快 **2.36×**、相对已有 SpMM 实现快 **19.92×**、相对 A100 上的 cuSPARSE 快 **5.82×**，矩阵输入存储与峰值 HBM 占用平均降 **48.24% / 22.98%**，与专用 GPU Tensor Core SpMM kernel 相当；代码已开源（[GitHub](https://github.com/LfLab-AI/Ascend-NPU-SpMM)）。
- **[Exploiting Native Hardware Heterogeneity: Custom GPSIMD Operators for Accelerating Emerging LLM Memory on AWS Trainium](http://arxiv.org/abs/2611.11820v1)**（`2611.11820`，cs.DC，11-09）：DeepSeek-V4.1-Flash 的 Engram 把 64-bit n-gram 哈希、稀疏表查找这类不规则整数运算与 GEMM、gate 混在一起，在 GPU-FPGA 系统上按设备拆分可加速 **5.9×**。AWS Trainium 的 NeuronCore 本身就并列 dense 引擎与可定制的 GPSIMD 引擎、共享 SBUF，但 NKI 只通过固定内建指令暴露 GPSIMD，官方自定义算子工具链也不支持 Trainium 2/3。作者搭了一套把 C++ 自定义算子编译进 GPSIMD、再注入 NKI kernel 生成的 NEFF 的工具链；相对手写优化的 NKI 基线，n-gram 哈希在 decode 快 **1.79×**，表检索最高 **4.95×**，整个 Engram 计算最高 **1.14×**。

### Attention 与低精度量化

- **[Coverage-Verified Sparse Attention: Closed-Loop Quality Control for Long-Context LLM Decoding](http://arxiv.org/abs/2611.10634v1)**（`2611.10634`，cs.LG/cs.CL，11-07）：现有稀疏 attention 是开环的——kernel 选定 KV 子集后，没有任何信号告诉它 attention mass 是否保住，也没有恢复路径。CVSA 用「coverage」（被选 token 上的 softmax mass 占比）作为信号：它与每步输出误差有精确的代数关系，可同时当接受判据和恢复时的插值权重；据此做 training-free 的 verify-then-correct 解码循环，高覆盖 draft 直接接受，只对不达标的 head 做恢复，不需要采样、draft 模型或投机解码设施。LongBench 7B/70B 上距 dense attention **1.13 分**以内（70B 统计上持平），RULER 掉 0.8 分，128K 上下文下 decode 步中位加速 **6.21×**。
- **[MACaQ: Fast, Model Agnostic Calibrated Quantization for Low-Precision Weights and Activations](http://arxiv.org/abs/2611.10649v1)**（`2611.10649`，cs.LG，11-07）：MACaQ 是 **MXFP4 的 W4A4 PTQ** 方法，每个 linear 层独立处理：对激活在线做 per-channel scale 与 block-Hadamard 变换，并把逆变换吸收进权重，因此无需跨层传播基变换，可流式实现（内存里只驻留一层）。为优于 round-to-nearest，它用 Hessian-aware 的 **Flip-Refinement** 坐标下降，把单个 FP4 code 贪心移到相邻可表示值。Qwen3-32B 在 vLLM 里只比纯 MXFP4 多 **2.3%** 端到端开销，perplexity 低于 DuQuant++ 与 MR-GPTQ，Llama-3-8B 上以 **1/11** 的量化时间追平二者最好结果；122B 模型用 64 分钟、32.5 GB 显存完成量化，并可直接用于图像扩散模型。

### Kernel 工程与集合通信

- **[Correct First, Fast Second: What a Local 7B Code Model Can and Cannot Do for GPU Kernel Optimization](http://arxiv.org/abs/2611.12079v1)**（`2611.12079`，cs.LG，11-09）：论文固定使用本地 7B 的 Qwen2.5-Coder-7B-Instruct，在 30 个 KernelBench 任务（level 1–3 各前 10 个，V100 评测）上比较四种交互策略：zero-shot、profiling 引导、执行反馈迭代、多 agent 循环。结论是**正确性而非速度是瓶颈**——即便生成的 kernel 正确，平均也只有 eager PyTorch 的 **0.97–0.99×**；7B 模型写不过 cuBLAS/cuDNN 的稠密 GEMM 与卷积，真正的收益只在 fused memory-bound epilogue。profiling 引导提示整体最强（**43%** 正确、27% 比基线快），agent 循环是唯一解出 level-2 fusion 的方法（**3/10** vs 其余 0/10）；但粗糙的 agent 脚手架有害（首版 3% 正确，低于 zero-shot 的 30%），把算力花在执行 oracle 下的 best-of-n（n=1→8 从 3% 到 **33%**）比堆推理步数有效。
- **[Reproducible LoRA Inference Through Deterministic Split-K Reduction](http://arxiv.org/abs/2611.10334v1)**（`2611.10334`，cs.LG/cs.AI/cs.DC/cs.PF，11-07）：vLLM 为 LoRA 低秩乘引入的 Split-K 用浮点 atomic add 合并部分和，带来数值不确定性。作者在 GSM8K、MBPP 与 Civil Comments 毒性判定上复现：Atomic Split-K=8 在 **97.61%** 的同输入对比中给出不同的 FP32 shrink 输出，重复推理分别有 **17.30% / 14.96%** 的 GSM8K / MBPP 输出变化（有时改变正确性），结构化决策概率在 **81.47%** 的对比中变动并出现决策翻转。其确定性 Triton Split-K=8 保留并行度、未观察到变化，相对 unsplit 归约仍有 **16.8–21.1%** 延迟收益，只比 atomic Split-K=8 多 **1.02–2.62%** 延迟。
- **[Network Contention State Aware Dynamic Optimizer for Accelerator Collective Communication Libraries](http://arxiv.org/abs/2611.10984v1)**（`2611.10984`，cs.DC/cs.LG/cs.MA，11-07）：集合通信库的配置由静态启发式决定，不看实时网络状态：作者实测这些启发式在无争用环境接近最优，但背景流量下明显退化。他们给 NCCL 加了协作式多 agent 运行时优化器：LSTM + 序贯概率比检验的 detector 区分持续争用与瞬时抖动，Dueling Double DQN + CQL 的 selector 据此调配置。Chameleon Cloud 裸金属集群上 AllReduce 延迟相对 NCCL 降 **36%**、相对收敛的 AutoCCL 降 **32%**；六个真实负载（CNN / Transformer / 推荐）在 Delta 集群 A100 + Slingshot-11 上最高 **65%** 加速。
