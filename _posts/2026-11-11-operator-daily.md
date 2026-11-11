---
layout: post
title: "算子日报 · 2026-11-11"
permalink: /operator-daily/2026-11-11/
date: 2026-11-11 23:59:00 +0800
tags: [算子日报]
---

> 每日汇总：CUDA 开源仓 Release Notes（算子）、芯片动态、arXiv 算子内核论文。

## TL;DR

- **NVLinkOneSided 把 MoE 的 EP all-to-all 做成单边推送**：源 rank 不先交换 metadata，直接把 token 写进目标 GPU 的 rank-major buffer，grouped GEMM 原地读取，省掉 per-expert 拷贝；Blackwell 上 dispatch/combine 比 DeepEP 快 **1.2–3.0×**（小 batch 收益最大），MoE layer 最多快 **2.4×**，≤128 token/GPU 时仍胜过 fused DeepGEMM MegaMoE。
- **cuDNN Frontend v1.32.3 正式转正**（昨日的 RC 已 promote）：SM100 上 DSA backward 的确定性 dKV relay 提速并可选 relay cluster size，SM90 补齐 DSA backward/indexer 的确定性执行，CSA compressor 在 SM90 打开 2/4 压缩比并调优 ratio-2 backward schedule。`cuda.core` 1.3.0 给 kernel launch 加了 `priority` 与 sm90+ 的 `cluster_scheduling_policy_preference`，`Linker` 支持增量链接。
- **精度与稀疏两侧的边界**：Ozaki 用低精度 tensor core 模拟 FP64 只对「大、稠密、GEMM-shaped」负载成立，这类负载不到 HPC 生态的 **10%**；tritify 把整条 forward 跑成精确整数算术，跨 CPU ISA / CPU-GPU 输出 bit-identical。

## 一、CUDA 开源仓 Release Notes（算子）

### cuDNN Frontend

- **v1.32.3**（11-10 18:11 UTC，本地上 11-11 02:11）— [链接](https://github.com/NVIDIA/cudnn-frontend/releases/tag/v1.32.3)：昨日 `v1.32.3-rc.80792049` 的正式转正版——RC 的正文只有 promotion 元数据，正式 tag 才带 release notes，以下按正式版条目收录。DSA backward 在 SM100 上优化确定性 dKV relay 性能，并新增 relay cluster-size 选择（[#2072](https://github.com/NVIDIA/cudnn-frontend/pull/2072)、[#2082](https://github.com/NVIDIA/cudnn-frontend/pull/2082)）；SM90 上为 DSA backward 加入确定性执行与可配置 relay cluster size（[#2209](https://github.com/NVIDIA/cudnn-frontend/pull/2209)），DSA indexer 也在 SM90 支持确定性 backward，并改进 top-k ≥512 时的 dW reduction（[#2249](https://github.com/NVIDIA/cudnn-frontend/pull/2249)）。CSA compressor 在 SM90 打开压缩比 2 和 4，并调优 ratio-2 的 backward schedule（[#2247](https://github.com/NVIDIA/cudnn-frontend/pull/2247)）；PyTorch SDPA 侧改为检测 `torch.nn.attention` 里的 cuDNN SDPA 开关，避免重复注册 provider（[#2368](https://github.com/NVIDIA/cudnn-frontend/pull/2368)）。

### cuda-python

- **cuda-core v1.3.0**（11-11 00:33 UTC，本地上 08:33）— [链接](https://github.com/NVIDIA/cuda-python/releases/tag/cuda-core-v1.3.0)：面向 kernel 启动与 JIT 的几处直接改动。`LaunchConfig` 新增 `priority`（沿用 stream priority range，越界的非零值报 `ValueError`，默认 0 保持 stream 原优先级）和 `cluster_scheduling_policy_preference`（`"DEFAULT"` / `"SPREAD"` / `"LOAD_BALANCING"`，要求 sm90+）；`host_launch` 可按 stream 顺序入队 Python callable 或 ctypes 的 `CUhostFn`，编译好的 host 函数指针原样交给 driver、绕过 GIL，是 `GraphBuilder.callback` 的 stream 侧对应物。`Linker` 支持 `incremental=True`（nvJitLink 13.2+，partial native result 以 CUBIN 返回、可继续链接）与 `"ltoir"` 目标（13.3+），增量 LTO 链能把 IR 留到最后一次链接。修复 `PinnedMemoryResource` 的 host memory pool 默认只对 host 可见、kernel 读写会 `CUDA_ERROR_ILLEGAL_ADDRESS` 的问题（[#3205](https://github.com/NVIDIA/cuda-python/issues/3205)）；`DeviceMemoryResource` 等 pool 的 `release_threshold` / `reuse_*` 属性可写、新增 `trim()` 归还空闲内存，流捕获期间调用也保持 capture 有效。`VirtualMemoryResource` 的 buffer 可通过 POSIX fd 或 64-byte fabric handle 跨进程共享（fabric 需 NVLink fabric 与 IMEX channel，未进 CI）；`CUDAOutOfMemoryError` 现在同时是 `MemoryError`。依赖面：最低 `cuda-bindings` 要求 CUDA 12 为 12.9.8、CUDA 13 为 13.5.0，移除 Python 3.10。

## 二、芯片动态

> 本期无新条目。窗口内 RSS 只返回 11-09 的 [NVIDIA 与韩国 MSIT 扩大主权 AI 合作](https://blogs.nvidia.com/blog/korea-sovereign-ai-collaboration/)，已收录于 11-09 期。

## 三、arXiv 论文（算子内核）

> 本期增量为 **11-08～11-10 提交的 50 条**，其中 18 条（11-08 两条、11-09 十六条）是 arXiv 追补公告补出的迟到条目；逐 id 与往期各期比对无重复，按算子开发相关性精选 6 篇。

### 通信与 MoE 算子

- **[One-Sided Expert-Parallel All-to-All for Mixture-of-Experts Inference over NVLink](http://arxiv.org/abs/2611.13188v1)**（`2611.13188`，cs.DC/cs.CL/cs.LG/cs.PF，11-09）：Expert parallelism 把 MoE 专家切分到多卡，每层要两次 all-to-all（dispatch 与 combine）；推理侧每卡每次处理的 token 少，小 batch 下延迟由不随 token 数增长的固定开销主导，而不是数据搬运。NVLinkOneSided 在单个 NVLink 域内做 one-sided EP all-to-all：每个源 rank 不先交换 metadata，直接把 token push 到目标 GPU 的 rank-major buffer，grouped GEMM 原地读取，省掉「先拷到 per-expert buffer」这一步；通信同步给两种协议，Fence 面向高吞吐、Sentinel 面向低延迟。Blackwell 上 dispatch/combine 比 DeepEP 快 **1.2–3.0×**（小 batch 加速最大），整个 MoE layer 比 DeepEP + DeepGEMM 最多快 **2.4×**，在每 GPU 至多 128 token 时仍快过 fused DeepGEMM MegaMoE，decode 阶段也有竞争力。接口与 activation 量化格式无关、和 MoE compute backend 解耦，便于接入推理框架。

### 精度、稀疏与异构加速

- **[FP64 Performance and the Limits of Ozaki Emulation on HPC Systems](http://arxiv.org/abs/2611.13332v1)**（`2611.13332`，cs.DC/math.NA，11-09）：论文用代表性 HPC 负载的 roofline profiling 检验「加速器可以放弃原生 FP64、靠低精度 tensor core 按需模拟」这一说法。结论是 Ozaki 模拟不应被当作 FP64 的通用替代，而只适合 dominant computation 足够大、稠密、GEMM-shaped 的负载——这类负载不到 HPC 生态的 **10%**；即便在 GEMM 上单独有加速，数值行为、workspace 需求、矩阵形状与集成开销也会决定应用级收益能否存活。更进一步，dense GEMM 只占 HPC 全部 FP64 计算的一小部分，大部分应用级 FP64 工作落在 GEMM 模拟够不到的地方。对算子侧的提示：FP64 模拟应按「形状 + 数值 + 集成成本」设准入条件，而不是按 GEMM 峰值算。
- **[Deterministic, Reproducible Inference in Exact Integer Arithmetic, and Verifiable Distributed Inference as Its Consequence](http://arxiv.org/abs/2611.13730v1)**（`2611.13730`，cs.LG/cs.DC/cs.SE，11-10）：优化过的浮点推理在不同硬件、编译器、线程数下会产生不同 bit；tritify 是开源 runtime，把 decoder-only LM、MoE decoder 以及 SAM 2.1 图像/视频栈的整个 forward 路径跑在精确整数算术上，浮点只留在离线量化和验证用的 float reference。它给出三条可验证的性质：run-to-run、跨 CPU 指令集、跨 CPU/GPU 的输出 bit-identical（附输出的 SHA-256）；整数路径复现浮点参考的决策（OLMoE-1B-7B 与一个 106B MoE 逐层到 16 层无 fidelity cliff）；计算可拆到不同指令集、CPU/GPU 与联网机器而输出 bit 不变——GLM-4.5-Air（1068 亿参数、427 GB 整数码）在三台单机都放不下整数码的机器上分片推理，结果与单机参考 bit-identical。作者不声称 per-device 加速，也未保证浮点中间值的 bit 相等。代码已开源（[GitHub](https://github.com/pelloran-inc/tritify)）。
- **[SkipLane: Write-Once, Conflict-Free Accumulation for Zero-Skipping CNN Engines on FPGAs](http://arxiv.org/abs/2611.14495v1)**（`2611.14495`，cs.AR，11-10）：跳过零激活能省乘法，但压缩窗口内不同输出列的乘积必须合并；作者此前的引擎用 row buffer 的 read-modify-write 做合并，结果在 **87%** 稀疏度下仍比 dense 慢。SkipLane 把整条 accumulation path 改成 write-once：每个 kernel row 在自己的 lane 里处理，用三个寄存器保存部分和，每周期最多把三个完成的列 retire 到互不冲突的 memory bank；lane 与 input channel 只在 readout 时求和，任何部分和都不回读；零权重空出的乘法器在同一次 pass 里算第二个 filter。RTL 仿真中，引擎在 **70%** 稀疏度以内保持在前端 bound 的 **0.5%** 以内，50–80% 稀疏度下比 18-bank、理想 RMW 的 SCNN-style scatter accumulator 快 **1.10–1.26×**；PYNQ-Z2 上六个引擎跑到 140 MHz、占 21.1k LUT，LeNet-5 板级 bit-exact，conv1 比 dense 引擎少 **1.94×** 周期（端到端 1.26× 快），conv2（行宽仅 13 像素）则为 dense 速度的 0.87×。

### Attention 与 kernel 生成

- **[SinkState: On-the-Fly SVD of Recurrent States for Efficient Hybrid LLM Serving](http://arxiv.org/abs/2611.14246v1)**（`2611.14246`，cs.LG，11-10）：hybrid 模型用 linear attention 把历史压进固定大小的 recurrent state，但现有 serving 系统始终按 dense 形式保存，每个 decode step 全量读写，内存流量随 batch 增长并主导成本。作者发现 linear layer 的 query/key/value 集中在稳定低维子空间（linear attention sink），state 实际是低秩的、且秩随 state 变化；SinkState 据此在推理全程用自适应低秩形式存储每个 recurrent state，配合 buffered update 和只作用于 top subspace 的局部 SVD，其余组件异步放到 CPU 上整理。在 4B–120B 的四个 Mamba-2 / Gated DeltaNet hybrid 模型上，recurrent state 峰值内存最多降 **2.6×**（平均 1.9×），decode 比 Mamba-SSM 与 FLA 最多快 **1.58×**（平均 1.25×），精度不掉。
- **[AutoScaffold: AI-Guided Scaffolding for Reinforcement Learning of Language Models](http://arxiv.org/abs/2611.14515v1)**（`2611.14515`，cs.LG/cs.CL，11-10）：RLVR 的 outcome reward 把每次尝试压成一个标量，训练时加的 scaffold（策略、提示）怎么设计和调整基本靠人，且按领域定制。AutoScaffold 让一个 frozen teacher 周期性检查学生的训练经验、诊断瓶颈，生成两类 scaffold——跨任务共享的策略与单题的 hint——并随学生进步调整支持量，评测时全部移除。除 ALFWorld、检索式 QA、数学推理外，论文把它用于 GPU kernel generation：在移除 scaffold 的评测下，把「生成的 kernel 正确且跑得比 PyTorch 参考快」的任务比例最多提高 **5.0 个百分点**；在数学上以约 **1/3** 的学生训练步数超过 QuestA 报告的成绩。分析显示 teacher 在不同任务类别抓到不同失败模式，并把 scaffold 从分步 workflow 逐步收敛到针对性修补。
