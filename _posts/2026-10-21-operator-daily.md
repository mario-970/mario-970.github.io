---
layout: post
title: "算子日报 · 2026-10-21"
permalink: /operator-daily/2026-10-21/
date: 2026-10-21 23:59:00 +0800
tags: [算子日报]
---

> 每日汇总：CUDA 开源仓 Release Notes（算子）、芯片动态、arXiv 算子内核论文。

## TL;DR

- **CUTLASS 4.8.1** 修复 `tensormap_replace` 在 TMA descriptor 被缩小后（grouped GEMM 等）的设备侧更新安全问题，避免 Blackwell/Rubin 上的非法访存；同时把 Rubin（sm107a/sm107f）上 packed mixed-input 与 packed FP8 减法的错误代码生成列为已知问题，相关 API 需避开或等 CTK 上游修复。
- **稀疏 attention 继续向 kernel 要算力**：KPage 在加载时把 16 行 K/V page 跨逻辑块打包，对 NVIDIA Block-Sparse Attention 前向 kernel 几何平均 **1.77×**；MLIPS 在 vLLM 中跨层共享 indexer 候选池，三个 consumer 层共享一个 producer 时 **1.14–1.40×**、GPU 能耗降 **14.8–27.4%**。
- **GEMM 与低精度**：Intel Arc Xe2 上用 INT8 引擎仿真 DGEMM/ZGEMM，DGEMM 最高 **21.1 TFLOP/s**（同板原生 FP64 的 12–28×）；3×3 矩阵乘的加法数降到 **55**（23 次乘法，Lean 4 验证）；跨 draft/target 的联合编译把两类模型的 block 交错进同一 tile，B300/RTX 5090 上 **1.37×**、交错 kernel 下 **1.45×**。

## 一、CUDA 开源仓 Release Notes（算子）

### CUTLASS

- **v4.8.1** — [链接](https://github.com/NVIDIA/cutlass/releases/tag/v4.8.1)（10-21）：`tensormap_replace` 现在保证 TMA descriptor 被缩小到原始 extent 以下时设备侧更新仍然安全，修复 grouped GEMM 等场景在 Blackwell/Rubin 上的潜在非法访存（可用 `keep_oob_fetch_window=True` 退回旧行为）；另修复 `cute.extern` 零参数函数生成非法类型签名导致二次编译失败、`CopyUniversalOp` 缺失内存属性错误/警告，以及无 Torch 环境下误改 `sys.path` 破坏 JAX 等 import 的问题。**已知问题**：CTK 缺陷会使部分 packed 指令生成非法代码——不要用 inline PTX 的 `sub.rn.e4m3x4.e4m3x4` / `sub.rn.e5m2x4.e5m2x4` 做高吞吐 FP8 减法；Rubin（sm107a/sm107f）上 bf16x2/fp16x2 结果、f32x2 输入且后续转 fp32 的 `mul/sub/add_packed_*_f32x2_f32x2`，以及首操作数为编译期常量的混合 `mul_packed_*` 会编译出错误结果。绑定测试覆盖 FlashAttention、FlashInfer、Quack、cuDNN-Frontend、PyTorch、TensorRT-LLM 的指定 commit。

> 同期 `cudnn-frontend` 只有 `v1.32.0.dev75554020`（10-21，24/24 wheels）这个 CI 自构建 nightly，正文自述 *Not a supported release*，与往期处理一致不计入。

## 二、芯片动态

- **[Google Cloud and NVIDIA Announce General Availability of NVIDIA Vera Rubin Infrastructure](https://blogs.nvidia.com/blog/google-cloud-vera-rubin-infrastructure/)**（10-21）：Google Cloud 将于下月 GA A5X 实例，组合 Vera Rubin NVL72、ConnectX-9 SuperNIC、Virgo 网络与第四代液冷，面向训练、推理与大规模 RL，之后扩展到欧洲。对算子侧的含义是 Rubin 平台从纸面进入云端可测阶段，FP4/FP8 blockscaled、TMA 与 TMEM 相关 kernel 将拿到真实部署面。

## 三、arXiv 论文（算子内核）

> 本期抓取窗口覆盖 10-18 至 10-20 共 **128 条**（28 / 58 / 42），未触及 `max_results=150` 上限，10-21 提交的论文尚未被 arXiv 索引；逐 id 与往期各期比对后，从未收录条目中精选 5 篇。

### 稀疏 attention 与长上下文

- **[KPage: Cross-Block Page Compaction for Hardware-Efficient Sparse Attention](http://arxiv.org/abs/2610.28096v1)**（`2610.28096`，cs.DC，10-20）：长序列视频 Transformer 的 3D token 布局下，被选中的块常常只填满部分物理 tile，浪费张量核容量。KPage 在加载阶段把有效的 **16 行 K/V page 跨逻辑块边界打包**进大 tile，不在 HBM 中物化压缩张量；Q 侧在 64×256 与 128×128 tile 间路由以减少 padding。精确掩码与 K/V 归属的 page 打包保证前向/反向的稀疏图不变。相对 NVIDIA Block-Sparse Attention（BSA）前向 kernel 加速 **1.35–2.06×**（几何平均 **1.77×**），部分占用块为主的 3D 布局上几何平均 **2.29×**；物理 QK 对容量从 334.1G 压到 177.5G，仍保留 160.6G 有效交互；对 SSTA/STA/VSA 的 8 种布局，前向 1.13–4.25×、前向+反向 1.02–2.71×。作者将开源并向 cuDNN Frontend 提 PR。
- **[MLIPS: Multi-Layer Indexer Pool Sharing](http://arxiv.org/abs/2610.27438v1)**（`2610.27438`，cs.LG/cs.AI/cs.PF，10-20）：稀疏 attention 限制了每个 query 读取的 key，但**找 key** 仍需逐层全前缀扫描。MLIPS 让 producer 层发布一个大于 attention 预算 `K` 的 top-`M` 候选池，后续 consumer 层从池中各自选 top-`K`，保留自己的打分与选择，训练-free；在 vLLM 中为 DeepSeek-V4-Flash 实现层间共享、decode 持久复用与粗粒度 producer。每 producer 带三个 consumer 时，四个长上下文套件上 **1.14–1.40×** 加速、GPU 能耗降 **14.8–27.4%**，单次精度变化 **−1.49 至 +0.29** 个百分点；共享范围继续扩大则出现任务相关掉点。对算子侧的提示是 indexer 的访存可以被组织成跨层共享的生产者-消费者流水线。

### GEMM、低精度与编译

- **[Emulating DGEMM and ZGEMM with INT8 Matrix Engines on Intel Arc GPUs: Ozaki Scheme II on the Xe2 Architecture](http://arxiv.org/abs/2610.26945v1)**（`2610.26945`，cs.MS/cs.DC/math.NA，10-18）：Intel GPU 上首个 Ozaki scheme II 实现：把 FP64 矩阵乘分解为若干**精确 INT8 乘**后按中国剩余定理重构，SYCL + oneMKL、完整 BLAS `dgemm`/`zgemm` 接口。为在 Xe2（Battlemage）上跑快，所有 memory-bound kernel 不碰 FP64、residue 计算分支消除、CRT 商用带证明安全裕度的 FP32、重构用 16-bit limb，并绕开厂商 INT8 GEMM 的性能悬崖。Arc Pro B60 上 DGEMM 在 `s=8/12/16`、`N=8192` 时达 17.3/12.1/9.3 TFLOP/s，最高 **21.1 TFLOP/s**，为同板原生 FP64 DGEMM 的 **12–28×**；ZGEMM 最高 **27.4 TFLOP/s**，`N≥4096`、`s=16` 时为原生 ZGEMM 的 13.7–17.8×。`s≥16` 且缩放良好时结果正确舍入，宽指数范围可用 `s=20–24` 的扩展 residue 路径。
- **[A 55-Addition Rank-23 Scheme for 3×3 Matrix Multiplication via Exact Two-Sided Minimization](http://arxiv.org/abs/2610.27451v1)**（`2610.27451`，cs.MS/cs.CC/cs.DS，10-20）：给出任意环上 **23 次乘法、55 次加法**的 3×3 矩阵乘方案，比 Sun（2026）的 56 次少一次，也是不换基的精确非交换 3×3 首个低于 56 的结果：13 个 A 的有符号组合 + 14 个 B 组合、23 次乘积、28 次重构加法，正确性经 Lean 4 机器验证。方法上输入侧用 chain covering 精确最小化、输出侧借转置原理精确求解；并在 Heule–Kauers–Seidl 数据库上证明该记录类唯一达到 55。对算子侧是算法层线索（尚无 GPU 实现），但小尺寸 batched matmul / 微 kernel 的加法数与寄存器压力可直接受益。
- **[Conjugate Compilation: Self-Improving Agentic Co-Compilation for Speculative Decoding](http://arxiv.org/abs/2610.25012v1)**（`2610.25012`，cs.LG/cs.DC/cs.PL，10-18）：把投机解码的 draft 与 target 两模型放进**同一个编译边界**联合优化，而不是分别编译。关键手法是交错 operand layout——把两模型的 block 交错进公共 tile，使 GEMM 与非 GEMM 工作共处且保持数学等价；再用 LLM 驱动的 agentic 搜索，以编译器和 GPU 实测为环境、MCTS 累积奖励来修改优化计划。6 个单卡配置（NVIDIA B300、RTX 5090）上每轮几何平均 **1.37×**，交错 kernel 提升到 **1.45×**、最好 **1.59×**；GPU 利用率平均提升 RTX 5090 **1.46×**、B300 **1.50×**。

> 未收录条目多为应用/训练方法向（医疗影像、EEG、自动驾驶、3D 高斯、检索与 RL 等）或与 GPU 算子无关的理论工作，与算子内核无直接关系。
