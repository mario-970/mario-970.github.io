---
layout: post
title: "算子日报 · 2026-09-27"
permalink: /operator-daily/2026-09-27/
date: 2026-09-27 23:59:00 +0800
tags: [算子日报]
---

> 每日汇总：CUDA 开源仓 Release Notes（算子）、芯片动态、arXiv 算子内核论文。

## TL;DR

- **KernelOPT** 把「让 LLM 改 kernel」从单打独斗推进到**在编译产物内部动手**：保留 cuBLAS/cuDNN 调用，只改 PyTorch Inductor 生成的 Triton 子 kernel，用五个 profiling 驱动的 agent 搜候选、再用四道闸门（静态校验 / 多种子正确性 / 模型级 float64 回退 / 性能门）筛，全不过就退回编译器基线；250 道 KernelBench 上相对 `torch.compile` 几何平均提速 **1.40× / 1.15× / 1.07×**（Level 1/2/3）。
- **TileBench** 在 B200 上把 **Triton 与 cuTile** 拉到同一套算子语义下正面比：45 个算子、统一的 dtype/尺寸扫描、default 与 autotune 两档配置、roofline 指标加 profiling 诊断。结论是**没有全局赢家**——cuTile 只在少数对 Tensor Core / TMA 友好的 kernel 上占优，Triton 在大量不规则、streaming、带宽受限算子上更强；LLM 生成 kernel 时 Triton 的 token 效率也稳定更高。
- **KREX** 解决 kernel agent 的评测效率：靠 **region-granular exclusivity**（只在标记的计时敏感区内独占 GPU，区域外允许并发）让多个 benchmarking 命令共享 GPU，吞吐最高 **3.4×**，而 p95 计时膨胀仅 0.30% / 1.58% / 3.90%（kernel >10ms / >1ms / >0.1ms）。Release Notes 与芯片动态本期均无新条目。

## 一、CUDA 开源仓 Release Notes（算子）

> 本期无新条目。窗口内（≥ 09-26）只有 cudnn-frontend 的 nightly dev 构建（`v1.31.0.dev*`），非正式 release；其余正式发布（CUTLASS v4.8.0、TensorRT v11.3、NCCL v2.32.3-1 / nccl4py v0.6.0、cuda-python v13.4.3 / v12.9.9 / cuda-core v1.2.1、CCCL python-1.2.0、cuda-quantum 0.16.0）均已在 09-21 ~ 09-26 各期收录。

## 二、芯片动态

> 本期无新条目。NVIDIA Developer Blog 与 NVIDIA Blog 两个源在 7 天窗口内无命中关键词的新文章。

## 三、arXiv 论文（算子内核）

- **[KernelOPT: Dispatch-Aware Agentic Search for GPU Kernel Optimization](https://arxiv.org/abs/2609.30059)**（`2609.30059`，09-24）：现有 LLM kernel 优化器把编译好的模型当黑盒，逐个优化独立 kernel，既不管编译器已经做出的结构决策，也不在模型端验证，因此和 `torch.compile` 的差距补不干净。KernelOPT 改为把编译产物当结构化对象：**保留 cuBLAS/cuDNN 等 vendor 库调用，只针对 Inductor 生成的 Triton 子 kernel**下手，由五个 profiling 驱动的 agent 协同搜索，并用四道闸门逐级过滤——静态校验、多种子正确性、模型级 float64 回退校验、性能门；**若没有候选能全部过关，就保留编译器基线**，不会为了提速牺牲正确性。系统接受 PyTorch `nn.Module`、独立 Triton kernel 与 Helion kernel；在 250 道 KernelBench 上相对 `torch.compile` 几何平均提速 **1.40×（Level 1，51/100 题有提升）、1.15×（Level 2，31/100）、1.07×（Level 3，12/50）**。

- **[TileBench: A Controlled Benchmark for Performance Evaluation and Bottleneck Diagnosis of Tile-Based Programming Models](https://arxiv.org/abs/2609.29067)**（`2609.29067`，09-24）：Triton 与 cuTile 这类 tile 级编程模型降低了写高性能 kernel 的门槛，但两者的实际性能、调参行为和易用性一直缺少可比的系统评测——现有对比往往算子语义不一致、实现结构不对等。TileBench 在 **NVIDIA B200** 上用统一语义与相当实现结构评测两者，含 **45 个算子**，覆盖多种 AI kernel 形态与访存/计算特征；每个任务都配 PyTorch 参考、经核验的 Triton 与 cuTile 实现、dtype 与输入尺寸扫描、default 与 autotuned 两档配置、roofline 指标与 profiling 诊断。评测显示性能差距**依 workload 而定**：cuTile 只在少数对 **Tensor Core / TMA 友好**的 kernel 上领先，Triton 则在一大批**不规则、streaming、带宽受限**算子上更强；对 LLM 生成的 kernel，在同样的迭代精炼协议下 **Triton 的 token 效率稳定高于 cuTile**。项目已开源于 https://github.com/Deep-Learning-Profiling-Tools/Tilebench。

- **[KREX: Concurrent Kernel Benchmarking on Shared GPUs via Region-Granular Exclusivity](https://arxiv.org/abs/2609.30057)**（`2609.30057`，09-24）：LLM agent 做 kernel 优化靠的是反复构造候选并在真机上计时，现有系统为保证计时保真度，会把整块 GPU 在一次 agent session 或一条 benchmarking 命令期间独占——但真正需要独占的只是命令里极短的一段计时关键区，其余时间 GPU 白白闲置。KREX 让 agent **显式标出计时敏感区域**，运行时只在区域内强制独占、区域外允许并发：区域内通过**阻断新的竞争性 GPU 提交并把在途工作排空**，再冻结同级进程、隔离 CPU 核，从而同时保护 GPU 执行与驱动计时的 host 线程；区域外则把 GPU context 常驻在持久进程中复用，避开节点级串行的 context 创建开销。在 NVIDIA 与 AMD GPU 上，相比命令粒度的独占基线，KREX 把 benchmarking 吞吐提到最高 **3.4×**，而 p95 计时膨胀仅 **0.30% / 1.58% / 3.90%**（kernel 长于 10ms / 1ms / 0.1ms）。

- **[Compiler and Hardware Co-Design for Accelerator Architectures](https://arxiv.org/abs/2609.30099)**（`2609.30099`，09-24）：异构加速器是算力密集型负载的高效路径，但全栈打通成本高。EAAC（Extensible Accelerator Architecture）给出一套**编译器 + 硬件架构协同设计框架**，目标是把加速器快速原型开发的接入成本压下来：基于 **MLIR** 以便对接各类前端，只提供数据编排所必需的最小编译器能力，从而在不预设架构倾向的前提下把一块简单加速单元跑起来。作者用**脉动阵列 + 内嵌 RISC-V 核的 GEMM 加速器原型**端到端跑通 EAAC 的 MLIR 流水线：在一个合成全连接层负载上相对纯 RISC-V 基线取得 **28× 执行时间加速**，其中 **GEMM 本身只占总执行时间的 1.64%**（瓶颈已在访存与编排而非乘加）。文中还刻画了编译器侧的**硬件信号量分配**行为——最坏情况下所需信号量数随指令数线性增长，并把它点名为后续优化目标。
