---
layout: post
title: "算子日报 · 2026-09-25"
permalink: /operator-daily/2026-09-25/
date: 2026-09-25 23:59:00 +0800
tags: [算子日报]
---

> 每日汇总：CUDA 开源仓 Release Notes（算子）、芯片动态、arXiv 算子内核论文。

## TL;DR

- **KernelOPT** 把 `torch.compile` 产物当结构化对象做 agentic 优化：保留 cuBLAS/cuDNN 调用、只改 Inductor 生成的 Triton 子 kernel，并用四道验证关卡保证端到端不回退，KernelBench 上相对 `torch.compile` 几何平均最高 **1.40×**。
- 配套的测量侧：**KREX** 用「计时敏感区域才算独占」把多 agent 共享 GPU 做 kernel 搜索的吞吐提到最高 **3.4×**（>10ms kernel 的 p95 计时代价仅 0.30%）；**TileBench** 在 B200 上给出 Triton vs cuTile 的负载相关结论。
- 另收录 tile 结构自适应 Cholesky **sTiles**、MLIR 软硬件协同的 GEMM 加速器 **EAAC**、looped Transformer 免训练推理加速 **FlashLoop**。Release Notes 与芯片动态本期均无新条目。

## 一、CUDA 开源仓 Release Notes（算子）

> 本期无新条目。截至 09-24 的正式 release（CUTLASS 4.8.0、cuDNN Frontend 1.30.0、nccl4py 0.6.0、TensorRT 11.3、cuda-python 13.4.3）已在上一期收录；窗口内其余更新仅 cudnn-frontend 的 nightly dev 构建（`v1.31.0.dev*`），非正式 release，不做收录。

## 二、芯片动态

> 本期无新条目。NVIDIA Developer Blog 与 NVIDIA Blog 两个源在 7 天窗口内无命中关键词的新文章。

## 三、arXiv 论文（算子内核）

- **[KernelOPT: Dispatch-Aware Agentic Search for GPU Kernel Optimization](https://arxiv.org/abs/2609.30059)**（`2609.30059`，09-24）：现有 LLM kernel 优化器把编译后的模型当黑盒，只针对孤立 kernel 优化，既不管编译器的结构决策，也不做端到端验证。KernelOPT 改为**把编译产物当结构化对象**：保留 cuBLAS/cuDNN 等厂商库调用，只改写 Inductor 生成的 Triton 子 kernel，由五个 profiling 驱动的 agent 协同搜索，再用「静态校验 → 多种子正确性 → 模型级 float64 回退验证 → 性能门控」四道关卡过滤候选，全不通过则原样保留编译器基线。支持 PyTorch `nn.Module`、独立 Triton kernel 与 Helion kernel；KernelBench 250 题上相对 `torch.compile` 的几何平均加速比为 Level 1 **1.40×**（51/100）、Level 2 **1.15×**（31/100）、Level 3 **1.07×**（12/50）。

- **[KREX: Concurrent Kernel Benchmarking on Shared GPUs via Region-Granular Exclusivity](https://arxiv.org/abs/2609.30057)**（`2609.30057`，09-24）：agent 自动化 kernel 优化要反复编译候选并实测耗时，现有系统为保计时保真度把整段会话独占一张 GPU，而真正需要独占总量的只是命令里的一小段。KREX 让 agent 在 benchmark 命令中标记**计时敏感区域**，区域内强制独占（阻断新提交、排空在途工作、冻结兄弟进程并隔离 CPU 核，同时保住驱动测量的 host 线程），区域外放开并发；并复用常驻 context 进程，避免反复创建 context 带来的节点级串行化。NVIDIA 与 AMD GPU 上相对命令级独占基线最高 **3.4×** 吞吐，p95 计时膨胀 0.30%（>10 ms kernel）、1.58%（1 ms）、3.90%（0.1 ms）。

- **[TileBench: A Controlled Benchmark for Performance Evaluation and Bottleneck Diagnosis of Tile-Based Programming Models](https://arxiv.org/abs/2609.29067)**（`2609.29067`，09-24）：tile-based 编程模型（Triton、cuTile）的性能与调优行为一直缺少可控对比。TileBench 在 B200 上以**匹配的算子语义和可比的实现结构**评估两者，含 45 个算子、PyTorch 参考实现、dtype/输入尺寸 sweep、默认与 autotune 两套配置、roofline 指标与 profiling 诊断。结论是性能差距强依赖负载：cuTile 在一小簇 Tensor Core / TMA 友好 kernel 上占优，Triton 在大量不规则、流式与带宽受限算子上更强；同一迭代精修协议下，LLM 生成的 **Triton kernel token 效率稳定优于 cuTile**。基准已开源。

- **[Dense Matrices Are Alike; Sparse Matrices Are Sparse in Their Own Way: A Structure-Adaptive Tile Cholesky Factorization](https://arxiv.org/abs/2609.29765)**（`2609.29765`，09-24）：稀疏直接 Cholesky 求解器通常给整个矩阵钉死一种数据结构，但对称正定系统的稀疏度从近稠密到极不规则都有，甚至同一矩阵内混合。sTiles 在数值计算前用轻量 selector 读 Cholesky factor 的稀疏模式，把矩阵**乃至矩阵内的每个 tile** 路由到 dense / semisparse / sparse 三种 regime，共用一套静态共享内存调度；新引入的 **active-column tile** 只保留分解会碰到的列，使带状、arrowhead 结构也能拿到 BLAS-3 效率。60 个 SPD 矩阵上，其总分解时间在每种 regime 都低于 MUMPS、PaStiX、CHOLMOD、symPACK 与 Intel oneMKL PARDISO，总和比最优固定结构快 1.6–2.6×，代价是一次性分析更贵，从同一模式的第三次分解起开始反超——正对应 INLA 这类「一个稀疏模式、上千次数值」的场景。GPU 侧 dense 路径在单张 A100 上比更快的 CPU 节点还快 1.2–6.3×，优势随 factor 增大而扩大。

- **[Compiler and Hardware Co-Design for Accelerator Architectures](https://arxiv.org/abs/2609.30099)**（`2609.30099`，09-24）：把加速器从编译器到硬件一次打通仍然很重。EAAC 给出一套可扩展的编译器 + 硬件架构，基于 **MLIR** 便于对接各类前端，只提供数据编排所必需的最小编译器能力，让一个硬件加速单元从零跑到能用的门槛尽量低，且不过早替使用者做设计决定；目标负载是访存模式可预测的静态数据流。作者用「脉动阵列 + 嵌入式 RISC-V」的 GEMM 加速器原型端到端跑通 EAAC 流程：合成的全连接层负载上相对纯 RISC-V **28×** 加速，其中 GEMM 运算本身只占 1.64% 总执行时间（瓶颈已在数据编排侧）。文中还刻画了编译器的硬件信号量分配，最坏情况下所需信号量随指令数线性增长，并把它列为后续优化目标。

- **[FlashLoop: Fast and Memory-Efficient Looped Transformers via Lazy Updates](https://arxiv.org/abs/2609.29812)**（`2609.29812`，09-24）：Looped Transformer 靠反复过同一个 block 换深度，但每多一圈就多一遍前向、多缓存一份 KV，循环深度一大、上下文一长，推理 FLOPs 与 KV cache 就线性上涨，参数效率兑现不成推理效率。论文指出这些开销大多是冗余的：随递推推进，状态变化越来越集中在少数 token；attention 输出差异由一小撮**稀疏且稳定**的 key 列主导；相邻圈之间的 KV 残差越来越适合低位量化。FlashLoop 据此做**免训练**推理框架，组合 token 稀疏更新、稀疏 attention 与 KV 残差量化，在多个 Looped Transformer 上保持精度无损，端到端最高 **1.64×** 加速、KV cache 最高 **6×** 压缩。
