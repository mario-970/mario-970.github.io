---
layout: post
title: "算子日报 · 2026-10-16"
permalink: /operator-daily/2026-10-16/
date: 2026-10-16 23:59:00 +0800
tags: [算子日报]
---

> 每日汇总：CUDA 开源仓 Release Notes（算子）、芯片动态、arXiv 算子内核论文。

## TL;DR

- **本期增量 = 10-15 全天（40 条，arXiv 首次索引）+ 10-14 补录 17 条**。上一期的 150 条窗口被上限截断：10-14 共 72 条只见到 55 条，这 17 条任何一期都没见过（10-13 已无遗漏，73 条全部在往期覆盖范围内）。逐 id 与历史各期比对无重合，本期精选 16 篇。
- **编译器 / kernel 4 篇**：Little Dragon（8B 小模型生成 TVM 优化脚本，构造正确率 91.27%）、Cyclotron（用 recurrence 语言直接表达处理器间通信）、Rondo（DL 编译器的**有状态 bug** 测试，查出 35 个）、多阈值算子编译（推广到非单调函数与多维阈值）。
- **稀疏 / 加速器 4 篇**：AdaBool（布尔矩阵乘自适应编码，快一个数量级）、Helix（Krylov 求解器可编程加速器）、PSA（SNN 稀疏卷积，EDP 降 18.52×）、GSRS（视频 DiT 稀疏 attention 1.55–2.21×）。
- **量化 5 篇**：RoLaQ（MLA 2-bit KV）、FPoT（基-√2 免乘法器 4-bit）、ShiftQAT（定点加速器数值契约）、CP（查表 2-bit 重建）、以及**一条重要的负结果**——MoE 路由统计并非量化敏感度的可靠代理。
- **训练显存 2 篇**：LCT（无损张量压缩，Qwen3-4B 峰值显存 −23.8% 换来 3.5× 上下文）、BitSparse（ReLU 稀疏激活的无损压缩 Triton kernel）。
- **发布面本期最薄**：仅 **cuda-core v1.2.2**（纯补 Python 3.15 wheel，无 API 变化）；**芯片动态本期无新条目**（命中的仍是 10-12 已收录的 Vera Rubin）。

## 一、CUDA 开源仓 Release Notes（算子）

### cuda-python

- **cuda-core v1.2.2**（10-16）— [链接](https://github.com/NVIDIA/cuda-python/releases/tag/cuda-core-v1.2.2)：release note 只有一条——**新增 Python 3.15 wheel，含实验性 free-threaded 构建**；3.15 上要求 `cuda.bindings` ≥ 12.9.10（CUDA 12）或 ≥ 13.4.4（CUDA 13），Python 3.10 仍支持。**纯打包发布，无 API 或精度层面变化**，升级与否只取决于是否要用 Python 3.15 / free-threaded 解释器。

> 窗口（≥ 10-15）内其余更新均不计入：`cudnn-frontend` 两个自构建 nightly（`v1.32.0.dev74371060` 10-16、`v1.32.0.dev74082381` 10-15，正文自述 *Automated nightly build from NVIDIA-internal CI. Not a supported release.*）；以及 `cudnn-frontend v1.29.1-rc.74184706`（10-15）——**release candidate**，尚未 promote 到 PyPI，且其自述校验存在缺口（`test_sdpa_bwd_dsl_sm120.py` 的 develop 同步检查被豁免、SM90 验证因 GPU OOM 超时未完成），不值得在算子侧跟进。30 天窗口内的其他正式 release（CUTLASS v4.8.0、CCCL v3.5.0 / v3.4.3 / python-1.2.1、cuDNN Frontend v1.31.0、TensorRT v11.4 / v11.3、NCCL v2.32.3-1 与 nccl4py v0.6.0、cuda-python v13.4.4 / cuda-core v1.2.1 / cuda-pathfinder v1.8.3、cuda-quantum 0.16.0）均已在各期收录。

## 二、芯片动态

> 本期无新条目。NVIDIA Blog 在 RSS 窗口内确有 10-15 的新帖（RTX Spark、Nemotron 家族扩容、DLSS 4.5 on GeForce NOW 等），但均为消费/产品向，**标题不命中任何芯片关键词**；窗口内唯一命中仍是 10-12 已收录的 [NVIDIA 开放 Vera Rubin 架构](https://blogs.nvidia.com/blog/ocp-global-summit-vera-rubin-scale-ai-factories/)。Developer Blog 源侧继续返回 0 条。

## 三、arXiv 论文（算子内核）

> 本期为 **10-15 全天（40 条）+ 10-14 补录 17 条**：上一期的窗口被 `max_results=150` 截断，10-14 的 72 条里只见到 55 条，剩余 17 条在往期任何一期的抓取中都不存在；10-13 的 73 条则全部已被往期覆盖（上期见 59 条 + 其后新公布 14 条），故本期不再回捞。

### kernel 与编译器

- **[How to Train Your Little Dragon: Teaching a Small Language Model to Construct and Refine GPU Optimization Scripts](http://arxiv.org/abs/2610.20980v1)**（`2610.20980`，cs.LG/cs.AI，10-14）：GPU kernel 优化的难点在于编译器变换彼此耦合，需要搜索**保计算、提性能的变换组合**。本文让 8B 小模型**不生成 kernel 代码、而是生成优化脚本**：设计一种紧凑无损表示，确定性解码即可重建出完整编译 trace（保留变换参数与依赖）。基于 TVM MetaSchedule 在 54,116 个 workload 上造 108,594 条样本（60,236 条「从无优化 kernel 生成优化脚本」+ 48,358 条「给定已部分优化的脚本改得更好」），SFT 后加**整形奖励 RL——只对数值正确的 kernel 给延迟奖励**。每次尝试正确率 91.27%（构造）/ 92.01%（改进）；RL 后在 102 个构造任务中 88 个、170 个改进提示中 145 个优于 SFT，**8B 参数在正确率与 kernel 性能上超过六个千亿至万亿级前沿模型**。
- **[Cyclotron: A Recurrence Language for Expressing Interprocessor Communication](http://arxiv.org/abs/2610.20981v1)**（`2610.20981`，cs.DC，10-14）：把流式数据流算法写成 recurrence 方程、再编译成互联处理器拓扑上的分布式程序。关键设计是**数据访问不限于对外部内存的访问——数据可以「活在」iteration space 自身之上**，让内部点之间直接互通取值，于是 nearest-neighbor systolic 交换、one-to-all 广播、双向 split ring、非对称分段广播这些**标准分布式线代库不支持的通信模式**都能直接表达。用户只声明跨 iteration space 的数据依赖，再自选哪些维度跨处理器、哪些随时间展开（**无需重写整个程序**），编译器安全地按 space-time 映射放置通信与计算算子。在最多 16 CPU 节点 / 16 GPU 上跑分布式 matmul、TRSM、Cholesky 与融合 Cholesky-TRSM，对标 ScaLAPACK / SLATE / cuSolverMp 有竞争力且常更快；Cholesky 的最佳通信策略**随问题规模与架构而变**。
- **[Stateful Testing for Deep Learning Compilers](http://arxiv.org/abs/2610.22003v1)**（`2610.22003`，cs.SE/cs.LG，10-15）：现代 DL 编译器惰性编译并跨调用缓存产物，**残留的编译器状态会让一次「单独看正确」的调用出错**；对 PyTorch / JAX / TensorFlow 中 26 个公开 issue 的分析表明，检出这类 bug 需要「先建立状态、再暴露失败」的序列。Rondo 把测试用例表示为**模型 + 调用序列**，用 MCMC 采样器按编译行为引导变异以偏向罕见行为，并用三路差分 oracle 对比「不编译 / 重置状态编译 / 保留状态编译」；确认后再保持输入与程序状态、只在状态保留/重置间复现，并缩短序列。四个框架 24 个常见模型上查出 **117 个 bug（含 35 个有状态 bug）**，提交 39 份报告、开发者确认 35 份、修复 18 个（含 11 个有状态 bug）；告警精度 99.26%，并把 44 条序列平均缩短 83.34%。
- **[Compiling Multi-Threshold Operators for Quantized Non-Monotonic and Fine-Granular Activations in Custom-Tailored DNN Accelerators](http://arxiv.org/abs/2610.21825v1)**（`2610.21825`，cs.AR/cs.LG，10-15）：低比特网络的激活函数常实现为**多阈值（multi-threshold）算子**以利于硬件，但现有实现只支持**单调递增函数、一维（per-channel）粒度**与少量预定义「激活-量化器」组合。本文把多阈值表示形式化推广到**非单调函数与多维阈值的细粒度参数**，并给出代数规则做自动转换、集成进基于 ONNX 的编译器基础设施；在 FINN 框架内扩展二分搜索多阈值算子做端到端综合验证，刻画了支持非单调性/细粒度的硬件代价，并与图级分解方案和 hls4ml 对比。展望部分演示了把多阈值表示用于训练（一种 QAT 形式）。

### 稀疏与专用加速器

- **[AdaBool: Sparse Boolean Matrix Processing with Adaptive Data Representation](http://arxiv.org/abs/2610.21264v1)**（`2610.21264`，cs.DB/cs.DC，10-15）：布尔语义下矩阵不必存数值、只需编码非零位置，因而**天然适合压缩以降低内存访问开销**；难点在于最优压缩方式强烈依赖稀疏度分布，而真实矩阵各区域的稀疏度差异很大，单一整体表示会浪费大量优化空间。AdaBool 把矩阵切分成细粒度元素块，**每块用最匹配其局部稀疏度的编码**，再配一个异构 GPU 执行模型：按编码类型分组、映射到匹配的处理路径以高效调度。端到端实现并加入低层系统优化，在生产与公开负载上**比代表性基线快一个数量级以上**。
- **[Helix: A Unified Programmable Accelerator for Diverse Krylov Subspace Solvers](http://arxiv.org/abs/2610.21096v1)**（`2610.21096`，cs.AR/cs.DC，10-15）：Krylov 子空间求解器（迭代解稀疏线性系统）的既有 FPGA 加速器通常只支持单一求解器、或每种求解器各配一套硬件。Helix 先抽象出多个求解器的**公共计算原语**、围绕高效可编程原语 kernel 建架构，再用可编程数据通路协调原语，从而**解耦硬件实现与求解器专有执行**，实现 HBM 复用与**零求解器切换延迟**；另提供 DSL 编译器把不同求解器映射到架构上并优化向量算子性能。Xilinx Alveo U55C 上编译执行 8 个求解器：相对只支持 JPCG 的 XcgSolver 与 SOTA FPGA 加速器 Callipepla 分别最高 **7.91× / 1.16×**；8 个求解器上达到 RTX PRO 6000 吞吐的 47.5%–96.9%，能效高 2.29–4.66×，单位峰值内存带宽吞吐高 1.89–3.85×；**可编程带来的开销仅 0.25%–0.57% 执行时间**。
- **[PSA: A Pruning-Friendly Sparse-Aware Accelerator for Efficient Sparse Convolution in SNNs](http://arxiv.org/abs/2610.21313v1)**（`2610.21313`，cs.AR，10-15）：脉冲神经网络兼具脉冲稀疏与剪枝稀疏，但两者叠加产生**高度不规则的负载**，难以高效执行，既有方案无法同时支持稀疏输入执行、低开销稀疏卷积 lowering 与剪枝友好执行。PSA 用 **SAMA（Sparse-Aware Merge-Accumulation）数据流**对齐稀疏脉冲索引、把不规则激活转成连续计算流；再用 **SIG-im2col** 卷积 lowering 以低映射代价与内存开销把稀疏卷积在片上转成 SpMM；另配轻量**剪枝友好门控**直接支持剪枝网络。ResNet-18/CIFAR-10 上能效较稠密基线 **4.45×**、EDP 降 **18.52×**。
- **[Sparse Attention for Video Generation Acceleration via Growing Sparsity and Reduced Search](http://arxiv.org/abs/2610.21129v1)**（`2610.21129`，cs.CV，10-15）：视频生成的 DiT 被自注意力二次复杂度拖累；已有稀疏 attention 方法全程用固定稀疏率、且难以准确判断该跳过哪些计算。作者从 attention 熵与平均 attention 距离观察到**去噪过程中 attention 越来越集中**，且**早期选中的 block 在后续步仍占大部分权重**。据此提出免训练框架 **GSRS**：稀疏率随去噪**递增**（早期多算、后期少算），并把后续搜索限制在早期候选集内以降低搜索开销；用 **Top-p** 做自适应稀疏控制，配套一套在 FlashAttention 计算内**精确求注意力权重**、开销可忽略的评估方案。HunyuanVideo / Wan 2.1 上加速 **1.55–2.21×**，生成质量与全注意力相当。

### 量化与低精度

- **[RoLaQ: Joint Rotation and Range Calibration for 2-Bit KV Caches in Multi-Head Latent Attention](http://arxiv.org/abs/2610.20686v1)**（`2610.20686`，cs.LG/stat.ML，10-14）：MLA 把 KV 内容压成每 token 一个共享 latent，低比特存储很有吸引力，但 **2-bit 下误差会同时扰动 attention 匹配与 value 读出**，且影响取决于误差方向与重建尺度。RoLaQ 是**离线联合旋转与范围校准**方法：在冻结的 key/value 投影所定义的子空间内精修共享旋转，并**联合学习量化范围、把阈值与重建增益耦合**起来控制共享信号的响应；校准由预测匹配与 attention 输出监督引导，权重全程冻结。三个 MLA 模型六个基准上，2-bit 方法中平均精度最高（用与不用 token 保护都成立）；DeepSeek-V2-Lite 上 SGLang 实现给出**同显存预算下 3.96× 的 BF16 KV 容量、256K 输入时 2.51× 的 decode 吞吐**。
- **[Task-Phase-Conditioned Expert Routing Is Not a Reliable Proxy for Quantization Sensitivity in MoE FFN Tensors](http://arxiv.org/abs/2610.20826v1)**（`2610.20826`，cs.LG，10-14）：**一条对实践直接有用的负结果**。作者用 Qwen-3.6-35B-A3B 在 `ik_llama.cpp` 里搭路由捕获与 profiling 流水线，从路由熵/决策度、变异系数、Jaccard/阶段脆弱性、top-1 专家强度合成敏感度分数，再用坐标下降在固定平均 bpw 预算下调层级分配。结论是**重分配跑不赢同档的均匀基线**：4.0 bpw 平坦档统计中性（z≈0.9）；2.7 bpw 陡峭档下**随机重分配与路由引导重分配相差不到 1 SE**（该档的结构性损失来自强制降位与 KLD 尾部注入），而激进预算条件出现 z>4 的确定损失并在三个独立权重向量上复现。作者的判断是：**路由行为统计是「逐层量化代价」的一个未经校验的代理**——MoE 混合精度不该按路由统计来分位宽。
- **[Fractional Power-of-Two Quantization for Efficient and Effective Multiplier-Free LLM Inference](http://arxiv.org/abs/2610.21257v1)**（`2610.21257`，cs.LG/cs.AI，10-15）：线性投影里的乘累加主导算力与能耗，PoT 量化能用移位-加实现**免乘法器**推理，但基-2 网格过粗、低位宽掉点严重。**FPoT 改用基 √2（半八度）的分数 PoT 网格**做 4-bit 权重，同时保持**纯整数移位-加数据通路**：权重走 √2 网格、激活走均匀整数；标定时做网格对齐精修并吸收进静态 per-channel scale；再用一个 **1-adder 的 √2 近似**、误差在权重校准阶段经「近似网格量化」吸收，使 PE 门数显著少于 4-bit 整数乘法器。Llama-3-8B 上 W4A4 相对 PoT 基线零样本常识推理**提升 8.72%**，与均匀整数基线仅差 **0.4 个百分点**；七个模型三个位宽均成立，且可与现有 SOTA PTQ 方法组合。
- **[ShiftQAT: Quantization-Aware Training for the Numerical Contract of a Fixed-Function Edge CNN Accelerator](http://arxiv.org/abs/2610.21695v1)**（`2610.21695`，cs.LG，10-15）：MAX7800x 这类**固定功能 CNN 加速器只执行满足精确数值契约的网络**——幂次二、per-tensor、对称 scale，8-bit 激活与 bias，纯整数算术，**每层一个静态配置的移位**。ShiftQAT 是该家族开源工具链 `ai8x-training` 的默认 QAT 路径，**按构造满足契约**：解决 residual、concatenation 与权值共享的依赖，使激活 scale 静态传播、每层重量化归约为一个离线配置的移位；权重与 bias 在联合幂次二网格上训练以适配 8-bit bias 预算。对比方法中**只有 ShiftQAT 满足契约的全部约束**（含 signed 8-bit 激活存储），即便如此 ImageNet 上仍能匹配约束更松的 QAT 方法的 W8A8 掉点、W4A8 距 FP32 在 2.9 个百分点内；MAX78002 芯片上 ShiftQAT 训练的 MobileNetV2 / ResNet-18 单次推理 **14.6 ms / 8.4 ms**，耗能 5.1 mJ / 2.8 mJ。
- **[N-Gram Memories at Two Bits: Quantizing Lookup Tables in Large Language Models](http://arxiv.org/abs/2610.21810v1)**（`2610.21810`，cs.LG/cs.CL，10-15）：条件记忆靠选择性查表扩容量，但**整张 embedding 表必须对任意输入随时可用**。CP（Conditional Program）是只重建「被请求的那些行」的压缩表示：每行用**短码 + 局部 scale**，重建程序**全表共享**，用 Gaussian trellis、范数恢复与可独立解码的行记录落地。2-bit 码相对 2-bit round-to-nearest **把输出 KL 降 41–63%**；三个已训练的 n-gram memory 家族上，同等存储预算下共享 trellis 都优于拟合 codebook 与低秩残差变体；DeepSeek 上把表存储**从 203 GB 降到 51 GB**，留给更大地表常驻 GPU 的余地。

### 训练显存与压缩

- **[LCT: Losslessly Compressed Tensors with GPU friendly execution for Training](http://arxiv.org/abs/2610.22179v1)**（`2610.22179`，cs.LG，10-15）：无损张量压缩能省训练显存，但训练要**反复重编码**，编解码开销是成败关键。LCT 面向这个反复编解码场景做 GPU 执行框架，利用浮点数据的天然冗余压缩权重、激活与优化器状态，用**预计算码本的 Huffman 编码**加高度优化的 GPU kernel 压低编解码开销，并分析了存储代价与对幅值漂移的适应性。压缩比最高 **1.455×**；162M 的 NanoGPT 上显存降 **24.6%** 而单步时间仅 **+7.2%**；Qwen3-4B 微调峰值显存最多降 **23.8%**，同一张卡上**上下文长度可扩 3.5×**。
- **[BitSparse: Compact Activation Storage for ReLU Feed-Forward Networks](http://arxiv.org/abs/2610.22256v1)**（`2610.22256`，cs.LG，10-15）：激活检查点靠重算省内存、有损压缩会改变存下来的激活；而 **ReLU / ReLU² 稀疏激活的回潮**带来了第三条路——**无损压缩激活缓存本身**。BitSparse 是 GPU 常驻的无损稀疏激活压缩格式，压缩与重建跑在**融合进 autograd function 的自定义 Triton kernel** 里，前向存得少、反向精确还原；支持 BF16 与 FP8，可选符号位打包可进一步逼近理论上限（文中与理论最优做了对比分析）。Modded-NanoGPT / Nemotron-H 上**每 token 显存降 31% / 30%**，实用序列长度下开销不超过 1%；另配新的预分配缓冲系统，避免变长张量分配时的主机同步。

### 推理系统与 KV 布局

- **[When the KV Cache Is Slower Than the GPU: Scaling Remote KV Tiers for Context-Parallel Prefill](http://arxiv.org/abs/2610.21500v1)**（`2610.21500`，cs.DC/cs.LG，10-15）：在 8×B300、GLM-5.3（FP8 MoE + MLA）、prefill 上下文并行 PCP8 的分离式部署上研究**远程 KV tier**。打开远程层（LMCache + Valkey）后中位 TTFT 不变，但 **16k token 的 p99 从 3.96 s 涨到 17.3 s**；在隔离引擎上回放生产请求流发现**远程命中 0.88 s、而重算只需 0.31 s**，同步加载把命中为主的流变成不稳定队列（p99 83.32 s 而 GPU 仅 7% 忙）。一个两段式成本模型同时解释了这两点：命中只有在 tier 带宽超过 **B\* = sR**（该模型与 GPU 上为 2.53 GB/s）时才划算，而**固定的每次命中引擎开销 T0≈0.33 s 意味着约 17k token 以下的提示在任何带宽下都赢不过重算**。据此做了四件事：**把 MLA KV store 跨 PCP rank 分片**（并证明现有的 min-over-ranks 查找返回最长缓存前缀、给出永不失同步的集合通信加载协议）、**按 rank 分区远程 tier**（GET 带宽 6.3× 到 10.76 GB/s）、为分片存储加异步加载、修掉三个造成无界停顿的客户端 bug。命中为主的流上 p99 从 83.32 s 降到 **0.67 s**；分片存储把 prefill 吞吐提升 14%，加上一处上游 kernel 修复共 34%。全部改动已提交上游 LMCache。
