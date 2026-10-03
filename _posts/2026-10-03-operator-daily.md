---
layout: post
title: "算子日报 · 2026-10-03"
permalink: /operator-daily/2026-10-03/
date: 2026-10-03 23:59:00 +0800
tags: [算子日报]
---

> 每日汇总：CUDA 开源仓 Release Notes（算子）、芯片动态、arXiv 算子内核论文。

## TL;DR

- **超低位宽这一侧，今天的两个新点都落在「重构通路的成本」上**：**XOR-Trellis** 把 trellis 量化的反量化器换成**结构化的 state→value 映射**（对硬件友好、易并行，同时保留 trellis 搜索所需的重构多样性），并用**曲率感知目标**在**原坐标系**里做路径优化，从而**不再依赖 Hadamard 类 incoherence 变换**；**LSP** 让低秩方案在训练后**合并回标准低秩因子**（绑定层组只共享一个因子），于是**解码热路径上不会多出运行时投影算子**，在 attention 里还能**用一个窄 latent 顶替完整的 K/V 缓存**（128k 上下文下权重+KV 总内存缩 **13.5×**）。
- **「稀疏/剪枝以什么为单位记账」继续细化**：**CommunityKV** 把稀疏 attention 变成**社区检测**——用 prefill 阶段**已经算出的 QK^T** 建 token 图并划社区，新 token 以 **O(1)** 局部规则归入，因此**流式解码全程无需全局重划分**（免训练；每 head 一张图 **1.25×**、query-group 聚合图 **1.71×** 端到端生成吞吐）；**QK-Wanda** 让 **Q 与 K 共享一份剪枝预算**，闭式删除代价在 50% 稀疏下把重构误差比 Wanda 压低 **60%**，但它同时给了个**反例**：Llama-3.1-70B 重构误差更低、困惑度反而更高。
- **状态与内存这一侧今天给的是可换算的数字**：**TACO** 用「每列取幅值最大项的符号」（三值、列 one-sparse）把优化器持久状态从 **27.7 GB 压到 0.16 GB（174×）**、OPT-13B 峰值训练内存 **80.6→27.5 GB（2.9×）**，单张 80 GB H100 就能全参数微调 **30–32B**；**RiP 卷积**修正了 Gural–Murmann 原位卷积闭式解的两处错误（其一会在**他们自己部署的网络每一层**上静默破坏仍然存活的输入），peak activation memory 降 **12.5%–33.3%**，**周期数不变、输出逐位一致**。

## 一、CUDA 开源仓 Release Notes（算子）

> 本期无新条目。30 天窗口内 10 个仓库的正式 release 仍只有 09-12～09-30 那一批（CUTLASS v4.8.0、TensorRT v11.3、NCCL 2.32.3-1 与 nccl4py v0.6.0、CCCL v3.4.3 与 python-1.2.1、cuda-python v13.4.3 / cuda-core v1.2.1 / cuda-pathfinder v1.8.3、cuda-quantum 0.16.0），均已在 09-12～10-02 各期收录。窗口内唯一的新 tag 是 `cudnn-frontend` 的 `v1.31.0.dev71332972`（10-03），仍是 NVIDIA 内部 CI 的 nightly 自构建（自述「Not a supported release」、只保留最近 14 个 tag），与前几期口径一致不计入。

## 二、芯片动态

> 本期无新条目。7 天窗口内两个 RSS 源仍只命中一条——NVIDIA Blog 的《How NVIDIA GPUs Help Accelerate OpenAI's GPT-6 Astra Ultrafast》（10-01），内容是 **OpenAI 模型上线**（跑在 Blackwell 上、靠推理优化拿到最高 8× token 生成加速），属应用/产品发布，无新架构也无 kernel 级细节，与昨日口径一致不计入。

## 三、arXiv 论文（算子内核）

- **[XOR-Trellis: Ultra-Low-Complexity Dequantization and Curvature-Aware Hadamard-Free LLM Quantization](https://arxiv.org/abs/2610.00432)**（`2610.00432`，09-30）：trellis 编码量化能在**超低位宽**下压缩 LLM 权重，而且不需要常规矢量量化那种**指数级大的码本**；但真要落地卡在两个环节：**重构速度**（反量化要有足够并行吞吐，否则反量化本身就成了推理瓶颈）与**精度**（不靠昂贵的 incoherence 变换也得准）。这篇的两手正对这两处：其一，**超低复杂度的 trellis 反量化器**，用**结构化的 state→value 映射**（对硬件友好）替代一般查表，同时**保留 trellis 搜索所需的多种重构选择**；其二，把离散 trellis 的路径优化重写成**曲率感知目标**，让模型敏感度**直接在原坐标系里**被反映，于是**不再依赖 Hadamard 类 incoherence 处理**。对算子组，这两条分别对应**解码热路径上一个更便宜的重构算子**与**省掉一次全维随机旋转**（后者在大模型上是实打实的带宽与算力开销）。边界要说白：摘要只给方法与设计目标，**没有任何吞吐或困惑度数字**，「足够并行」是设计意图而非实测结论。

- **[Learning Functional Subspaces for Neural Network Compression](https://arxiv.org/abs/2609.40127)**（`2609.40127`，09-30）：低秩分解的好处是**矩阵保持稠密**、在标准硬件上照样高效，但既有方法选「丢哪个子空间」用的是**局部闭式准则**（激活能量、逐层重构误差、损失的二次近似），忽略了误差在网络内的传播，压缩率高时误差随深度叠加、性能直接崩。**LSP** 改为**端到端学习要丢弃的子空间**：每个线性层（或读取同一激活的**绑定层组**）配一个**正交投影算子**，所有投影算子对**全局目标**（与稠密模型输出分布的 KL，或原训练损失）联合优化，**预训练权重保持冻结**；投影算子由**白化 SVD 截断**初始化，秩按**「每个投影算子每省一个参数所对应的输出 KL」**分配。训练完成后投影算子**可合并回标准低秩因子**，同一绑定组只共享一个因子——这正是它对算子的价值：**解码热路径上不会多出一个运行时投影算子**；在 attention 里还让模型**用一个窄 latent 代替完整的 K 与 V 缓存**。结果：OPT-125M/1.3B、Qwen3-4B、Llama-2-7B 与 ViT-B/16 上均优于基线，且压缩越狠优势越大——**−70% 压缩**时 Llama-2-7B 的 WikiText-2 困惑度 **10.9**、平均 zero-shot **42.2%**，最强基线为 **13.3 / 36.0%**；小 batch 下分解模型解码**比稠密快至 1.6×**，缓共享 latent 后 **128k 上下文**下权重+KV cache 总内存缩 **13.5×**（未绑定的基线分解最多 6.5×）。

- **[Right In-Place (RiP) Convolution: A Simple, General, and Near-Optimal Strategy for Memory-Efficient CNN Inference](https://arxiv.org/abs/2610.00586)**（`2610.00586`，09-30）：在微控制器这类受限硬件上，限制 CNN 推理的是**激活内存**而非算力；直接的原位（in-place）卷积能省掉双缓冲，代价是访存不再顺序——Gural 与 Murmann 的「内存最优」闭式解需要**非顺序遍历**（其转置代价是 2× 推理时间）。这篇先**证伪**该闭式解的两段区间：（1）**恰好少分配 $(k-1)C_{in} \bmod (C_{out}-C_{in})$ 个标量**——这一项在**他们自己部署的网络里每一层卷积都生效**，表现为对**仍然存活的输入**的静默破坏；（2）一旦关键 leg 离开输出网格，闭式解**无上界地高估**，最高达 **2432×**。作者修正两处并把结论推广到**任意 stride、dilation、padding 与矩形核**。其 **RiP** 是**逐位等价**的重写：每层在共享工作区里**右对齐读输入、左对齐从 0 号位置写输出**；由于「欠账」是输出像素索引的**分段仿射函数**，**O(1) 求断点**即得最小安全间隔，同时**保持 row-major 访存**。验证：**10,000 个随机层无一次破坏**；25 个架构共 84 个卷积层中，**58 个与其 herringbone 工作区完全一致、81 个在 5% 以内**，平均比双缓冲**省 24.8% 内存**。写进 **TinyEngine** 的 kernel 并部署到 **Raspberry Pi Pico 1 / Pico 2**：11 个 MCUNet 模型的 peak activation memory 降 **12.5%–33.3%**，**周期数不变、输出逐位一致**，能装进 Pico 1 **256 KB SRAM** 的模型由 6 个增至 **9 个**。对算子组的意义偏方法论：**内存布局的闭式解必须把「少分配」和「越界高估」两个方向一起验证**——一个错项完全可以在「跑得通、不掉精度」的前提下静默破坏数据。

- **[CommunityKV: Efficient Long-Context Decoding via Graph Partitioning](https://arxiv.org/abs/2610.00418)**（`2610.00418`，09-30）：稀疏 attention 的两难在于：要么训练一个选择器，要么在免训练一侧依赖**语义粗糙的启发式**或**昂贵聚类**——而后者在解码过程中**难以增量更新**。CommunityKV 把稀疏 attention 形式化为**社区检测**：用**标准 prefill 中已经算出的 QK^T** 构建 token 图（不额外增加打分扫描），把图划分成社区，从而按**语义内聚的 token 组**检索；对**新生成的 token**，一条**局部更新规则在常数时间**内把它归入社区，于是**整个流式解码期间不需要全局重新划分**。在 Qwen3 与 Llama-3.1、三个长上下文基准上：**每个 query head 一张图**时端到端生成吞吐最高为 dense attention 的 **1.25×**；把同一 query group 的图**聚合**后最高 **1.71×**，精度相当。对算子组可迁移的一点是**「增量状态必须是 O(1) 的」**：图本身是 prefill 的副产品，解码期只维护一次常数时间的归属判定——这与昨日 SparseEngine / Vosti 那条「调度状态进热路径会威胁确定性验证」的张力同源，但这里把代价压到了常数时间。

- **[QK-Wanda: Coupling Queries and Keys for Unstructured Pruning](https://arxiv.org/abs/2610.01554)**（`2610.01554`，10-01）：Wanda 在**每个线性投影内部独立**给权重打分，但 Q 与 K 是通过**点积相互作用**的。QK-Wanda 在**无掩码的 pre-RoPE 重构目标**下，按**逐个删除的代价**给 Q、K 权重打分，并把**对面投影的信息**（给 Q 的分数掺入 K、给 K 的分数掺入 Q）作为增量，使两个投影**共享同一份剪枝预算**。分数有**闭式解**，无需梯度或权重更新；整轮剪枝耗时**只比 Wanda 多 1.3%（A100）与 3.1%（H200）**。在 TinyLlama / Llama 2 / Llama 3 / Qwen2.5 共 **15 个模型、0.5B–72B** 上评测：**50% 稀疏**时 QK 重构误差比 Wanda 平均低 **60%**、**80%** 时低 **45%**。但下游收益**依模型而异**——Llama 2 70B 在 80% 稀疏下 WikiText-2 困惑度降 **20.3%**、C4 降 **13.5%**、平均 zero-shot 升 **5.94 个百分点**，Qwen2.5-72B 同向改善，而 **Llama-3.1-70B 是重构误差更低、困惑度却明显更高**。作者自己的总结是「耦合型剪枝准则有前景，**但局部重构误差不是模型质量的可靠预测器**」——对算子组，这又是一次提醒：**别把 layer-wise 重构指标当作剪枝后端的验收标准**。另需注意这是**非结构化**剪枝，省的是内存与带宽，不直接换算成 kernel 加速。

- **[Joint Branch-Space Transform Coding for Diffusion Activation Quantization with Classifier-Free Guidance](https://arxiv.org/abs/2610.00930)**（`2610.00930`，10-01）：扩散模型 PTQ 已经在利用 timestep / feature / layer 结构，CFG 结构也开始被引入，但**激活量化仍然在条件与无条件坐标上各自独立进行**，跨激活的结构没被用上。这篇的观察是：**配对的 CFG 激活构成一个强相关的二维源**，在固定比特预算下**分支编码基的选择会实质影响量化保真度**。据此提出 **branch-space transform coding**：用一个**离线导出的 2×2 正交矩阵**旋转配对的 CFG 分支，对模型参数与量化管线的改动都极小；进一步给出 **GCBT**，把 **CFG 引导方向**与**跨分支二阶矩**一起纳入，在等速率量化噪声代理下**存在逐层闭式解**，不需要梯度优化或角度搜索。叠加在既有扩散 PTQ 方法之上，多数可比项取得**统计显著的保真度提升、没有统计显著的退化**，且**底层宿主量化管线保持不变**。对算子组，这属于「**把旋转折进权重、运行时只多一个 2×2 变换**」的量化前处理，几乎是零算子成本那一档；但摘要只报保真度、**没有吞吐或延迟数字**，而且它优化的是扩散采样的生成质量，不是通用 GEMM 场景。

- **[TACO: Ternary Absolute-max Column-wise One-sparse Optimizer for LLM Fine-Tuning](https://arxiv.org/abs/2610.02199)**（`2610.02199`，10-01）：全参数微调的开销大头是**优化器状态**，而既有路线要么压缩状态、要么放弃一阶梯度、要么在**保留稠密状态**的前提下改更新几何。Muon 通过矩阵值更新压低优化器内存，但其几何与 AdamW 不同，微调 AdamW 预训练模型时可能掉点。**TACO** 顺着 Muon 的**算子范数最速下降**视角再进一步：在**按维度归一化的 $1\to1$ 算子范数**下求**精确的最速下降方向**——做法是**对二维权重矩阵的每一列取幅值最大项的符号**，更新因此天然是**三值且列 one-sparse**，并且**保留了一阶梯度**。工程上每列只维护**一小组低精度梯度分量**：相对 AdamW8bit，持久优化器状态由 **27.7 GB 降到 0.16 GB（174×）**，OPT-13B 上峰值训练内存由 **80.6 GB 降到 27.5 GB（2.9×）**，精度与运行时间相当；由此可在**单张 80 GB H100** 上对 **30–32B** 模型做全参数微调（跨多个模型族与任务）。对算子组的看点是：更新算子被压成**逐列取幅值最大项 + 符号**，基本是**归约 + 掩码**级别的算子而非稠密乘，内存与带宽都友好——但前提是**把优化器几何从 Adam 的一阶矩换成算子范数最速下降**，属算法层面的替换，不是无损压缩。

- **[MoE-CORE: Coordinated Expert Offloading and Residency for Memory-Constrained MoE Inference](https://arxiv.org/abs/2610.01950)**（`2610.01950`，10-01）：MoE 的稀疏激活省下了算力，但**专家权重可能超出设备内存**；offloading 让紧凑设备也能推理，代价是把 **host-to-device 传输暴露进推理路径**。MoE-CORE 做的协调是：**prefill 阶段把完整专家层分批装入交替缓冲**；**decode 阶段**组合四件事——**非均匀的逐层缓存容量**、**领域信息初始化**、**路由历史感知的替换**、**跨层预取**。主配置**精确执行 router 选中的专家**，另有一条可选的**按分数替换（substitution）**路径处理「低分未命中」。实测：DeepSeek-V4-Flash-W4A8 上五类负载平均 **TPOT 38.0–44.8 ms**，对照的 vLLM Prefetch 配置为 **1268.9–1269.1 ms**；GLM-5.2-W4A8C8 上为 **206.6–220.5 ms** 对 **5941.5–5941.8 ms**；在 **84 GB NPU 内存上限**下，DeepSeek GSM8K 的最佳配置以**近似专家替换 + MTP（深度 2）**达到 **TPOT 21.5 ms**。两点边界必须说清：对照设置里两个系统的**输出长度上限不同（1K 对 128 token）**，所以上面的比值**不能当作干净的端到端加速**；此外**替换路径是近似的**（会改变输出），只有主配置才是精确执行。

> 收录范围说明：本期的 arXiv 窗口里**没有提交日期 ≥ 10-02 的条目**（arXiv 索引滞后约一天，10-02 提交尚未入库），因此本节收录的是 09-30 与 10-01 提交、且**尚未被 09-30～10-02 各期报道过**的条目。其中前四条（XOR-Trellis、LSP、RiP 卷积、CommunityKV）是 09-30 提交、当日索引延迟漏报的补录，后四条（QK-Wanda、GCBT、TACO、MoE-CORE）为 10-01 提交；09-30 期已收录的 RLX / Parameterized Stripe Attention / QuantMLA / WUSH-KV / STEPQuant / Delta-Matching / FP64-INT8-FP4 / Mira / Behavioral Capacity Certificates，10-01 期已收录的 Prefix-Invariant / Vosti / LampAttention / Switching Linear Attention / Low-Discrepancy Dither / PARK / SparseEngine / SpTRSV / SparLeak，以及 10-02 期已收录的 Stochastic Rounding / VASC / MWOP / ZTA-Q，均不再重复。
