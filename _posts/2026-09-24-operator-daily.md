---
layout: post
title: "算子日报 · 2026-09-24"
permalink: /operator-daily/2026-09-24/
date: 2026-09-24 23:59:00 +0800
tags: [算子日报]
---

> 每日汇总：CUDA 开源仓 Release Notes（算子）、芯片动态、arXiv 算子内核论文。
> 本期合并 09-23 / 09-24 两期（09-23 未单独成刊），Release Notes 一并补录 09-22 发布但上一期未及收录的 CUTLASS 4.8.0 与 TensorRT 11.3。

## TL;DR

- **CUTLASS 4.8.0** 落地 Rubin (sm107)：CuTe DSL 首发 dense GEMM 支持，高吞吐 FP8（MMA_K=64）与 FP4（MMA_K=128）Tensor Core MMA、TMEM 512→576 COL、共享内存放宽到 328KB，Primitives 侧另有 FP4 的 2:4 稀疏。
- **cuDNN Frontend 1.30.0** 把 FlashInfer serving 路径端到端跑通 FROST SDPA 引擎：graph declaration 即 operand contract、SDPA 的 THD gate 由 BSHD 改为 (H,S,D)、THD 模板支持动态 batch/head extent（此前每套 shape 铸一个 2.6s 的 kernel）。
- arXiv 收录 7 篇：block-sparse attention 运行时 **Tessera**（最高 6.79×）、用 gated-delta 线性 attention 状态做 KV 驱逐信号 **DeltaS**、补偿感知的稀疏选择 **CompKV**，以及 4 篇量化相关工作。芯片动态本期无新条目。

## 一、CUDA 开源仓 Release Notes（算子）

### CUTLASS

- **v4.8.0**（09-22）— [链接](https://github.com/NVIDIA/cutlass/releases/tag/v4.8.0)：**CuTe DSL 首发 Rubin (sm107) dense GEMM 支持**，可用特性包括：更高吞吐的 FP8（MMA_K=64）与 FP4（MMA_K=128）Tensor Core MMA、B collector reuse、TMEM 由 512 COL 扩到 576 COL、共享内存分配放宽到 328KB、FP8/FP4 混合精度吞吐增强与 softmax 加速相关特性；Primitives 侧同步获得上述能力（外加 FP4 的 **2:4 稀疏**支持）。
  - **CuTe DSL extensions (`cute_ext`)**：TMA load/store/multicast/reduce-store 的 CTA-V map 改为自动推导（显式 map 仍可作为覆盖传入）；新增异步 atomic TMA reduce-store 与 sparse MMA；新增可复用的 GEMM mainloop 与 TMA epilogue helper；可选 TMEM 累加器 buffer 规划（容量受限时用 ping-pong 重叠存储）；grouped GEMM 通过 SMEM 暂存 TMA descriptor 更新、workspace 复用与削减 prologue/同步开销改善性能。
  - **编译器流水线 opt-in preview**：`cute_ext` API 可混入普通 `@cute.jit` / `@cute.kernel`（`CUTE_DSL_USE_EXTENSION_COMPILER=1` 试用，预计不早于 4.10 转默认；行为与性能预期不变，但生成的 PTX/SASS 可能不同）。
  - **IKET Profiler** 支持对 Rubin (sm107) kernel 采样，并可按 cluster 单独 dump timing 以降低 profiling 开销。
  - 新增示例覆盖 Rubin blockscaled GEMM（FP4/FP6/FP8 混合精度、UE5M3 / block-32 scale factor）、Rubin grouped GEMM 与 blockwise GEMM，以及 Blackwell 系列的 back-to-back / persistent / CLC 调度 / GLU / mixed input / planar complex / input transform / GeForce pingpong / Blackwell Ultra blockscaled GEMM 与隐式 GEMM 卷积。

### cuDNN Frontend

- **v1.30.0**（09-23）— [链接](https://github.com/NVIDIA/cudnn-frontend/releases/tag/v1.30.0)：**FlashInfer serving 路径在 FROST SDPA 引擎上端到端跑通**——把 FlashInfer 实际构建的图在树内重新声明为一致性测试集，逐处修掉「引擎读 buffer 而非读声明」的问题，FlashInfer serving shape 上的剩余 decline 清零。对算子开发者，本期几处是通用的 graph API 语义修正：
  - **声明即 operand contract**：backend 按指针读 variant pack，调用方绑定的 buffer 形状本不必与声明一致（2-D 矩阵绑到 `[1, m, k]`、flat quantizer blob 绑到重排过的 scale 张量、0-d 标量绑到 `(1,1,1)`），但 Python 引擎此前读的是 buffer 自身形状，导致同一个 `execute()` 在 backend plan 成功、在 FROST plan 被拒。`_normalize` 现在只要调用方几何与声明不符却覆盖其字节数，就按声明描述该 slot：buffer 携带**声明**的 extent 时保留自己的 stride（线性 attention 引擎就这样服务 strided 输入），其他 extent 则仅当其 slot 构成一整段连续区间、覆盖声明的字节时才重述——FlashInfer 的 column-major B 正是连续块的转置 view。**dtype 也由声明提供**：FlashInfer 把 packed fp4 数据与 E4M3 scale blob 都以 `uint8` 绑定，故存储槽宽与声明一致的 buffer 按声明 dtype 解读，更窄/更宽的 buffer 保留自身 dtype 且永不重述。
  - **SDPA 的 THD gate 改为 (H, S, D) 而非 BSHD**：ragged offset 下 Q/K/V/O 的 batch stride 根本不会被读（每个序列的 base 来自 offset 表，lowering 把 batch 轴按 extent 1 绑定），但引擎 gate 与 DSL adapter 却要求四个轴都是 BSHD 物理序，于是 FlashInfer 的 ragged prefill 在 `b > 1` 时全被拒。`packed_layout_ok` 现在依次检查 head dim 最内、再 head、再 token，mismatch gate 与 SM100/SM120 两个 adapter 共用同一判定。
  - **THD 下支持 FlashInfer 的 padded `(b, s_max, h)` Stats**（即无 ragged offset 的 `return_lse` buffer）：此前 packed 路径按 token 行连续写入，`b > 1` 被拒、`b == 1` 靠「序列长度之后的行未写入」侥幸吻合。20 个 SM100/SM107 THD 模板现接受 `lse_padded_rows`，按静态 fake 选择 per-batch 形式，adapter 在启动流上以 `-inf` 初始化该 buffer。**不增加 kernel ABI**。
  - **THD 模板支持动态 batch 与 head extent**：`compile()` 原本把 `b`/`qh`/`kh` 与所有 stride 钉死在 fake 里，FlashInfer 的 serving shape 因此每套 `(b, qh, kh, strides)` 都新铸一个 2.6s 的 kernel（其 cuDNN attention 测试里累计 126 个 forward kernel），而 `_host` 本就是从 `problem_size` 运行时读 `B`/`QH`/`KH`。`compile(dynamic_bhk=True)` 在取完 cache key 后把它们重绑为 `cute.sym_int()`，每类 layout 收敛到一个 kernel。
  - dense padded-Q trim 推广到所有 kernel（此前 SM100 fp8/mxfp8 d128 与 (192,128) 等 flavor 需另走 host 读取路径）。

### NCCL

- **nccl4py v0.6.0**（09-23）— [链接](https://github.com/NVIDIA/nccl/releases/tag/nccl4py-v0.6.0)：新增**实验性 CuTe DSL ReduceCopy API**——`lsa_reduce_sum()`、`multimem_reduce_sum()`、`lsa_copy()`、`multimem_copy()`、`lsa_reduce_sum_copy()`、`multimem_reduce_sum_copy()`、`local_reduce_sum_copy()`，覆盖 LSA window、multimem 指针与本地张量三类目标。Host 侧补齐 NCCL 2.32 接口：`NCCLCollConfig.launch_completion_event` / `NcclEventSpec`（per-collective 的 CUDA launch 完成事件）、NVLS host 模式配置、CFT 能力上报（`cft_support` / `cft_multicast_support` / `cft_counted_support`）与窗口注册标志 `WindowFlag.GIN_ONLY` / `CFT_COUNTED`。CuTe DSL barrier 生命周期新增显式销毁（`LsaBarrierSession.destroy()` / `GinBarrierSession.destroy()` / `BarrierSession.destroy()`），使 barrier handle 与 index 可安全复用；并修正 `ThreadScope.THREAD` 与 libcu++ 对齐。示例见 `examples/cute/08_reduce_copy.py`。

### TensorRT

- **v11.3**（09-22）— [链接](https://github.com/NVIDIA/TensorRT/releases/tag/v11.3)：**默认 CUDA 版本升到 13.4**；`IParserRefitter` 的外部权重处理重构；移除 `demoDiffusion` 与 detectron2 python sample；Polygraphy 升到 v0.53.6。

### cuda-python

- **v13.4.3 / v12.9.9 / cuda-core v1.2.1**（09-23）— [链接](https://github.com/NVIDIA/cuda-python/releases/tag/v13.4.3)：均为补丁版本，无实质 changelog（cuda-core 1.2.1 仅外链 release notes），与算子内核关联弱。

> 注：cudnn-frontend 的 nightly dev 构建（`v1.30.0.dev*`、`v1.31.0.dev*`）非正式 release，不做收录。

## 二、芯片动态

> 本期无新条目。NVIDIA Developer Blog 与 NVIDIA Blog 两个源在 7 天窗口内无命中关键词的新文章。

## 三、arXiv 论文（算子内核）

- **[Decoupling Logical Masks from GPU Execution for Dynamic Block-Sparse Attention](https://arxiv.org/abs/2609.25869)**（`2609.25869`，09-22）：视频扩散 Transformer 用 block-sparse attention（BSA）只算逻辑 mask 选中的块来降本，但**把逻辑块几何与执行选择耦合**会限制对不同 mask / 不同 GPU 的适配，而运行时 kernel 特化本身的开销可能吃掉执行省下的时间。Tessera 把逻辑 mask 与 GPU 执行解耦并保持指定的 attention 交互：物理映射层把逻辑块保留 / 合并 / 细分成适配不同 mask 形状与 GPU 架构的物理 tile；任务组织层在 GPU task 内分组调度 tile 以复用数据、暴露并行、重叠搬运与计算；再以 profile 引导的 regime 选择用离线打表做低开销执行计划选择。特化 CUDA kernel 覆盖四代 NVIDIA GPU，在 2315 个真实 attention mask 与工业视频扩散模型上最高 **6.79×** 加速。

- **[DeltaS: Reading the Gated Linear Attention State for KV Cache Eviction in Streaming Video](https://arxiv.org/abs/2609.27470)**（`2609.27470`，09-23）：视频语言模型越来越多采用「线性 attention 层与 full attention 层交错」的混合架构——线性 attention 的循环状态大小固定，full attention 的 KV cache 却随流增长，必须驱逐；而**流式场景下驱逐要在问题到来之前决定**，无法依赖 query。现有方法从 KV cache 自身（位置、attention、key-value 表示）推分数，attention 类分数还需 proxy query 或额外计算。DeltaS 转而利用混合骨架里现成的信号：gated-delta 线性 attention 的状态按「输入减去已能从状态中取回的部分」的残差更新，所以一个 chunk 引起的**归一化状态漂移**（state drift）正反映它带来多少新信息。方法 query-agnostic 且免训练，保留状态漂移大的 chunk。在预算与保留策略固定的受控对比中，state drift 优于位置 / attention / key-value 类信号；信号开销仅占前向的 **1.9%**，六个长视频基准平均 **+2.1** 分、最长基准 **+5.6** 分。

- **[CompKV: Compensation-Aware KV Selection for Long-Context LLM Inference](https://arxiv.org/abs/2609.26300)**（`2609.26300`，09-22）：稀疏 attention 只对选中子集算精确 attention，被排除的 token 需靠**补偿**追回贡献，但现有方法先按 attention mass 选 token、再对未选中的做粗粒度补偿，把两者的交互忽略了——真正该优先选的，是省略后会留下最大补偿误差的 token。CompKV 从理论上给出块级均值补偿的残差由**块 attention mass 与块内 logit 方差共同支配**，用紧凑的块级统计量近似该残差，得到一个可直接部署的选择准则，并配了高效的异步实现。RULER 与 LongBench-Pro 上优于所评估的稀疏基线，self-attention 相对 full attention 最高 **6.85×** 加速。

- **[Disaggregated Quantization: Specializing LLM Prefill and Decode](https://arxiv.org/abs/2609.26333)**（`2609.26333`，09-22）：prefill 与 decode 对量化的诉求本就不同——低精度算术加速 prompt 处理，紧凑权重降低生成期的内存流量。DQ 把**计算格式、权重与存储位置按两个阶段分别特化**。在 Qwen 3 与 Gemma 3 上：仅在 decode 侧去掉激活量化即可提升 decode-heavy 任务精度且不增加推理成本；为 prefill 单独训练 compute-native 权重（NVFP4）比 weight-only 推理更快，同时在 2–3 bit decode 下精度持平或更好；用已发布的 Qwen3.8-27B GGUF decoder 训练 NVFP4 prefiller，**1-bit 精度在 MMLU-Pro 上 +32.5 点、MMMU-Pro 上 +35.3 点，且不改动 decode checkpoint**。为在单卡容纳额外 checkpoint，offloaded disaggregated prefill（ODP）从 SSD 流式载入权重并按 prompt 长度摊薄，同一 27B 模型 8K prompt 下 llama.cpp TTFT **1.78×**。另在 vLLM 中评估 disaggregated serving 下的精度，并通过 PTQ 把共享权重格式的解耦验证到 **2.8T** 参数规模。

- **[QuantWM: Temporally Consistent 2-Bit KV Cache Quantization for World Models and Video Generation](https://arxiv.org/abs/2609.26425)**（`2609.26425`，09-22）：视频生成与世界模型的 2-bit KV cache 量化在 VBench 一类基准上近乎无损，却仍造成严重**时序闪烁**。作者定位到一个反直觉现象：**Key 的重建误差小于 Value，输出退化却大得多**——根源在 attention，Key 的小扰动会改变 attention logits（`QK^T`），进而改变 Query 选中的时空 token。QuantWM 训练无关、严格因果，用两项技术抑制这种偏移：QSAC（quantization-sensitivity-aware clustering）联合历史 Query 敏感度与残差范围，挑选对 INT2 友好的 Key 聚类中心；PSAC（principal-subspace attention compensation）沿主导 Query 子空间用低秩投影恢复剩余 Key 误差，直接稳定 attention logits。在 Causal-Forcing、LingBot-World-v2、HY-World 1.5、Matrix-Game-2、Longcat-Video 上提升视觉质量与时序一致性，KV cache 最高 **6.20×** 压缩且额外开销有限。

- **[Implementation and Evaluation of BitNet Inference on a CGLA by Signed-Int4 Instructions](https://arxiv.org/abs/2609.27453)**（`2609.27453`，09-23）：BitNet b1.58 用三值权重加整数激活，这套算术与常规 int8 / 浮点 GEMM 并不匹配，现有 BitNet 加速器都走专用数据通路。本文改为映射到 **CPU-Grounded Linear Array (CGLA)**——一个可编程 ASIC，具备显式 DMA、本地内存与编译器可见的可复用整数 lane——并新增 **OP_SMA4 可复用 signed-int4 乘累加指令**，而非 BitNet 专用通路：每个三值权重占一个 signed 4-bit lane，每个 int8 激活拆成两个 signed-int4 片段、用 shift-and-add 重建。把 145 MHz FPGA 实测按频率缩放到 840 MHz 28nm CGLA，得到每个 signed-int4 乘积 **0.390 ns**；CGLA offload 的 BitNet C++ 执行实测 **2.52 tokens/s**。

- **[Beyond Scalar Sensitivity: Activation-Aware Mixed-Precision LLM Quantization with Cross-Layer Refinement](https://arxiv.org/abs/2609.25916)**（`2609.25916`，09-22）：混合精度权重量化常被建模为多选背包（MCKP），但求解器依赖**标量敏感度代理**——把每个权重矩阵的 Hessian 压成一个数，且各模块独立处理。作者证明即使最优标量代理，相对完整 activation-aware 二次型仍有最高 `√(κ(A)κ(B))` 的乘性失真（典型 LLM 模块上该上界从 `10^1` 到 `10^13`），**模块间敏感度排序因而不可靠**。CASA 分两阶段：Stage 1 用 Kronecker 分解 Hessian 导出的 activation-aware 度量替代标量代理，把 MCKP 化为连续松弛有闭式解的形式；Stage 2 用端到端模型 loss 做跨层感知的局部搜索来评估 bit-width 更新。多 LLM、多 bit 预算下困惑度优于最新标量代理基线，在 **< 3 bit/weight** 的超低比特区间尤其明显——为 mixed-precision kernel 的比特分配提供了一条可用的判定依据。
