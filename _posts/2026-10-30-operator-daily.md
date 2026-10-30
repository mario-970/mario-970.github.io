---
layout: post
title: "算子日报 · 2026-10-30"
permalink: /operator-daily/2026-10-30/
date: 2026-10-30 23:59:00 +0800
tags: [算子日报]
---

> 每日汇总：CUDA 开源仓 Release Notes（算子）、芯片动态、arXiv 算子内核论文。

## TL;DR

- **低精度 kernel 的「正确性」成了主线**：numoracle 用带精度标注的参考程序生成逐元素接受域，13 个 Triton 算子、每块 GPU 33,960 次正确运行零误报，RTX PRO 6000 上杀死 116/124 个变异体；Evo 2 的 FP8 checkpoint 在 Blackwell 上失败，根因是单个 GEMM 的累加顺序，只模拟该 projection 就把 40B 的 BRCA1 AUROC 从 **0.434** 恢复到 **0.869**。发布面只有 cuDNN Frontend **v1.32.1** 一个正式 patch（DSA/CSA backward 的原生 H96/H32 与长上下文 workspace 优化）。
- **Kernel 工程侧**：多 agent 在 Furiosa RNGD NPU 上用 12 天把三个 Gemma-4-12B decode kernel 做到 **9.293×** geomean，距权重搬运下限 1.25×；CACPU 用分层 cache blocking 把 CPU causal attention 做到比 PyTorch SDPA/FlexAttention/SGLang 快 **2.01×**；AB-Attention 跳过 89–92% 注意力仍把 PSNR 提高最多 4.4 dB、端到端 **2.23×**。
- **低精度算法侧**：LinQuant 用 7 个函数不变性让线性递归 prefill 适配低精度；GDN 的 state 量化配合 FP8 最近写入窗口，在 Qwen3.5-4B/9B 上比 STEPQuant@4 高 **38.39/41.21** 个百分点。

## 一、CUDA 开源仓 Release Notes（算子）

### cuDNN Frontend

- **v1.32.1**（10-30）— [链接](https://github.com/NVIDIA/cudnn-frontend/releases/tag/v1.32.1-release)：DSA/CSA backward 的补丁版，补原生 head 支持并压缩长上下文 workspace。
  - **DSA backward**：SM100/SM103 上为 Top-K 宽度 **2049–2051** 增加原生 **H96**，普通与确定性模式都支持（[#2042](https://github.com/NVIDIA/cudnn-frontend/pull/2042)）。
  - **DSA indexer backward**：SM100/SM103 的 opt-in `sm100_v2` 后端增加原生 **H32**，普通模式支持非对齐 Top-K 宽度（[#1908](https://github.com/NVIDIA/cudnn-frontend/pull/1908)）。
  - **CSA backward**：加速 ratio-2/4 的 packed metadata lookup，碎片化序列多时收益更明显（[#1932](https://github.com/NVIDIA/cudnn-frontend/pull/1932)）；确定性 ratio-4 backward 在 ≥65,536 compressed rows 的长上下文上 workspace 降约 **2/3**（[#1968](https://github.com/NVIDIA/cudnn-frontend/pull/1968)）。

### cuda-python

- **cuda-pathfinder v1.8.5**（10-30）— [链接](https://github.com/NVIDIA/cuda-python/releases/tag/cuda-pathfinder-v1.8.5)：新增 Windows CTK 13.2.2 的 CUPTI DLL 名 `cupti64_2026.1.2.dll`，`load_nvidia_dynamic_lib("cupti")` 能识别已安装的库（[#3279](https://github.com/NVIDIA/cuda-python/pull/3279)）。对 Windows 上做 kernel profiling 的算子开发者是直接修复；release 正文只有文档链接，以上内容取自仓库内 `1.8.5-notes.rst`。

> 本期其余更新不计入：`cudnn-frontend` 同日还有 `v1.32.1-rc.77808318`（与正式版同内容，promote 后即 1.32.1）和 nightly `v1.33.0.dev77874593`（正文自述 *Automated nightly build from NVIDIA-internal CI. Not a supported release.*），延续既有口径；30 天窗口内其余正式 release（CUTLASS v4.8.1、CCCL python-1.3.0 / v3.5.0 / v3.4.3、TensorRT v11.4、cuda-python 其余版本）均已在往期各期收录。

## 二、芯片动态

> 本期无新条目。NVIDIA Developer Blog 源继续返回 0 条；NVIDIA Blog 在 7 天窗口内无新的芯片/架构类命中。本节连续为空属**数据源侧**状况，不是抓取故障。

## 三、arXiv 论文（算子内核）

> 本期按 `fetch_state` 从 10-29 起抓增量，实际返回 **150 条**（触及 `max_results` 上限，按提交时间倒序）：10-29 全部 **58 条**、10-28 全部 **44 条**、10-27 被截断后 **48 条**。10-29 当天因 arXiv 最新仍只到 10-27 而未出刊；10-30 提交的论文尚未被 API 索引，留待下一期。与前一期逐 id 比对：10-28/10-29 共 **102 条**全为新条目；10-27 有 **18 条晚到**，其中 FAD-KV 与算子相关，补录于本期；其余晚到条目属应用/理论方向。按算子开发相关性精选 **13 篇**。

### 低精度与数值正确性

- **[Beyond atol and rtol: Sound Test Oracles for Low-Precision GPU Kernels](http://arxiv.org/abs/2610.38983v1)**（`2610.38983`，cs.MS/cs.PL/cs.SE，10-28）：低精度 GPU kernel 通常用手工 atol/rtol 对照参考实现，但合法误差随输入、格式和精度计划变化，固定阈值只能在误报和漏检之间取舍。numoracle 从带精度标注的参考程序推导**逐元素接受域**，计算过程跟踪共享舍入误差、按 provenance 排序的乘积累加、block-scaled 量化器与 math library 误差，并用 rounding-interval 检查抓出对称界会放过的错误 store。对 Liger-Kernel、Unsloth、vLLM 的 **13 个 Triton 算子**、两代 GPU 评估：每块 GPU 上 **33,960** 次正确 kernel/行为保持变体运行零误报；输入漂移下调好的固定容差会误报最多 **30%** 的正确运行，numoracle 仍为零；RTX PRO 6000 上杀死 **116/124** 个有效变异体（四个固定基线为 59–83），并在历史回放中定位 8 组精度故障改动。
- **[Accuracy is not compatibility: two Evo 2 checkpoints hinge on one FP8 matrix multiplication](http://arxiv.org/abs/2610.39415v1)**（`2610.39415`，cs.LG，10-29）：FP8 格式只规定数值怎么存，不规定乘积累加怎么求和；这个空隙会让 checkpoint 在「支持 FP8」的硬件上直接失效。Evo 2 的 1B/40B checkpoint 声明依赖 Hopper FP8，在两块 Blackwell（sm_120）上过不了官方 forward 测试（7B 正常），RTX 5070 Ti 上 40B 的 next-base 准确率只有 **25.7%**，接近瞎猜；更精确的算术反而更糟，fp32 累加同样失败。作者用位精确的 H100 FP8 GEMM 模拟器 mmaemu 把依赖定位到 1B 的第 1 个 projection（共 21 个）：按 H100 实测的 **1.4%** 收缩后通过，而不是 2.1%；只模拟 40B 的第一个 projection，就把 BRCA1 变异分类 AUROC 从 **0.434 拉到 0.869**（H100 为 0.868）。结论：release 应注明参考算术，FP8「格式兼容」不等于「数值兼容」。
- **[LinQuant: Functional Invariances for Low-Precision Linear Recurrences](http://arxiv.org/abs/2610.39418v1)**（`2610.39418`，cs.LG，10-29）：线性递归（linear/gated/delta-rule）在 prefill 的 chunkwise-parallel 算法里会变成大量 dense matmul，本来适合低精度，但直接按原形式量化很脆弱。LinQuant 找出 **7 个保持全精度计算结果不变的函数不变性**（旋转、对角缩放、decay/write 因子的换位等），并给出它们在各递归族里成立的条件，用这些自由度改善量化误差。把适用不变性接进 Mamba-2 的低精度 prefill 管线 **SSDi8** 后，在 Mamba-2 1.3B/2.7B/8B 与 Nemotron-H 8B 上，匹配延迟下 perplexity 均有改善。
- **[Low-Bit GDN Recurrent-State Quantization: Global Gradient-aware Allocation and Pure-Write Sliding Windows](http://arxiv.org/abs/2610.39931v1)**（`2610.39931`，cs.LG，10-29）：线性注意力的 recurrent state 固定大小，降低 state 精度能省显存，但量化写回误差会进入后续递归，长推理任务上掉点并诱发重复/超长输出。GGA 把候选量化残差与「对未来预测损失的梯度敏感度」结合，在模型级 mixed-precision 预算下分配每个 head 的位宽；PW 把最近写入单独保存为 **FP8 vector pair**，走完整的 GDN 状态转移，离开窗口后再合并进低比特历史 state。Qwen3.5-4B/9B 的七项评测上，GGA@4 比 STEPQuant@4 高 **19.48/29.84** 个百分点；再加 PW-S8 后七任务宏平均 **69.83%/78.63%**，比 STEPQuant@4 高 **38.39/41.21** 个百分点，同时缩短平均生成长度。

### KV cache、Attention 与序列 kernel

- **[FAD-KV: Adaptive Dictionary KV Caching for Efficient Reasoning Decoding](http://arxiv.org/abs/2610.37921v1)**（`2610.37921`，cs.CL/cs.LG，10-27 晚到补录）：字典式稀疏 KV 压缩把每个 token 的 KV 表示成学习字典中少量原子的稀疏组合，但解码时 OMP 之类的迭代稀疏编码会拖慢 decode，静态字典又会和测试时演化的 KV 分布失配。FAD-KV 用**一次性 Top-n 筛选 + 轻量系数精修**替代迭代稀疏编码，并用固定大小字典的「复用 inactive atom」机制做自适应更新，避免字典随生成无限增长。在 reasoning/long-context 基准上给出质量-效率折中，且解码过程中字典有界。
- **[AB-Attention: Accurate Block Attention for Accelerating Video Diffusion Transformers](http://arxiv.org/abs/2610.39663v1)**（`2610.39663`，cs.CV，10-29）：视频 DiT 的 3D full attention 是二次开销；dynamic block-sparse 在激进稀疏下会因粗粒度打分、被剪块直接消失、整 tile 保留/丢弃而掉精度。AB-Attention 用**块内 query 聚类**低成本估计 block 重要性，再按 inclusion probability 采样并做逆概率重加权，使一次前向估计 dense attention 而不是截断；配套 kernel 把远小于 tile 的 K/V 块聚成完整 tensor-core tile，不增加 tensor-core 成本。Wan2.1/HunyuanVideo 上跳过 **89–92%** 的注意力（基线要算 1.8–3.6× 的量），PSNR 仍比最强基线高最多 **4.4 dB**，端到端加速最多 **2.23×**。
- **[Causal Attention on CPU: A Cache-Aware and Load-Balanced Scheduling Kernel for Prefill](http://arxiv.org/abs/2610.38809v1)**（`2610.38809`，cs.LG，10-28）：Agentic AI 的并发长上下文 prefill 会打满 GPU，CPU 是可行替代，但现有 CPU attention kernel 不适应多级 cache，也缺线程调度。CACPU 针对 causal full-sequence prefill 做**分层 cache blocking** 和 **kernel 内任务分发**，比 PyTorch SDPA、FlexAttention 与 SGLang 的 attention kernel 快最多 **2.01×**；接进 SGLang 后 Llama-3.1-8B 的 TTFT 最多降低 **1.74×**，精度不变。
- **[Dual DeltaNet: Value-Side Memory Reorganization for Linear Attention](http://arxiv.org/abs/2610.38578v1)**（`2610.38578`，cs.LG，10-28）：线性注意力的 state `S∈R^{d_k×d_v}` 可以看成 `d_k` 个地址、每个存 `d_v` 维 value 的 bank：key-side read 取回与 key 关联的 value，value-side read 则聚合 value 特征匹配的地址。Dual DeltaNet 在 Kimi Delta Attention 上再加一条 **value-side delta rule**，用当前 value 的选定特征重排它占用的地址；配 **chunkwise 并行训练**与 **rank-2 递归**提高推理算术强度。相对 KDA、GDN-2、Mamba-3，在语言建模、零样本下游与检索上综合最强，训练/推理开销相对 KDA 只有小幅增加。
- **[Attention, Dynamic Wave and Hybrid Sequence Mixers in Language Models](http://arxiv.org/abs/2610.39854v1)**（`2610.39854`，cs.LG/cs.CL，10-29）：这篇用 10.94M 参数模型在 TinyStories 上比较 Transformer、Dynamic Wave 递归与 Hybrid，结论对算子开发者更有价值的是**效率数字取决于实现而不是架构**：相对未融合 attention，Pure Wave 在 4096 token 时峰值显存少 **78.7%**（1404.3 vs 6601.6 MB），但换成融合 SDPA kernel 后优势完全消失（1404.3 vs 1403.6 MB），训练还贵 **8.3%**。把逐元素项提出递归循环、不写自定义 kernel 就把该层加速 **2.53×**，与 attention 的延迟差距从 2.48× 缩到 1.12×；Hybrid 会继承较贵 mixer 的成本，峰值显存与纯 Transformer 相差 0.1% 以内。报告递归替代方案时必须注明 attention backend 和实现。

### GPU kernel 工程、通信与数值内核

- **[A Case Study of a Multi-Agent Kernel-Optimization Loop That Learns a New AI Accelerator](http://arxiv.org/abs/2610.38329v1)**（`2610.38329`，cs.LG/cs.AI/cs.AR/cs.PL，10-28）：前沿 LLM 能给大多数单算子写出正确 CUDA，但换到架构和编程语言都陌生的加速器上准确率会崩。作者在 FuriosaAI RNGD NPU 的稀疏文档 Rust DSL 上跑一个多 agent kernel 优化循环，12 天内在共享且稀缺的设备上优化三个 Gemma-4-12B decode kernel：agent 自己写 DSL 文档摘要、跑 microbenchmark、记录编译器合法性规则，并在工具链变化时重验；再用 per-kernel Design Tree 的角色化搜索探索整 kernel 设计、累积小改动。先用廉价检查筛掉候选再上设备，后期只有反复对比当前最优后才接受改动；最终三个 kernel 对官方基线 **9.293×** geomean，每个都在「搬权重所需周期」的 **1.25×** 以内。
- **[The Endgame of FSDP Communication: Layout-Aware Collectives on Copy Engines, RDMA, and NVSwitch](http://arxiv.org/abs/2610.38228v1)**（`2610.38228`，cs.DC/cs.LG/cs.PF，10-28）：FSDP2 的参数/梯度是 per-parameter tensor，collective 路径却用 flat、rank-major buffer，布局转换在 Llama-3.1 70B、8K tokens/GPU 下占 step 时间 **6.4%**；默认 NCCL reduce-scatter 还能把本可单 wave 完成的并发 GEMM 拖慢 **69%**。TIDE 用 NCCL 2.31+ 的公开 API 做零拷贝 collective：TIDE-AG 用 NVSwitch multicast + 受控 recursive-doubling RDMA 把 shard 直接送到参数存储，传输和放置都不占 SM；TIDE-RS 直接消费 per-parameter bf16 梯度、归约成 fp32 shard（NCCL 没有的 mixed-precision 模式），节点内用 copy engine + 短生命周期 CTA、跨节点用 NVSwitch。64 GPU 上，Llama-3.1 70B/Wan2.2/Qwen3-30B-A3B 的 step 时间相对 Stock FSDP2 最多降 **27.1%**、相对 FSDP2+SymmMem 降 **6.0%**；reduce-scatter SM 时间降 **12.5–14.5×**，对并发计算的干扰最多降一个数量级。
- **[Learning to Formulate: Agentic Hardware-Software Co-Design of AI Infrastructure through Progressive Optimization and Verification](http://arxiv.org/abs/2610.39380v1)**（`2610.39380`，cs.AR/cs.AI/cs.DC，10-29）：AI 基础设施按层设计，层与层之间的边界会提前排除「芯片和软件必须一起改」的优化。Formulaic 让 agent 写 formulation、数学求解器在 formulation 内优化、验证与实测再检查假设和结果，agent 从计算图向芯片逐步细化；每步是一个带正确性条件的程序模板族，wall relaxation 通过给硬件约束定价来搜索芯片。GB10 上 agent 写的模板跑完整 Qwen3-8B 层比 SGLang 快 **1.075×**；模拟 Ventus GPU 上联合设计把 Qwen2.5-0.5B 最坏情况周期降到参考的 **0.428**（只改软件为 0.934）；五步 GEMM 细化链上复用证明让检查快 **2.51×**。作者明确说这是分项研究而非端到端系统。
- **[Reduced and Mixed-Precision Block Householder QR Factorization](http://arxiv.org/abs/2610.38537v1)**（`2610.38537`，math.NA，10-28）：用 WY 表示做 block Householder QR 时，统一半精度会不稳定；实验表明大矩阵上**范数计算必须用 fp32 累加**才能避免上溢/下溢。mpWYQR 只提高选定计算的精度，并避免现有多精度 block QR（mpBQR3）的大部分精度转换；误差界只差一个常数因子，启发式界相同。矩阵尺寸和条件数扫描中保持 mpBQR3 精度，速度最多 **18×**、内存更少；作者还讨论了高精度 inner product 的性能代价与高效 GPU 实现所需的库支持。

> 其余新入库条目多为应用/训练方法向（视觉生成、医疗影像、遥感、机器人、推荐、科学计算应用等）或与 GPU 算子无直接关系的理论工作，未收录。
