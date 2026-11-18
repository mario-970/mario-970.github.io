---
layout: post
title: "算子日报 · 2026-11-18"
permalink: /operator-daily/2026-11-18/
date: 2026-11-18 23:59:00 +0800
tags: [算子日报]
---

> 每日汇总：CUDA 开源仓 Release Notes（算子）、芯片动态、arXiv 算子内核论文。

## TL;DR

- **KV cache 的压缩开始往「不改 kernel」走**：Gemma 4 的 K=V 结构让一份 640 值行就能同时当 key 和 value cache，paged attention kernel 原样读取，32K–64K 上下文下 KV 显存降 1.34–1.40×；XM-Sparse 则在 scale-up 内存池上把热 KV 与 K-index 分别 offload，decode 吞吐比 PCIe 方案高 4.98×；RISA 用跨去噪步复用的 read plan 把 block-diffusion LM 的 32K attention 加速 2.67×。
- **量化要看「能不能按原生 kernel 跑」**：Qwen3-8B 在 B200 上的部署级实验表明 INT2 全部退化，最终 16-bit 激活 + NVFP4/INT8 混合配置在 144 个 fused linear 模块上验证原生 packed 执行后才算过关，TPOT 为 BF16 的 0.939；CPTC 把低 bit 解码压到两次乘法 + 64 字节表，已做进 Trainium 4 且吞吐等同原生 FP4/FP8。
- **编译器与微架构两侧都在提醒「可移植不等于可优化」**：Mirage 经 Triton 迁到 Ascend 910B3，融合负载最多快 87%，但形状相关回退明显；GPU LSU 的逆向测试给出 bank conflict 与 L1 tag lookup 的周期级延迟公式，并显示 MPS 下一个 warp 可把同机 cuBLAS matmul 拖慢最多 750×。

## 一、CUDA 开源仓 Release Notes（算子）

> 本期无新条目。窗口内只有 cuDNN Frontend 的 nightly 自构建：`v1.33.0.dev83259061`、`v1.33.0.dev83068401`（均 11-18）与 `v1.33.0.dev82913247`（11-17），正文自述 *Automated nightly build from NVIDIA-internal CI. Not a supported release.*，按既有口径不计入（dev82913247 已在 11-17 期注记）；30 天窗口内其余正式 release（CUTLASS v4.8.1、CCCL python-1.3.0、NCCL v2.32.6-1、cuda-pathfinder v1.9.0、cuda-core v1.3.0、cuda-python v13.5.0、cuda-quantum 0.17.0）均已收录于往期。

## 二、芯片动态

> 本期无新条目。两个 RSS 源在 7 天窗口内共返回 4 条，全部发布于 11-16：Rubin HPC 发布已收录于 11-17 期，MLPerf Training/Endpoints 榜单、CUDA-X 科学计算库与 CUDA 20 周年回顾也已在 11-17 期文末注记中判为非算子内核更新；窗口内没有新的芯片 / 架构动态。

## 三、arXiv 论文（算子内核）

> 本期增量窗口（提交日 11-15 起）返回 **150 条**，触及抓取上限；逐 id 与 11-17 期抓取结果比对后，真正新进索引的有 **74 条**：11-16 追补 18 条、11-17 提交 56 条（今日窗口内 11-15 的 25 条与 11-16 的其余 51 条均已在上一期审阅过）。按算子开发相关性精选 8 篇。

- **[Storing the Gemma 4 K=V Cache Once: Exact Global-KV Memory Savings in a Paged Serving Engine with Unchanged Attention Kernels](http://arxiv.org/abs/2611.23035v1)**（`2611.23035`，cs.LG/cs.CL/cs.DC，11-17）：Gemma 4 31B 的 global attention 层里 V 是归一化后的 K 投影、K 是 V 乘以固定权重、RoPE 只作用在四分之一维度上，但现有 serving 引擎仍按每个 KV head / token 存满 1024 个值。该工作只存一份 640 值行，利用两个重叠的 strided view 把它分别当作 key cache 和 value cache，**paged attention kernel 一行不改**；vLLM 原型中输出与旧 cache 上的同算术模拟逐位一致，32K–64K 上下文下每请求稳态 decode KV 显存降 **1.34–1.40×**，双 A100 + NIXL 的 prefill/decode 分离场景里 15 GiB 池的 decode 引擎每步 token 数为原版的 1.38×（16K prompt）。代价是同 batch 下 eager step 慢 4–9%，且目前无法与 8-bit KV cache 叠加。
- **[RISA: Recoverable Sparse Attention for Block-Diffusion Language Models](http://arxiv.org/abs/2611.22191v1)**（`2611.22191`，cs.CL/cs.LG，11-17）：block 式扩散 LM 的长上下文推理卡在 memory-bound attention 上——每个去噪步都要重读不断增长的 KV cache，而直接套稀疏 attention 会因不同 query 选中不同 page 放大整个 block 的读取集合，支撑集重建也有开销。RISA 利用「block 内历史 KV 固定、query 逐去噪步演化」的特点：新生成历史分成精确优先层与可恢复层，用 seed query 建一个受保护、有界的 read plan 并在整个 block 内复用，attention 仍用最新 query 与当前 block KV 重算。DreamReasoner-8B 上 32K 上下文 attention 快 **2.67×**、固定轨迹请求快 1.36×，128K 时加速 **5.53×**；三个模型八组设置下同 KL 均快于 MAGE / QUEST / LoSA。
- **[XM-Sparse: High-Throughput Sparse-Attention LLM Serving over Scale-up Memory Pools](http://arxiv.org/abs/2611.22280v1)**（`2611.22280`，cs.DC/cs.LG，11-17）：稀疏 attention 仍要保留完整 KV 历史，HBM 容量因此直接限制 serving 并发；把 KV offload 到主机又受 PCIe 带宽和内存容量双重限制。作者先指出即使有 X-Mem 这类 scale-up 内存池，瓶颈仍是 HBM 容量：热 KV buffer 和常驻的 indexer key（K-index）cache 占住大量 HBM，且两者访问模式完全不同。XM-Sparse 把两类 cache 都 offload，并各配一套 I/O：动态选中的 KV 条目用 **TMA 取回并与 attention 做 tile 级流水融合**以掩盖延迟，可预测的 K-index 用 SM 高效的 GPU 发起 DMA 预取。实现于 SGLang，DeepSeek-V3.2 / GLM-5.3 在 16K–1M 上下文下 decode 吞吐比 PCIe 的 ECHO / HiSparse 高 **4.98× / 4.52×**，比同平台朴素集成高 1.34×。
- **[The Hidden Cost of MoE Diffusion Decoding: Profiling and Fixing the Sampler](http://arxiv.org/abs/2611.21852v1)**（`2611.21852`，cs.LG/cs.AI/cs.CL，11-16）：masked diffusion LM 每次前向都要在整个 canvas×vocab 的 logits 上跑逐 token 采样与停止流水线，去噪 S 步就意味着每个输出 token 处理 S 倍的 logit。在 GB10 上剖析 26B 的 diffusion MoE 后发现：权重量化之后，**这条 logits 流水线而非 expert GEMM 才是 decode 的最大耗时项**。作者做了两级无损优化：bit-exact 级输出与参考解码器逐字节一致，accuracy-parity 级用单遍融合 entropy kernel + Gumbel-max 采样；再叠加 W4A16 expert kernel，单卡端到端达到 BF16 的 **2.4–2.7×**，GSM8K / HumanEval / MATH 精度不变。LLaDA2.0-mini 上复现（流水线占 decode 28.8%，W4A16 快 3.3×），而 dense diffusion 模型上该现象消失（仅 3.6%）。
- **[Beyond Bitwidth: Deployment-Constrained Mixed-Precision Quantization for Multilingual LLM Inference on NVIDIA Blackwell](http://arxiv.org/abs/2611.21974v1)**（`2611.21974`，cs.IR/cs.CL/cs.LG，11-16）：低 bit 量化只有同时满足「精度守住、走 GPU 原生 kernel、实测显存与延迟达标」才可用。作者在 B200 上用 vLLM 系统评估 Qwen3-8B 的 INT2/INT3、GPTQ、NVFP4 与混合精度配置，并对全部 **144 个 fused linear 模块**逐一确认原生 packed 执行与精度：packed INT2/INT3 能跑，但七种 INT2 网格全部在生成上退化；隔离模块的恢复效果强烈非可加，韩语与推理的敏感度排序甚至相反。最终配置为 16-bit 激活 + 98 个 NVFP4 weight-only 模块 + 46 个 INT8 模块 + INT8 输出头，在未参与选择的 FINAL-81（5,168 条）上全部门槛通过：权重加载显存 **5.917 GiB**（上限 5.98）、韩语教师正确保留率 0.904（门槛 0.90）、TPOT 为 BF16 的 **0.939**，成对准确率比 NVFP4 W4A4 高 1.4 个百分点。
- **[Computed Permutation Trellis Coding: Near-Optimal Quantization from Two Multiplications and a 64-Byte Table](http://arxiv.org/abs/2611.22019v1)**（`2611.22019`，cs.IT/cs.AI/cs.AR/cs.LG，11-17）：INT4/FP4 这类逐值舍入格式浪费了相当一部分比特，trellis 码联合编码能补回来，但已有解码器要么需要大查找表、要么用哈希限制可解码范围。CPTC 的解码器只做两次乘法、取乘积高位查一张 **64 字节表**；由于这个和不均匀分布，小表两端恰好能放尾部数值。同一解码器覆盖 1–6 bit，能精确复现标准整数与浮点格式，4 bit 时距理论极限仅 0.07 bit，匹配查找表大数千倍的随机置换 trellis 码并全面超过 QTIP codebook；Llama 3.1 8B/70B 无微调权重量化下各 bit 宽 KL 最低（最多好 27%）。解码器已做进 **AWS Trainium 4**，吞吐等同原生 FP4/FP8、只占脉动阵列可忽略面积；融合编码 kernel 单 GPU 两分钟内量化 8B 模型线性层。
- **[How well does graph superoptimization transfer? Adapting tensor-graph superoptimizer to a Triton-based NPU](http://arxiv.org/abs/2611.22844v1)**（`2611.22844`，cs.PF/cs.AR/cs.PL，11-17）：把面向 CUDA 的 Mirage 经 Triton 编译栈迁移到华为 Ascend 910B3，考察图级超优化能否跨架构迁移。测试覆盖 matmul、softmax、RMSNorm+MatMul、gated MLP、FNN 与 attention：融合负载上最多比参考实现快 **87.0%**，但性能高度依赖 tensor shape 与负载结构，部分配置出现明显回退。作者把限制归结为目标特定张量切分、执行资源建模、后端 lowering/fusion 以及搜索与验证成本——**可移植的图表示与编程模型不等于可移植的优化空间**，需要在可复用编译基础设施里保留模块化的目标特定机制。
- **[Dissecting the GPU Load/Store Unit and Exposing Its Security Risks](http://arxiv.org/abs/2611.23022v1)**（`2611.23022`，cs.AR/cs.CR，11-17）：用微基准逆向 NVIDIA GPU 的 load/store unit，给出 global L1 hit 与 shared memory 访问的**周期级延迟公式**：地址相关项就是 bank conflict 与 L1 tag lookup，且 load 会在这两处 contention；LSU 流水打满时，后启动 block 的 warp 会被先来的 warp 饿死。安全侧，仅靠 global bank conflict 或 tag lookup 就能在既有 GPU AES 计时攻击的威胁模型下恢复完整 AES-128 密钥；MPS 下两者都能形成跨进程隐蔽信道且速率逐代上升， starvation 还能让一个 warp 把同机 cuBLAS matmul 在 A100 上拖慢最多 **750×**、H100 上 **563×**。对算子开发者而言，这条延迟模型也是解释 shared memory 与 L1 访问性能边界的直接依据。
