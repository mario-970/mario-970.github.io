---
layout: post
title: "算子日报 · 2026-10-23"
permalink: /operator-daily/2026-10-23/
date: 2026-10-23 23:59:00 +0800
tags: [算子日报]
---

> 每日汇总：CUDA 开源仓 Release Notes（算子）、芯片动态、arXiv 算子内核论文。

## TL;DR

- **低精度 GEMM 的瓶颈被拆到比特层面**：FIGFLEX 用 radix-4 shift-add 把 FP-INT 乘加融成统一的部分积累加、部分积数量约减半，并针对 INT3 回收闲置数据通路；LLaMA-2-7B 在 3/2-bit 平均精度下 WikiText-2 perplexity 为 **5.85 / 8.77**，线性 GEMM 每面积吞吐比 AxCore、FIGLUT 高 **1.56× / 1.42×**。
- **稀疏 attention 的「尾部」有了更便宜的近似**：OpenAttention 用 count-aware mean proxy 代替对未读尾部的采样，Llama-3.1-8B 在 RULER-32K HARD 上 **1% 读取保留 99% dense 精度**，定制 decode kernel 比 dense attention 快 **10×**；RCR 则把 KV 重选交给 drift 触发、由 per-sample 控制器控制频率，比 InfiniGen 高 **1.6–3.8 ROUGE-L**。
- **kernel 工程侧**：三维 Poisson multigrid 把 patch 更新放进寄存器、用 warp shuffle 交换，数据搬运达普通 device copy 带宽的 **87%**；Apple Silicon 评测显示 CUDA kernel 设计只能部分迁移——Metal 缺少 async copy、batch-one decode 让多项 serving 机制失效。

## 一、CUDA 开源仓 Release Notes（算子）

### cuda-python

- **cuda-pathfinder v1.8.4**（10-23）— [链接](https://github.com/NVIDIA/cuda-python/releases/tag/cuda-pathfinder-v1.8.4)：新增对 **cuDSS 1.0** 动态库名 `libcudss.so.1` / `cudss64_1.dll` 的识别，同时保留 cuDSS 0.x 旧名；加载 cuDSS 时会按 1.0 的要求预加载可用的 NVRTC。另新增 Python 3.15 支持（并因 EOL 移除 3.10）。对算子的落点是**稀疏直接求解器**：通过 cuda-python 定位 cuDSS 1.0 的路径被修通，Windows 上尤其直接受益。release 正文只有文档/PyPI/conda 链接，以上内容取自仓库内的 `1.8.4-notes.rst`。

> 本期其余更新均不计入：`cudnn-frontend` 新增 `v1.32.0.dev76128831`（10-23），正文自述 *Automated nightly build from NVIDIA-internal CI. Not a supported release.*，延续既有口径；前一天的 `dev75836140` 已在上期说明。30 天窗口内的其他正式 release（CUTLASS v4.8.1、CCCL v3.5.0 / v3.4.3 / python-1.2.1、TensorRT v11.4、nccl4py v0.6.0、cuda-python v13.4.4 / cuda-core v1.2.2）均已在往期各期收录。

## 二、芯片动态

> 本期无新条目。NVIDIA Developer Blog 源继续返回 0 条；NVIDIA Blog 在 7 天窗口内唯一命中的仍是 10-21 的 Google Cloud × Vera Rubin GA，已在上期收录。本节连续为空属**数据源侧**状况，不是抓取故障。

## 三、arXiv 论文（算子内核）

> 本期增量窗口覆盖 10-20 至 10-22 共 **140 条**（未触及 `max_results` 上限）。按 id 与往期各期文章比对后，从尚未收录的条目中精选 6 篇：10-22 提交 4 篇，10-21、10-20 各补录 1 篇（arXiv 索引滞后，上一期窗口左端被 150 条上限截断）。

### 低精度与 GEMM

- **[FIGFLEX: A Mixed-Precision FP-INT GEMM Accelerator Co-Designed with Low-Bit LLM Quantization](http://arxiv.org/abs/2610.30480v1)**（`2610.30480`，cs.AR/cs.LG，10-21）：weight-only 量化把权重压到低比特整数、激活保持浮点，主力算子随之变成 **FP-INT GEMM**；但现有 GPU 要先反量化到 FP16 再进 tensor core，比特并行加速器又只支持单一权重精度——省了内存却没省计算。FIGFLEX 用 **shift-add、radix-4** 重写 FP-INT 乘法，把乘加融合成精度可变的统一部分积累加、部分积约减半，并用 INT3 优化回收 radix-4 在该精度下闲置的数据通路；配套量化器按「GPTQ 无法补偿的误差」给每个 channel group 分配比特，per-input-channel 的 2 的幂列缩放把激活挡在 FP16 次正规范围之外，使量化模型能直接映射到该数据通路。LLaMA-2-7B 上 3/2-bit 平均精度的 WikiText-2 perplexity 为 **5.85 / 8.77**，低于 GPTQ、AWQ、SliM-LLM、AMQ、QuIP#（均未微调）；3-bit 混合精度模型的线性 GEMM 上，整芯片每面积吞吐比 AxCore、FIGLUT 分别高 **1.56× / 1.42×**。对算子侧的启示是：**反量化开销和 FP16 次正规下溢**会直接吃掉低比特的收益，量化格式与 kernel 数据通路需要一起设计。
- **[Transformers as Generic Deep Neural Networks: Exact Compilation, the Cost of Removing Activation-Activation Products, and Multi-Agent Attention Clustering](http://arxiv.org/abs/2610.28958v1)**（`2610.28958`，cs.LG/cs.MA，10-20）：作者证明每个 Transformer block（attention、LayerNorm、GELU MLP）都能**精确编译**成由固定仿射映射、routing、square/sigmoid/rsqrt 加上 GELU 组成的通用 DNN，因此 NPU 不必为 attention/normalization 造专用算子、量化也不必依赖 Transformer 专用算法。代价可以量化：若不允许「activation-activation 乘积」这一种逐元素运算，按 transform 编译时每个 head 处理 dense score 至少多出 **N² 次标量非线性求值**；保留它（survey 的 NPU 都有）即可消掉这项开销。对 top-k attention 的可交换尾部，作者证明 exact / conditional-mean tail mass 是无偏的 MMSE 估计，而直接丢弃尾部是有条件有偏的；当 query 或 agent 成簇时，先用质心选 key，可把每 head 计算分数从 **N² 降到 Nk**。DeiT-Tiny、MambaIRv2-Light 与 Pythia-70M 上编译模型在全 key 下与参考实现一致，标准 PTQ 可直接套用；DeiT-Tiny 及 Pythia-70M 的 k≤8 设置下，精确 tail-mass 估计用一半 key 就优于 tail dropping。

### 稀疏 attention 与 KV cache

- **[OpenAttention: One Percent Is Enough with Tail Proxies](http://arxiv.org/abs/2610.30951v1)**（`2610.30951`，cs.LG/cs.AI，10-22）：稀疏 attention 只读 KV 的一小部分，真正的难点是**未读尾部怎么处理**：丢掉会有偏，采样在 5% 以下读取率会失效（尾部很可能藏着 selector 漏掉的重击 token）。作者发现最简单的 proxy——把整个尾部折成一个 **count-aware mean token**——在每种读取率下都比采样误差更低，并从闭式上说明低读取率下采样的权重误差近乎丢弃的两倍、单 proxy 始终更低。OpenAttention 由此组合 clustering、ErrorGate 与 top-k selection，再加 count-aware mean proxy；训练无关的 Query-Geometry Clustering/Selector 在压缩 key 时尊重 query 方向，ErrorGate 按各簇的估计权重误差分配精确读取量，而不是只看质心排名。Llama-3.1-8B 在 RULER-32K HARD 上以 **1% 读取保留 99% dense 精度**，跨模型/上下文长度/数据集均达 SOTA；定制 kernel **OpenAttention.flash** 在 1% 读取下 decode attention 比 dense 快 **10×**。
- **[Drift Says When, Budget Says How Much: Rate-Controlled KV Re-selection for Long-Form Multimodal Generation](http://arxiv.org/abs/2610.31433v1)**（`2610.31433`，cs.LG，10-22）：多模态 LLM 的 KV 压缩通常在 prefill 时一次性选好要保留的 visual token，但长文生成中模型读取的证据会随描述推进而移动——即使 oracle 用未压缩答案的 attention 来选，与未压缩输出的 ROUGE-L 一致度也只有 **0.44–0.52**。固定 drift 阈值重选又会系统性失败：drift 幅度跨样本不可比，低 drift 样本几乎从不刷新，而它们收益最大。RCR 让 **drift 决定何时、何处重选**，另用 per-sample 控制器调节刷新频率；配合低秩 drift detector、候选受限重选，以及**留在 GPU 上完成决策**。统一一套超参下，六个模型-任务设置中有五个比 InfiniGen 高 **1.6–3.8 ROUGE-L** 点、第六个无显著差异，同时与逐步精确重选相差不超过 1.7 点，每步估计数据移动相当，GPU 常驻 detector 小 **4×**；在 DetailCaps 上，drift 定时比同刷新率的随机定时高 **2.0–5.0** 点。

### GPU kernel 工程

- **[Mixed-Precision Coalesced Matrix-Free Multigrid with Vertex-Patch Smoothing](http://arxiv.org/abs/2610.31496v1)**（`2610.31496`，math.NA，10-22）：三维 Poisson 方程的几何多重网格求解器，用连续三次元的 multiplicative vertex-patch Schwarz 光滑子做精确局部求解。数据布局是**持久的 cell-wise 存储**——每个 cell 持有共享自由度的自己那份拷贝；GPU 上 patch 更新在**寄存器**里算、一个 cell 一个线程，patch 内 cell 之间用 **warp shuffle** 交换，几乎不用 shared memory。尽管占用率低，patch 更新的算术吞吐接近 FP32 峰值；再用 fast diagonalization、合并变换与 patch 的反射对称性压缩运算量，一次 sweep 的数据搬运达到普通 device copy 带宽的 **87%**。外层双精度迭代里套单精度层级，并用 **rebalancing**（在舍入前用双精度交换共享自由度的正负拷贝以完成抵消）保住双精度级别的收敛率，且与网格规模无关。对写 stencil/multigrid 类访存密集算子的人，这是一份「寄存器驻留 + warp shuffle + 混合精度误差抵消」的完整工程样本。
- **[Evaluating the Applicability of LLM Inference Optimizations on Apple Silicon](http://arxiv.org/abs/2610.31479v1)**（`2610.31479`，cs.LG/cs.DC，10-22）：在四台 Apple Silicon 机器上、以统一的单请求 decode 负载评估四类优化（KV-cache pruning、投机解码、移植 GPU kernel、训练后量化），并与 RTX 4090 做匹配比较。结论对「优化是否可移植」很有参考价值：KV pruning 在 Apple 上最高带来 **1.75×** 端到端吞吐，却可能降低桌面 NVIDIA GPU 的吞吐；所有被测投机解码方法在 Apple 上都**降低**吞吐，尽管在服务器 GPU 上报告过 2–6× 提升；CUDA kernel 设计只能部分迁移——公开的 Metal 没有 async copy，batch-one decode 又让多项 serving 机制不适用，但可迁移部分仍带来最高 **1.20×** decode 吞吐；量化收益则因机器而异。作者的核心结论是：**报告的加速比不是技术本身的属性**，而是技术、runtime、硬件契约与负载形态的联合函数——跨平台算子移植时应先核对这几项，而不是照搬 CUDA 上的 kernel 结构。

> 其余新入库条目多为应用/训练方法向（医疗影像、EEG、机器人、3D 高斯、遥感、推荐与安全攻击等）或与 GPU 算子无直接关系的理论工作，未收录。
