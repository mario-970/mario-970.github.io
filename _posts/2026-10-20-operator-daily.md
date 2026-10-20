---
layout: post
title: "算子日报 · 2026-10-20"
permalink: /operator-daily/2026-10-20/
date: 2026-10-20 23:59:00 +0800
tags: [算子日报]
---

> 每日汇总：CUDA 开源仓 Release Notes（算子）、芯片动态、arXiv 算子内核论文。

## TL;DR

- **本期增量覆盖 10-17 至 10-19 三天共 90 条 arXiv 论文**（上期只覆盖到 10-16，10-17、10-18 两批此前未出刊），未触及 150 条抓取上限，逐 id 与往期各期比对无重合，精选 11 篇。发布面只有 `cudnn-frontend` 的 nightly CI 构建，芯片 RSS 无命中。
- **稀疏 attention 与低精度仍是主线**：DASA（自回归视频 DiT 2.16×）和 RECAP（HunyuanVideo 3.05×）分别从动态区域选择与缓存纠错切入，Certified Sparse Attention 则把每个 query 的读取误差做成可证明的停止条件；量化侧覆盖 MX-like Tensor Core 的 scale 格式选择、NVFP4 LoRA 训练、NVFP4 预训练配方，以及量化内积的存储/读取预算分离。
- **kernel 与数值侧**：Polar Switch 用多项式电路减少 Muon 正交化的对称 matmul 次数（方阵快 15.5–18.7%），张量缩并聚合给出指数优于 Strassen 的中等规模快速 matmul 构造，混合精度 TRMM 在 A100/GH200 上达到 FP64 峰值的 6.36–11.03×；另有手机 NPU 上的 LoRA input-gradient kernel。

## 一、CUDA 开源仓 Release Notes（算子）

> 本期无新正式 release。窗口（≥ 10-17）内 10 个仓库只出现 `cudnn-frontend` 的两个 nightly 自构建——`v1.32.0.dev74996650`（10-19，24 个 wheel 缺 3 个 win_arm64）与 `v1.32.0.dev75272740`（10-20，24/24 齐），正文均自述 *Automated nightly build from NVIDIA-internal CI. Not a supported release.*，与上期处理一致，不计入。

## 二、芯片动态

> 本期无新条目。两个 RSS 源本次均未返回命中芯片关键词的新帖。

## 三、arXiv 论文（算子内核）

> 本期为 **10-17 至 10-19 的 90 条增量**（24 / 26 / 40 条），`max_results=150` 未截断；逐 id 与往期各期比对无重合。按算子开发相关性精选 11 篇。

### kernel、GEMM 与数值

- **[Polar Switch: Fast Muon Orthogonalization with Polynomial Circuits](http://arxiv.org/abs/2610.26742v1)**（`2610.26742`，math.NA/cs.LG/math.OC，10-19）：Muon 用 Newton–Schulz 多项式在 GPU 上正交化梯度矩阵；Polar Switch 把迭代写成**多项式电路**——在少量方形 buffer 上做一串对称矩阵乘，`k` 次乘积对应最高 `2^{k+1}+1` 次的奇多项式，再按 FLOP 预算（计入对称 matmul kernel 成本）与半精度舍入约束，数值搜索乘积顺序、系数与电路组合。方阵上比 Polar Express 快 **15.5–18.7%**，长宽比 3.25/3.5/8 的矩形矩阵上比 Gram Newton–Schulz 快 **8.6–16.2%**，精度相当；Modded-NanoGPT #91、Qwen3-0.6B、Gemma 3-1B 最终 val loss 持平，1400 万个矩阵上谱范数越界不超过 3%。对算子侧的直接含义是**每次 Muon step 的对称 matmul 次数更少**，且半精度路线有明确的稳定性约束。
- **[Simultaneous Matrix Multiplication by Tensor Contraction Aggregation](http://arxiv.org/abs/2610.26563v1)**（`2610.26563`，math.NA，10-19）：把 Coppersmith–Winograd 快速矩阵乘法解释为**张量缩并的聚合**，并给出把多个独立矩阵乘精确聚合的新构造。由此得到一批中等规模、指数优于 Strassen 的显式算法（并未逼近 `ω` 的最好界），同时改进若干固定数量独立方阵乘的 rank / border rank 上界。作者还按误差模型量化了有限精度可用性：border-rank 构造以保留至少 **1/2、1/3、1/4** 精度为代价换成本下降。对算子侧是「批量独立 matmul 能否共享缩并」的一条算法线索，但精度折损必须在 kernel 设计时显式核算。
- **[Mixed-Precision GPU Acceleration for Confidence Regions of Spatial Excursion Sets](http://arxiv.org/abs/2610.25039v1)**（`2610.25039`，cs.DC，10-18）：面向空间置信域的 GPU 框架，把自适应 tile 混合精度用在 Cholesky 与 **TRMM** 上：三角矩阵乘继承 Cholesky 因子的分块精度布局、模拟操作数用 FP16，再用协方差 scaling 与补偿性 rescale 撑住数值范围。4×A100 上混合精度 TRMM 达到理论 FP64 峰值的 **6.36–6.93×**，4×GH200 上 **10.20–11.03×**，覆盖度与双精度参考接近。给出了一条「FP16 操作数 + 分块精度继承 + 补偿缩放」的 TRMM/GEMM 类算子模板，但收益绑定在低精度可容忍的统计场景。

### 稀疏 attention

- **[Dynamics-Aware Sparse Attention for Efficient Autoregressive Video Diffusion](http://arxiv.org/abs/2610.26262v1)**（`2610.26262`，cs.CV，10-19）：自回归视频 DiT 的 attention 在时空上均匀分配算力，而运动通常只占少区域。DASA 借视频编解码思路把内容分成动态/静态区：历史 KV cache 只保留当前生成 chunk 内的动态特征以压显存，计算侧按局部动态强度做 **region 级配额**，再用轻量蒸馏训练补偿。VBench 上端到端 **2.16×** 且保真度保持。对算子的重点是「动态区域掩码 + KV 选择性保留」同时作用于访存与计算两条路径，且掩码粒度是 region 级、便于映射到 block-sparse attention kernel。
- **[RECAP: Reciprocal Sparse Attention and Hierarchical Caching for Efficient Video Diffusion](http://arxiv.org/abs/2610.26244v1)**（`2610.26244`，cs.CV/cs.LG，10-19）：稀疏 attention 与跨去噪步缓存直接组合时，稀疏估计的误差会沿采样轨迹被缓存复用放大。RECAP 观察到相邻步的稀疏化误差方向一致，于是用在线误差反馈做互惠校正：缓存的稠密精修补上当前稀疏估计漏掉的交互，新的稀疏统计跟踪输入变化并调节旧校正的衰减，校正后的预测再作为整步复用的源。免训练，HunyuanVideo 720p/5s 上 **3.05×**，保真度和速度都优于直接组合。对算子侧的提示是稀疏 attention 可以把「遗漏块的稠密补偿」与缓存管理做成一条流水线，而不是两个独立优化。
- **[Certified Sparse Attention via Perturbed Retrieval](http://arxiv.org/abs/2610.24991v1)**（`2610.24991`，cs.LG/cs.CL，10-18）：现有稀疏 attention 只控制读多少 key、不控制单个 query 的误差，且真实 KV cache 上误差重尾。本文给每个 key 的分数加独立 Exp(1) 扰动，使阈值检索等价于**按 attention 权重采样**；沿几何阈值网格下降得到嵌套样本，用 Horvitz–Thompson 估计与同时置信界，每个 head 在首个满足目标误差 `ε` 的层级停止，整个生成过程违约概率 ≤ `δ`。读取量 `O(ε^{-2} log(1/δ))` 与上下文长度无关，且同阶意义下最优。实现上用 **4-bit 扫描 + 确定性界**并在 GQA 组内共享候选（这一步仍要读全部 key）；3 个 LLM、16K–128K 十个设置零违约，等 key 读取量下 90 分位误差比 15 个基线家族中最好的低 **4.6–7.9×**。摘要只给了 CPU decode 1.1–1.4×，GPU kernel 收益待验证。

### 量化与低精度

- **[MX-Like Tensor Core for LLM Inference with NPU Integration via Hardware-Software Co-Design Exploration](http://arxiv.org/abs/2610.26083v1)**（`2610.26083`，cs.AR，10-19）：在开源 SNAX NPU 上系统扫描 MX-like 块缩放格式的设计空间：把 **8-bit scale 格式族**与不对称 weight/activation 精度联合起来，为每种格式建专用 MAC 数据通路模板以公平比较硅成本，并接自动化软硬协同流程。结论是 scale 格式只在**激进 FP4** 下才显著影响精度；**UE5M3** 在 FP4 区间相对无 per-tensor scaling 的 NVFP4-like 基线把困惑度退化最多降 **25.3%**，代价仅 +0.06% 面积、+0.49% 功耗。对 NVFP4/MXFP GEMM 的 scale 格式与 MAC 数据通路选择有直接参考，但结论在 NPU 侧，迁移到 Tensor Core 需重新核算。
- **[PACT: NVFP4 Data-Flow LoRA Fine-Tuning on Consumer GPUs](http://arxiv.org/abs/2610.25048v1)**（`2610.25048`，cs.LG，10-18）：在消费级 GPU 上做 NVFP4 LoRA：冻结投影走**原生 NVFP4 GEMM**，同时压掉量化/反量化开销；复用已量化的 RMSNorm 输出，并把 gate/up 激活以 NVFP4 存到反向；前向非线性保持 BF16 输入，冻结权重用适配 NVFP4 scale 与量化激活的 GPTQ 离线校准。再配融合非线性 kernel 与前/反向权重布局缓存。16 GB RTX 5080 上多模型多任务显存更低、微调更快且精度有竞争力。对算子侧的价值在于把 NVFP4 从推理推进到训练：**激活在反向也可低精度存，但非线性入口必须保 BF16**。
- **[What Matters for NVFP4 Training? A Comparative Study of Low-Precision Pretraining Recipes](http://arxiv.org/abs/2610.24343v1)**（`2610.24343`，cs.CL/cs.AI/cs.LG，10-17）：在稠密 8B 与 30B-A3B MoE、最多 1T token 上系统比较 NVFP4 训练配方，并提出一套最终配方：采用比 stochastic rounding 量化误差更低的**无偏梯度估计**，同时保留**选择性高精度层**。消融显示配方中每项都改善 loss 轨迹；高精度层在规模化时是恢复精度的必要条件；反向传播中并非所有张量都同等受益于去偏，剩余张量还有互补的误差抑制空间。这是该规模下相对 BF16 loss gap 最强的 FP4 训练结果，直接回答「哪些张量/环节必须留在高精度」这一 FP4 训练 kernel 的分层精度设计问题。
- **[Query-Dependent Reading for Quantized Inner-Product Estimation: A Minimax Separation](http://arxiv.org/abs/2610.24890v1)**（`2610.24890`，cs.IT/cs.DS/cs.LG，10-17）：把量化内积拆成「存储预算 `w`」与「读取预算 `r<w`」两个独立约束：在静态 bit-probe 模型下证明 **query-independent 读取无法从额外存储获益**，归一化 minimax 误差收敛到 `1/(4^R−1)`、与 `W` 无关；而当读取可以依赖 query 时额外存储才有用——用高斯投影与随机阈值的 two-batch 协议可让误差随 `W/R` 增长而消失，并给出有限维下界，双向稀疏区间内只差 `π/2` 的常数因子。对量化 dot-product / 压缩 KV 的 kernel 设计含义是：**与其多存比特，不如把读取做成 query-dependent 的随机探测**；但这是信息论结果，没有实现与 GPU 数据。

### 边缘与 NPU

- **[First-Order LoRA Training with Phone NPUs](http://arxiv.org/abs/2610.24816v1)**（`2610.24816`，cs.LG/cs.AI，10-17）：手机上做 LoRA 微调时，冻结权重的前向、激活重算与 input-gradient 三个乘积仍是大头，而 NPU 的推理接口不支持 input gradient。系统把这三个乘积全部放 NPU，训练图、adapter 与优化器留在 CPU；**input-gradient kernel 复用 NPU 常驻的权重 tile 与 forward dequantizer**，避免转置权重拷贝，并保证前向/反向使用同一套有效权重。为保住梯度，按梯度敏感度选精度并对 FP16 输出做 **per-row range guard**；其余 CPU 算子融合，NPU 与 adapter 计算重叠，支持可恢复执行与热节流。两代 Snapdragon、四个 Qwen/Llama 模型上验证。对边缘算子侧的参考是「推理常驻权重 + 反向复用」的 kernel 组织方式，以及 FP16 梯度输出的逐行防溢出。

> 未收录条目多为应用/训练方法向（医疗影像、EEG、自动驾驶、3D 高斯、检索与 RL 等）或与 GPU 算子无关的理论工作，与算子内核无直接关系。
