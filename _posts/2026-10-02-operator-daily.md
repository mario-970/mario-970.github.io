---
layout: post
title: "算子日报 · 2026-10-02"
permalink: /operator-daily/2026-10-02/
date: 2026-10-02 23:59:00 +0800
tags: [算子日报]
---

> 每日汇总：CUDA 开源仓 Release Notes（算子）、芯片动态、arXiv 算子内核论文。

## TL;DR

- **「舍入规则该选哪个」今天有了一个按位置分配答案**：SR（随机舍入）与 RN（就近舍入）之争不是全局开关问题。线性投影的概率前向误差界显示，SR 的误差包络按 **O(√n·u)** 随归约长度增长，RN 是 **O(n·u)**——低精度下差距迅速拉开，在 MLP 的长 down-projection 上最明显。实测 DistilGPT-2（尾数 t=6）：MLP 里用 SR 把困惑度压到全精度的 **1.15×**，RN 是 **2.21×**；但在 **LM head 上次序反转**，因为 MLP 噪声主要是**均匀的 logit 平移**（softmax 对其不变，SR 的方差被 discount 掉），head 噪声在词表上非均匀。混合配置（MLP 用 SR、head 用 RN）落在 **1.10×**，比同比特 RN 好 **28%**。
- **同一追问下沉到电路层**：**ZTA-Q** 是一个开源 RISC-V + TFLite INT8 推理平台，把 requantization 的后处理数据通路做成可配置，用来研究**乘数位宽降低、共享移位缩放、简化舍入**这三种电路级近似各自对精度的影响——在 Arty A7-100T 上把 top-1/top-5 掉点限制在 **0.25 个百分点**内。它与上面那篇是同一问题的 FP 侧与 INT8 侧镜像。
- **稀疏侧两篇都在改「稀疏以什么粒度、什么粒度上记账」**：**VASC** 指出块稀疏检索的第二个失配——**「被高度注意」不等于「有价值」**，于是把 value 对比度加进块打分，并引入**跨层「欠服务额度」记账**，让被固定预算长期压住的块在后续层里有竞争机会（比 dense VGGT 快 2.29×）；**MWOP** 把 attention 的稀疏粒度从 head 细化到**模态路径（V2V/T2V/T2T）**、FFN 通道按模态分别选，并落成 **path-sparse Triton kernel**（LLaVA-OneVision-7B 上 prefill 1.6×、12 个基准平均保留 99.7% 性能）。

## 一、CUDA 开源仓 Release Notes（算子）

### cuda-python

- **cuda-pathfinder v1.8.3**（2026-10-02）— [链接](https://github.com/NVIDIA/cuda-python/releases/tag/cuda-pathfinder-v1.8.3)：CUDA 工具链组件定位库（定位 headers / 动态库 / bitcode / 静态库 / 二进制工具），本次新增**支持 Windows 上 CUDA 13 的 cuSPARSELt DLL 名 `cusparseLt_13.dll`**，使 `load_nvidia_dynamic_lib("cusparseLt")` 能识别 **cuSPARSELt 0.10.0** 的 wheel，同时保留对旧名 `cusparseLt.dll` 的支持（PR #2982）。对算子的落点很直接：**在 Windows 上通过 cuda-python 取结构化稀疏（2:4）矩阵乘库的路径被修通**，此前 0.10.0 wheel 会因 DLL 改名而定位失败。注意发布正文本身只有文档/PyPI/conda 链接，无 changelog，上述内容取自仓库内的 `1.8.3-notes.rst`。

> 同一窗口内 `cudnn-frontend` 的三条 `v1.31.0.dev…`（09-27～09-29）仍是 NVIDIA 内部 CI 的 nightly 自构建（自述「Not a supported release」、只保留最近 14 个 tag），与前几期一致不计入；CUTLASS v4.8.0、TensorRT v11.3、NCCL 2.32.3-1 / nccl4py-v0.6.0、CCCL v3.4.3 与 python-1.2.1、cuda-python v13.4.3 / cuda-core-v1.2.1、cuda-quantum 0.16.0 等正式发布均已在 09-12～09-30 各期收录。

## 二、芯片动态

> 本期无新条目。7 天窗口内两个 RSS 源只有一条命中关键词的更新——NVIDIA Blog 的《How NVIDIA GPUs Help Accelerate OpenAI's GPT-6 Astra Ultrafast》（10-01），内容是 **OpenAI 模型上线**：运行在 Blackwell 上，靠推理优化拿到最高 **8×** 的 token 生成加速。它属于**应用/产品发布**，既无新架构也无 kernel 级细节，按惯例不计入芯片动态。

## 三、arXiv 论文（算子内核）

- **[Stochastic Rounding in Low-Precision Transformer Inference: A Variable-Precision Emulation Study of a Small GPT-2](https://arxiv.org/abs/2610.01889)**（`2610.01889`，10-01）：低精度推理到底该用**随机舍入（SR）还是就近舍入（RN）**？这篇的答案是「**取决于它落在网络的哪个位置**」，做法是把数值格式**固定住**，只改单个运算是点上（operation site）的舍入规则。为在任意精度上做实验，作者把矢量化的 PRISM 舍入库扩展到任意虚拟精度（VPSR 算法），并**证明舍入判定可在硬件浮点下被精确求值**。随后给出两条互补分析：其一，线性投影的**概率前向误差界**表明 SR 的误差包络随归约长度 n 按 **O(√n·u)** 增长，RN 是 **O(n·u)**，该差距在低精度下迅速拉大，并在 MLP 的长 down-projection 上最突出；其二，在输出 softmax 处把期望交叉熵变化做二阶分解为**带符号漂移、漂移曲率与 Fisher 加权方差罚项**，这解释了为什么两个位置行为相反——**MLP 的噪声主要是一个均匀的 logit 平移，softmax 对其不变，于是 SR 的方差被大部分 discount 掉；head 的噪声在词表上非均匀，不被 discount**。实验与理论吻合：DistilGPT-2 在 **t=6** 位有效位下，MLP 用 SR 使困惑度升到全精度参考的 **1.15×**，RN 则是 **2.21×**；而在 LM head 上次序反转，因为 SR 引入了非均匀方差、确定性 RN 不引入。混合精度配置（MLP 输出 t=6、MLP 用 SR、head 用 RN）把困惑度拉回 **1.10×**，相对同比特 RN **降低 28%**。对写低精度 GEMM 与量化 kernel 的人，两条直接可用的结论：**舍入规则是 per-site 的调度决策，不是全局开关**；以及**误差随归约长度的增长指数（√n 对 n）决定了长归约的位置该选谁**——统一用 RN 会在长归约上付出更高的代价，统一用 SR 又会在输出层引入非均匀方差。需注意其结论来自**仿真与理论**（小型 GPT-2、PRISM 仿真），尚未在真实 kernel 上实测。

- **[VASC: Value-Aware Sparse Attention with Cross-Layer Memory for Efficient 3D Reconstruction](https://arxiv.org/abs/2610.01013)**（`2610.01013`，10-01）：VGGT 这类前馈 3D 视觉模型把相机估计与稠密场景重建压进**单次前向**，但**全局注意力的二次复杂度**让长图像序列变得昂贵，而现有稀疏方法可能**偏向「被高度注意但 value 冗余」的区域**。**VASC** 是免训练的稀疏 attention：**value 感知的块选择**把池化的 query–key 相关性与**邻域 value 对比度**结合起来，在保留 query 相关且内容独特的部分的同时削掉冗余；**执行感知的跨层记忆**则**跨层追踪「未被服务的需求」**，并按实际执行结果更新该状态，使**此前被欠服务的块能在固定计算预算下参与竞争**。在 7Scenes 与 NeuralRGB-D 上、对 VGGT 与 π³ 的评测中，姿态估计与重建质量均优于 FasterVGGT，推理比 dense VGGT 快 **2.29×**；代码已开源于 `github.com/kosakayamahoo-design/VASC`。与昨日收录的 PARK 属同一条修正路线（PARK 修的是「检索度量」，VASC 加的是「**注意力高不等于价值高**」这一条），更值得注意的是它的第二个机制：把「每层独立按预算取 top-k」换成**跨层配额的记账与调度**——代价是**调度状态进了 kernel 的热路径**（需按层维护、按实际执行回写），这与昨日 SparseEngine 的跨请求状态管理、Vosti「kernel 选择不得依赖运行时状态」构成一组张力：**稀疏化收益越大，越需要跨层/跨请求的可变状态，而可变状态正是确定性验证的敌人**。

- **[MWOP: Modality-aware Width-wise Operation Pruning for Efficient MLLMs](https://arxiv.org/abs/2610.01434)**（`2610.01434`，10-01）：MLLM 处理长图文序列时推理代价高昂，现有的「算子压缩」利用了模态级冗余，但**把 attention head 内部与共享 FFN 通道上的计算当作统一单位**，更细的冗余没被挖。作者的观察是：**同一个 attention head 内，不同模态交互路径（V2V / T2V / T2T）的冗余程度不同**；**同一个 FFN 通道在视觉与文本执行下的重要性也不同**。**MWOP** 据此**逐层独立地剪 V2V/T2V/T2T 三条 attention 路径**，并**为视觉与文本输入分别选择 FFN 通道**；剪枝准则是一阶 Taylor，**attention 剪完后重新评估 FFN 重要性**，再用 LoRA 做恢复训练。为把这种细粒度稀疏变成真实加速，作者实现了 **path-sparse Triton attention kernel** 与**紧凑的视觉侧 FFN 执行**。因为**保留 token 序列**，它与 token 压缩类方法正交、可叠加：在 LLaVA-OneVision-7B 上，单独使用 MWOP 取得 **1.6× prefill 加速**、12 个基准**平均保留 99.7%** 性能；与两种代表性 token 压缩方法叠加后，把它们的 prefill 加速从 **2.0× / 1.9×** 提升到 **2.9× / 2.7×**，在 Qwen2.5-VL-7B 上也成立；代码开源于 `github.com/EIT-NLP/MWOP`。对算子组而言，这里的可迁移点是**稀疏的粒度与时机**：把可剪单元从 head/通道细化到「head 内的模态路径」，得到的是一个**编译期确定的结构化稀疏模式**，而不是运行期的动态掩码——这正是它能被写成一个快 kernel（而非只省 FLOPs）的前提。

- **[ZTA-Q: an Open-source RISC-V Platform for Accurate Quantized CNN Inference](https://arxiv.org/abs/2610.01867)**（`2610.01867`，10-01）：边缘 AI 普遍采用低精度推理，但开源加速器平台对**标准 TFLite 整数推理方案**的端到端支持有限。**ZTA-Q** 是一个开源 RISC-V 平台，支持 TFLite INT8 模型的**精确部署**：除扩展算子支持外，它提供一个**可配置的后处理数据通路**，用于研究**电路级近似**——**降低乘法器精度、共享移位缩放、简化舍入**——如何影响模型精度。系统实现在 Digilent Arty A7-100T FPGA 上，工作频率 **83.3 MHz**；在代表性 CNN 上，相对基线的 LUT / 寄存器 / DSP 开销为 **26.3% / 12.6% / 150%**，而 top-1 与 top-5 精度掉点均**限制在 0.25 个百分点以内**。把这篇与今日首条对照看最有用：**requantization 的后处理通路（乘数位宽、移位、舍入）本身就是一组精度旋钮**，而这里的「simplified rounding」正是 RN 在硬件上的廉价版本——FP 侧刚被证明「舍入规则要按位置分配」，INT8 datapath 侧则是**这些选择被固化在电路里、改一次要重新流片**。边界要说清楚：面向边缘 CNN 与 INT8/TFLite，吞吐与位宽都不在数据中心级显卡的语境里。
