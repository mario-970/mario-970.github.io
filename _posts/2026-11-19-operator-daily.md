---
layout: post
title: "算子日报 · 2026-11-19"
permalink: /operator-daily/2026-11-19/
date: 2026-11-19 23:59:00 +0800
tags: [算子日报]
---

> 每日汇总：CUDA 开源仓 Release Notes（算子）、芯片动态、arXiv 算子内核论文。

## TL;DR

- **cuDNN Frontend v1.33.0 把 sparse attention 和训练算子一次补齐**：PyTorch/JAX 获得简化版 sparse attention，Hopper 增加融合 compressed DSA indexer + top-k 与 prepared plan，SM120 有了 prefill / split-KV / 量化 KV decode 的预编译 attention（免运行时编译）；Cake 加入确定性 GDN/KDA 训练引擎，SDPA 在 Hopper/SM120 上补齐 native 路径与 FP8/MXFP8 反向。注意移除 v1.32.0 的实验性 triangle attention API。
- **SASS 里还有编译器不生成的调度字段**：barrier exemption 与可编程 warp priority 被逆向出来后做成编译后二进制优化器——6 块 GPU、4 代架构上代表性 kernel 最多快 **31.9%**，llama.cpp / vLLM 生产 kernel 最多快 **5.4%**、端到端推理最多快 3.6%。
- **稀疏 attention 继续往「更细粒度 + 分布式持久 kernel」走**：PQRS 让每个 query 自选 key cluster（视频 DiT 端到端 1.66–2.10×），Spark-H3 做自适应 block 稀疏（DiT 2.49×），DexAttention 用 persistent kernel 在 16 GPU 上跑动态 mask（训练 1.39×、视频 attention 2.11–2.37×）。量化侧则提醒：sub-4-bit 的软件 bit-unpacking 可带来 1.48–21.5× 执行 slowdown，VLA 的逐 step 精度需要专用硬件才能吃到。

## 一、CUDA 开源仓 Release Notes（算子）

### cuDNN Frontend

- **v1.33.0** — [链接](https://github.com/NVIDIA/cudnn-frontend/releases/tag/v1.33.0)：一次性铺开 LLM 推理/训练算子。sparse attention 侧，PyTorch/JAX 获得简化版 sparse attention（训练含 dense indexer KL loss），Hopper 新增融合 compressed DSA indexer + top-k 与可复用的 prepared plan，block-sparse attention 支持 per-token Q/K FP8 scale；SDPA 在 Hopper 补齐 native D64/D128/D256 前向和 paged KV 前向、D64/D128 确定性反向与 bottom-right causal mask，SM120 新增 paged decode split-KV、D512 paged attention、半精度 D576/V512 前向，反向扩到 FP8/MXFP8（含可选 amax），SM100 改进 MXFP8 反向、SM103 支持 NVFP4 QAT 反向；SM120 另有预编译 attention（prefill / split-KV / 量化 KV decode，C++ header 调用、免运行时编译）。训练侧 Cake 加入确定性 GDN/KDA 引擎、GQA 与 sparse attention、DSA indexer 训练，并覆盖 GEMM、SwiGLU/MXFP8、causal conv、RMSNorm、LM-head 等；linear attention 侧融合 GDN decode 与多 token 验证支持 SM100/SM120，重复 KDA/GDN 调用可从 C++ 回放 launch 以省掉 Python dispatch。破坏性变更：移除 v1.32.0 的实验性 `triangle_attention` / `triangle_multiplicative_update`，mHC、DSA top-k 等 API 迁到 `cudnn.ops.*`。

### TensorRT

- **v11.5** — [链接](https://github.com/NVIDIA/TensorRT/releases/tag/v11.5)：工具链侧常规维护——默认 CUDA 升到 13.5，解析器支持 ONNX opset 25 / ONNX 1.22，插件源码要求 C++20，`trtexec` 新增 `--independentWeightLayouts` 与安全引擎的 `--saveEngineSo` / `--loadEngineSo`。没有新算子、新精度格式或新架构支持。

### cuda-python

- **cuda-core v1.3.1** — [链接](https://github.com/NVIDIA/cuda-python/releases/tag/cuda-core-v1.3.1)：修 `Buffer.from_ipc_descriptor` 的并发竞态——另一线程同时释放同一 descriptor 时不再错误 unmap 新导入的内存池 buffer，导入会等待 pending release；`PinnedMemoryResource` 无参构造不再打断另一线程进行中的 CUDA graph capture（1.3.0 引入的 `cuMemPoolSetAccess` 会让 global-mode capture 失效，现套用 relaxed-mode wrapper，`torch.cuda.graph` 这类默认 global mode 的调用方受益）；新增 `minimal-sysctk12/13` extras 只装 `cuda-bindings` floor，1.3.0 已从 PyPI yank。
- **cuda-pathfinder v1.9.1** — [链接](https://github.com/NVIDIA/cuda-python/releases/tag/cuda-pathfinder-v1.9.1)：Windows Arm64 上可从 `nvidia-cudnn-cu13` wheel 找到 cuDNN——该 wheel 把 `cudnn64_9.dll` 放在与 x64 相同的 `nvidia/cudnn/bin`，原先只在 x64 搜索；现在会校验 DLL 架构，避免不兼容 wheel 阻断后续安装根，Conda / `CUDA_PATH` / Program Files 仍仅限 x64。

> 窗口内 cuDNN Frontend 另有 nightly 自构建 `v1.34.0.dev83610449`（11-19）与 `v1.33.0.dev83259061`（11-18），正文自述 *Not a supported release*，按既有口径不计入。其余正式 release（CUTLASS v4.8.1、CCCL python-1.3.0、NCCL v2.32.6-1、cuda-pathfinder v1.9.0、cuda-quantum 0.17.0 等）均已在往期收录。

## 二、芯片动态

> 本期无新条目。两个 RSS 源在 7 天窗口内仍只返回 11-16 发布的 4 条：Rubin HPC 已收录于 11-17 期，MLPerf Training/Endpoints 榜单、CUDA-X 科学计算库（cuPhoton / cuPotential / cuSpa）与 CUDA 20 周年回顾均非算子内核更新、已在往期注记；NVIDIA Developer Blog 仍返回 0 条。

## 三、arXiv 论文（算子内核）

> 本期增量窗口（提交日 11-18 起）抓取 **150 条**，触及上限。与 11-18 期快照逐 id 比对，新进索引 **61 条**——11-18 提交 49 条、11-17 追补 12 条（11-16 的条目均在上一期审阅范围内）；11-19 当日投稿尚未进入 arXiv 索引，下期补录。按算子开发相关性精选 8 篇。

- **[Dissecting an Unexplored SASS Control-Code Field: Barrier Exemption and Programmable Warp Priority](http://arxiv.org/abs/2611.24437v1)**（`2611.24437`，cs.AR/cs.DC/cs.PF，11-18）：SASS 每条指令都带调度控制码，但其中一个字段在 cuBLAS/cuDNN 等闭源库里出现超过 50 万次，nvcc 却从不生成，此前未被解读。作者用指令级微基准在 Volta–Blackwell 上确认它控制两个独立机制：**barrier exemption**（EXEMPT，被 barrier 阻塞的指令可在等待期提前发射）与**可编程 warp 优先级**（START/TILE/END 三档）。实现为不改源码、不改编译器的编译后二进制优化器，6 块 GPU、4 代架构上代表性 kernel 最多快 **31.9%**，llama.cpp 生产 kernel 最多快 5.4%、端到端 LLM 推理最多快 3.6%。
- **[Every Query Routes Itself: Accelerating Video Diffusion Transformers with Per-Query Routed Sparse Attention](http://arxiv.org/abs/2611.24479v1)**（`2611.24479`，cs.CV/cs.LG，11-18）：聚类稀疏 attention 让同一 query cluster 共享 key cluster 选择，精度受限于分组的粗细。PQRS 是 training-free 方法：只对 key 聚类，每个 query 自选 key cluster，并用质心补偿未选块；query tile 不再共享 key list，于是改为遍历 key cluster、把选择相同的 query 分桶成 dense task，并以固定 softmax offset 一次合并部分结果。Wan / HunyuanVideo 四个模型上端到端 **1.66–2.10×** 加速，Wan2.1-T2V 28.9 dB、HunyuanVideo 35.4 dB PSNR。
- **[Spark-H3: Adaptive Block Sparse Attention for Video Diffusion Models](http://arxiv.org/abs/2611.24256v1)**（`2611.24256`，cs.CV，11-18）：block sparse attention 的固定分组会把 attention 偏好不同的 token 分到一组，均匀 key 平均又会低估压缩块的 attention mass。Spark-Attn 的 Spark-Reblock 按 attention 偏好递归分组、同时保持规则 block 形状以便 GPU 执行；Spark-Reweight 用全局 mean query 作锚构造 attention 加权的 KV 表示与 logit bias。MiniMax-H3 上 10% 精确块密度时 DiT **2.49×** 加速、24.81 dB PSNR；20% 密度 26.76 dB、2.26×。
- **[DexAttention: Distributed Sparse Attention through Rapid Online Planning and Persistent GPU Execution](http://arxiv.org/abs/2611.23512v1)**（`2611.23512`，cs.DC，11-17）：分布式稀疏 attention 的 mask-aware 调度必须让在线重规划比省下的执行时间更便宜。DexAttention 用 GPU 加速的 topology-aware 聚类做 assigner，轻量贪心 scheduler 排粗粒度通信/计算，每块 GPU 上一个 persistent kernel 维护细粒度依赖并派发 ready task，从而避免重复 launch。4 节点 16 GPU、11 组配置上几何平均端到端训练快 **1.39×**；动态视频 attention 比 Ring Attention 快 **2.11–2.37×**（含规划）。
- **[A Fast, Programmable GPU-to-CPU Invocation Layer for GPU-Initiated Communication](http://arxiv.org/abs/2611.23848v1)**（`2611.23848`，cs.DC/cs.LG，11-18）：现有 GPU 发起通信多把传输控制放在 GPU 线程（GDA），把反向 offload 到 host proxy 当慢速回退。KIWI 论证反向 offload 可以更快，并把它做成可编程接口：单向 device-to-host **亚微秒**延迟、数亿次/秒，同时处理共享队列争用、轮询开销与 proxy 负载不均。基于它的 KIWI-EP 重写 MoE dispatch/combine kernel，AMD GPU 上 expert-parallel 通信延迟最多降 **2.3×**，vLLM 每输出 token 时间最多降 29%。
- **[Hardware Acceleration of Vision-Language-Action Models with Weight Precision Rising over the Flow Steps](http://arxiv.org/abs/2611.24234v1)**（`2611.24234`，cs.AR，11-18）：VLA 的 action head 每个 flow step 都要从 DRAM 重读全部权重，却只作用于约 50 个 token，推理是 memory-bound；作者发现后续 step 会衰减前一步的量化误差，因此权重精度应随 step 上升——2→4 bit 的 per-channel 舍入无需校准或训练即可保持闭环成功率，且比每步 4-bit 少传 **25%** 比特。但 Jetson AGX Thor 上现有 GPU kernel 因固定开销无法利用逐 step 精度；作者据此实现按传输比特数缩放延迟/能耗的加速器，RTL + LPDDR5X 时序下 GR00T N1.6 单次推理 **8.02 ms**，比 Thor NVFP4 快 1.7×、能耗低 9.1×。
- **[Why Sub-4-Bit Vision CNNs Collapse and Where Mixed-Precision Recovery Stops](http://arxiv.org/abs/2611.23445v1)**（`2611.23445`，cs.AI/cs.CV/cs.LG，11-17）：系统分析 sub-4-bit 视觉 CNN 的崩溃位置：退化高度非线性，集中在 late-stage pointwise projection 与 depthwise 层；保留不到 5% 的关键层激活为 8-bit 精度可恢复最多 **92%** 的精度损失。关键部署发现是：低 bit 虽然降低存储，但标准 runtime 的软件 bit-unpacking 会带来 **1.48–21.5×** 执行 slowdown（随 batch size 变化），模型 footprint 与实际 edge 吞吐之间存在硬取舍。
- **[Optimize the Cob, Then the Kernel: Optimizing Beam Search for Generative Recommendation Inference](http://arxiv.org/abs/2611.23339v1)**（`2611.23339`，cs.LG/cs.AI，11-17）：生成式推荐的 beam search 推理是长 prompt、宽 beam、少量固定输出 token；vLLM 把各 beam 当独立序列批处理，反复读共享前缀的 KV cache，且每步单独调用引擎带来 host 侧开销。作者用 Tree Attention 沿序列维拼接 beam 消除冗余 KV 访问，再叠加预生成 partial-attention mask、静态 KV 分配、循环展开 dispatch、TorchAir 图编译、自动算子融合与词表约束解码；**不用自定义 kernel**，在 Ascend NPU 上端到端延迟比 vLLM 最多降 **25×**，与使用自定义 kernel 的 SOTA 相当——说明应先做完整条生成流水线，再决定是否值得写 custom kernel。
