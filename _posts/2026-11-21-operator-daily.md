---
layout: post
title: "算子日报 · 2026-11-21"
permalink: /operator-daily/2026-11-21/
date: 2026-11-21 23:59:00 +0800
tags: [算子日报]
---

> 每日汇总：CUDA 开源仓 Release Notes（算子）、芯片动态、arXiv 算子内核论文。

## TL;DR

- **本期只有一条正式更新**：cuda-python v12.9.11（CUDA 12.9 维护线）修掉 CUDA graph 中 host callback 第二次启动必崩的问题——旧实现的内置 trampoline 在首次调用后就释放状态；同时把 driver/runtime 结构体视图的所有权与数组返回值改成安全拷贝，`nvJitLink` / `nvvm` / `nvfatbin` 的指针参数开始接受多维连续缓冲。
- **arXiv 与芯片动态本期为空**：arXiv 增量窗口 119 条全部落在 11-18 / 11-19，与上期快照逐 id 比对无新增（周五投稿尚未公告）；芯片 RSS 仍只有 11-16 的 4 条旧闻，Developer Blog 源返回 0 条。

## 一、CUDA 开源仓 Release Notes（算子）

### cuda-python

- **v12.9.11**（2026-11-20 17:37 UTC，本地上 11-21 01:37，晚于上期出刊）— [链接](https://github.com/NVIDIA/cuda-python/releases/tag/v12.9.11)：CUDA 12.9 维护线（与 11-06 收录的 CUDA 13 线 v13.5.0 并行，本次随包更新的是 `cuda.bindings 12.9.11`）。对算子最直接的是 **CUDA graph 内的 host callback**：`cuLaunchHostFunc` / `cudaLaunchHostFunc` 旧实现走内部 trampoline，首次调用后即释放状态，导致被捕获进 graph 的 host function 在第二次 launch 时崩溃；现在函数指针与 user data 原样交给 CUDA，调用方必须自行保证 callback 对象与 user data 活到所有使用结束（含 graph 与 executable 的生命周期）。结构体层面，driver / runtime 的嵌套 struct / union 视图会持有 owner，struct-array getter 返回独立副本，修改返回值不再回写父结构（需读出 → 修改 → 整体赋值）；`reserved` 成员不再暴露为 Python 属性（访问抛 `AttributeError`），底层存储已零初始化，IPC handle 需改用 `getPtr()` + `ctypes` 序列化。JIT 工具链侧，`nvfatbin` / `nvjitlink` / `nvvm` 的 buffer-like 指针参数不再限一维，可直接传多维连续 NumPy 数组或 ctypes 对象；另修正 `cuLogs*` 的 driver 版本探测，12.9 驱动上不再误报 `FunctionNotFoundError`。打包面：Python 3.10 已 EOL 移除（支持 3.11–3.15），wheel 不再携带 `.pyx` / `.in`，源码构建改为预生成、不再解析 CUDA 头文件。

> 窗口内 cuDNN Frontend 另有 nightly 自构建 `v1.34.0.dev84275295`（11-21），正文自述 *Automated nightly build from NVIDIA-internal CI. Not a supported release.*，按既有口径不计入（`dev83981319` 已在 11-20 期注记）。其余正式 release（CUTLASS v4.8.1、CCCL python-1.3.0、cuDNN Frontend v1.33.0、TensorRT v11.5、NCCL v2.32.6-1、cuda-core v1.3.1、cuda-pathfinder v1.9.1、cuda-quantum 0.17.0 等）均已收录于往期。

## 二、芯片动态

> 本期无新条目。NVIDIA Developer Blog 源本期返回 **0 条**（源侧仍空）；NVIDIA Blog 虽有 11-17～11-19 的 5 条更新（SC26 主题演讲、Azure / Windows PC 加速计算、RTX Spark 台式机、Omniverse、GeForce NOW 促销），但均为平台 / 消费级 / 活动类内容，不涉及算子内核或架构层面的开发更新；命中关键词的仍只有 11-16 的 4 条旧闻（Rubin HPC、MLPerf、CUDA-X 科学计算库、CUDA 20 周年），已在 11-17～11-19 期注记。

## 三、arXiv 论文（算子内核）

> 本期无新条目。增量窗口（提交日 ≥ 11-18）返回 **119 条**，与 11-20 期快照逐 id 比对**无新增**：11-18 的 63 条、11-19 的 56 条均已在上一轮抓取范围内并完成审阅。11-20（周五）及之后的投稿尚未进入 arXiv 索引（周末无公告），按脚本的 2 天回补窗口，待下一次公告后自动补录。
