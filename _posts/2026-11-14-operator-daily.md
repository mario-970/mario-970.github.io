---
layout: post
title: "算子日报 · 2026-11-14"
permalink: /operator-daily/2026-11-14/
date: 2026-11-14 23:59:00 +0800
tags: [算子日报]
---

> 每日汇总：CUDA 开源仓 Release Notes（算子）、芯片动态、arXiv 算子内核论文。

## TL;DR

- **cuda-pathfinder v1.9.0 修通 toolkit 安装下的 NVRTC JIT**：Linux 上从 CUDA Toolkit 加载 NVRTC 时，会从同一目录预加载 `libnvrtc-builtins.so.<major>.<minor>`，不再依赖 `LD_LIBRARY_PATH` 或 `ldconfig` 条目；静态库 / bitcode / headers 的查找结果也统一返回解析后的绝对路径，工作目录变化后缓存仍有效。
- **其余两类来源今日无新增**：cuDNN Frontend 只有 11-14 的内部 CI nightly（自述 *Not a supported release*）；芯片 RSS 仍是 11-09 的旧条目；arXiv 增量窗口返回的 75 条与上一期抓取结果逐 id 比对后无任何新增。

## 一、CUDA 开源仓 Release Notes（算子）

### cuda-python

- **cuda-pathfinder v1.9.0**（11-13 17:02 UTC，本地上 11-14 01:02）— [链接](https://github.com/NVIDIA/cuda-python/releases/tag/cuda-pathfinder-v1.9.0)：CUDA 工具链组件定位库，release 正文只有文档链接，以下取自仓库内 `1.9.0-notes.rst`。最直接的一条落在 **Linux 上的 NVRTC JIT**：从 CUDA Toolkit 安装加载 NVRTC 时，现在会从 NVRTC 所在目录预加载 `libnvrtc-builtins.so.<major>.<minor>`，因此即使 toolkit 目录不在 `LD_LIBRARY_PATH`、也没有对应的 `ldconfig` 条目，JIT 编译也能工作（[#3439](https://github.com/NVIDIA/cuda-python/pull/3439)）；wheel 包只保留原有的 NVRTC 13.0 workaround，Conda 包不变。查找语义上，headers / 二进制工具 / 静态库 / bitcode 库的查找结果统一返回**解析后的绝对路径**，符号链接会被跟随，返回值可能与配置的 alias 不同（包括被链接文件本身的 basename），但 `LocatedStaticLib` / `LocatedBitcodeLib` 的 `filename` 字段仍报告实际搜索的文件名——公开返回类型、搜索顺序与缓存生命周期均未变。这修掉了相对路径 root（wheel / Conda / toolkit）的静态库与 bitcode 路径在工作目录变化后缓存失效的问题，并正确处理 Windows 的 `D:toolkit` 这类盘符相对 root（[#3363](https://github.com/NVIDIA/cuda-python/issues/3363)、[#3418](https://github.com/NVIDIA/cuda-python/pull/3418)）；含「符号链接后接 `..`」的 root 也不会再被按字面折叠到别的位置（[#3413](https://github.com/NVIDIA/cuda-python/issues/3413)）。

> 本期其余更新不计入：cuDNN Frontend 新增 `v1.33.0.dev82169718`（11-14），它与 11-13 的 `v1.33.0.dev81902178` 同为内部 CI 的 nightly 自构建，正文自述 *Automated nightly build from NVIDIA-internal CI. Not a supported release.*，按既有口径不计入；30 天窗口内其余正式 release（CUTLASS v4.8.1、CCCL python-1.3.0、cuDNN Frontend v1.32.3、NCCL v2.32.6-1、cuda-core v1.3.0、cuda-python v13.5.0、cuda-pathfinder v1.8.6）均已收录于往期。

## 二、芯片动态

> 本期无新条目。窗口内 RSS 仍只返回 11-09 的 [NVIDIA 与韩国 MSIT 扩大主权 AI 合作](https://blogs.nvidia.com/blog/korea-sovereign-ai-collaboration/)，已收录于 11-09 期。

## 三、arXiv 论文（算子内核）

> 本期增量窗口（提交日 11-11 起）返回 **75 条**，但逐 id 与上一期抓取结果比对后**无任何新增**：全部是 11-13 期已抓到的 11-11/11-12 投稿（其中 9 篇已收录于往期，其余经上一期筛选后属应用 / 理论方向）。arXiv API 的最新提交日仍停在 11-12，11-13 起的投稿尚未出现在索引中，本期无新论文可收。
