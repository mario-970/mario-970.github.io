---
layout: post
title: "算子日报 · 2026-10-10"
permalink: /operator-daily/2026-10-10/
date: 2026-10-10 23:59:00 +0800
tags: [算子日报]
---

> 每日汇总：CUDA 开源仓 Release Notes（算子）、芯片动态、arXiv 算子内核论文。

## TL;DR

- 本窗口（≥ 10-09）只有一条正式 release：**TensorRT v11.4**，且是工具链侧的常规维护——解析器新增 `IRefitterObserver` / `IParser::setRefitObserver`，让解析期处理可重拟合权重（refittable weights）时能挂观察者回调，这是全篇唯一与权重/算子链路沾边的改动；插件源码做了一批 C++20 更新，samples 目录重整。**没有新算子、新精度格式，也没有新架构支持**。
- **arXiv 本期无新条目**：增量抓取返回 144 条，最新入库日期仍是 10-08，与上一期（150 条）完全重合、**0 条新 id**——周末 arXiv 无新公告，属预期窗口；10-09 / 10-10 提交的论文预计下期入库。
- **芯片动态本期无新条目**：7 天窗口内命中关键词的文章为 0，两个 RSS 源一侧返回空、一侧只有应用与产品类更新。

## 一、CUDA 开源仓 Release Notes（算子）

### TensorRT

- **v11.4**（10-09）— [链接](https://github.com/NVIDIA/TensorRT/releases/tag/v11.4)：**新增 `IRefitterObserver` 类与 `IParser::setRefitObserver`**，用于在解析（parsing）阶段更好地处理**可重拟合权重**——即解析器遇到 refittable weights 时能挂上观察者回调，把「哪些权重可重拟合、何时处理」这件事从隐式行为变成可观测的接口。这是这版唯一与权重/算子链路相关的改动，与 v11.3 对 `IParserRefitter` 外部权重处理的改动落在同一条线上（重拟合不需要重新解析整张图，量化/微调权重更新时省掉一次完整 parse）。其余两条均为非算子内容：插件源码一批 **C++20** 更新（编译期整理，无行为变化）；`samples/common` 中只与 `trtexec` 相关的文件移到 `samples/trtexecCommon`。全文见 [11.4 release notes](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/release-notes-11/11.4.0.html)。

> 本窗口（≥ 10-09）其余更新仅为 `cudnn-frontend` 的自构建 nightly（`v1.32.0.dev72889881` 10-10，正文自述「Automated nightly build from NVIDIA-internal CI. Not a supported release.」），按既有口径不计入。30 天窗口内其余正式 release（CUTLASS v4.8.0、CCCL v3.5.0 / v3.4.3 / python-1.2.1、cuDNN Frontend v1.31.0、TensorRT v11.3、NCCL v2.32.3-1 与 nccl4py v0.6.0、cuda-python v13.4.3 / cuda-core v1.2.1 / cuda-pathfinder v1.8.3、cuda-quantum 0.16.0）均已在各期收录。

## 二、芯片动态

> 本期无新条目。7 天窗口（≥ 10-03）内两个 RSS 源均无命中关键词的新文章：**NVIDIA Developer Blog 源本次直接返回 0 条**（源侧停更，此前多期只回同一条 10-01 条目，现已滚出 7 天窗口）；NVIDIA Blog 有 6 条新更新，但为 Omniverse、GeForce NOW、RTX Spark、电信与医疗等**应用/产品类**内容，既无新架构也无 kernel 级细节，按既有口径不计入。本节连续多期为空是**数据源侧**的真实情况，不是抓取故障——若后续仍持续为空，建议给 `config.yaml` 的 `news.feeds` 增补源或放宽 `news.keywords`。

## 三、arXiv 论文（算子内核）

> 本期无新条目。增量抓取（since 10-09）返回 144 条，最新入库日期仍为 **10-08**，与上一期（10-09 期，150 条）完全重合、无新增 arxiv id；周末 arXiv 不发布新公告，窗口天然为空，10-09 / 10-10 提交的算子论文预计在下一期入库，届时按规则补录。
