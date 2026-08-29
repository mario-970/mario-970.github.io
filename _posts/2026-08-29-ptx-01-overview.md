---
layout: post
title: "PTX 全景：ISA 是什么、解决什么问题"
permalink: /ptx/01-overview/
date: 2026-08-29 00:00:00 +0800
tags: [PTX]
---

> 本文是 [PTX 专栏](/ptx/) 第 1 期，对应官方文档 §1.1–1.4。

## TL;DR

PTX 是 NVIDIA 为 GPU 定义的一台「虚拟机器」和一套「虚拟指令集」（ISA）。它位于 CUDA/C++ 源码与真实硬件指令之间：nvcc 先把高级语言编译成 PTX，GPU 驱动再在安装/加载期把 PTX 翻译成目标硬件的原生指令。正是这层「安装期翻译」，让 PTX 能用一套稳定的指令集横跨多代 GPU。这是专栏第 1 期，讲清 PTX 是什么、为什么存在。

<!-- more -->

## 一、概念铺垫：GPU 与「数据并行」

受实时高清 3D 图形的需求驱动，可编程 GPU 演变成一台高度并行、多线程、众核的处理器，拥有巨大算力和极高内存带宽。它特别适合「数据并行计算」——同一个程序作用在许多数据元素上并行执行——并且具备高「算术强度」（算术运算与内存操作的比值）。

数据并行带来两个直接推论：

1. 同一个程序在众多数据元素上重复执行，因此对复杂控制流的需求较低；
2. 高算术强度意味着可以用「计算」掩盖内存访问延迟，而不必依赖大容量数据缓存。

数据并行处理把「数据元素」映射到「并行线程」：3D 渲染把像素/顶点映射到线程；视频编解码、图像缩放、立体视觉、模式识别把图像块/像素映射到线程；信号处理、物理仿真、计算金融、计算生物等领域同样受益。

## 二、文档语义：定位与七大目标

§1.1 末段给出 PTX 的定义：**一台「虚拟机器」和一套面向通用并行线程执行的指令集**。关键在于下一句——**PTX 程序在「安装期」被翻译成目标硬件的指令集**，由「PTX→GPU 翻译器」与驱动共同完成，从而把 NVIDIA GPU 变成可编程的并行计算机。

§1.2 列出 PTX 的七大目标：

1. 提供跨越**多代 GPU** 的稳定 ISA；
2. 编译后性能可比肩**原生 GPU** 性能；
3. 为 C/C++ 等编译器提供**机器无关**的目标 ISA；
4. 为应用和中间件开发者提供**代码分发**用的 ISA；
5. 为优化代码生成器和翻译器提供**公共源级 ISA**；
6. 便于**手写**库、性能内核与架构测试；
7. 提供可扩展的编程模型，覆盖从单个单元到众多并行单元的 GPU 规模。

§1.3 是 9.3 版新增内容，本专栏将陆续覆盖：`mma_throughput` pragma、`clmad` 指令、`mbarrier.test_wait/try_wait` 的 `.phase_type` 限定符与 report 操作数、`mbarrier.check_layout`、`multimem.st.async`/`multimem.red.async`、`cp.async.bulk` 族的 `.sem`/`.scope`、`fabric.*` 指令族、`fence.proxy` 的 fabric 限定符，以及 `.language` 指令。

§1.4 给出文档十章结构：Programming Model、PTX Machine Model、Syntax、State Spaces/Types/Variables、Instruction Operands、Abstracting the ABI、Instruction Set、Special Registers、Directives、Release Notes——这正是本专栏 48 期目录的骨架来源。

## 三、第一眼 PTX：一个最小内核

编译链路如下（伪代码）：

```text
CUDA 源码 (.cu)
      │  nvcc 离线编译
      ▼
PTX（虚拟 ISA · 文本 · 机器无关）
      │  驱动在安装/加载期翻译
      ▼
SASS（目标硬件原生指令 · 机器相关）
```

CUDA 源码：

```cuda
__global__ void addKernel(float *a, float *b, float *c) {
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}
```

真实 PTX（Compiler Explorer，NVCC 13.3.0，`-arch=sm_90a -ptx -O3`，原样贴入）：

```nasm
.visible .entry addKernel(float*, float*, float*)(
        .param .u64 addKernel(float*, float*, float*)_param_0,
        .param .u64 addKernel(float*, float*, float*)_param_1,
        .param .u64 addKernel(float*, float*, float*)_param_2
)
{

        ld.param.u64    %rd1, [addKernel(float*, float*, float*)_param_0];
        ld.param.u64    %rd2, [addKernel(float*, float*, float*)_param_1];
        ld.param.u64    %rd3, [addKernel(float*, float*, float*)_param_2];
        cvta.to.global.u64      %rd4, %rd3;
        cvta.to.global.u64      %rd5, %rd2;
        cvta.to.global.u64      %rd6, %rd1;
        mov.u32         %r1, %tid.x;
        mul.wide.s32    %rd7, %r1, 4;
        add.s64         %rd8, %rd6, %rd7;
        ld.global.f32   %f1, [%rd8];
        add.s64         %rd9, %rd5, %rd7;
        ld.global.f32   %f2, [%rd9];
        add.f32         %f3, %f1, %f2;
        add.s64         %rd10, %rd4, %rd7;
        st.global.f32   [%rd10], %f3;
        ret;

}
```

逐行讲解（点到为止，细节留到后面各期）：

- `.visible .entry` 声明一个**内核入口**：`.entry` 表示这是可被主机端调用的内核，`.visible` 表示对外可见。
- `.param .u64` 是三个**参数**，每个都是 64 位无符号整数——这里是三个 `float*` 指针。
- `ld.param.u64 %rd1, [...]` 从**参数状态空间**读出指针，放进 64 位寄存器 `%rd1`。
- `cvta.to.global.u64` 做**地址转换**，把「泛型地址」转成「全局地址」（第 32 期展开）。
- `mov.u32 %r1, %tid.x` 读取特殊寄存器 `%tid.x`（即 `threadIdx.x`）到 32 位寄存器 `%r1`（第 42 期展开）。
- `mul.wide.s32 %rd7, %r1, 4` 计算 `i * 4`，因为 `float` 占 4 字节，得到字节偏移。
- `ld.global.f32 %f1, [%rd8]` 从**全局状态空间**读 `a[i]` 到浮点寄存器 `%f1`（第 31 期展开）。
- `add.f32 %f3, %f1, %f2` 执行真正的加法。
- `st.global.f32 [%rd10], %f3` 把结果写回全局内存 `c[i]`。

这一段已经能看到 PTX 的三个特征：**显式状态空间**（`.param`/`.global`）、**类型化寄存器**（`%rd` 64 位、`%r` 32 位、`%f` 浮点）、**一条指令一个操作**的 RISC 风格。

关于文件头：godbolt 后处理裁剪了三行模块指令（§11.1）。完整 PTX 以 `.version`、`.target`、`.address_size` 开头——`.version` 声明 PTX 语言版本（major 表示不兼容变更、minor 表示新增特性，NVCC 13.3 对应 `.version 9.3`），`.target` 声明目标架构（`-arch=sm_90a` 对应 `.target sm_90a`），`.address_size` 声明地址位宽（缺省为 32）。

这里有个容易忽略的细节：`.target` 的**基线架构**（如 `sm_90`）遵循「洋葱模型」——新一代保留旧一代的全部特性，因此为旧目标生成的 PTX 能在更新一代设备上运行，这正是「稳定 ISA 横跨多代」的实现机制。但带 `a` 后缀的目标（如 `sm_90a`）含**架构专属特性**，不遵循洋葱模型，不能跨代运行。本示例用 `sm_90a` 只是贴近 Hopper 新特性，与「跨代兼容」是两个不同概念——这一点后文会反复遇到。

## 小结

- PTX 是**虚拟 ISA**，不是真实硬件指令；它把「编译」拆成两段：nvcc 离线产出 PTX，驱动在线翻译成原生指令。
- 这层间接带来的最大红利是**跨代稳定**：同一份 PTX 能运行在多代 GPU 上，因为翻译发生在安装/加载期。
- PTX 用显式状态空间 + 类型化寄存器 + 逐条简单指令，来描述并行线程的执行。
