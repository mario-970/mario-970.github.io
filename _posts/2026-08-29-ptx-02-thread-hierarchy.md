---
layout: post
title: "编程模型·线程层级：从线程到 CTA、Cluster、Grid"
permalink: /ptx/02-thread-hierarchy/
date: 2026-08-29 12:00:00 +0800
tags: [PTX]
---

> 本文是 [PTX 专栏](/ptx/) 第 2 期，对应官方文档 §2.1–2.2.3。

## TL;DR

GPU 是主 CPU（host）的协处理器，数据并行、计算密集的部分被「卸载」到设备上，以海量线程执行一个 kernel。这些线程分层组织：**thread → warp → CTA → cluster → grid**。每层都有只读特殊寄存器暴露 ID 与形状——线程用 `%tid`/`%ntid`，CTA 用 `%ctaid`/`%nctaid`，cluster 用 `%cluster_*`。反汇编里它们不是函数调用，而是直接读寄存器，这正是 PTX「显式并行」编程模型的体现。

<!-- more -->

## 一、概念铺垫：为什么需要线程层级

数据并行把「数据元素」映射到「线程」，一次 kernel 启动的线程数可以非常庞大。但线程之间的通信与同步是有代价的：让全部线程互相通信不现实，也不必要。

于是 PTX 把线程**分层**，每层的通信能力不同：

- **CTA 内**：线程可以通信、同步；
- **cluster 内**：不同 CTA 可以经共享内存通信、同步；
- **cluster 之间**：不能通信、同步。

越往上，通信越弱，但可并行的线程总数越大。这就是 thread → CTA → cluster → grid 分层的由来。

## 二、文档语义：线程层级

**§2.1** GPU 是一台能并行执行极多线程的计算设备，作为主 CPU（host）的**协处理器**工作：应用里数据并行、计算密集的部分从 host 卸载到 device。一段「执行多次、但每次数据独立」的程序，可以隔离成一个 **kernel 函数**，在 GPU 上以「许多不同的线程」执行；它被编译成 PTX，再在安装期翻译成目标 GPU 指令集。

**§2.2** 执行一个 kernel 的那批线程，组织成一个 **grid**。grid 由 CTA 或 cluster 组成——**CTA（cooperative thread array）就是 CUDA 的 thread block，cluster 就是 CUDA 的 thread block cluster**。

**§2.2.1 CTA** PTX 编程模型是**显式并行**的：一个 PTX 程序描述的是「并行线程数组中某一个线程」的执行。CTA 是一组并发/并行执行同一 kernel 的线程数组。CTA 内线程可通信，可指定**同步点**让线程等所有线程到齐。每个线程在 CTA 内有唯一的线程标识 `tid`——一个三维向量（`tid.x`/`tid.y`/`tid.z`），表示线程在 1D/2D/3D CTA 中的位置，分量从 0 到该维度线程数。CTA 的形状由 `ntid`（也是三维向量）指定各维度线程数。CTA 内线程按 **warp** 以 SIMT（单指令多线程）方式执行：warp 是「同时执行相同指令」的最大线程子集，线程按序编号；warp 大小是机器相关常量，通常 32，PTX 提供运行时立即数 `WARP_SZ`。

**§2.2.2 cluster** cluster 是一组并发/并行运行、且能经共享内存互相同步通信的 CTA。通信前需确保对端 CTA 的共享内存已存在、且尚未退出。cluster 级 barrier 可同步 cluster 内全部线程。每个 CTA 在 cluster 内有唯一的 `cluster_ctaid`；cluster 形状由 `cluster_nctaid`（1D/2D/3D）指定；另有扁平化的一维序号 `cluster_ctarank` 与总数 `cluster_nctarank`。对应只读特殊寄存器 `%cluster_ctaid`/`%cluster_nctaid`/`%cluster_ctarank`/`%cluster_nctarank`。cluster 仅适用于 `sm_90` 及以上；启动时指定 cluster 维度即为**显式 cluster 启动**，否则为隐式 1×1×1，`%is_explicit_cluster` 可区分二者。

**§2.2.3 grid** CTA 有最大线程数、cluster 有最大 CTA 数；但跑同一 kernel 的 cluster 可批量放进一个 grid，使单次启动的线程总数极大，代价是不同 cluster 间不能通信同步。每个 cluster 在 grid 内有唯一 `clusterid`，grid 形状由 `nclusterid` 指定，另有全局时序标识 `gridid`；寄存器 `%clusterid`/`%nclusterid`/`%gridid`。每个 CTA 在 grid 内有唯一 `ctaid`，grid 形状由 `nctaid` 指定；寄存器 `%ctaid`/`%nctaid`。整体上，kernel = 一批线程，组织成「grid of clusters」，其中 cluster 是仅 `sm_90+` 的可选层。

层级结构如下：

```text
grid（%nctaid 个 CTA，或 %nclusterid 个 cluster）
 │
 └─ cluster（可选，仅 sm_90+；%cluster_nctaid 个 CTA）
     │
     └─ CTA / thread block（%ntid 个线程）
         │
         └─ warp（SIMT 执行单位，通常 32 线程）
             │
             └─ thread（%tid 唯一标识）
```

## 三、反汇编与讲解：四个层级寄存器

用一个把 CUDA 内置变量写回全局内存的最小内核，看它们在 PTX 里长什么样。

CUDA 源码：

```cuda
__global__ void idxKernel(int *out) {
    out[0] = threadIdx.x;
    out[1] = blockIdx.x;
    out[2] = blockDim.x;
    out[3] = gridDim.x;
}
```

真实 PTX（Compiler Explorer，NVCC 13.3.0，`-arch=sm_90a -ptx -O3`，原样贴入）：

```nasm
.visible .entry idxKernel(int*)(
        .param .u64 idxKernel(int*)_param_0
)
{

        ld.param.u64    %rd1, [idxKernel(int*)_param_0];
        cvta.to.global.u64      %rd2, %rd1;
        mov.u32         %r1, %tid.x;
        st.global.u32   [%rd2], %r1;
        mov.u32         %r2, %ctaid.x;
        st.global.u32   [%rd2+4], %r2;
        mov.u32         %r3, %ntid.x;
        st.global.u32   [%rd2+8], %r3;
        mov.u32         %r4, %nctaid.x;
        st.global.u32   [%rd2+12], %r4;
        ret;

}
```

逐行讲解：

- `ld.param.u64 %rd1, [...]` + `cvta.to.global.u64 %rd2, %rd1`：把参数 `out` 指针转成全局地址，放进 `%rd2`。
- `mov.u32 %r1, %tid.x`：读特殊寄存器 `%tid.x`，即 `threadIdx.x`。
- `st.global.u32 [%rd2], %r1`：写 `out[0]`。
- `mov.u32 %r2, %ctaid.x` → `out[1]`：即 `blockIdx.x`。
- `mov.u32 %r3, %ntid.x` → `out[2]`：即 `blockDim.x`。
- `mov.u32 %r4, %nctaid.x` → `out[3]`：即 `gridDim.x`。

对应关系一目了然：

| CUDA 内置变量 | PTX 特殊寄存器 | 含义 |
|--------------|----------------|------|
| `threadIdx.x` | `%tid.x` | 线程在 CTA 内的 ID |
| `blockDim.x` | `%ntid.x` | CTA 形状（线程数） |
| `blockIdx.x` | `%ctaid.x` | CTA 在 grid 内的 ID |
| `gridDim.x` | `%nctaid.x` | grid 形状（CTA 数） |

关键观察：这四个 CUDA 内置变量在 PTX 里不是函数调用，也不是内存读取，而是**直接读特殊寄存器**——它们是硬件为每个线程预置好的只读寄存器。每个线程独立地从自己的 `%tid`/`%ctaid` 出发，算出自己的地址和分工，这就是「显式并行」：同一段 PTX 代码，每个线程读到的 `%tid` 不同，于是各干各的活。

`%cluster_*` 系列需要显式 cluster 启动才会被填充；普通 1×1×1 隐式启动下读它们得到默认值，所以这个最小例子看不到它们。cluster 的完整用法留到第 43 期展开。

## 小结

- GPU 是 host 的协处理器；kernel 是在设备上以海量线程执行的一段程序。
- 线程分层：thread → warp → CTA → cluster → grid。CTA 内可同步；cluster 内 CTA 可经共享内存通信；不同 cluster 间不可通信。
- 各层 ID/形状由只读特殊寄存器暴露（`%tid`/`%ntid`/`%ctaid`/`%nctaid`/`%cluster_*`），反汇编里直接读寄存器，不是函数调用。
