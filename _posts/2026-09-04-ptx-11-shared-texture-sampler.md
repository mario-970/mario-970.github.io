---
layout: post
title: "状态空间·共享内存与纹理 + 采样器类型：cluster 里的窗口"
permalink: /ptx/11-shared-texture-sampler/
date: 2026-09-04 12:00:00 +0800
tags: [PTX]
---

> 本文是 [PTX 专栏](/ptx/) 第 11 期，对应官方文档 §5.1.7–5.1.8, §5.3。

## TL;DR

这一期收尾状态空间，讲三样东西。`.shared` 是 CTA 拥有的共享内存，从 `sm_90` 起它不只是「本 CTA 内可见」，而是「cluster 内所有 CTA 都可见」——所以有 `::cta`（默认）和 `::cluster` 两个寻址窗口，`mapa` 指令能拿到别的 CTA 的 shared 地址；静态容量 48 KB，`sm_90a` 起可到 228 KB。`.tex` 纹理空间已经弃用，改由 `.texref` 这类不透明类型接管。§5.3 讲的 `.texref`/`.samplerref`/`.surfref` 是不透明类型——布局对 PTX 程序隐藏，只能通过纹理指令读写。

<!-- more -->

## 一、概念铺垫：shared 的边界在扩大，纹理在改名

第 3、5 期已经从「软件可见性」和「硬件本体」两个角度讲过共享内存，这一期补上 PTX 状态空间层面的细节——尤其是一件事：**从 `sm_90` 引入 cluster 之后，shared 的可见边界从「本 CTA」扩大到了「整个 cluster」**。这个变化让 `.shared` 状态空间多了「寻址窗口」的概念，也让一个新指令 `mapa` 登场。

至于纹理，则是一条反方向的线索：`.tex` 这个旧的状态空间正在被淘汰，取而代之的是一套「不透明类型」（`.texref` 等）。所以这一期一边讲 shared 的新边界，一边交代纹理的旧貌新颜。

## 二、文档语义：shared 的窗口、纹理的退场、采样器的类型

### §5.1.7 共享空间 `.shared`：两个寻址窗口

共享空间由**执行中的 CTA 拥有**，从 cluster 概念引入后，**cluster 内所有 CTA 的线程都能访问**它。为了说清「访问的是谁的 shared」，指令带两个子限定符：`::cta`（当前执行 CTA 的共享内存窗口）和 `::cluster`（cluster 内任意 CTA 的窗口）。先看图：

```text
.shared 的两个寻址窗口（§5.1.7）：

  cluster 内所有 CTA 的共享内存连成一个大窗口 = .shared::cluster
  ┌──────────────────────────────────────────────────┐
  │  CTA 0 的 shared    CTA 1 的 shared    ...       │
  │  ┌────────────────┐                               │
  │  │  .shared::cta  │  ← 自己 CTA 的窗口，默认用它    │
  │  └────────────────┘                               │
  └──────────────────────────────────────────────────┘

  mapa 指令：把「本 CTA 里某变量」映射成它在 ::cluster 窗口里的地址，
            于是能拿到别的 CTA 的对应 shared 地址
```

文档明确了两点：`.shared::cta` 窗口的地址**同时落在** `.shared::cluster` 窗口里（前者是后者的子集）；不写子限定符时默认是 `::cta`（所以 `ld.shared` 等价于 `ld.shared::cta`）。`.shared` 里声明的变量指的是**当前 CTA** 的内存地址，要拿别的 CTA 的对应变量，用 `mapa` 指令把它映射到 `.shared::cluster` 窗口。

共享内存通常有专门优化来支持「共享」这个用途：**广播**（所有线程读同一个地址）和**顺序访问**（相邻线程顺序访问相邻地址）。容量上，静态分配的共享内存上限是每 CTA 48 KB；`sm_90a` 等新架构支持扩展容量——`sm_90a`、`sm_100a`、`sm_103a`、`sm_110a` 都是 228 KB，`sm_120a`、`sm_121a` 是 100 KB。

### §5.1.8 纹理空间 `.tex`：已弃用

纹理空间 `.tex` 是通过纹理指令访问的全局内存，context 内所有线程共享，只读、被缓存，所以「对纹理内存的访问与对同一图像的全局内存写**不连贯**」。历史上有 128 个硬件纹理绑定、`.tex` 变量绑到纹理标识符等机制。

关键在于：**显式声明 `.tex` 变量已弃用**。程序应改用 `.texref` 类型变量引用纹理内存；`.tex` 指令仅为向后兼容保留，`.tex` 变量等价于 `.global` 作用域的 `.texref` 变量。换句话说，旧的 `.tex` 空间并入全局空间，纹理的访问交给新的 `.texref` 类型。

### §5.3 采样器、表面、纹理类型：不透明类型

PTX 内置了三个**不透明（opaque）类型**：`.texref`（纹理）、`.samplerref`（采样器）、`.surfref`（表面）。「不透明」的意思是：它们有类似结构体的命名字段，但布局、字段顺序、基地址、总大小这些信息**对 PTX 程序全部隐藏**。

这些类型的使用被严格限定在几个场景：模块作用域或 kernel 入口参数里定义变量；用静态赋值表达式初始化字段；通过纹理/表面指令（`tex`、`suld`、`sust`、`sured`）读写；用 `txq`、`suq` 查询字段；用 `mov` 取指针。但不能用 `ld`/`st` 访问指针、不能做指针算术、不能出现在初始化器里。

它们的关键字段（§5.3.1–5.3.2）决定了纹理的行为：`width`/`height`/`depth` 是各维大小，`channel_data_type`/`channel_order` 是数据格式，`normalized_coords` 决定坐标是否归一化到 [0,1)，`filter_mode`（`nearest`/`linear`）决定取值的插值方式，`addr_mode_0/1/2`（`wrap`/`mirror`/`clamp_*`）决定越界坐标怎么处理。

纹理和采样器有两种工作模式：**统一模式**下，纹理和采样器信息装在一个 `.texref` 句柄里；**独立模式**下，两者各有句柄（`.texref` 管纹理、`.samplerref` 管采样器），在使用处才组合。独立模式额外多一个 `force_unnormalized_coords` 字段，用于 OpenCL 编译时的坐标归一化覆盖。

## 三、反汇编与讲解：广播，shared 最拿手的活

§5.1.7 说共享内存有「广播」优化——所有线程读同一个地址。下面这个内核正好演示广播：线程 0 把一个值写进 shared，其余线程都读它。

CUDA 源码：

```cuda
__global__ void broadcastKernel(int *out, const int *in) {
    __shared__ int val;
    if (threadIdx.x == 0) val = in[0];
    __syncthreads();
    out[threadIdx.x] = val;
}
```

真实 PTX（Compiler Explorer，NVCC 13.3.0，`-arch=sm_90a -ptx -O3`，指令原样贴入）：

```nasm
        ld.param.u64    %rd1, [broadcastKernel(int*, int const*)_param_0];  // out
        ld.param.u64    %rd2, [broadcastKernel(int*, int const*)_param_1];  // in
        mov.u32         %r1, %tid.x;                // threadIdx.x
        setp.ne.s32     %p1, %r1, 0;                // threadIdx.x != 0 ?
        @%p1 bra        $L__BB0_2;                  // 非 0 线程跳过写，直接去屏障

        cvta.to.global.u64      %rd3, %rd2;         // in 转 global 地址
        ld.global.u32   %r2, [%rd3];                // 线程 0 读 in[0]
        st.shared.u32   [_ZZ15broadcastKernelPiPKiE3val], %r2;   // 写 shared 变量 val

$L__BB0_2:
        bar.sync        0;                          // __syncthreads()
        ld.shared.u32   %r3, [_ZZ15broadcastKernelPiPKiE3val];   // 所有线程读同一个 shared 地址
        cvta.to.global.u64      %rd4, %rd1;         // out 转 global 地址
        mul.wide.u32    %rd5, %r1, 4;               // i*4
        add.s64         %rd6, %rd4, %rd5;           // out[i] 地址
        st.global.u32   [%rd6], %r3;                // out[i] = val
        ret;
```

几个点值得讲：

**其一，`ld.shared` 没带子限定符，默认就是 `::cta`。** 这正是 §5.1.7 说的「不写子限定符时默认 `::cta`」——本 CTA 自己的共享内存窗口。如果代码访问了 cluster 里别的 CTA 的 shared，这里才会出现 `ld.shared::cluster`，并伴随 `mapa` 先算出远端地址。

**其二，广播的现场：所有线程读同一个 shared 地址。** `_ZZ15broadcastKernelPiPKiE3val` 是编译器给 shared 变量 `val` 起的 mangled 符号名。关键是——`st.shared` 只被线程 0 执行一次（`@%p1 bra` 把其余线程分流了），而 `ld.shared` 这一条被**所有**线程执行，且它们读的是**同一个**地址 `[val]`。这就是 §5.1.7 说的「广播」：一条 `ld.shared`，几十个线程同时从同一个地址取数，硬件为此做了专门的加速。

**其三，`bar.sync 0` 是广播正确性的前提。** 线程 0 的写和其余线程的读之间必须隔一道屏障，否则别的线程可能在 `val` 还没写好时就抢先读。这也呼应了前面几期反复出现的模式：shared 的「共享」必然伴随同步来协调「谁先写、谁后读」。

## 小结

- `.shared` 从 `sm_90` 起可见边界扩到整个 cluster：`::cta`（默认）和 `::cluster` 两个窗口，`mapa` 映射远端地址。
- shared 有广播、顺序访问两种优化；静态容量 48 KB，`sm_90a` 起 228 KB。
- `.tex` 纹理空间已弃用，改由 `.texref` 类型接管；旧 `.tex` 变量等价于 `.global` 的 `.texref`。
- `.texref`/`.samplerref`/`.surfref` 是不透明类型，布局隐藏，只能经纹理/表面指令访问；分统一模式、独立模式。
- 反汇编里，`ld.shared` 默认 `::cta`；广播表现为「一条写、所有线程读同一地址」，`bar.sync` 保证先写后读。
