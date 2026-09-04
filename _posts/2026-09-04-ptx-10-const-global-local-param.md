---
layout: post
title: "状态空间·常量/全局/本地/参数：四块内存的性格"
permalink: /ptx/10-const-global-local-param/
date: 2026-09-04 09:00:00 +0800
tags: [PTX]
---

> 本文是 [PTX 专栏](/ptx/) 第 10 期，对应官方文档 §5.1.3–5.1.6。

## TL;DR

上一期立起了八个状态空间的总表，这一期逐个讲清其中四个「普通内存」空间：`.const` 是 host 初始化、只读、大小只有 64 KB + 640 KB 的常量区；`.global` 是容量最大、跨 CTA/grid 通信的主通道；`.local` 是每线程私有的「溢出兜底」，寄存器放不下的数据落到这里；`.param` 最特殊——它不只是内核参数，还有三种用途：host 传参给 kernel、设备函数的形式参数、以及传大结构体的字节数组。

<!-- more -->

## 一、概念铺垫：四块内存，四种性格

第 9 期的属性表（Table 7）用「可寻址、可初始化、访问权限、共享级别」四列刻画了八个状态空间。但表只是速查，每块内存背后还有自己的「性格」——大小多大、给谁用、怎么访问、有什么坑。这一期把 `.const`、`.global`、`.local`、`.param` 四块拿出来逐个讲透。

为什么是这四个放一起？因为它们都是「普通内存」——数据本体在 device memory 上（对照第 5 期讲过的：它们不是寄存器和共享内存那种「本体在片上」的存储），而且都是「有地址、要 `ld`/`st` 访问」的空间。但它们的职责天差地别：一个只读、一个全局通信、一个私有兜底、一个专门传参。搞清这四块，就搞清了 GPU 内存的大部分日常使用场景。

## 二、文档语义：四个空间逐个拆

### §5.1.3 常量空间 `.const`：只读、小、但有两块

常量空间是「host 初始化、只读」的内存，用 `ld.const` 访问。它最特别的地方是**大小分两层**，先看图：

```text
常量空间（.const）的容量布局（§5.1.3）：

  ┌────────────────┐
  │ 静态常量变量区  │  64 KB     ← __constant__ 声明的变量放这里
  ├────────────────┤
  │ 区域 1         │  64 KB  ┐
  │ 区域 2         │  64 KB  │
  │  …             │         │  额外的 640 KB：10 个独立的 64 KB 区域
  │ 区域 10        │  64 KB  ┘  ← 由 driver 分配，指针作为内核参数传入
  └────────────────┘
```

第一层是 **64 KB**，用来放「静态大小的常量变量」——就是源码里 `__constant__` 声明的那些。第二层是**额外的 640 KB**，组织成 10 个独立的 64 KB 区域，由 driver 分配、初始化常量缓冲区，再把指针作为内核参数传给 kernel。因为 10 个区域**不连续**，driver 必须保证每个缓冲区完整地落在单个 64 KB 区域内、不跨区域边界。

文档还交代了两点：静态常量变量可以有可选的初始化器，没有显式初始化器的默认初始化为 0；历史上（PTX 2.2 之前）常量内存是「11 个 64 KB bank」的形态，用 `.const[bank]` 指定 bank，那个 banked 形式已经弃用，现在统一成「64 KB + 640 KB」这套。

### §5.1.4 全局空间 `.global`：所有人的通信信道

全局空间是「context 内所有线程都能访问」的内存，是**不同 CTA、不同 cluster、不同 grid 的线程之间通信的机制**。访问方式三种：`ld.global`（读）、`st.global`（写）、`atom.global`（原子操作）。

和常量一样，全局变量有可选的初始化器，没有显式初始化器的默认初始化为 0。全局内存的容量远大于常量，是数据的主存储区——前面所有反汇编里 `in`、`out` 指针指向的都是全局空间。

### §5.1.5 本地空间 `.local`：每线程私有的溢出兜底

本地空间是**每个线程私有的内存**，用来存放线程自己的数据。它「通常是带 cache 的标准内存」，大小受限——因为要按线程逐一分配。用 `ld.local` 和 `st.local` 访问。

理解 `.local` 的关键，是它和寄存器的关系。第 9 期讲过：寄存器放不下的变量会「溢出（spill）到内存」，这个「内存」就是 `.local`。所以 `.local` 的定位是**寄存器的兜底**——当寄存器不够用、或者变量需要按地址访问（比如局部数组用动态下标）时，数据就从寄存器挪到本地内存。

文档还讲了 ABI 相关的细节：使用 ABI 编译时，`.local` 变量必须声明在函数作用域内、分配在**栈**上；在那些不支持栈的实现里，本地变量存在固定地址，递归调用不被支持，`.local` 变量可以声明在模块作用域。

### §5.1.6 参数空间 `.param`：三种用途，不止传参

参数空间最容易被人误解成「就是内核参数」，其实文档说它有**三种用途**：

1. **把输入参数从 host 传给 kernel**——最常见的用法，参数是 per-grid、只读的。
2. **声明设备函数的形式输入参数和返回参数**——在 kernel 执行期间调用的设备函数，它的形参和返回值也走 `.param` 空间，是 per-thread、可读写的。
3. **声明本地作用域的字节数组变量**，用来做函数调用参数——典型场景是把一个大结构体按值传给函数。

用途 1 和用途 2/3 的差异，文档用两个维度区分：内核参数是**只读**、**per-kernel**；设备函数参数是**读写**、**per-thread**。为了在指令层面区分「这个 `.param` 地址是内核参数还是设备函数参数」，指令可以带 `::entry` 或 `::func` 子限定符——`ld.param::entry` 指内核参数，`ld.param::func` 指设备函数参数。不写子限定符时，默认值取决于具体指令（比如 `st.param` 等价于 `st.param::func`）。

一个实现细节值得知道：参数空间的位置是**实现相关**的——有的实现里内核参数就放在全局内存里，此时参数空间和全局空间之间没有访问保护。所以 PTX 代码不能对 `.param` 变量的相对位置或顺序做任何假设。

## 三、反汇编与讲解：三个空间同台，一个空间缺席

下面这个内核同时踩到常量、全局、参数三个空间，还故意用了一个局部数组，看它会不会落到本地内存。

CUDA 源码：

```cuda
__constant__ int c_scale = 3;

__global__ void spacesKernel(int *out, const int *in, int n) {
    int local_arr[8];
    int i = threadIdx.x;
    local_arr[i & 7] = in[i];
    out[i] = local_arr[i & 7] * c_scale;
}
```

真实 PTX（Compiler Explorer，NVCC 13.3.0，`-arch=sm_90a -ptx -O3`，指令原样贴入）：

```nasm
.visible .entry spacesKernel(int*, int const*, int)(
        .param .u64 spacesKernel(int*, int const*, int)_param_0,
        .param .u64 spacesKernel(int*, int const*, int)_param_1,
        .param .u32 spacesKernel(int*, int const*, int)_param_2
)
{
        ld.param.u64    %rd1, [spacesKernel(int*, int const*, int)_param_0]; // .param：out
        ld.param.u64    %rd2, [spacesKernel(int*, int const*, int)_param_1]; // .param：in
        cvta.to.global.u64      %rd3, %rd1;         // out 转 global 地址
        cvta.to.global.u64      %rd4, %rd2;         // in 转 global 地址
        mov.u32         %r1, %tid.x;                // %tid.x（.sreg）
        mul.wide.s32    %rd5, %r1, 4;               // i*4（字节偏移）
        add.s64         %rd6, %rd4, %rd5;           // in[i] 的 64 位地址
        ld.global.u32   %r2, [%rd6];                // .global：读 in[i]
        ld.const.u32    %r3, [c_scale];             // .const：读常量 c_scale
        mul.lo.s32      %r4, %r3, %r2;              // c_scale * in[i]
        add.s64         %rd7, %rd3, %rd5;           // out[i] 的 64 位地址
        st.global.u32   [%rd7], %r4;                // .global：写 out[i]
        ret;
}
```

三个空间的性格，在反汇编里一目了然：

**`.param`——参数入场的通道。** 三个入参 `out`、`in`、`n` 都声明在 `.param` 空间，用 `ld.param` 读进来。注意一个细节：第三个参数 `n` 在源码里根本没用到，所以编译器**没有**为它生成 `ld.param`——只有被真正用到的参数才会被读入。

**`.global`——要算 64 位地址才能碰。** `in`、`out` 的访问都要先 `cvta.to.global` 转地址，再 `mul.wide` + `add.s64` 算出 64 位字节地址，最后 `ld.global`/`st.global`。全局空间是片下的大空间，地址计算这套流程一步不能少。

**`.const`——直接按符号名取。** 对比之下，读常量只用了 `ld.const.u32 %r3, [c_scale]` 一条指令，直接写符号名 `c_scale`，既没有 `cvta`，也没有 64 位地址计算。常量空间有一条独立的、更短的只读路径——这正是 §5.1.3 说的「只读、用 `ld.const` 访问」在指令层面的投影。

**`.local`——缺席了。** 源码里明明声明了 `int local_arr[8]`，反汇编里却没有一条 `st.local` 或 `ld.local`。原因是编译器看穿了这段代码：`local_arr[i & 7]` 先被写、紧接着又被读，中间没有别的线程碰它，于是直接把 `in[i]` 留在寄存器 `%r2` 里中转，整个数组压根不需要落地。这反过来说明 `.local` 的本质——它是「寄存器放不下、或必须按地址访问」时才启用的兜底。想要看到 `.local` 真正现身，得让数组大到放不进寄存器（上一期那个 `a[64]` 的溢出内核就是，`st.local.v4.u32` 冒了出来）。

## 小结

- `.const`：只读、host 初始化、`ld.const` 直接符号访问；容量 64 KB（静态）+ 640 KB（10 个独立 64 KB 区域，driver 分配）。
- `.global`：context 内所有线程共享，是跨 CTA/cluster/grid 通信的机制；`ld.global`/`st.global`/`atom.global`，默认 0 初始化。
- `.local`：每线程私有的溢出兜底，寄存器放不下的数据落到这里；ABI 下分配在栈上。
- `.param` 三种用途：host→kernel 传参（只读、per-grid）、设备函数形参/返回值（读写、per-thread）、传大结构体的字节数组；用 `::entry`/`::func` 子限定符区分。
- 反汇编里，`.param` 读参数、`.global` 算 64 位地址、`.const` 直接符号寻址；`local_arr` 因「写后立即读」被优化掉，反证 `.local` 只在寄存器放不下时才启用。
