---
layout: post
title: "编程模型·内存层级：寄存器到全局内存的距离"
permalink: /ptx/03-memory-hierarchy/
date: 2026-08-29 18:00:00 +0800
tags: [PTX]
---

> 本文是 [PTX 专栏](/ptx/) 第 3 期，对应官方文档 §2.3。

## TL;DR

GPU 的数据存储不是平铺的一大块，而是按「谁看得见、活多久」分层的一张阶梯：**线程私有**的 local 离执行单元最近，**CTA 内共享**的 shared 次之，**所有线程共享**的 global 最远、容量最大。此外还有 constant、param、texture、surface 这几个所有线程都能访问的状态空间。PTX 不靠「猜」，而是让每一条 `ld`/`st` 指令用后缀（`.global`/`.shared`/`.const`/`.param`）显式声明自己访问哪一层。本期讲清这张阶梯。

<!-- more -->

## 一、概念铺垫：为什么要分层

存储有两个互相打架的指标：**速度**和**容量**。越快、越靠近计算单元的存储，单位成本越高，容量越小；越慢、越远的存储，容量越大、越便宜。没有任何一种存储能同时满足「快、大、便宜」，所以硬件只能把它们排成一张阶梯，让程序把「热数据」放近处、把「冷数据」放远处。

CPU 靠的是「大而多级的缓存 + 程序员无感的透明管理」：数据在内存和寄存器之间来来回回，多数情况下不用写代码的人操心。GPU 走的是另一条路——**显式暴露存储层级**，把「数据放在哪一层」这个决定权交给程序员（以及编译器）。好处是：热数据可以被精确地钉在 shared 里反复用，避免昂贵的全局内存往返；代价是：写代码的人得清楚每一层的可见范围和生命周期。

于是 §2.3 回答的核心问题就是：**我的数据能放在哪些地方，谁能看到它，它能活多久。**

## 二、文档语义：七种状态空间

§2.3 的措辞可以整理成三句话：**按线程粒度分层**、**各层生命周期不同**、**另有四个跨线程状态空间**。

**第一句，按线程粒度分层。** 每个线程有一块**私有的 local memory**；每个 CTA（thread block）有一块 **shared memory**，对 block 内所有线程可见，也对同一 cluster 内的所有活跃 block 可见，生命周期与 block 相同；而所有线程都能访问**同一块 global memory**。从 `sm_90` 起，这张图里插进了 cluster 这一层——cluster 内的 CTA 之间也能互相看到对方的 shared。

层级结构如下：

```text
thread
 └─ local：线程私有（寄存器优先，溢出才落 local memory）
        ▲
CTA / thread block
 └─ shared：block 内所有线程可见；cluster 内活跃 block 也可见
        ▲
grid
 └─ global：所有线程可见，容量最大
```

![Memory Hierarchy](/images/ptx/memory-hierarchy-with-clusters.png)

*图 3：Memory Hierarchy（官方文档 §2.3 Figure 3）——内存层级全景：线程私有的 local，CTA 的 shared（cluster 内活跃 block 也可见），以及所有线程共享的 global；自 `sm_90` 起引入 cluster 层。*

**第二句，生命周期。** global、constant、texture 这三个状态空间**跨 kernel 启动持久**——同一次应用先后多次启动 kernel，它们的数据还在。而 shared 的生命周期只跟 block 绑定：block 结束，shared 也就没了。

**第三句，四个额外状态空间。** 所有线程还能访问 constant、param、texture、surface：

- **constant**：只读。
- **param**：内核参数所在的空间（第 2 期反汇编里见过的 `.param .u64`）。
- **texture**：只读，提供不同的**寻址模式**（如归一化坐标）和针对特定数据格式的**数据过滤**。
- **surface**：可读可写。

这几个空间各自针对不同用途做了优化。其中有一条容易被忽略的**一致性陷阱**：texture 和 surface 的数据是**被缓存**的，而在同一次 kernel 调用内，这份缓存**不会**相对 global 写、surface 写保持连贯。也就是说——如果同一 kernel 里先写了某个 global 地址、再把它当 texture 读回来，读到的是**未定义数据**。安全的用法只有一种：这个位置必须由「上一次 kernel 调用」或「内存拷贝」更新过，而不是被同一 kernel 里的任何线程写过。

最后一段讲 host 与 device 的分工：host 和 device 各自维护自己的内存（host memory / device memory）。device memory 可以被 host **映射**后直接读写，也可以走设备的高性能 **DMA（直接内存访问）引擎**做更高效的批量拷贝。

## 三、反汇编与讲解：后缀标出每一层

下面这个内核，一口气踩过五个状态空间：参数（param）、全局内存（global）、共享内存（shared）、常量（constant），以及线程私有的局部变量。

CUDA 源码：

```cuda
__constant__ float kConst = 3.0f;

__global__ void memKernel(float *out, const float *in) {
    __shared__ float tile[128];
    int i = threadIdx.x;
    tile[i] = in[i];
    __syncthreads();
    out[i] = tile[(i + 1) & 127] * kConst;
}
```

真实 PTX（Compiler Explorer，NVCC 13.3.0，`-arch=sm_90a -ptx -O3`，指令原样贴入，行尾 `//` 注释是我标注的对应 CUDA 源码行）：

```nasm
.visible .entry memKernel(float*, float const*)(
        .param .u64 memKernel(float*, float const*)_param_0,
        .param .u64 memKernel(float*, float const*)_param_1
)
{
        ld.param.u64    %rd1, [memKernel(float*, float const*)_param_0];   // 读参数 out
        ld.param.u64    %rd2, [memKernel(float*, float const*)_param_1];   // 读参数 in
        cvta.to.global.u64      %rd3, %rd1;         // out 转 global 地址
        cvta.to.global.u64      %rd4, %rd2;         // in 转 global 地址
        mov.u32         %r1, %tid.x;                // int i = threadIdx.x;
        mul.wide.s32    %rd5, %r1, 4;               // i * 4（字节偏移）
        add.s64         %rd6, %rd4, %rd5;           // in[i] 的字节地址
        ld.global.f32   %f1, [%rd6];                // 读 in[i]
        shl.b32         %r2, %r1, 2;                // i * 4（字节偏移）
        mov.u32         %r3, memKernel(float*, float const*)::tile;   // tile 的 shared 基址
        add.s32         %r4, %r3, %r2;              // tile[i] 的 shared 地址
        st.shared.f32   [%r4], %f1;                 // tile[i] = in[i]
        bar.sync        0;                          // __syncthreads()
        add.s32         %r5, %r2, 4;                // (i+1)*4
        and.b32         %r6, %r5, 508;              // ((i+1) & 127) 的字节偏移
        add.s32         %r7, %r3, %r6;              // tile[(i+1)&127] 的 shared 地址
        ld.shared.f32   %f2, [%r7];                 // 读 tile[(i+1)&127]
        ld.const.f32    %f3, [kConst];              // 读常量 kConst
        mul.f32         %f4, %f2, %f3;              // ... * kConst
        add.s64         %rd7, %rd3, %rd5;           // out[i] 的字节地址
        st.global.f32   [%rd7], %f4;                // out[i] = 结果
        ret;
}
```

逐行对应已标在注释里，这里只补几个注释里放不下的语义：

- `mov.u32 %r3, memKernel(...)::tile`：`::tile` 是编译器为 shared 数组分配的符号，取出来就是它在 shared 空间里的基址。
- `and.b32 %r6, %r5, 508`：这是「取模 128」的强度削减——128 是 2 的幂，`(i+1) & 127` 等价于模 128，字节偏移上再乘 4 就是 `& 508`。
- `ld.const.f32 %f3, [kConst]`：直接以符号名 `kConst` 寻址，不像 global 那样需要 `cvta` 转换——constant 空间有独立的寻址方式。
- `bar.sync 0` 是 `__syncthreads()` 编译成的同步屏障（第 35 期展开）；`cvta.to.global` 的地址转换细节留到第 32 期。

把后缀和状态空间对照起来，一目了然：

| `ld/st` 后缀 | 状态空间 | 可见范围 | 读写 |
|--------------|----------|----------|------|
| `.param` | param | 所有线程（内核参数） | 读为主 |
| `.global` | global | 所有线程 | 可读写 |
| `.shared` | shared | block 内 + cluster 内活跃 block | 可读写 |
| `.const` | constant | 所有线程 | 只读 |
| （寄存器） | local/register | 线程私有 | 可读写 |

这正印证了 TL;DR 里的那句话：**PTX 用指令后缀显式声明状态空间**，一条 `ld` 到底是读全局内存还是读共享内存，看后缀一眼便知，不存在「默认空间」。

关于 local 这一层，值得多说一句：反汇编里 `int i` 落在寄存器 `%r1`，而不是「local memory」。这是 PTX 的一个精细之处——线程私有的存储实际分两档：**寄存器**（`.reg`，最快，绝大多数局部变量都在这）和 **local memory**（`.local`，寄存器放不下、或需要按地址访问时才用，物理上落在 device memory 的线程私有区域）。§2.3 用「private local memory」一词概括这一整层，具体到 `.reg` 和 `.local` 的区分，留到第 9–10 期状态空间章节再展开。

另外补一个能说明「shared 是软件管理的」的点：这个例子里 `-O3` 和 `-O0` 产出的 PTX 几乎一模一样，shared 的 `st`/`ld` 都被原样保留。原因在于 `tile[i]` 被 `__syncthreads()` 之后的**另一个线程**读取——数据跨线程流动，编译器无法把它优化成纯寄存器中转。反过来说，如果 shared 只被「自己写、自己读」，编译器就完全可能把它消掉（上一版去掉 `__syncthreads()` 后，`-O3` 直接把 `tile` 折叠成一条 `fma`）。这提醒我们：**shared 的价值在于跨线程共享数据**，它的生命周期和可见性才是它不可被省略的根本原因。

## 小结

- 存储按「可见范围 + 生命周期」分层：**local（线程私有）→ shared（block 内 + cluster 内活跃 block）→ global（所有线程）**。
- 另有四个跨线程状态空间：constant（只读）、param（内核参数）、texture（只读、带寻址与过滤）、surface（可读写）。
- global/constant/texture 跨 kernel 持久；shared 随 block 生灭。
- texture/surface 被缓存且同 kernel 内不相对 global/surface 写保持一致，同一 kernel 里「先写 global 再读 texture」会得到未定义数据。
- PTX 用 `ld`/`st` 的后缀（`.global`/`.shared`/`.const`/`.param`）显式声明状态空间，反汇编里一眼可辨。
