---
layout: post
title: "操作数·类型转换与舍入：cvt 的格式、舍入与代价"
permalink: /ptx/16-type-conversion-rounding/
date: 2026-10-19 21:00:00 +0800
tags: [PTX]
---

> 本文是 [PTX 专栏](/ptx/) 第 16 期，对应官方文档 §6.5–6.6（§6.5.1–§6.5.2、§6.6）。

## TL;DR

算术、逻辑、数据搬移指令的操作数必须同类型、同大小，除非「改变类型或大小」本身就是这条指令的定义。类型或大小不同的操作数，必须先转换。PTX 的通用转换指令是 `cvt`，写法为 `cvt{.舍入修饰}.目标类型.源类型`，例如 `cvt.rni.s32.f32`。表 15、表 16 规定哪些源/目标格式组合可以直接转换，以及转换行为：`sext`（符号扩展）、`zext`（零扩展）、`chop`（只保留低位）、`s2f`/`u2f`（有符号/无符号整数转浮点）、`f2s`/`f2u`（浮点转有符号/无符号整数）、`f2f`（浮点转浮点）。浮点舍入修饰符有 `.rn`、`.rna`、`.rz`、`.rm`、`.rp`、`.rs`；整数舍入修饰符有 `.rni`、`.rzi`、`.rmi`、`.rpi`。§6.6 给出不同状态空间操作数的代价估计：寄存器、共享、常量、参数、立即数记为 0，本地、全局、纹理、表面记为 >100 clocks；内存延迟可以用多线程和提前发射 load 来隐藏。

<!-- more -->

## 一、概念铺垫：为什么需要 cvt

前几期讲了操作数的类型规则和地址写法，但有一个前提没有展开：**一条指令的操作数必须类型一致、大小一致**。C 语言里 `int + float` 会自动提升，到了 PTX 没有这种自动提升；类型或大小不一致时，必须先用 `cvt` 转换。

`cvt` 的名字来自 convert，指令结构如下：

```text
cvt.rni.s32.f32   %r1, %f1
└─┬─┘ └┬─┘ └┬┘ └┬┘   └┬─┘  └┬─┘
 cvt  舍入 目标 源    目标   源
      修饰 类型 类型  寄存器 寄存器
```

点号后面依次是：可选的舍入修饰、目标类型、源类型。上例把 `f32` 源转换成 `s32` 目标，舍入方式为 `.rni`（就近取整，等距时取偶数）。

这里要和第 14 期的规则分清：**同大小且兼容的类型可以静默 cast**（例如 `.u32` 直接当 `.s32` 用，不生成指令）；**改变大小，或跨整数/浮点类别时，必须用 `cvt`**。本节讲的正是后一种情况。

## 二、文档语义

### §6.5 总则：操作数先统一类型和大小

文档开宗明义：所有算术、逻辑、数据搬移指令的操作数，都必须是相同类型、相同大小；只有那些「改变大小或类型是指令定义的一部分」的操作除外。大小或类型不同的操作数，必须在执行操作之前完成转换。

`mul.wide` 属于「改变大小是定义的一部分」的例子：它明确要求结果比源宽一倍，所以不需要先 `cvt`。`cvt` 本身也是例外：它的全部工作就是改变类型或大小。

### §6.5.1 标量转换：表 15 和表 16

`cvt` 的目标类型和源类型由助记符后缀给出，例如 `cvt.s32.u16` 表示「源 `u16`，目标 `s32`」。这个例子中，`u16` 会被零扩展到 `s32`。

表 15、表 16 规定不同源/目标格式组合下 `cvt` 使用的精度和格式。转换行为可以归成四类：

```text
整数 → 更宽整数：sext（符号扩展）或 zext（零扩展）
整数 → 更窄整数：chop（只保留放得下的低位）
整数 ↔ 浮点：s2f / u2f / f2s / f2u
浮点 ↔ 浮点：f2f
```

| 转换类别 | 行为 | `cvt` 示例 |
|----------|------|------------|
| 有符号整数 → 更宽整数 | 符号扩展（sext） | `cvt.s64.s32` |
| 无符号整数 → 更宽整数 | 零扩展（zext） | `cvt.u64.u32` |
| 任意整数 → 更窄整数 | 截断（chop） | `cvt.u16.u32` |
| 有符号整数 → 浮点 | s2f | `cvt.rn.f32.s32` |
| 无符号整数 → 浮点 | u2f | `cvt.rn.f32.u32` |
| 浮点 → 有符号整数 | f2s | `cvt.rzi.s32.f32` |
| 浮点 → 无符号整数 | f2u | `cvt.rni.u32.f32` |
| 浮点 → 浮点 | f2f | `cvt.rn.f16.f32` |

表 15 的行是源格式，列是目标格式，单元格给出转换行为：行 `s32`、列 `s64` 是 `sext`，行 `s32`、列 `f32` 是 `s2f`，行 `f32`、列 `s32` 是 `f2s`。表 15 覆盖 `s8/s16/s32/s64`、`u8/u16/u32/u64`、`f16/f32/f64/bf16/tf32` 等格式；其中为 `–` 的单元格表示没有直接转换。表 16 覆盖另一组浮点格式：`f16`、`f32`、`bf16`、`e4m3`、`e5m2`、`e2m3`、`e3m2`、`e2m1`、`ue8m0`、`s2f6`，它们之间的组合同样以 `f2f` 或 `–` 标记。表描述的是 `cvt` 指令的行为；编译器也可能用其他指令实现同样的转换语义，示例一里的 `and` 就是截断的一种实现。

`chop` 的含义是只保留低位。例如把一个 32 位值截到 16 位，结果只保留低 16 位，高 16 位被丢弃；`0x12345678` 截到 16 位得到 `0x5678`。

表 15 有一条关于「目标寄存器比目标格式宽」的说明：结果先按目标格式截断，再扩展到寄存器宽度；扩展方式（符号还是零）由**目标格式**决定。文档的例子是 `cvt.s16.u32` 写入 32 位寄存器：先把 `u32` 截到 16 位，再按目标格式 `s16` 符号扩展到 32 位。

转换到浮点时，如果值超出浮点数的表示范围，文档规定用最大浮点值表示：`f32` 和 `f64` 用 IEEE 754 的 Inf，`f16` 用约 131,000。

### §6.5.2 舍入修饰符

转换指令可以带舍入修饰符。PTX 有四种整数舍入修饰符和六种浮点舍入修饰符。

浮点舍入修饰符（表 17）：

| 修饰符 | 含义 |
|--------|------|
| `.rn` | 就近舍入，等距时取偶数 |
| `.rna` | 就近舍入，等距时远离零 |
| `.rz` | 朝零舍入 |
| `.rm` | 朝负无穷舍入 |
| `.rp` | 朝正无穷舍入 |
| `.rs` | 把随机位与尾数丢弃位做整数加法，根据进位决定朝零还是远离零 |

整数舍入修饰符（表 18）：

| 修饰符 | 含义 |
|--------|------|
| `.rni` | 就近取整，等距时取偶数 |
| `.rzi` | 朝零取整 |
| `.rmi` | 朝负无穷取整 |
| `.rpi` | 朝正无穷取整 |

浮点转整数时，用 `.rni/.rzi/.rmi/.rpi` 指定「浮点值怎么变成整数」；浮点转浮点时，用 `.rn/.rna/.rz/.rm/.rp/.rs` 指定「结果落到目标浮点精度时的舍入方式」。整数之间的扩展或截断不涉及舍入，不需要修饰符；有浮点参与的转换才需要按表 17、表 18 选择。

### §6.6 操作数代价：不同状态空间的距离

操作数来自不同状态空间时，操作的快慢不同。文档给出的相对代价估计如下：

```text
寄存器   | 0
立即数   | 0
参数     | 0
共享     | 0
常量     | 0（首次访问成本高，之后摊销）
本地     |> 100 clocks
全局     |> 100 clocks
纹理     |> 100 clocks
表面     |> 100 clocks
```

完整的表 19：

| 空间 | 时间 | 说明 |
|------|------|------|
| 寄存器 | 0 | |
| 共享 | 0 | |
| 常量 | 0 | 摊销成本低，首次访问高 |
| 本地 | > 100 clocks | |
| 参数 | 0 | |
| 立即数 | 0 | |
| 全局 | > 100 clocks | |
| 纹理 | > 100 clocks | |
| 表面 | > 100 clocks | |

寄存器最快，全局内存最慢。内存访问的大部分延迟可以通过多种方式隐藏：

1. **多线程**：硬件发出一个内存操作后，可以切换到其他线程执行；
2. **提前发射 load**：把 `ld` 尽量往前放，因为执行不会在发出 load 时阻塞，直到后续指令真正用到结果才需要等待；
3. **store 的寄存器很快可用**：`st` 用到的寄存器不需要等内存写完就能继续使用。

表中的数字是文档给出的估计值，不是精确周期数；它表达的是不同状态空间之间的量级差异。

## 三、反汇编与讲解

### 示例一：符号扩展、零扩展与截断

CUDA 源码：

```cuda
__global__ void intConv(long long *out, int a, unsigned int b,
                        unsigned int e, unsigned int x, unsigned int y) {  // L1
    out[0] = a + 1;                    // L2
    out[1] = b + 1u;                   // L3
    out[2] = (unsigned short)(e + 1u); // L4
    out[3] = (short)(x + y);           // L5
}

__global__ void chopSign(int *out, unsigned int x, unsigned int y) {  // L6
    out[0] = (short)(x + y);           // L7
}
```

真实 PTX（Compiler Explorer，NVCC 13.3.0，`-arch=sm_90a -ptx -O3`，指令原样贴入）：

{% raw %}
```nasm
.visible .entry intConv(long long*, int, unsigned int, unsigned int, unsigned int, unsigned int)(
        .param .u64 intConv(long long*, int, unsigned int, unsigned int, unsigned int, unsigned int)_param_0,
        .param .u32 intConv(long long*, int, unsigned int, unsigned int, unsigned int, unsigned int)_param_1,
        .param .u32 intConv(long long*, int, unsigned int, unsigned int, unsigned int, unsigned int)_param_2,
        .param .u32 intConv(long long*, int, unsigned int, unsigned int, unsigned int, unsigned int)_param_3,
        .param .u32 intConv(long long*, int, unsigned int, unsigned int, unsigned int, unsigned int)_param_4,
        .param .u32 intConv(long long*, int, unsigned int, unsigned int, unsigned int, unsigned int)_param_5
)
{
        ld.param.u64    %rd1, [intConv(long long*, int, unsigned int, unsigned int, unsigned int, unsigned int)_param_0];  // L1: out
        ld.param.u32    %r1, [intConv(long long*, int, unsigned int, unsigned int, unsigned int, unsigned int)_param_1];   // L1: a
        ld.param.u32    %r2, [intConv(long long*, int, unsigned int, unsigned int, unsigned int, unsigned int)_param_2];   // L1: b
        ld.param.u32    %r3, [intConv(long long*, int, unsigned int, unsigned int, unsigned int, unsigned int)_param_3];   // L1: e
        ld.param.u32    %r4, [intConv(long long*, int, unsigned int, unsigned int, unsigned int, unsigned int)_param_4];   // L1: x
        ld.param.u32    %r5, [intConv(long long*, int, unsigned int, unsigned int, unsigned int, unsigned int)_param_5];   // L1: y
        cvta.to.global.u64      %rd2, %rd1;
        add.s32         %r6, %r1, 1;            // L2: a + 1（32 位加法）
        cvt.s64.s32     %rd3, %r6;              // L2: s32 → s64，符号扩展
        st.global.u64   [%rd2], %rd3;           // L2: out[0]
        add.s32         %r7, %r2, 1;            // L3: b + 1
        cvt.u64.u32     %rd4, %r7;              // L3: u32 → u64，零扩展
        st.global.u64   [%rd2+8], %rd4;         // L3: out[1]
        add.s32         %r8, %r3, 1;            // L4: e + 1
        and.b32         %r9, %r8, 65535;        // L4: 截到低 16 位（chop）
        cvt.u64.u32     %rd5, %r9;              // L4: 零扩展到 u64
        st.global.u64   [%rd2+16], %rd5;        // L4: out[2]
        add.s32         %r10, %r5, %r4;         // L5: y + x
        cvt.u64.u32     %rd6, %r10;             // L5: 先零扩展成 64 位
        cvt.s64.s16     %rd7, %rd6;             // L5: 保留低 16 位并按 s16 符号扩展
        st.global.u64   [%rd2+24], %rd7;        // L5: out[3]
        ret;
}

.visible .entry chopSign(int*, unsigned int, unsigned int)(
        .param .u64 chopSign(int*, unsigned int, unsigned int)_param_0,
        .param .u32 chopSign(int*, unsigned int, unsigned int)_param_1,
        .param .u32 chopSign(int*, unsigned int, unsigned int)_param_2
)
{
        ld.param.u64    %rd1, [chopSign(int*, unsigned int, unsigned int)_param_0];  // L6: out
        ld.param.u32    %r1, [chopSign(int*, unsigned int, unsigned int)_param_1];   // L6: x
        ld.param.u32    %r2, [chopSign(int*, unsigned int, unsigned int)_param_2];   // L6: y
        cvta.to.global.u64      %rd2, %rd1;
        add.s32         %r3, %r2, %r1;          // L7: y + x
        cvt.s32.s16     %r4, %r3;               // L7: 取低 16 位，按 s16 符号扩展到 s32
        st.global.u32   [%rd2], %r4;            // L7: out[0]
        ret;
}
```
{% endraw %}

**`cvt.s64.s32`：有符号扩展。** `a + 1` 先按 32 位算出 `%r6`，再转成 64 位。结果要存进 `long long`，所以必须补足高 32 位；源是 `s32`，用符号扩展。

**`cvt.u64.u32`：零扩展。** `b + 1u` 的结果是 `u32`，转成 `u64` 时高 32 位补零。扩展方式由源类型的有符号性决定：有符号源用 sext，无符号源用 zext。

**`and.b32 ..., 65535`：截断的另一种写法。** `(unsigned short)` 只保留低 16 位，语义上对应表 15 的 `chop`。编译器没有生成 `cvt.u16.u32`，而是用 `and 0xFFFF` 完成截断，再用 `cvt.u64.u32` 零扩展到 64 位。指令选择不同，语义相同。

**`cvt.u64.u32` + `cvt.s64.s16`：先扩展到 64 位，再做有符号截断。** `(short)(x + y)` 只保留 `x+y` 的低 16 位，并按 `s16` 解释。`cvt.s64.s16` 完成「保留低 16 位 + 符号扩展」，前面的 `cvt.u64.u32` 先把源准备成 64 位寄存器宽度。

**`cvt.s32.s16`：截断并符号扩展到 32 位。** `chopSign` 把 `x+y` 的低 16 位当作 `s16`，再符号扩展到 `s32`。这条指令同时体现了表 15 的两种行为：`chop`（只保留低位）和 `sext`（按有符号源做符号扩展）。

### 示例二：浮点与整数互转的舍入

CUDA 源码：

```cuda
__global__ void floatConv(int *iout, unsigned int *uout, float *fout,
                          float f, int s, unsigned int u) {  // L1
    iout[0] = __float2int_rn(f);   // L2
    iout[1] = __float2int_rz(f);   // L3
    iout[2] = __float2int_rd(f);   // L4
    iout[3] = __float2int_ru(f);   // L5
    uout[0] = __float2uint_rn(f);  // L6
    fout[0] = (float)s;            // L7
    fout[1] = (float)u;            // L8
}
```

真实 PTX（同编译器与参数）：

{% raw %}
```nasm
.visible .entry floatConv(int*, unsigned int*, float*, float, int, unsigned int)(
        .param .u64 floatConv(int*, unsigned int*, float*, float, int, unsigned int)_param_0,
        .param .u64 floatConv(int*, unsigned int*, float*, float, int, unsigned int)_param_1,
        .param .u64 floatConv(int*, unsigned int*, float*, float, int, unsigned int)_param_2,
        .param .f32 floatConv(int*, unsigned int*, float*, float, int, unsigned int)_param_3,
        .param .u32 floatConv(int*, unsigned int*, float*, float, int, unsigned int)_param_4,
        .param .u32 floatConv(int*, unsigned int*, float*, float, int, unsigned int)_param_5
)
{
        ld.param.u64    %rd1, [floatConv(int*, unsigned int*, float*, float, int, unsigned int)_param_0];  // L1: iout
        ld.param.u64    %rd2, [floatConv(int*, unsigned int*, float*, float, int, unsigned int)_param_1];  // L1: uout
        ld.param.u64    %rd3, [floatConv(int*, unsigned int*, float*, float, int, unsigned int)_param_2];  // L1: fout
        ld.param.f32    %f1, [floatConv(int*, unsigned int*, float*, float, int, unsigned int)_param_3];   // L1: f
        ld.param.u32    %r1, [floatConv(int*, unsigned int*, float*, float, int, unsigned int)_param_4];   // L1: s
        ld.param.u32    %r2, [floatConv(int*, unsigned int*, float*, float, int, unsigned int)_param_5];   // L1: u
        cvta.to.global.u64      %rd4, %rd3;
        cvta.to.global.u64      %rd5, %rd2;
        cvta.to.global.u64      %rd6, %rd1;
        cvt.rni.s32.f32         %r3, %f1;       // L2: 就近取整，等距取偶
        st.global.u32   [%rd6], %r3;            // L2: iout[0]
        cvt.rzi.s32.f32         %r4, %f1;       // L3: 朝零取整
        st.global.u32   [%rd6+4], %r4;          // L3: iout[1]
        cvt.rmi.s32.f32         %r5, %f1;       // L4: 朝负无穷取整
        st.global.u32   [%rd6+8], %r5;          // L4: iout[2]
        cvt.rpi.s32.f32         %r6, %f1;       // L5: 朝正无穷取整
        st.global.u32   [%rd6+12], %r6;         // L5: iout[3]
        cvt.rni.u32.f32         %r7, %f1;       // L6: 无符号目标，就近取整
        st.global.u32   [%rd5], %r7;            // L6: uout[0]
        cvt.rn.f32.s32  %f2, %r1;               // L7: s32 → f32，就近取偶
        st.global.f32   [%rd4], %f2;            // L7: fout[0]
        cvt.rn.f32.u32  %f3, %r2;               // L8: u32 → f32，就近取偶
        st.global.f32   [%rd4+4], %f3;          // L8: fout[1]
        ret;
}
```
{% endraw %}

**四个 `cvt` 的助记符只差舍入修饰。** `cvt.rni.s32.f32`、`cvt.rzi.s32.f32`、`cvt.rmi.s32.f32`、`cvt.rpi.s32.f32` 的源和目标格式完全相同，都是 `f32 → s32`，区别只在整数舍入修饰：

- `__float2int_rn` → `.rni`：就近取整，等距取偶；
- `__float2int_rz` → `.rzi`：朝零；
- `__float2int_rd` → `.rmi`：朝负无穷；
- `__float2int_ru` → `.rpi`：朝正无穷。

**无符号目标把 `s32` 换成 `u32`。** `__float2uint_rn` 生成 `cvt.rni.u32.f32`；舍入修饰仍是 `.rni`，目标格式变成 `u32`。目标是有符号还是无符号，由 `cvt` 的目标类型决定。

**整数转浮点用浮点舍入修饰。** `(float)s` 生成 `cvt.rn.f32.s32`，`(float)u` 生成 `cvt.rn.f32.u32`。源是有符号还是无符号，由 `cvt` 的源类型决定；两者都用了 `.rn`，即就近取偶。

### 示例三：f32 与 f16 之间的转换

CUDA 源码：

```cuda
#include <cuda_fp16.h>
__global__ void halfConv(__half *out, float *fout, float x, float y, __half z) {  // L1
    out[0] = __float2half_rn(x);   // L2
    out[1] = __float2half_rz(y);   // L3
    fout[0] = __half2float(z);     // L4
}
```

真实 PTX（同编译器与参数）：

{% raw %}
```nasm
.visible .entry halfConv(__half*, float*, float, float, __half)(
        .param .u64 halfConv(__half*, float*, float, float, __half)_param_0,
        .param .u64 halfConv(__half*, float*, float, float, __half)_param_1,
        .param .f32 halfConv(__half*, float*, float, float, __half)_param_2,
        .param .f32 halfConv(__half*, float*, float, float, __half)_param_3,
        .param .align 2 .b8 halfConv(__half*, float*, float, float, __half)_param_4[2]
)
{
        ld.param.u16    %rs3, [halfConv(__half*, float*, float, float, __half)_param_4];  // L1: z
        ld.param.u64    %rd1, [halfConv(__half*, float*, float, float, __half)_param_0];  // L1: out
        ld.param.u64    %rd2, [halfConv(__half*, float*, float, float, __half)_param_1];  // L1: fout
        ld.param.f32    %f1, [halfConv(__half*, float*, float, float, __half)_param_2];   // L1: x
        ld.param.f32    %f2, [halfConv(__half*, float*, float, float, __half)_param_3];   // L1: y
        cvta.to.global.u64      %rd3, %rd2;
        cvta.to.global.u64      %rd4, %rd1;
        {  cvt.rn.f16.f32 %rs1, %f1;}       // L2: f32 → f16，就近取偶
        st.global.u16   [%rd4], %rs1;       // L2: out[0]
        {  cvt.rz.f16.f32 %rs2, %f2;}       // L3: f32 → f16，朝零
        st.global.u16   [%rd4+2], %rs2;     // L3: out[1]
        {  cvt.f32.f16 %f3, %rs3;}          // L4: f16 → f32
        st.global.f32   [%rd3], %f3;        // L4: fout[0]
        ret;
}
```
{% endraw %}

**f32 → f16 带浮点舍入修饰。** `__float2half_rn` 生成 `cvt.rn.f16.f32`，`__float2half_rz` 生成 `cvt.rz.f16.f32`。目标格式 `f16` 只有 16 位，`f32` 的尾数位放不下，必须决定丢弃位怎么处理；`.rn` 和 `.rz` 就是两种选择。

**f16 → f32 不需要舍入修饰。** `__half2float` 生成 `cvt.f32.f16`，没有点号修饰。`f16` 能表示的值在 `f32` 中都能精确表示，这个方向不涉及「丢弃尾数位后如何舍入」。

**目标寄存器宽度跟着目标格式走。** `f16` 结果写进 16 位寄存器 `%rs1`/`%rs2`，用 `st.global.u16` 存储；`f32` 结果写进 `%f3`，用 `st.global.f32`。`cvt` 的目标类型决定了结果寄存器的形态。

### 示例四：操作数来自哪个状态空间

CUDA 源码：

```cuda
__constant__ int c[32];                          // L1
__global__ void costs(int *out, int i) {         // L2
    __shared__ int s[32];                        // L3
    s[threadIdx.x] = i;                          // L4
    __syncthreads();                             // L5
    int x = s[threadIdx.x];                      // L6
    int y = c[i];                                // L7
    out[i] = x + y + 1;                          // L8
}
```

真实 PTX（同编译器与参数）：

{% raw %}
```nasm
.visible .entry costs(int*, int)(
        .param .u64 costs(int*, int)_param_0,
        .param .u32 costs(int*, int)_param_1
)
{
        ld.param.u64    %rd1, [costs(int*, int)_param_0];  // L2: out（参数空间）
        ld.param.u32    %r1, [costs(int*, int)_param_1];   // L2: i（参数空间）
        cvta.to.global.u64      %rd2, %rd1;
        mov.u32         %r2, %tid.x;               // L4: 特殊寄存器
        shl.b32         %r3, %r2, 2;               // L4: threadIdx.x × 4
        mov.u32         %r4, costs(int*, int)::s;  // L3: 共享数组地址
        add.s32         %r5, %r4, %r3;
        st.shared.u32   [%r5], %r1;                // L4: 写共享内存
        bar.sync        0;                         // L5
        ld.shared.u32   %r6, [%r5];                // L6: 读共享内存
        mul.wide.s32    %rd3, %r1, 4;              // L7: i × 4
        mov.u64         %rd4, c;                   // L7: 常量数组地址
        add.s64         %rd5, %rd4, %rd3;
        ld.const.u32    %r7, [%rd5];               // L7: 读常量内存
        add.s32         %r8, %r6, %r7;             // L8: 寄存器运算
        add.s32         %r9, %r8, 1;               // L8: 立即数 1
        add.s64         %rd6, %rd2, %rd3;
        st.global.u32   [%rd6], %r9;               // L8: 写全局内存
        ret;
}
```
{% endraw %}

这段指令序列把表 19 里的几类操作数都摆了出来：

**`ld.param.u32`：参数空间。** 内核参数 `i` 从 `.param` 空间读入，表中参数记为 0。

**`ld.shared.u32` / `st.shared.u32`：共享空间。** 数组 `s` 的地址来自 `mov.u32 %r4, costs(int*, int)::s`，读写都走共享内存，表中共享记为 0。

**`ld.const.u32`：常量空间。** `c` 是 `__constant__` 数组，`mov.u64 %rd4, c` 取得常量数组地址，`ld.const.u32` 读取。表中常量记为 0，但注明「摊销成本低，首次访问高」。

**`add.s32 ..., 1`：立即数。** 常量 `1` 直接编码在指令里，表中立即数记为 0。

**`st.global.u32`：全局空间。** 结果写回 `out`，表中全局记为 >100 clocks。

**寄存器运算。** `add.s32 %r8, %r6, %r7` 的操作数都在寄存器里，表中寄存器记为 0。

文档对延迟隐藏的三条建议在这个内核里也能对应上：多线程让硬件在某个线程等待内存时切换到其他线程执行；`ld.const` 越早发出越好；`st.global` 使用的 `%r9` 在指令发出后即可继续使用。表中的 0 和 >100 是量级估计，不代表精确周期。

## 小结

- 算术、逻辑、数据搬移指令的操作数必须同类型、同大小；改变大小或类型要先 `cvt`，除非改变本身就是指令定义的一部分（如 `mul.wide`）。
- `cvt{.舍入修饰}.目标类型.源类型`：`cvt.rni.s32.f32` 把 `f32` 按就近取整转成 `s32`。
- 表 15/16 定义允许的格式组合：sext、zext、chop、s2f、u2f、f2s、f2u、f2f；为 `–` 的组合没有直接转换。
- 扩展方式看源类型：有符号源 sext，无符号源 zext；截断（chop）只保留低位。目标寄存器比目标格式宽时，先截断再按目标格式扩展。
- 浮点舍入：`.rn`（就近取偶）、`.rna`（就近远离零）、`.rz`（朝零）、`.rm`（朝负无穷）、`.rp`（朝正无穷）、`.rs`（按随机位决定方向）。
- 整数舍入：`.rni`（就近取偶）、`.rzi`（朝零）、`.rmi`（朝负无穷）、`.rpi`（朝正无穷）。
- 超出浮点范围时，文档规定用最大浮点值表示：`f32`/`f64` 为 Inf，`f16` 约 131,000。
- 操作数代价：寄存器、共享、常量、参数、立即数记为 0；本地、全局、纹理、表面 >100 clocks。延迟可用多线程、提前 load 和 store 的快速寄存器释放来隐藏。
- 反汇编里：`cvt.s64.s32`/`cvt.u64.u32` 分别是符号/零扩展；`and 0xFFFF` 和 `cvt.s64.s16` 实现截断；四种浮点转整数舍入对应 `.rni/.rzi/.rmi/.rpi`；`cvt.rn.f16.f32`/`cvt.rz.f16.f32` 展示 `f2f` 舍入。
