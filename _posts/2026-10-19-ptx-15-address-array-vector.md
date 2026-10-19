---
layout: post
title: "操作数·地址、数组、向量：内存指令如何定位数据"
permalink: /ptx/15-address-array-vector/
date: 2026-10-19 20:00:00 +0800
tags: [PTX]
---

> 本文是 [PTX 专栏](/ptx/) 第 15 期，对应官方文档 §6.4（§6.4.1–§6.4.4）。

## TL;DR

标量变量作操作数很直接，内存操作数的关键在**地址**。PTX 的地址操作数有六种写法：`[var]`、`[reg]`、`[reg+immOff]`、`[var+immOff]`、`[immAddr]`、`var[immOff]`。地址按字节计，可以是 32 位或 64 位，必须按访存大小自然对齐，否则行为未定义。数组标识符本身是一个**地址常量**，下标只能是常量、寄存器、或寄存器加减常量，复杂下标要先算成地址。向量操作数用 `.x/.y/.z/.w`（或 `.r/.g/.b/.a`）取元素，用花括号列表拆装，向量 `ld`/`st` 可以一次搬多个元素。内存指令不写状态空间时走**通用寻址**：`.const`、`.param`、`.local`、`.shared` 是通用地址空间里的窗口，落在哪个窗口就映射到哪个状态空间，其余按 global 处理。标签只能作 `bra`/`brx.idx` 的操作数，函数名只能作 `call` 的操作数，`mov` 可以取函数地址供间接调用。

<!-- more -->

## 一、概念铺垫：地址是内存操作数的核心

一条内存指令要回答两个问题：访问**哪个**内存位置、访问**多少**字节。后者由指令类型决定；前者由地址操作数给出。

{% raw %}
```text
ld.global.v4.f32  {%f1, %f2, %f3, %f4}, [%rd6+16]
└──────┬───────┘  └────────┬────────┘   └───┬────┘
  指令 + 状态空间        目标（寄存器）      地址操作数
  （决定访存大小 16 字节） （接收 4 个 f32）  （reg + 立即数偏移）
```
{% endraw %}

标量变量作操作数时，直接写变量名就够了；一旦涉及地址、数组、向量，操作数就要编码「数据在哪、怎么排布」。§6.4 按这个顺序展开：地址的写法（§6.4.1）、通用寻址（§6.4.1.1）、数组索引（§6.4.2）、向量拆装（§6.4.3），最后是标签和函数名这两种只能出现在特定指令里的操作数（§6.4.4）。

## 二、文档语义

### §6.4.1 地址操作数：六种写法

所有内存指令都接受一个地址操作数，用来指定要访问的内存位置。文档列出六种形式：

| 写法 | 含义 |
|------|------|
| `[var]` | 可寻址变量 `var` 的名字 |
| `[reg]` | 存放字节地址的整数类型或位类型寄存器 `reg` |
| `[reg+immOff]` | 寄存器中的字节地址 + 常量整数字节偏移（有符号，32 位） |
| `[var+immOff]` | 变量 `var` 的地址 + 常量整数字节偏移（有符号，32 位） |
| `[immAddr]` | 立即数绝对字节地址（无符号，32 位） |
| `var[immOff]` | 数组元素，见 §6.4.2 |

寄存器可以用位类型（如 `.b32`、`.b64`）或整数类型（如 `.u32`、`.s64`）声明，两者都按位存放地址。

**访存大小**是这条指令在内存中访问的总字节数，由指令本身决定。例如 `ld.v4.b32` 的访存大小是 16 字节（4 个 32 位元素），`atom.f16x2` 的访存大小是 4 字节（2 个 16 位元素）。地址必须按访存大小**自然对齐**：地址是访存大小的整数倍。不对齐时行为未定义，文档给出的可能表现包括静默屏蔽低位地址来凑对齐，或者直接触发 fault。

**地址宽度**可以是 32 位或 64 位，不支持 128 位。地址需要时零扩展到指定宽度；寄存器宽度超过目标架构状态空间地址宽度时，地址被截断。同一段代码里可以同时出现 32 位和 64 位地址，转换时按这套规则扩展或截断（示例三会看到共享地址到通用地址的扩展）。

**地址运算按字节进行**，使用整数算术和逻辑指令（`add`、`mul`、`shl` 等），没有 C 风格的指针算术。C 里的 `p + i` 会按元素大小自动缩放；到了 PTX，这个缩放必须由编译器（或程序员）先算出来，变成「索引 × 元素字节数」，再和基地址相加。

`mov` 指令可以把变量的地址装进一个指针寄存器。这个地址是变量所在状态空间内的偏移量。`ld`/`st` 则在寄存器和可寻址状态空间之间搬数据，语法和许多汇编语言类似：变量直接写名字，地址表达式放在方括号里解引用。

文档给出的例子：

{% raw %}
```nasm
.shared .u16 x;
.reg    .u16 r0;
.global .v4 .f32 V;
.reg    .v4 .f32 W;
.const  .s32 tbl[256];
.reg    .b32 p;
.reg    .s32 q;

ld.shared.u16   r0,[x];
ld.global.v4.f32 W, [V];
ld.const.s32    q, [tbl+12];
mov.u32         p, tbl;
```
{% endraw %}

`[x]`、`[V]` 是 `[var]`，`[tbl+12]` 是 `[var+immOff]`；`mov.u32 p, tbl` 取的是数组 `tbl` 的地址，不是 `tbl` 里的数据。

### §6.4.1.1 通用寻址：状态空间是通用地址空间里的窗口

```text
通用地址空间（64 位地址）

┌──────────────────────────────────────────────┐
│ .global 窗口                                  │
│   ┌──────────────────────────────┐            │
│   │ .param 窗口（嵌在 .global 内）│            │
│   └──────────────────────────────┘            │
├──────────────────────────────────────────────┤
│ .const 窗口                                   │
├──────────────────────────────────────────────┤
│ .local 窗口                                   │
├──────────────────────────────────────────────┤
│ .shared 窗口                                  │
└──────────────────────────────────────────────┘

命中 .const / .local / .shared 窗口：
    通用地址 − 窗口基址 = 对应状态空间内的地址
命中 .global 窗口（含 .param）或不在任何窗口内：
    按 global 内存地址解释
```

内存指令**不写状态空间**时，操作按通用寻址进行。文档把 `.const`、内核函数参数（`.param`）、`.local`、`.shared` 建模为通用地址空间中的**窗口**：每个窗口由窗口基址和窗口大小定义，窗口大小等于对应状态空间的大小。通用地址落在 `.const`、`.local`、`.shared` 的窗口内时，映射到对应状态空间；落在这些窗口之外时，映射到 global 内存。`.param` 窗口包含在 `.global` 窗口内部。落在窗口内时，用通用地址减去窗口基址，就得到底层状态空间中的地址。

这解释了 `ld.global.u32` 和 `ld.u32` 的区别：前者直接按 global 地址访问，后者由通用寻址按窗口信息解析这个通用地址属于哪个状态空间。下面示例三里的 `ld.u32 %r1, [%rd1]` 就是通用寻址，同一个指令既能读共享内存指针，也能读全局内存指针。

### §6.4.2 数组：标识符就是地址常量

所有类型都可以声明数组。数组的标识符是它所声明状态空间中的一个**地址常量**；数组大小是程序中的常量。

数组元素有两种访问方式：显式计算出字节地址，或用方括号下标。方括号内的表达式只能是以下三种之一：

1. 常量整数；
2. 寄存器变量；
3. 简单的「寄存器 + 常量偏移」表达式，常量偏移可以是加到寄存器上，也可以是从寄存器上减掉。

更复杂的下标必须先写成地址计算，再用于访存。文档的例子：

{% raw %}
```nasm
ld.global.u32  s, a[0];
ld.global.u32  s, a[N-1];
mov.u32        s, a[1];  // move address of a[1] into s
```
{% endraw %}

`a[0]`、`a[N-1]` 是访存：把数组元素的值读进 `s`。`mov.u32 s, a[1]` 取的是 `a[1]` 的**地址**，因为 `mov` 不访问内存，方括号里的数组下标在这里参与的是地址计算。

### §6.4.3 向量：花括号列表与元素后缀

向量操作数既可以作源，也可以作目标。文档特别规定：**作目标操作数时，向量表达式中的所有元素必须互不相同**，否则行为未定义。向量也可以作为参数传给被调用的函数。

向量元素用 `.x`、`.y`、`.z`、`.w` 后缀提取，也可以用颜色字段 `.r`、`.g`、`.b`、`.a`。花括号列表用于模式匹配，把向量拆成若干标量：

{% raw %}
```nasm
.reg .v4 .f32 V;
.reg .f32     a, b, c, d;

mov.v4.f32 {a,b,c,d}, V;
```
{% endraw %}

花括号列表中的元素与向量元素按下标位置一一对应：

```text
Ra = V.x = V.r
Rb = V.y = V.g
Rc = V.z = V.b
Rd = V.w = V.a
```

向量 `ld`/`st` 可以用来实现宽 load/store，文档指出这可能改善内存性能。寄存器可以是向量变量，也可以是类型相同的标量组成的花括号列表：

{% raw %}
```nasm
ld.global.v4.f32  {a,b,c,d}, [addr+16];
ld.global.v2.u32  V2, [addr+8];
```
{% endraw %}

第一条一次读 16 字节（4 个 `.f32`），第二条一次读 8 字节（2 个 `.u32`）。`[addr+16]`、`[addr+8]` 同时展示了 `[reg+immOff]` 形式。

### §6.4.4 标签与函数名：只能出现在特定指令里

标签（label）和函数名（function name）是两种特殊操作数，使用范围受到限制：

- 标签只能用于 `bra` / `brx.idx` 指令，作为分支目标；
- 函数名只能用于 `call` 指令，作为被调用者；
- 函数名还可以用于 `mov` 指令，把函数地址装进寄存器，供间接调用使用。

从 PTX ISA 3.1 开始，`mov` 还可以取**内核函数**的地址，传给一个在 GPU 上发起内核启动的系统调用。这是 CUDA Dynamic Parallelism（动态并行）支持的一部分。

## 三、反汇编与讲解

### 示例一：数组下标的四种形态

CUDA 源码：

```cuda
__global__ void arrayIndex(int *out, const int *in, int i) {  // L1
    int x = in[3];              // L2
    int y = in[i];              // L3
    int z = in[i + 2];          // L4
    int w = in[i - 1];          // L5
    out[0] = x + y + z + w;     // L6
}
```

真实 PTX（Compiler Explorer，NVCC 13.3.0，`-arch=sm_90a -ptx -O3`，指令原样贴入）：

{% raw %}
```nasm
.visible .entry arrayIndex(int*, int const*, int)(
        .param .u64 arrayIndex(int*, int const*, int)_param_0,
        .param .u64 arrayIndex(int*, int const*, int)_param_1,
        .param .u32 arrayIndex(int*, int const*, int)_param_2
)
{
        ld.param.u64    %rd1, [arrayIndex(int*, int const*, int)_param_0];   // L1: out 指针
        ld.param.u64    %rd2, [arrayIndex(int*, int const*, int)_param_1];   // L1: in 指针
        ld.param.u32    %r1, [arrayIndex(int*, int const*, int)_param_2];    // L1: i
        cvta.to.global.u64      %rd3, %rd1;         // out：通用指针 → 全局地址
        cvta.to.global.u64      %rd4, %rd2;         // in：通用指针 → 全局地址
        ld.global.u32   %r2, [%rd4+12];             // L2: in[3] → 常量下标变立即数偏移 3*4=12
        mul.wide.s32    %rd5, %r1, 4;               // L3–L5: i × 4（int 的字节大小）
        add.s64         %rd6, %rd4, %rd5;           // L3–L5: in + i*4，得到 in[i] 的字节地址
        ld.global.u32   %r3, [%rd6];                // L3: in[i] → [reg]
        ld.global.u32   %r4, [%rd6+8];              // L4: in[i+2] → [reg+immOff]
        ld.global.u32   %r5, [%rd6+-4];             // L5: in[i-1] → [reg+immOff]（负偏移）
        add.s32         %r6, %r3, %r2;              // L6: y + x
        add.s32         %r7, %r6, %r4;              // L6: + z
        add.s32         %r8, %r7, %r5;              // L6: + w
        st.global.u32   [%rd3], %r8;                // L6: out[0] → [reg]
        ret;
}
```
{% endraw %}

这段代码把 §6.4.1 的地址形式和 §6.4.2 的下标规则全部落到了指令上：

**`[%rd4+12]`：常量下标变成立即数偏移。** `in[3]` 没有单独的地址计算，下标 3 乘以 `int` 的字节大小 4 得到 12，直接作为 `immOff` 写进地址操作数。这就是 §6.4.2 允许的「常量整数」下标，也是 §6.4.1 的 `[reg+immOff]` 形式。

**`mul.wide.s32` 与 `add.s64`：字节地址由整数指令算出。** `in[i]` 先算 `i × 4`，再加到基地址上。`mul.wide` 把 32 位的索引扩展成 64 位结果，`add.s64` 做 64 位加法。PTX 没有「指针 + 整数自动乘元素大小」的规则，缩放这一步在指令里是显式的。

**`[%rd6+8]` 与 `[%rd6+-4]`：寄存器加减常量。** `in[i+2]` 和 `in[i-1]` 都属于「寄存器 + 常量偏移」形式，编译器直接把 `+2`、`-1` 换算成字节偏移 `+8`、`-4`，没有为每个下标重新算一遍地址。`[%rd6+-4]` 里的负号落在有符号 32 位 `immOff` 上，地址本身仍是 64 位寄存器。

**`[%rd3]`：纯寄存器地址。** `out[0]` 的下标是 0，偏移被省掉，只剩 `[reg]`。同理，三个 `ld.param` 的 `[arrayIndex(...)_param_0]` 是 `[var]` 形式——内核参数名本身就是可寻址变量。

**对齐。** `ld.global.u32` 的访存大小是 4 字节，偏移 12、8、-4 都是 4 的倍数；基地址按 `int` 指针对齐时，这些访问都满足自然对齐。

### 示例二：float4 的宽 load/store 与元素重排

CUDA 源码：

```cuda
__global__ void vecCopy(float4 *out, const float4 *in, int i) {  // L1
    float4 v = in[i];       // L2
    v.x += 1.0f;            // L3
    v.w = v.y;              // L4
    out[i] = v;             // L5
}
```

真实 PTX（同编译器与参数）：

{% raw %}
```nasm
.visible .entry vecCopy(float4*, float4 const*, int)(
        .param .u64 vecCopy(float4*, float4 const*, int)_param_0,
        .param .u64 vecCopy(float4*, float4 const*, int)_param_1,
        .param .u32 vecCopy(float4*, float4 const*, int)_param_2
)
{
        ld.param.u64    %rd1, [vecCopy(float4*, float4 const*, int)_param_0];  // L1: out
        ld.param.u64    %rd2, [vecCopy(float4*, float4 const*, int)_param_1];  // L1: in
        ld.param.u32    %r1, [vecCopy(float4*, float4 const*, int)_param_2];   // L1: i
        cvta.to.global.u64      %rd3, %rd1;
        cvta.to.global.u64      %rd4, %rd2;
        mul.wide.s32    %rd5, %r1, 16;              // L2/L5: i × 16（sizeof(float4)）
        add.s64         %rd6, %rd4, %rd5;           // L2: &in[i]
        ld.global.v4.f32        {%f1, %f2, %f3, %f4}, [%rd6];  // L2: 一次读 16 字节
        add.s64         %rd7, %rd3, %rd5;           // L5: &out[i]
        add.f32         %f8, %f1, 0f3F800000;       // L3: v.x += 1.0f
        st.global.v4.f32        [%rd7], {%f8, %f2, %f3, %f2};  // L4–L5: v.w = v.y 后一次写 16 字节
        ret;
}
```
{% endraw %}

**宽访存。** `ld.global.v4.f32` 和 `st.global.v4.f32` 的访存大小都是 16 字节，一条指令搬完整个 `float4`。文档说这种方式可能改善内存性能，具体收益取决于访问模式和硬件，不能把它当成必然的加速手段。

**花括号列表按位置对应元素。** {% raw %}`ld.global.v4.f32 {%f1, %f2, %f3, %f4}, [%rd6]`{% endraw %} 执行后，`%f1` 是 `v.x`、`%f2` 是 `v.y`、`%f3` 是 `v.z`、`%f4` 是 `v.w`。这正是 §6.4.3 的映射关系，只是源码里的 `.x/.y/.z/.w` 后缀在 PTX 里被位置关系取代了。

**`v.x += 1.0f` 只改一个槽位。** `add.f32 %f8, %f1, 0f3F800000` 只处理 `v.x`，立即数 `0f3F800000` 是 `1.0f` 的位模式。其余元素保持原值。

**源向量可以重复，目标向量不行。** `v.w = v.y` 让 `st` 的第四个槽位也取 `%f2`，于是列表是 {% raw %}`{%f8, %f2, %f3, %f2}`{% endraw %}。这里向量是 `st` 的**源**操作数，重复元素合法；§6.4.3 的「元素必须互不相同」约束的是**目标**向量表达式，如果 `ld` 的目标列表里出现重复寄存器，行为未定义。

**对齐。** 访存大小是 16 字节，地址是 `基址 + i*16`；`float4` 指针对齐时，每个 `in[i]`/`out[i]` 都按 16 字节自然对齐。

### 示例三：通用寻址、数组地址常量与函数调用

CUDA 源码：

```cuda
__noinline__ __device__ int loadGen(const int *p) { return *p; }  // L1

__global__ void mixed(int *out, const int *in, int useShared, int i) {  // L3
    __shared__ int s[32];                         // L4
    s[threadIdx.x] = threadIdx.x;                 // L5
    __syncthreads();                              // L6
    const int *p = useShared ? &s[threadIdx.x] : &in[i];  // L7
    out[i] = loadGen(p);                          // L8
}
```

真实 PTX（同编译器与参数）：

{% raw %}
```nasm
.visible .func  (.param .b32 func_retval0) loadGen(int const*)(
        .param .b64 loadGen(int const*)_param_0
)
{
        ld.param.u64    %rd1, [loadGen(int const*)_param_0];
        ld.u32  %r1, [%rd1];                        // L1: *p，未写状态空间 → 通用寻址
        st.param.b32    [func_retval0+0], %r1;
        ret;
}

.visible .entry mixed(int*, int const*, int, int)(
        .param .u64 mixed(int*, int const*, int, int)_param_0,
        .param .u64 mixed(int*, int const*, int, int)_param_1,
        .param .u32 mixed(int*, int const*, int, int)_param_2,
        .param .u32 mixed(int*, int const*, int, int)_param_3
)
{
        ld.param.u64    %rd4, [mixed(int*, int const*, int, int)_param_0];  // out
        ld.param.u64    %rd5, [mixed(int*, int const*, int, int)_param_1];  // in
        ld.param.u32    %r3, [mixed(int*, int const*, int, int)_param_2];   // useShared
        ld.param.u32    %r2, [mixed(int*, int const*, int, int)_param_3];   // i
        mov.u32         %r1, %tid.x;
        shl.b32         %r4, %r1, 2;                // L5: threadIdx.x × 4
        mov.u32         %r5, mixed(int*, int const*, int, int)::s;  // L4: 数组标识符 = 地址常量
        add.s32         %r6, %r5, %r4;              // L5: &s[threadIdx.x]
        st.shared.u32   [%r6], %r1;                 // L5: s[tid] = tid
        bar.sync        0;                          // L6
        setp.eq.s32     %p1, %r3, 0;                // L7: useShared == 0 ?
        @%p1 bra        $L__BB1_2;                  // L7: 标签作 bra 的操作数
        mul.wide.u32    %rd6, %r1, 4;
        { .reg .b64 %tmp;
          cvt.u64.u32   %tmp, %r5;                  // 32 位 shared 地址 → 64 位
          cvta.shared.u64       %rd7, %tmp;         // L7: shared 地址 → 通用地址
        }
        add.s64         %rd12, %rd7, %rd6;
        bra.uni         $L__BB1_3;
$L__BB1_2:
        mul.wide.s32    %rd8, %r2, 4;               // L7: i × 4
        add.s64         %rd12, %rd5, %rd8;          // L7: &in[i]（通用地址）
$L__BB1_3:
        { // callseq 0, 0
        st.param.b64    [param0+0], %rd12;          // L8: 传参（var+immOff）
        call.uni (retval0), 
        loadGen(int const*),                        // L8: 函数名作 call 的操作数
        ( 
        param0 
        );
        ld.param.b32    %r8, [retval0+0];           // L8: 取返回值
        } // callseq 0
        cvta.to.global.u64      %rd9, %rd4;
        mul.wide.s32    %rd10, %r2, 4;
        add.s64         %rd11, %rd9, %rd10;
        st.global.u32   [%rd11], %r8;               // L8: out[i] = 返回值
        ret;
}
```
{% endraw %}

这段代码同时覆盖了 §6.4.1.1、§6.4.2 和 §6.4.4：

**`mov.u32 %r5, mixed(...)::s`：数组标识符是地址常量。** `s` 是共享内存数组，它的标识符就是 shared 状态空间里的地址常量；`mov` 把这个地址装进 32 位寄存器 `%r5`。这与文档 `mov.u32 p, tbl` 的例子同型。这里的共享内存地址以 32 位形式存放，所以后面的 `add.s32` 做 32 位地址加法。

**`st.shared.u32 [%r6], %r1`：显式状态空间 + `[reg]`。** 这里写了 `.shared`，地址 `%r6` 直接按共享内存地址解释，不需要通用寻址的窗口判断。

**`ld.u32 %r1, [%rd1]`：通用寻址。** `loadGen` 的指针参数 `p` 可能是共享内存地址，也可能是全局内存地址。指令没有写 `.shared` 或 `.global`，所以按 §6.4.1.1 的规则运行：地址落在 `.shared` 窗口内就访问共享内存，落在 `.global` 窗口内就访问全局内存。调用点为此准备了两个分支：共享路径用 `cvta.shared.u64` 把共享地址转成通用地址；全局路径直接把通用指针 `%rd5` 加偏移。两条路径在 `$L__BB1_3` 汇合，`%rd12` 是最终的通用地址。

**`@%p1 bra $L__BB1_2` 与 `bra.uni $L__BB1_3`：标签作分支目标。** `$L__BB1_2` 和 `$L__BB1_3` 是标签，只能作为 `bra`/`brx.idx` 的操作数。这里 `useShared == 0` 时跳到 `$L__BB1_2` 走全局路径，否则顺序执行共享路径。

**`call.uni (retval0), loadGen(int const*), (param0)`：函数名作被调用者。** 函数名只出现在 `call` 指令里；调用序列用 `st.param.b64 [param0+0], %rd12` 传参、`ld.param.b32 %r8, [retval0+0]` 取返回值，`[param0+0]` 和 `[retval0+0]` 是 `[var+immOff]` 形式。

**地址宽度的扩展。** 共享地址 `%r5` 是 32 位，转通用地址前先用 `cvt.u64.u32` 零扩展到 64 位，再由 `cvta.shared.u64` 加上窗口基址。这一步对应 §6.4.1 的「地址按需零扩展」和 §6.4.1.1 的窗口映射。

## 小结

- 地址操作数有六种：`[var]`、`[reg]`、`[reg+immOff]`、`[var+immOff]`、`[immAddr]`、`var[immOff]`；地址按字节计，支持 32/64 位，必须按访存大小自然对齐。
- PTX 的地址运算就是整数/位运算，元素大小的缩放要显式算出来：`in[i]` 对应 `mul.wide ... 4` + `add.s64`。
- 不写状态空间时走通用寻址；`.const/.param/.local/.shared` 是通用地址空间里的窗口，`.param` 窗口嵌在 `.global` 窗口内。
- 数组标识符是地址常量；下标只能是常量、寄存器、寄存器 ± 常量，复杂下标要先算成地址。
- 向量用 `.x/.y/.z/.w` 或 `.r/.g/.b/.a` 取元素，花括号列表按位置对应；作目标时元素必须唯一。向量 `ld`/`st` 实现宽访存，访存大小是元素大小之和。
- 标签只能作 `bra`/`brx.idx` 的操作数，函数名只能作 `call` 的操作数；`mov` 可以取函数地址供间接调用，PTX 3.1 起还可取内核函数地址用于动态并行。
- 反汇编里：`in[3]` 变成 `[%rd4+12]`，`in[i±常量]` 变成 `[%rd6±字节偏移]`；`float4` 的一次拷贝是 16 字节的 `ld/st.global.v4.f32`；通用指针的读取是 `ld.u32`，共享地址转通用地址用 `cvt.u64.u32` + `cvta.shared.u64`。
