# 异步拷贝（cp.async）为什么能"绕过"L1 Cache？——L1 Cache和Shared Memory共享SRAM的疑问解析

> 原始问题：Ampere架构的异步拷贝指令可以把数据从global memory直接搬到shared memory，和计算异步进行，还能绕过register和L1 cache。但shared memory不是和L1 cache共用同一块物理SRAM吗？这不矛盾吗？

这个问题问得很准，这个"矛盾"确实是很多人会卡住的地方——答案的关键在于：**"绕过L1 cache"指的是绕过L1的"缓存逻辑/管理机制"，而不是绕过这块物理SRAM本身**。这两者是不同层面的东西，拆开讲。

---

## 先确认前提是对的

从Volta架构开始（包括Ampere），NVIDIA确实把**L1 cache和shared memory合并成了同一块物理SRAM**，可以通过配置动态划分两者的比例（比如128KB的统一空间，你可以配成"64KB L1 + 64KB shared memory"或者其他比例）。这个前提没错。

---

## 关键区分：物理存储 vs 访问路径/管理机制

同一块物理SRAM，可以被两种**完全不同的"使用模式"**访问：

**模式1：作为L1 Cache使用**——这时候硬件会给这块SRAM附加一套**缓存管理逻辑**：
- Tag比对（判断某个地址的数据在不在cache里）
- Cache line分配和替换策略（LRU之类）
- 和L2/global memory之间的一致性维护

**模式2：作为Shared Memory使用**——这时候完全是**软件直接寻址**，没有tag、没有replacement policy，kernel代码写`shared_mem[i]`就是直接按地址访问这块SRAM，没有"缓存命中/未命中"这个概念，因为它根本不是在"缓存"什么东西，它本身就是数据存放的终点。

---

## 传统（同步）拷贝路径 vs 异步拷贝路径的真正区别

### 传统方式（不用async copy，普通的 `LDG` + `STS` 指令组合）

```
Global Memory → L2 Cache → L1 Cache（缓存模式，占用tag/line资源） → Register → Shared Memory
```

数据要经过**两次搬运**：先用load指令把数据从global memory读到寄存器（这个过程中会经过L1 cache，如果cache命中可以加速，但也会占用L1 cache的缓存容量和管理开销），再用store指令把寄存器里的数据写到shared memory。这个过程**真实占用了寄存器资源**（数据要在寄存器里过一道），也**真实触发了L1 cache的缓存逻辑**（数据被当作普通的global memory访问，走了cache tag比对、可能发生cache line分配/驱逐）。

### Ampere的异步拷贝（`cp.async` 指令）

```
Global Memory → L2 Cache → 直接写入Shared Memory（物理上可能是同一块SRAM，但走的是直接寻址路径）
```

硬件专门加了一条**直接的数据通路**，数据从L2出来之后，直接被搬运引擎写到shared memory对应的物理地址，**不经过寄存器中转**，也**不经过L1 cache的tag比对/行分配这套缓存管理逻辑**——即使最终落地的物理位置可能和L1 cache是同一块SRAM（按配置划分出来的shared memory那部分），但这次访问压根没有"被当作一次cache访问"来处理，是直接按shared memory的寻址方式写进去的。

---

## 用一个类比理解这个"绕过"

可以把这块统一SRAM想象成一个仓库，这个仓库被隔成两个区域：**"临时缓存区"（L1模式）**和**"长期存放区"（shared memory模式）**。

- **传统路径**：快递（数据）先要在"临时缓存区"登记入库（走cache的tag/line分配这套登记流程），然后工作人员把它搬到手推车上（寄存器），再推到"长期存放区"卸货——整个过程多了"临时缓存区登记"和"手推车中转"这两步。
- **异步拷贝路径**：有一条**专用传送带**，直接把货物从卡车（L2）送到"长期存放区"，完全不经过"临时缓存区"的登记流程，也不用手推车中转——虽然"长期存放区"和"临时缓存区"是同一个仓库的两个隔间，但这次运货压根没碰"临时缓存区"这个流程。

---

## 为什么这样设计有好处

**1. 省掉寄存器占用**
传统方式里，数据要临时存一份在寄存器里才能转存到shared memory，这会占用寄存器资源——而register本身是稀缺资源（occupancy会受register数量限制），如果能跳过这一步，kernel能用更少的register，给occupancy留出更多空间。

**2. 省掉不必要的cache污染**
如果这块数据只是"路过"一下就要存进shared memory（典型场景就是tiling，把数据从global memory搬进shared memory准备复用），**用L1 cache去"缓存"它的意义不大**——因为很快就会主动把它存进shared memory这个"真正的目的地"了，经过L1 cache走一遭反而占用了L1 cache本该服务于其他真正需要缓存命中优化的访问的容量，这是一种不必要的资源浪费（cache pollution）。

**3. 真正实现异步**
因为不需要先搬进寄存器再搬出去（这个过程是同步阻塞的，线程要等数据到寄存器才能继续下一步），硬件直接处理从global memory到shared memory这条通路，可以让这个拷贝过程**和计算指令并行发生**——可以先发起几个异步拷贝请求，让SM继续执行别的计算指令，不用傻等拷贝完成，这是`cp.async`机制真正的核心价值，"computation和data movement overlap"指的就是这个。

---

## 一句话总结

**L1 cache和shared memory确实共享同一块物理SRAM，但"绕过L1 cache"说的是绕过这块SRAM在"扮演L1 cache角色时"所需要的那套tag比对、line管理、cache一致性维护机制，而不是绕过这块SRAM本身**——异步拷贝走的是一条直接写入shared memory地址空间的专用硬件通路，物理终点可能落在同一块SRAM上，但访问方式和传统的"先进cache、再进寄存器、再存shared memory"这条路径完全不同，这才是"bypass"真正的含义。
