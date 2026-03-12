# INT4 or FP8 KV Cache

## Motivation

LMSYS 这篇 [INT4 QAT](https://lmsys.org/blog/2026-01-26-int4-qat/) 文章里，训练侧做的是 fake quantization，rollout/inference 侧做的是 W4A16，即 INT4 weights + BF16 activations，并明确写到这一路径“依赖 BF16 Tensor Cores”。这意味着当前的低比特优化重点首先放在权重压缩上，而不是 KV cache。因此我下一步讨论 KV cache 精度时，核心问题是“在 attention 访存主导的 decode 阶段，是否值得把 KV cache 从 BF16 再压到 FP8 或 INT4，以及这样做在 H200 上是否真的能形成系统收益”？

## FP8

### Background
- 硬件侧：H200 属于 Hopper 架构，而 Hopper 的官方核心特征之一就是 Transformer Engine 对 FP8 的硬件支持。NVIDIA 官方将 Hopper 的低精度主路线明确定位在 [mixed FP8 / FP16](https://www.nvidia.com/en-us/data-center/technologies/hopper-architecture/)，而不是 INT4 训练/注意力主路径。
- 软件侧：vLLM 直接提供了 FP8 KV Cache 文档，说明它可以显著降低 KV cache 的内存占用，从而支持更多 token、提升吞吐并支持更长上下文；它还给出了 kv_cache_dtype="fp8"、fp8_e4m3、fp8_e5m2 等模式，以及默认、warmup 校准、数据集校准三类 scale 方案。更关键的是，[vLLM 文档](https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/#fp8-kv-cache-overview)写明：在 Flash Attention 3 backend 下，attention 可以直接在 量化的 FP8 域 中执行，此时 queries 也会被量化到 FP8。这个细节非常重要，因为它说明 FP8 KV cache 并不只是“存储压缩”，而是已经开始进入 kernel-level 的低精度 attention 路径。

### Advantage
在INT4 QAT这个前提下，把 KV cache 从 BF16 进一步降到 FP8，最直接的收益是显存减半级别的 KV cache 压缩，同时不必改写整个系统去适配一个新的 INT4 attention 栈。因为 W4A16 已经把权重压到 4 bit，剩余显存大头之一自然会越来越多地落到 KV cache 上，尤其在长上下文和高并发 rollout 时更明显。FP8 KV cache 因为和 Hopper 的 FP8 硬件能力、vLLM/TensorRT 类生态更一致，所以它非常适合被定义为 “在不改变整体系统形态的前提下，继续扩展上下文长度和 batch 容量的增量优化”

## INT4

PyTorch 的 [INT4 decoding](https://pytorch.org/blog/int4-decoding/) 文章明确指出，KV cache 随上下文长度线性增长，会给 attention decode 带来严重的内存压力。他们实验中还观察到，group-wise INT4 KV cache 在 decode phase 的精度上可与 BF16 KV cache 相比拟。这说明从算法可行性上，INT4 KV cache 不是空谈。Hugging Face 的 cache 文档也已经支持 quantized cache，其中 hqq 支持 int2/int4/int8，quanto 支持 int2/int4。这些都说明 INT 型 KV cache 作为研究与实验方向是成立的。 KV cache 的瓶颈，尤其在 decode 阶段，本质上是 attention 的 memory-bound 访存问题，而不是像权重量化那样主要受益于一个成熟的 weight-only GEMM kernel。PyTorch 的 INT4 decoding 文章把问题说得非常直接：他们最初虽然把 KV cache 读带宽降低了 4 倍，但并没有自然看到延迟改善，反而观察到 INT4 attention 对 HBM 带宽的利用效率比 BF16 attention 低得多；随后他们必须专门设计 CUDA 优化，才把 INT4 GQA decode 提升到比 BF16 更快。也就是说，INT4 KV cache 不是“有低比特就自然更快”，而是“只有专门 attention kernel 做得足够好时才会更快”。

### Why INT 4 is harder?
1. INT4 通常需要更复杂的 group-wise / per-token / per-channel scale 管理，scale 元数据本身会吞掉部分理论压缩收益。
2. INT4 往往涉及更复杂的 pack/unpack、dequant、layout 转换，而这些操作很容易把理论上的 4× 压缩变成运行时的额外开销。
3. INT4 KV cache 的效果高度依赖专用 attention kernel，而这类 kernel 的成熟度和普及度显著弱于 FP8 KV cache。Hugging Face 的 KV cache 文档甚至直接提醒：cache quantization 会伤害 latency，特别是在上下文不长、显存又够用的时候，要在 memory efficiency 和 latency 之间平衡。这一点对 INT4 会更敏感。



