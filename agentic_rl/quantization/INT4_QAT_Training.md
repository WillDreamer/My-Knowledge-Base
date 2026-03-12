# INT4 QAT Training

LLM 的瓶颈是 参数规模 + KV cache + rollout 计算，因此 INT4 权重理论上节省 75% 显存。然而在现有的autograd，gpu kernel的限制下，我们还无法进行INT 4 training。最近有工作提出《Squeezing 1TB Model Rollout into a Single H200: INT4 QAT RL End-to-End Practice》

**其中Quantization Aware Training** 的思想不是直接训练 INT4，而是**训练阶段仍然用 FP16，但在计算图中模拟量化误差。**

- 我们首先来看Rollout阶段，我们需要采用真实的INT 4权重来进行推理加速，即使用 GPU INT4 kernel进行INT4 weight × FP16 activation → FP16 output
- 然后是training阶段， 路线图如下：

```markdown
FP16 weight
↓
fake quantization
↓
INT4 representation (simulated)
↓
forward compute
↓
backward update FP16 weights
```


- **Quantization Aware Training**
    - 首先我们需要维护一份BF16的权重 $$\mathcal{w}_{bf16}$$，最后更新也是更新这个权重
    - 然后我们进行quantization，即 $$\text{round}(\mathcal{w}_{bf16}/s)$$,将数值映射到INT 4的grid
    - 然后立刻进行de-quantization, ![equation](https://latex.codecogs.com/svg.image?w_q=\mathrm{round}(w_{bf16}/s)*s) ,恢复到了BF16
    - 在前向阶段使用 $$w_q \times activation$$
    - 在反向传播阶段，由于计算 $$\partial\mathcal{L}$$ / $$\partial w_{q}$$ 时，round操作不可导，所以采用近似估计, $$\partial \mathcal{L}/ \partial w_{q} \approx \partial \mathcal{L}/ \partial w_{bf16}$$，直接用 $$\mathcal{w}_{bf16}$$的梯度更新权重就好


### Patch in Megatron
```
class _FakeInt4QuantizationSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, group_size):
        m, n = x.shape
        block_size_m, block_size_n = 1, group_size


        m_padded = ceil_div(m, block_size_m) * block_size_m
        n_padded = ceil_div(n, block_size_n) * block_size_n

        x_padded = torch.zeros(
            (m_padded, n_padded),
            dtype=x.dtype, device=x.device
        )
        x_padded[:m, :n] = x

        x_view = x_padded.view(
            m_padded // block_size_m,
            block_size_m,
            n_padded // block_size_n,
            block_size_n
        )

        x_max = x_view.abs().float().amax(dim=(1, 3), keepdim=True)
        q_max = 7
        x_scale = x_max / q_max

        x_scale = x_scale.clamp(min=1e-5)

        x_div = x_view / x_scale
        x_round = torch.round(x_div)

        x_q_clamped = x_round.clamp(-q_max, q_max)

        x_dequant_view = x_q_clamped * x_scale

        x_dequant_full = x_dequant_view.view_as(x_padded)
        x_out = x_dequant_full[:m, :n].contiguous().to(x.dtype)

        return x_out

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

def fake_int4_quantization_ste(x, group_size):
    x_out = _FakeInt4QuantizationSTE.apply(x, group_size)
```

- 在每一行上，按列分组做 group-wise quantization，每一行每 128 个元素共用一个 scale。
- 然后对每个 group 求最大绝对值，生成 scale。
- 然后量化：除 scale、round、clamp
- 然后反量化回浮点数：$x_dequant_view = x_q_clamped * x_scale$
- 恢复原始维度