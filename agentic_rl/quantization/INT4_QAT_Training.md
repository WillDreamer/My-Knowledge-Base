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
