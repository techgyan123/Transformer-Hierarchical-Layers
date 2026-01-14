# Transformer Hierarchical Layers (THL)

A Python library implementing the **Transformer Hierarchical Layers** architecture: a strictly non-Transformer, hierarchical recurrent computation graph designed for **low-budget LLMs** (4GB VRAM, Mobile).

THL solves the KV cache bottleneck by using **Sequence-Length Independent Memory** (O(1) memory per layer with respect to T).

## Key Features

- **Bounded Memory**: Uses fixed-slot routed memory (`J=1024` slots) instead of growing KV cache.
- **Hierarchical Recurrence**: Multi-timescale GRU tiers processing information at different frequencies $\tau_k$.
- **Sparse Routing**: TopK routing mechanism with load balancing.
- **Layered Inference**: Optimized engine for running 7B+ parameter models on 4GB VRAM GPUs via module streaming.
- **FP16 Optimization**: Built-in support for Automatic Mixed Precision (AMP).

## Installation

```bash
# Clone the repository
git clone https://github.com/erebustn/Core
cd Core

# Install dependencies
pip install torch matplotlib
```

## Usage

### Basic Language Modeling

```python
import torch
from thl.config import THLConfig
from thl.model import THLModel

# 1. Configure
config = THLConfig(
    num_tiers=3,
    memory_slots=1024,
    dim=768
)

# 2. Initialize
model = THLModel(config) # Auto-initialization with Xavier

# 3. Forward
input_ids = torch.randint(0, config.vocab_size, (1, 32))
logits, state = model(input_ids)
print(logits.shape) # [1, 32, vocab_size]
```

### Low-VRAM Inference (Layered Engine)

Run large models on small GPUs by streaming layers:

```python
from thl.inference.layered import LayeredInferenceEngine
from thl.inference.state import InferenceState

engine = LayeredInferenceEngine(model, device="cuda")
state = InferenceState.init(1, config, model.tiers, model.memory_bank)

token = torch.tensor([123])
logit, state = engine.step(token, state)
```

### Fine-Tuning Sequence Classification

THL supports specialized heads for downstream tasks:

```python
from thl.model import THLForSequenceClassification

model = THLForSequenceClassification(config, num_labels=2)
logits, loss = model(input_ids, labels=torch.tensor([1]))
```

## Architecture

| Component | Description |
|-----------|-------------|
| **Memory Bank** | $M_t \in \mathbb{R}^{J \times d}$. Holds long-term context. |
| **Sparse Router** | Reads relevant slots ($r_t$) using TopK queries. |
| **Hierarchical Tiers** | Stack of recurrent cells ($s_t^{(k)}$) updating at intervals $\tau_k$. |
| **Novelty Writer** | Writes new information to memory if novel ($w_t$). |

## Testing

Run the full verification suite:

```bash
./scripts/run_tests.sh
```

## License

MIT License.
