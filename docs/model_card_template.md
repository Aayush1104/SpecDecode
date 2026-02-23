---
language: en
license: mit
tags:
  - speculative-decoding
  - draft-model
  - text-generation
  - {{ domain }}
datasets:
  - {{ dataset }}
base_model: {{ base_model }}
pipeline_tag: text-generation
---

# {{ model_name }}

A draft model for speculative decoding, fine-tuned on **{{ domain }}** data.

## What This Model Does

This is a fine-tuned version of `{{ base_model }}` built to serve as a fast draft model in speculative decoding. It proposes candidate tokens that a larger target model (e.g., Qwen2.5-7B) verifies in parallel, producing the same output distribution as the target alone but faster.

This model is not meant for standalone generation. It should be paired with a larger target model through the SpecDecode library.

- **Base model** {{ base_model }}
- **Domain** {{ domain }}
- **Training data** {{ dataset }}
- **Training method** {{ training_method }}
- **Parameters** ~82M

## Usage

```python
from src.speculative.backends import create_backend
from src.speculative.decoding import speculative_decode

target = create_backend("Qwen/Qwen2.5-7B", dtype="bfloat16")
draft = create_backend("{{ hf_repo_id }}", dtype="float16")
# use speculative_decode(target, draft, input_ids, ...)
```

## Training Details

- **Steps** {{ num_steps }}
- **Batch size** {{ batch_size }} (effective, with gradient accumulation)
- **Learning rate** {{ learning_rate }}
- **Scheduler** Cosine with warmup
- **Precision** bfloat16
- **Hardware** {{ hardware }}

{{ #if distillation }}
### Knowledge Distillation

- **Teacher model** {{ teacher_model }}
- **Distillation temperature** {{ distill_temperature }}
- **Alpha (KD vs LM loss)** {{ distill_alpha }}
{{ /if }}

## Performance

| Metric | Value |
|--------|-------|
| Acceptance rate (on {{ domain }}) | {{ acceptance_rate }} |
| Tokens/second (speculative) | {{ tokens_per_second }} |
| Speedup vs baseline | {{ speedup }} |
| Validation loss | {{ val_loss }} |

## Limitations

- Built as a draft model, not for standalone text generation
- Performance depends on the target model and how well the domain matches
- Trained primarily on English data
- Acceptance rates may drop on out-of-domain inputs

## Citation

```bibtex
@software{specdecode2024,
  title={SpecDecode: Speculative Decoding with Adaptive Routing},
  author={Aayush},
  year={2024},
  url={https://github.com/Aayush1104/specdecode}
}
```
