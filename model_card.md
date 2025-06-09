---
language: en
tags:
- video-understanding
- reasoning
- multimodal
- reinforcement-learning
- question-answering
license: mit
datasets:
- CLEVRER
- NExT-QA
- MMVU
- MVBench
- TempCompass
- Video-MME
- STAR
---

# TW-GRPO: Token-Level Importance Weighting GRPO

TW-GRPO is a novel approach for improving video reasoning capabilities in multimodal large language models through focused thinking and soft multi-level rewards.

## Model Description

TW-GRPO (Token-Level Importance Weighting Group Relative Policy Optimization) integrates **focused thinking** and **soft multi-level rewards** for multi-choice video question answering. Unlike vanilla thinking which assigns uniform token importance, **focused thinking highlights critical tokens to dominate loss calculation**. By shifting single-choice QA's binary rewards to multi-choice QA's multi-level rewards, TW-GRPO enables fine-grained gradient estimation and training efficiency.

![Model Overview](https://raw.githubusercontent.com/longmalongma/TW-GRPO/main/assets/figs/intro.jpg)

### Key Innovations

- **🎯 Token-Level Importance Weighting**: Prioritizes tokens with high informational density (estimated by intra-group information entropy) during loss computation, enabling concise, task-focused reasoning chains.
- **🎨 Multi-grained Reward Modeling**: Uses multi-choice QA tasks with partial correctness evaluation to improve gradient estimation and policy stability.
- **🔄 Question-Answer Inverse**: A data augmentation technique converting single-choice QA into multi-choice formats via question negation and answer inversion, mitigating data scarcity.

## Training Methodology

TW-GRPO was trained on the CLEVRER dataset's counterfactual tasks using the Qwen2.5-VL-7B-Instruct model as the backbone. The training process incorporates:

- Entropy-guided vision-language reasoning through intra-group information entropy
- Dynamic token importance weighting based on semantic significance
- Multi-level reward signals for partial correctness assessment

## Performance

TW-GRPO demonstrates superior performance across multiple video reasoning benchmarks compared to existing methods:

| Dataset | TW-GRPO | Video-R1 | VideoChat-R1 | Qwen2.5-VL-7B-COT-SFT | Qwen2.5-VL-7B (zero-shot) |
|---------|---------|----------|--------------|------------------------|----------------------------|
| CLEVRER | **80.5** | 78.6 | 60.4 | 66.8 | 36.5 |
| NExT-GQA | **71.5** | 69.9 | 63.4 | 64.5 | 37.2 |
| MMVU | **40.5** | 39.9 | 36.5 | 37.3 | 29.1 |
| MVBench | **54.4** | 54.3 | 48.8 | 50.2 | 34.9 |
| TempCompass | **46.2** | 45.7 | 40.9 | 41.6 | 35.4 |
| Video-MME | **61.3** | 60.8 | 54.6 | 56.2 | 36.4 |
| STAR | **50.1** | 49.8 | 45.3 | 46.4 | 31.7 |

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import torch
import os
import decord
import numpy as np

# Load model and tokenizer
model_path = "Falconss1/TW-GRPO"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)

# Function to extract frames from video
def extract_frames(video_path, num_frames=8):
    vr = decord.VideoReader(video_path)
    total_frames = len(vr)
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = vr.get_batch(indices).asnumpy()
    return [Image.fromarray(frame) for frame in frames]

# Example usage
video_path = "path/to/your/video.mp4"
frames = extract_frames(video_path)
question = "What will happen if the red sphere is removed from the scene? A. The blue sphere will not collide with the yellow cube. B. The yellow cube will not move. C. The green sphere will not move. D. The blue sphere will not move."

# Format prompt
prompt = f"<image>\nUser: {question}\nAssistant:"

# Generate response
response = model.chat(tokenizer, prompt, frames)
print(response)
```

## Limitations

- The model's performance may vary depending on the complexity and specific nature of video content
- While focused on improving reasoning capabilities, the model may still struggle with particularly complex causal and counterfactual reasoning scenarios
- Performance is dependent on the quality and relevance of extracted video frames

## Citation

```bibtex
@article{dang2025reinforcing,
  title = {Reinforcing Video Reasoning with Focused Thinking},
  author = {Jisheng Dang, Jingze Wu, Teng Wang, Xuanhui Lin, Nannan Zhu, Hongbo Chen, Wei-Shi Zheng, Meng Wang, Tat-Seng Chua},
  booktitle = {arXiv preprint arXiv:2505.24718},
  year = {2025}
}
```

## Acknowledgements

This model builds upon the contributions from the open-source community, including works from [Open-R1-Video](https://github.com/Wang-Xiaodong1899/Open-R1-Video), [Video-R1](https://github.com/tulerfeng/Video-R1), and [VideoChat-R1](https://github.com/OpenGVLab/VideoChat-R1).

For more information, code, and resources, please visit our [GitHub repository](https://github.com/longmalongma/TW-GRPO). 