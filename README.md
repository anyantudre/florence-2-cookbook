# Florence-2 Cookbook

<p align="center">
  <img src="img/banner.png" alt="Florence-2 Cookbook" width="100%">
</p>

Hands-on notebooks and a short guide to Florence-2, Microsoft's lightweight vision-language model: run inference on 14 built-in tasks, auto-label images, fine-tune the model on your own data, score its boxes, answer questions and measure its speed.

For models, tools, papers and tutorials from the wider ecosystem, see the [Awesome Florence-2 list](AWESOME.md).

[![arXiv](https://img.shields.io/badge/arXiv-2311.06242-b31b1b.svg)](https://arxiv.org/abs/2311.06242)
[![Models on Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Models-Florence--2-yellow)](https://huggingface.co/florence-community)
[![Transformers docs](https://img.shields.io/badge/%F0%9F%A4%97%20Transformers-docs-blue)](https://huggingface.co/docs/transformers/model_doc/florence2)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> [!NOTE]
> This is an independent community project. Florence-2 was created by Microsoft; this repository is not affiliated with or endorsed by Microsoft. For the original work, see the [paper](https://arxiv.org/abs/2311.06242) and the [official model card](https://huggingface.co/microsoft/Florence-2-large).

## Why this repository

In early 2024, I was working on a project that needed a vision-language model for several different tasks, on limited hardware. None of the models available at the time gave satisfying results in that setting. When Microsoft released Florence-2, I was impressed by how well it performed across so many tasks for its size. This repository gathers the notebooks and notes I put together while exploring it.

## Contents

- [Notebooks](#notebooks)
- [Quickstart](#quickstart)
- [About Florence-2](#about-florence-2)
- [Awesome Florence-2 list](AWESOME.md)
- [Citation](#citation)
- [License](#license)

## Notebooks

| # | Notebook | What it covers | Open |
|---|----------|----------------|------|
| 1 | [Sample inference](1.%20Florence-2%20sample%20inference.ipynb) | All 14 built-in tasks: captioning, detection, grounding, segmentation, OCR, cascaded tasks | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/anyantudre/florence-2-cookbook/blob/main/1.%20Florence-2%20sample%20inference.ipynb) |
| 2 | [Data auto-labelling](2.%20Data%20auto-labelling%20with%20Florence-2.ipynb) | Label a folder of images with `<OD>` and export YOLO annotations | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/anyantudre/florence-2-cookbook/blob/main/2.%20Data%20auto-labelling%20with%20Florence-2.ipynb) |
| 3 | [Fine-tuning on DocVQA](3.%20Fine-tuning%20Florence-2.ipynb) | Full fine-tuning for document question answering | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/anyantudre/florence-2-cookbook/blob/main/3.%20Fine-tuning%20Florence-2.ipynb) |
| 4 | [Fine-tuning on a detection dataset](4.%20Finetuning%20Florence-2%20on%20detection%20dataset%20-%20Roboflow.ipynb) | LoRA fine-tuning on a Roboflow dataset, with mAP and confusion matrix | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/anyantudre/florence-2-cookbook/blob/main/4.%20Finetuning%20Florence-2%20on%20detection%20dataset%20-%20Roboflow.ipynb) |
| 5 | [Multi-region prompts](5.%20Multi-region%20prompts%20with%20Florence-2.ipynb) | Ask about many boxes in one batched call: describe, categorize or read every region | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/anyantudre/florence-2-cookbook/blob/main/5.%20Multi-region%20prompts%20with%20Florence-2.ipynb) |
| 6 | [Confidence scores](6.%20Confidence%20scores%20for%20Florence-2%20boxes.ipynb) | Score boxes from the model's token probabilities and filter boxes drawn for absent objects, tested on COCO | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/anyantudre/florence-2-cookbook/blob/main/6.%20Confidence%20scores%20for%20Florence-2%20boxes.ipynb) |
| 7 | [Speed benchmark](7.%20Florence-2%20speed%20benchmark.ipynb) | Latency, throughput and memory per task, model size, precision and decoding, on GPU and CPU | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/anyantudre/florence-2-cookbook/blob/main/7.%20Florence-2%20speed%20benchmark.ipynb) |
| 8 | [Question answering](8.%20Question%20answering%20with%20Florence-2.ipynb) | Answer open and multiple-choice questions with the `-ft` checkpoints, measured on A-OKVQA | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/anyantudre/florence-2-cookbook/blob/main/8.%20Question%20answering%20with%20Florence-2.ipynb) |

Notebook 1 is adapted from Microsoft's [sample inference notebook](https://huggingface.co/microsoft/Florence-2-large/blob/main/sample_inference.ipynb), notebook 3 from the Hugging Face [fine-tuning tutorial](https://huggingface.co/blog/finetune-florence2), and notebook 4 from [Roboflow's notebook](https://github.com/roboflow/notebooks). Credit goes to their authors.

## Quickstart

Recent versions of Hugging Face Transformers support Florence-2 natively through the [`florence-community`](https://huggingface.co/florence-community) checkpoints (tested with `transformers` 5.22):

```python
import requests
import torch
from PIL import Image
from transformers import AutoProcessor, Florence2ForConditionalGeneration

model_id = "florence-community/Florence-2-base-ft"
model = Florence2ForConditionalGeneration.from_pretrained(model_id, dtype=torch.float16).to("cuda")
processor = AutoProcessor.from_pretrained(model_id)

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg"
image = Image.open(requests.get(url, stream=True).raw)

task = "<OD>"
inputs = processor(text=task, images=image, return_tensors="pt").to("cuda", torch.float16)
generated_ids = model.generate(**inputs, max_new_tokens=1024, num_beams=3)
text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
print(processor.post_process_generation(text, task=task, image_size=image.size))
# {'<OD>': {'bboxes': [[34, 160, 597, 371], ...], 'labels': ['car', 'wheel', 'wheel']}}
```

> [!TIP]
> The original `microsoft/Florence-2-*` checkpoints load with `trust_remote_code=True`, and that custom code breaks on recent Transformers versions (errors such as `'_supports_sdpa'` or `forced_bos_token_id`). Use the `florence-community` checkpoints instead.

## About Florence-2

Florence-2 ([Xiao et al., 2023](https://arxiv.org/abs/2311.06242)) is an open-source (MIT) vision foundation model from Microsoft. It handles many vision and vision-language tasks with a single model: you choose the task with a text prompt, and the model answers in text. Despite its small size, it matches or beats much larger models such as Kosmos-2 (1.6B) in zero-shot evaluations.

### Models

| Model | Size | Description |
|-------|------|-------------|
| Florence-2-base ([original](https://huggingface.co/microsoft/Florence-2-base) · [transformers](https://huggingface.co/florence-community/Florence-2-base)) | 0.23B | Pretrained on FLD-5B |
| Florence-2-large ([original](https://huggingface.co/microsoft/Florence-2-large) · [transformers](https://huggingface.co/florence-community/Florence-2-large)) | 0.77B | Pretrained on FLD-5B |
| Florence-2-base-ft ([original](https://huggingface.co/microsoft/Florence-2-base-ft) · [transformers](https://huggingface.co/florence-community/Florence-2-base-ft)) | 0.23B | Fine-tuned on a collection of downstream tasks |
| Florence-2-large-ft ([original](https://huggingface.co/microsoft/Florence-2-large-ft) · [transformers](https://huggingface.co/florence-community/Florence-2-large-ft)) | 0.77B | Fine-tuned on a collection of downstream tasks |

### Tasks

| Task | Prompt | Extra input |
|------|--------|-------------|
| Caption | `<CAPTION>`, `<DETAILED_CAPTION>`, `<MORE_DETAILED_CAPTION>` | none |
| Object detection | `<OD>` | none |
| Dense region caption | `<DENSE_REGION_CAPTION>` | none |
| Region proposal | `<REGION_PROPOSAL>` | none |
| Phrase grounding | `<CAPTION_TO_PHRASE_GROUNDING>` | caption |
| Referring expression segmentation | `<REFERRING_EXPRESSION_SEGMENTATION>` | phrase |
| Region to segmentation | `<REGION_TO_SEGMENTATION>` | region (`<loc_x1><loc_y1><loc_x2><loc_y2>`) |
| Open-vocabulary detection | `<OPEN_VOCABULARY_DETECTION>` | phrase |
| Region to category / description | `<REGION_TO_CATEGORY>`, `<REGION_TO_DESCRIPTION>` | region |
| OCR | `<OCR>`, `<OCR_WITH_REGION>` | none |

Region coordinates are quantized to `[0, 999]`. Notebook 1 shows every task and its output format.

### How it works

Vision tasks differ in spatial hierarchy (image-level or pixel-level) and in semantic granularity (a label or a detailed description). Florence-2 covers all of them with one sequence-to-sequence model.

<p align="center">
  <img src="img/representation.jpeg" alt="Spatial hierarchy and semantic granularity across vision tasks" width="700">
  <br><em>Source: Florence-2 paper.</em>
</p>

A DaViT vision encoder turns the image into visual embeddings. These are concatenated with the embeddings of the text prompt and passed to a transformer encoder-decoder, which generates text. For region tasks, the vocabulary includes location tokens: boxes are written as `(x0, y0, x1, y1)` and polygons as a clockwise list of vertices.

<p align="center">
  <img src="img/architecture.png" alt="Florence-2 architecture" width="700">
  <br><em>Source: Florence-2 paper.</em>
</p>

### Training data: FLD-5B

The model's strength comes mostly from its data. The authors built FLD-5B, with 5.4 billion annotations (boxes, masks, captions, grounding) on 126 million images, generated largely automatically with specialist models and filtered with quality checks.

<p align="center">
  <img src="img/annotation.jpeg" alt="Example of FLD-5B annotations" width="700">
  <br><em>Source: Florence-2 paper.</em>
</p>

## Citation

If you use Florence-2, please cite the original paper:

```bibtex
@article{xiao2023florence,
  title   = {Florence-2: Advancing a Unified Representation for a Variety of Vision Tasks},
  author  = {Xiao, Bin and Wu, Haiping and Xu, Weijian and Dai, Xiyang and Hu, Houdong and Lu, Yumao and Zeng, Michael and Liu, Ce and Yuan, Lu},
  journal = {arXiv preprint arXiv:2311.06242},
  year    = {2023}
}
```

## License

The code and notebooks in this repository are released under the [MIT License](LICENSE). Florence-2's weights are MIT-licensed by Microsoft, and linked projects keep their own licenses.
