# Awesome Florence-2

A curated list of models, tools, papers and tutorials for Florence-2, Microsoft's lightweight vision-language model. Part of [Florence-2 Notebooks & Resources](README.md).

> [!NOTE]
> This is an independent community list, not affiliated with or endorsed by Microsoft.

## How to read this list

Every entry that ships code or weights carries a flag. Flags were checked against each project's code or config in November 2026.

| Flag | Meaning |
|------|---------|
| 🟢 native | Works with current tools: the native Transformers class (`Florence2ForConditionalGeneration`), or a runtime of its own (ONNX, MLX, Rust, C#, its own PyTorch code). |
| 🟠 remote-code | Loads Florence-2 through Microsoft's original `trust_remote_code` code, which breaks on recent Transformers versions. Works if you pin an older `transformers`. |
| ⚪ archived | Archived, offline, or not updated in over a year. |

Papers, articles and benchmark pages have no flag.

## Contents

- [Start here](#start-here)
- [Official resources](#official-resources)
- [When to use Florence-2](#when-to-use-florence-2)
- [Tutorials](#tutorials)
- [Fine-tuning](#fine-tuning)
- [Fine-tuned models](#fine-tuned-models)
- [Research built on Florence-2](#research-built-on-florence-2)
- [Deployment and runtimes](#deployment-and-runtimes)
- [Integrations and pipelines](#integrations-and-pipelines)
- [Demos](#demos)
- [Known limitations](#known-limitations)
- [Contributing](#contributing)

## Start here

Transformers has supported Florence-2 natively since [v4.56](https://github.com/huggingface/transformers/releases/tag/v4.56.0), through the [`florence-community`](https://huggingface.co/florence-community) checkpoints. Start there. Microsoft's original `microsoft/Florence-2-*` repos still load through `trust_remote_code`, and that code fails on recent versions with errors such as `'_supports_sdpa'` or `forced_bos_token_id` (see [#115](https://huggingface.co/microsoft/Florence-2-large/discussions/115) and [transformers#41622](https://github.com/huggingface/transformers/issues/41622)). Switching takes two lines:

```python
# Before: breaks on recent transformers
model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)

# After
from transformers import Florence2ForConditionalGeneration
model = Florence2ForConditionalGeneration.from_pretrained("florence-community/Florence-2-large")
```

For fine-tuning, use `transformers` 5.13 or later, which fixes a training-loss bug in the native class ([#46898](https://github.com/huggingface/transformers/releases/tag/v5.13.0)). Load the model in float32: the checkpoints are stored in float16, and float16 training diverges to `NaN`.

Almost every community fine-tune is still in the original format (🟠). To use one, pin an older `transformers` (projects commonly use 4.49 or earlier) or convert the weights with the [conversion script](https://github.com/huggingface/transformers/blob/main/src/transformers/models/florence2/convert_florence2_original_pytorch_to_hf.py) in the Transformers source (not tested here).

Microsoft has not released a successor, and it never published the FLD-5B training dataset ([request thread](https://huggingface.co/microsoft/Florence-2-large/discussions/24)).

## Official resources

- [Florence-2: Advancing a Unified Representation for a Variety of Vision Tasks](https://arxiv.org/abs/2311.06242) - The paper ([CVPR 2024 version](https://openaccess.thecvf.com/content/CVPR2024/papers/Xiao_Florence-2_Advancing_a_Unified_Representation_for_a_Variety_of_Vision_CVPR_2024_paper.pdf)).
- 🟢 [florence-community checkpoints](https://huggingface.co/florence-community) - Official Transformers conversions: [base](https://huggingface.co/florence-community/Florence-2-base), [large](https://huggingface.co/florence-community/Florence-2-large), [base-ft](https://huggingface.co/florence-community/Florence-2-base-ft), [large-ft](https://huggingface.co/florence-community/Florence-2-large-ft).
- 🟢 [Transformers documentation](https://huggingface.co/docs/transformers/model_doc/florence2) - Native classes, task list, quantization example.
- 🟠 Microsoft checkpoints - The original release: [base](https://huggingface.co/microsoft/Florence-2-base), [large](https://huggingface.co/microsoft/Florence-2-large), [base-ft](https://huggingface.co/microsoft/Florence-2-base-ft), [large-ft](https://huggingface.co/microsoft/Florence-2-large-ft).
- 🟠 [Sample inference notebook](https://huggingface.co/microsoft/Florence-2-large/blob/main/sample_inference.ipynb) - Microsoft's notebook covering every task.

## When to use Florence-2

Florence-2 answers fixed task prompts (labelling, grounding, captioning); it does not chat. The table shows where it still holds up in late 2026 and where newer models do better.

| Job | Florence-2? | Consider instead |
|-----|-------------|------------------|
| Captioning under 1B parameters | ✅ 143.3 CIDEr on COCO (large-ft) | |
| Finding an object from a text description | ✅ 93.4 on RefCOCO val (large-ft) | [Moondream 3.1](https://moondream.ai/blog/moondream-3-1-beyond-benchmarks) (larger) |
| OCR with a box for each line, in one pass | ✅ | |
| Detection on your own classes, with training data | ❌ 43.4 COCO mAP after fine-tuning | [RF-DETR](https://arxiv.org/abs/2511.09554) (48 to 60 AP), YOLO |
| Zero-shot detection with confidence scores | ❌ No scores | [Grounding DINO 1.5](https://arxiv.org/abs/2405.10300), [YOLOE](https://arxiv.org/abs/2503.07465) |
| Document parsing and document QA | ❌ | [PaddleOCR-VL](https://arxiv.org/abs/2510.14528), [Qwen3-VL](https://arxiv.org/abs/2511.21631) |
| Free-form questions, chat, other languages | ❌ | [Qwen3-VL](https://arxiv.org/abs/2511.21631), [SmolVLM](https://arxiv.org/abs/2504.05299), Moondream |

Florence-2 numbers come from the [model card](https://huggingface.co/microsoft/Florence-2-large). Every zero-shot detector, Florence-2 included, drops sharply on out-of-domain data ([Roboflow100-VL](https://arxiv.org/abs/2505.20612)), so plan to fine-tune for real projects.

## Tutorials

- 🟢 [Notebooks in this repository](README.md#notebooks) - Inference, auto-labelling with YOLO export, and two fine-tuning recipes on the native class.
- 🟠 [Fine-tuning Florence-2](https://huggingface.co/blog/finetune-florence2) - Hugging Face blog, fine-tuning on DocVQA. Its encoder-freezing loop sets `is_trainable` instead of `requires_grad`, so the encoder still trains.
- 🟠 [Fine-tune Florence-2 for object detection](https://blog.roboflow.com/fine-tune-florence-2-object-detection/) - Roboflow blog, LoRA on a custom dataset.
- 🟠 [Florence-2: open source vision foundation model by Microsoft](https://blog.roboflow.com/florence-2/) - Roboflow's overview of the model and tasks.
- 🟠 [Roboflow notebooks](https://github.com/roboflow/notebooks) - [Run different vision tasks](https://github.com/roboflow/notebooks/blob/main/notebooks/how-to-run-different-vision-tasks-with-florence-2.ipynb) and [fine-tune on a detection dataset](https://github.com/roboflow/notebooks/blob/main/notebooks/how-to-finetune-florence-2-on-detection-dataset.ipynb).
- ⚪ [andimarafioti/florence2-finetuning](https://github.com/andimarafioti/florence2-finetuning) - Code for the Hugging Face blog post.
- [Vision Language Models Explained](https://huggingface.co/blog/vlms) - Hugging Face's introduction to vision-language models, for background.
- ⚪ [AssemblyAI-Community/florence-2](https://github.com/AssemblyAI-Community/florence-2) - How Florence-2 works and how to use it.

## Fine-tuning

Tools:

- 🟠 [roboflow/maestro](https://github.com/roboflow/maestro) - Fine-tuning library and CLI for Florence-2, PaliGemma 2 and Qwen2.5-VL, with LoRA, QLoRA or a frozen encoder.
- ⚪ [microsoft/dstoolkit-finetuning-florence-2](https://github.com/microsoft/dstoolkit-finetuning-florence-2) - Microsoft accelerator with fine-tuning notebooks.
- ⚪ [autodistill-florence-2](https://github.com/autodistill/autodistill-florence-2) - Auto-labelling base model and a `Florence2Trainer`.

Data format: training pairs are a task prompt (`prefix`) and the expected text (`suffix`), with boxes written as `<loc_x1><loc_y1><loc_x2><loc_y2>` on a 0 to 999 grid. Roboflow documents the [Florence-2 dataset format](https://roboflow.com/formats/florence-2).

Pitfalls:

- Load in float32 for training, and use `transformers` 5.13 or later.
- To freeze the vision encoder, set `requires_grad = False` on `model.model.vision_tower` (native class).
- Check that the model still handles its original tasks after fine-tuning: public recipes rarely measure this.

## Fine-tuned models

Captioning and prompt generation:

- 🟠 [MiaoshouAI/Florence-2-large-PromptGen-v2.0](https://huggingface.co/MiaoshouAI/Florence-2-large-PromptGen-v2.0) - Tags and captions for Stable Diffusion and Flux training data ([base version](https://huggingface.co/MiaoshouAI/Florence-2-base-PromptGen-v2.0)).
- 🟢 [Disty0/Florence-2-large-PromptGen-v2.0](https://huggingface.co/Disty0/Florence-2-large-PromptGen-v2.0) - The same model converted to the native format ([base version](https://huggingface.co/Disty0/Florence-2-base-PromptGen-v2.0)).
- 🟠 [gokaygokay/Florence-2-SD3-Captioner](https://huggingface.co/gokaygokay/Florence-2-SD3-Captioner) - Long captions in the style of SD3 prompts.
- 🟠 [gokaygokay/Florence-2-Flux-Large](https://huggingface.co/gokaygokay/Florence-2-Flux-Large) - Flux-style prompts.
- 🟠 [thwri/CogFlorence-2.2-Large](https://huggingface.co/thwri/CogFlorence-2.2-Large) - Detailed captions distilled from CogVLM2.
- 🟠 [diffusers/shot-categorizer-v0](https://huggingface.co/diffusers/shot-categorizer-v0) - Labels film shots with metadata, for curating video datasets.

Screens and GUI agents:

- 🟠 [microsoft/OmniParser-v2.0](https://huggingface.co/microsoft/OmniParser-v2.0) - Screen parser whose icon captioner is a fine-tuned Florence-2-base ([code](https://github.com/microsoft/OmniParser)).
- 🟠 [AskUI/PTA-1](https://huggingface.co/AskUI/PTA-1) - Locates GUI text and elements from a prompt.
- 🟠 [HongxinLi/GoClick-Large](https://huggingface.co/HongxinLi/GoClick-Large) - GUI click grounding.

Documents:

- 🟠 [HuggingFaceM4/Florence-2-DocVQA](https://huggingface.co/HuggingFaceM4/Florence-2-DocVQA) - Document question answering.
- 🟠 [yifeihu/Florence-2-DocLayNet-Fixed](https://huggingface.co/yifeihu/Florence-2-DocLayNet-Fixed) - Document layout detection.
- 🟠 [ucsahin/Florence-2-large-TableDetection](https://huggingface.co/ucsahin/Florence-2-large-TableDetection) - Table detection.

Detection and spatial reasoning:

- 🟠 [Qingyun/Florence-2-large-DOTA-v1.0-lmmrotate](https://huggingface.co/Qingyun/Florence-2-large-DOTA-v1.0-lmmrotate) - Rotated boxes on aerial images ([code](https://github.com/VisionXLab/mllm-mmrotate)).
- 🟠 [remyxai/SpaceFlorence-2](https://huggingface.co/remyxai/SpaceFlorence-2) - Spatial question answering.
- 🟠 [anushreeberlia/fashion-florence](https://huggingface.co/anushreeberlia/fashion-florence) - Fashion attribute extraction.

Other languages and domains:

- 🟠 [gagan3012/Florence-2-FT-ArabicOCR](https://huggingface.co/gagan3012/Florence-2-FT-ArabicOCR) - Arabic OCR.
- 🟠 [aipib/Florence-2-FT-JP-OCR2](https://huggingface.co/aipib/Florence-2-FT-JP-OCR2) - Japanese OCR.
- 🟠 [m-aliabbas1/Florence-2-FT-path-vqa](https://huggingface.co/m-aliabbas1/Florence-2-FT-path-vqa) - Pathology question answering (research use).

## Research built on Florence-2

Florence-2 as a backbone:

- [Florence-VL](https://arxiv.org/abs/2412.04424) - Uses Florence-2 as the vision encoder of a multimodal LLM (CVPR 2025). ⚪ [Code](https://github.com/JiuhaiChen/CVPR2025-Florence-VL).
- [FLOWER](https://arxiv.org/abs/2509.04996) - Robot policy built on Florence-2, about 950M parameters (CoRL 2025). ⚪ [Code](https://github.com/intuitive-robots/flower_vla_calvin).
- [X-VLA](https://arxiv.org/abs/2510.10274) - Cross-embodiment robot policy with a Florence-2-large encoder (ICLR 2026). 🟠 [Code](https://github.com/2toinf/X-VLA).
- [OmniParser](https://arxiv.org/abs/2408.00203) - Vision-only GUI agent parser. 🟠 [Code](https://github.com/microsoft/OmniParser).
- [TinyClick](https://arxiv.org/abs/2410.11871) - Single-turn GUI agent on Florence-2-base.

Applications:

- [Fine-Tuning Florence2 for Enhanced Object Detection in Un-constructed Environments](https://arxiv.org/abs/2503.04918)
- [Fine-Tuning Vision-Language Model for Automated Engineering Drawing Information Extraction](https://arxiv.org/abs/2411.03707)
- [Multimodal AI for Gastrointestinal Diagnostics: Tackling VQA in MEDVQA-GI 2025](https://arxiv.org/abs/2507.14544)

Reliability:

- [InspectVLM: Unified in Theory, Unreliable in Practice](https://arxiv.org/abs/2508.01921) - Tests whether unified models such as Florence-2 hold up across tasks.
- [One Object, Multiple Lies](https://arxiv.org/abs/2507.07709) - Cross-task adversarial attacks on unified vision-language models.

## Deployment and runtimes

- 🟢 [onnx-community/Florence-2-base-ft](https://huggingface.co/onnx-community/Florence-2-base-ft) - ONNX export in several precisions, for ONNX Runtime and Transformers.js ([large-ft](https://huggingface.co/onnx-community/Florence-2-large-ft)).
- 🟢 [Florence-2 on WebGPU](https://github.com/huggingface/transformers.js-examples/tree/main/florence2-webgpu) - Runs in the browser with Transformers.js ([live demo](https://huggingface.co/spaces/Xenova/florence2-webgpu)).
- 🟢 [Blaizzy/mlx-vlm](https://github.com/Blaizzy/mlx-vlm) - Apple silicon through MLX.
- 🟢 [jamjamjon/usls](https://github.com/jamjamjon/usls) - Rust, on ONNX Runtime.
- 🟢 [curiosity-ai/florence2-sharp](https://github.com/curiosity-ai/florence2-sharp) - C#/.NET, on ONNX Runtime, with a [performance guide](https://docs.curiosity.ai/florence2-sharp/advanced/performance).
- 🟢 [mlboydaisuke/Florence-2-base-CoreML](https://huggingface.co/mlboydaisuke/Florence-2-base-CoreML) - iOS through Core ML, CPU only ([ExecuTorch version](https://huggingface.co/mlboydaisuke/Florence-2-base-ExecuTorch)).
- 🟢 [roboflow/inference](https://github.com/roboflow/inference) - Inference server with a Florence-2 Workflows block.
- 🟢 [ducviet00/fast-encdec](https://github.com/ducviet00/fast-encdec) - Experimental engine reporting 2.75 to 4.2× faster generation than Transformers on an L4 GPU.
- 🟠 [vllm-project/bart-plugin](https://github.com/vllm-project/bart-plugin) - vLLM dropped built-in Florence-2 support in September 2025; this plugin brings it back.
- 🟠 [OpenVINO notebook](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/florence2) - Converts and runs Florence-2 on Intel hardware.
- 🟠 [happyme531/Florence-2-base-ft-ONNX-RKNN2](https://huggingface.co/happyme531/Florence-2-base-ft-ONNX-RKNN2) - Rockchip NPU (AGPL-3.0).

Few speed numbers are published. [Roboflow](https://docs.roboflow.com/models/supported-models/florence-2) measures a 128-token caption at 652 ms (base) and 1,120 ms (large) on an L4 GPU, and [mlsysbook](https://mlsysbook.ai/kits/contents/raspi/vlm/vlm.html) times every task on a Raspberry Pi 5 (16 s for a caption with base). Latency grows with the length of the output, so captions are fast and segmentation is slow.

## Integrations and pipelines

Labelling and datasets:

- 🟢 [roboflow/supervision](https://github.com/roboflow/supervision) - Turns Florence-2 output into detections with `sv.Detections.from_vlm(sv.VLM.FLORENCE_2, ...)`.
- 🟢 [FiftyOne](https://github.com/voxel51/fiftyone) - Florence-2 in the FiftyOne model zoo.
- 🟠 [X-AnyLabeling](https://github.com/CVHub520/X-AnyLabeling) - Desktop annotation tool with Florence-2 auto-labelling.
- ⚪ [autodistill-grounded-sam-2](https://github.com/autodistill/autodistill-grounded-sam-2) - Florence-2 + SAM 2 for auto-labelling segmentation data.

Segmentation, tracking and editing:

- 🟠 [IDEA-Research/Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2) - Florence-2 + SAM 2 demos and an auto-labelling pipeline.
- 🟢 [D-Ogi/WatermarkRemover-AI](https://github.com/D-Ogi/WatermarkRemover-AI) - Florence-2 finds watermarks and LaMa removes them, on images and video.

ComfyUI and captioning tools:

- 🟢 [kijai/ComfyUI-Florence2](https://github.com/kijai/ComfyUI-Florence2) - The most widely used Florence-2 node for ComfyUI, with its own Florence-2 implementation; supports PromptGen and other fine-tunes.
- 🟢 [1038lab/ComfyUI-RMBG](https://github.com/1038lab/ComfyUI-RMBG) - Background removal and segmentation nodes, including Florence-2.
- 🟢 [gokayfem/ComfyUI_VLM_nodes](https://github.com/gokayfem/ComfyUI_VLM_nodes) - ComfyUI nodes for several VLMs, including Florence-2.
- 🟠 [MieMieeeee/ComfyUI-CaptionThis](https://github.com/MieMieeeee/ComfyUI-CaptionThis) - Captions a whole folder, one `.txt` per image.
- ⚪ [jhc13/taggui](https://github.com/jhc13/taggui) - Desktop tagger and captioner for image datasets.

Agents and robotics:

- 🟢 [jkawamoto/mcp-florence2](https://github.com/jkawamoto/mcp-florence2) - MCP server exposing Florence-2 OCR and captioning to AI agents.
- 🟠 [sandraschi/ocr-mcp](https://github.com/sandraschi/ocr-mcp) - MCP server bundling several OCR models, including Florence-2.
- 🟠 [florence2_ros2_wrapper](https://github.com/JEDominguezVidal/florence2_ros2_wrapper) - ROS 2 node.

## Demos

- 🟢 [prithivMLmods/Florence-2-Image-Caption](https://huggingface.co/spaces/prithivMLmods/Florence-2-Image-Caption) - Captioning demo.
- 🟢 [Xenova/florence2-webgpu](https://huggingface.co/spaces/Xenova/florence2-webgpu) - Runs entirely in your browser.
- 🟠 [gokaygokay/Florence-2](https://huggingface.co/spaces/gokaygokay/Florence-2) - The most-liked demo, covering all tasks.
- 🟠 [gokaygokay/Flux-Florence-2](https://huggingface.co/spaces/gokaygokay/Flux-Florence-2) - Image to Flux prompt.
- ⚪ [SkalskiP/florence-sam](https://huggingface.co/spaces/SkalskiP/florence-sam) - Florence-2 + SAM 2 (paused).
- ⚪ [andito/Florence-2-DocVQA](https://huggingface.co/spaces/andito/Florence-2-DocVQA) - DocVQA fine-tune (currently erroring).

## Known limitations

These come from the model itself, so the native implementation does not fix them.

| Limitation | Workaround | Discussion |
|------------|------------|------------|
| No confidence scores for boxes | Score boxes from the token probabilities of the `<loc_>` tokens (experimental) | [#88](https://huggingface.co/microsoft/Florence-2-large/discussions/88) |
| Grounding returns a box even when the object is absent | Check with a second model or a confidence threshold | [#88](https://huggingface.co/microsoft/Florence-2-large/discussions/88) |
| No free-form questions (VQA) | Fine-tune with your own task prefix (see [notebook 3](README.md#notebooks)), or use a chat model | [#58](https://huggingface.co/microsoft/Florence-2-large/discussions/58), [this repo #3](https://github.com/anyantudre/Florence-2-Vision-Language-Model/issues/3) |
| One region per `<REGION_TO_*>` prompt | Send one prompt per box in a single batch for the same image | [#85](https://huggingface.co/microsoft/Florence-2-large/discussions/85), [this repo #1](https://github.com/anyantudre/Florence-2-Vision-Language-Model/issues/1) |
| `<OCR>` merges lines | Use `<OCR_WITH_REGION>` to get one entry per line | [#90](https://huggingface.co/microsoft/Florence-2-large/discussions/90) |
| Misses small objects in busy scenes | Run on image tiles, e.g. with supervision's `InferenceSlicer` | [#96](https://huggingface.co/microsoft/Florence-2-large/discussions/96) |
| English output only | Fine-tune on your language | [#101](https://huggingface.co/microsoft/Florence-2-large/discussions/101) |
| Batched inference can be slower than one by one | Batch one task at a time, use greedy decoding, cap `max_new_tokens` | [#57](https://huggingface.co/microsoft/Florence-2-large/discussions/57) |

## Contributing

Suggestions are welcome through issues and pull requests. Each entry should:

- Use the format `- FLAG [Name](link) - What it does, in one sentence.`
- Carry a flag (🟢, 🟠 or ⚪) checked against the project's code or model config, not its README alone.
- Be useful to Florence-2 users today, or be historically important enough to keep with an ⚪ flag.
