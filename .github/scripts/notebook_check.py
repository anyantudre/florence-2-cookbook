"""Check that the code the notebooks rely on still works with the latest transformers and supervision.

Runs on CPU with the smallest checkpoint. Exits with an error if anything breaks.
"""
import requests
import supervision as sv
import torch
import transformers
from PIL import Image
from transformers import AutoProcessor, Florence2ForConditionalGeneration

MODEL_ID = "florence-community/Florence-2-base-ft"
IMAGE_URL = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg"

print(f"torch {torch.__version__}, transformers {transformers.__version__}, supervision {sv.__version__}")

model = Florence2ForConditionalGeneration.from_pretrained(MODEL_ID, dtype=torch.float32).eval()
processor = AutoProcessor.from_pretrained(MODEL_ID)
image = Image.open(requests.get(IMAGE_URL, stream=True, timeout=60).raw).convert("RGB")


def run(task, text_input=""):
    inputs = processor(text=task + text_input, images=image, return_tensors="pt")
    with torch.inference_mode():
        ids = model.generate(**inputs, max_new_tokens=128, num_beams=1, do_sample=False)
    text = processor.batch_decode(ids, skip_special_tokens=False)[0]
    return processor.post_process_generation(text, task=task, image_size=image.size)[task]


caption = run("<CAPTION>")
print("caption:", caption)
assert "car" in caption.lower(), "caption does not mention the car"

detections = run("<OD>")
print("detection:", detections)
assert "car" in detections["labels"], "object detection did not find the car"

grounding = run("<CAPTION_TO_PHRASE_GROUNDING>", "a green car")
print("grounding:", grounding)
assert grounding["bboxes"], "phrase grounding returned no box"

parsed = sv.Detections.from_vlm(sv.VLM.FLORENCE_2, {"<OD>": detections}, resolution_wh=image.size)
assert len(parsed) == len(detections["bboxes"]), "supervision could not parse the detections"

# One training step, as in the fine-tuning notebooks: the loss must be a finite number.
model.train()
inputs = processor(text="<CAPTION>", images=image, return_tensors="pt")
labels = processor.tokenizer("A green car parked in front of a yellow building.", return_tensors="pt").input_ids
loss = model(**inputs, labels=labels).loss
loss.backward()
print("training loss:", loss.item())
assert torch.isfinite(loss), "training loss is not finite"

print("All checks passed.")
