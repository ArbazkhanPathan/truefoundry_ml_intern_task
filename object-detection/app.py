import io
import torch
from PIL import Image
from fastapi import FastAPI, UploadFile
from transformers import DetrFeatureExtractor, DetrForObjectDetection
import aiohttp
import asyncio
model_name = "TahaDouaji/detr-doc-table-detection"
model = DetrForObjectDetection.from_pretrained(model_name)
feature_extractor = DetrFeatureExtractor.from_pretrained(model_name)
app = FastAPI()
@app.post("/predict")
async def detect_table(file: UploadFile):
    image = Image.open(io.BytesIO(await file.read()))
    image_tensor = feature_extractor(images=image, return_tensors="pt").pixel_values

    outputs = model(image_tensor)
    predicted_boxes = outputs.pred_boxes[0].detach().cpu().numpy().tolist()

    return {"predicted_boxes": predicted_boxes}
