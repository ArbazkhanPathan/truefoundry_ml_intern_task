from fastapi import FastAPI
from transformers import AutoModelForTokenClassification, AutoTokenizer
from pydantic import BaseModel
import torch
model_name = "d4data/biomedical-ner-all"
model = AutoModelForTokenClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
app = FastAPI()
class InferenceRequest(BaseModel):
    text: str
@app.post("/ner")
def ner(request: InferenceRequest):
    try:
        inputs = tokenizer.encode_plus(request.text, return_tensors="pt")
        input_ids = inputs["input_ids"].tolist()[0]
        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        print("Input IDs:", input_ids)
        print("Tokens:", tokens)
        outputs = model(**inputs)
        predicted_label_ids = torch.argmax(outputs.logits, dim=2)[0]
        print("Logits:", outputs.logits)
        print("Predicted Labels:", predicted_label_ids)
        predicted_labels = [model.config.id2label[label_id] for label_id in predicted_label_ids]
        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        entities = []
        entity = {"text": "", "start": None, "end": None, "type": None}
        for i, (token, label) in enumerate(zip(tokens, predicted_labels)):
            if label.startswith("B-"):
                if entity["text"]:
                    entities.append(entity.copy())
                entity["text"] = token
                entity["start"] = i
                entity["end"] = i
                entity["type"] = label[2:]
            elif label.startswith("I-"):
                entity["text"] += " " + token
                entity["end"] = i
            else:
                if entity["text"]:
                    entities.append(entity.copy())
                    entity = {"text": "", "start": None, "end": None, "type": None}
        if entity["text"]:
            entities.append(entity.copy())
        return {"entities": entities}
    except Exception as e:
        # Log the error or return an error response
        return {"error": str(e)}