from fastapi import FastAPI
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
from pydantic import BaseModel
model_name = "typeform/distilbert-base-uncased-mnli"
model = DistilBertForSequenceClassification.from_pretrained(model_name)
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
app = FastAPI()
class InferenceRequest(BaseModel):
    text: str
@app.post("/classify")
def classify_text(request: InferenceRequest):
    encoded_input = tokenizer.encode_plus(
        request.text,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )
    logits = model(**encoded_input).logits
    predicted_labels = logits.argmax(dim=1).tolist()
    return {"predicted_label": predicted_labels[0]}
