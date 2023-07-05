from typing import List
from fastapi import FastAPI
from transformers import GPT2LMHeadModel, GPT2Tokenizer
model_name = "sshleifer/tiny-gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
app = FastAPI()
@app.post("/generate")
def generate_text(prompt: str, max_length: int = 100, num_return_sequences: int = 1):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(
        input_ids,
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        do_sample=True,
        pad_token_id=model.config.pad_token_id,
        eos_token_id=model.config.eos_token_id,
        bos_token_id=model.config.bos_token_id,
        top_k=50,
        temperature=0.7
    )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"result": generated_text}
