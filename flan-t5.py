import time
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

print("Programs starts at ", time.asctime())
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")

print("Inference starts at ", time.asctime())
inputs = tokenizer("A step by step recipe to make bolognese pasta:", return_tensors="pt")
outputs = model.generate(**inputs)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
print("Inference ends at ", time.asctime())
