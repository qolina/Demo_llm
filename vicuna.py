import time
# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

print("Programs starts at ", time.asctime())
tokenizer = AutoTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5")
model = AutoModelForCausalLM.from_pretrained("lmsys/vicuna-7b-v1.5")

#from transformers import AutoTokenizer, LlamaForCausalLM
#model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)

print("Inference starts at ", time.asctime())
prompt = "Hey, are you conscious? Can you talk to me?"
inputs = tokenizer(prompt, return_tensors="pt")

# Generate
generate_ids = model.generate(inputs.input_ids, max_length=30)
print(tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])
print("Inference ends at ", time.asctime())
