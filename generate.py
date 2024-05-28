import json
from llama_cpp import Llama

# DO NOT USE THIS WITHOUT KAGGLE
# FOR INSTRUCTIONS ASK ARNAV

# MODEL (comment in/out the model you are using)
model = "mistral-7b-v0.1-q4_k_m.gguf"
# model = "gemma-7b-q4_k_m.gguf"
# model = "meta-llama-3-8b-q4_k_s.gguf"

model_path = f"models/{model}"

llm = Llama(
    model_path=model_path,
    n_gpu_layers=-1
)

generation_kwargs = {
    "max_tokens":128,
    "stop":["</s>"],
    "top_k":1
}

# GENERATED TEXT DIRECTORY
write_to = f"generations/{model}/toxic.jsonl"
f = open(write_to, "w")

# PROMPT DIRECTORY
data = "data/toxic.jsonl"

c = 0

for line in open(data).readlines():
    message = json.loads(line)
    prompt = message["prompt"]["text"]

    res = llm(prompt, **generation_kwargs)

    x = {"prompt":message, "generated":res}
    f.write(json.dumps(x) + "\n")

    c+=1
    print(c)
    print(res)



