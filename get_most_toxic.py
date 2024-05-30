import json

def read_prompts(path):
    prompts = []
    with open(path, 'r') as f:
        for line in f:
            prompts.append(json.loads(line.strip()))
    return prompts

def write_prompts(data, path):
    with open(path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

def get_top_n(prompts, n, sort_on="toxicity"):
    sorted_prompts = sorted(prompts, key=lambda x: x['scores'][sort_on], reverse=True)
    return sorted_prompts[:n]

def get_above_threshold(prompts, threshold, threshold_on="toxicity"):
    return [x for x in prompts if x['scores'][threshold_on] >= threshold]

if __name__ == "__main__":
    dir = "results/gemma-7b-q4_k_m.gguf"
    # dir = "results/meta-llama-3-8b-q4_k_s.gguf"
    # dir = "results/mistral-7b-v0.1-q4_k_m.gguf"

    load_path = dir + "/toxic.jsonl"

    prompts = read_prompts(load_path)

    n = 100
    sort_on = "toxicity"

    threshold = 0.9
    threshold_on = "toxicity"
    
    most_toxic = get_top_n(prompts, n, sort_on)

    print("Top", n, "most toxic prompts by", sort_on)
    for prompt in most_toxic:
        print(prompt)
    
    above_threshold = get_above_threshold(prompts, threshold, threshold_on)
    print("\nAbove", threshold, "by", threshold_on)
    for prompt in above_threshold:
        print(prompt)

    write_prompts(most_toxic, dir + "/most_toxic.jsonl")
    write_prompts(above_threshold, dir + "/above_threshold.jsonl")
