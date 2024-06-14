import json

# Run this file after score.py to get the top n most toxic prompts and prompts above a certain threshold.
# After running this file run attribute.ipynb to get the attribution scores.

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

def get_bottom_n(prompts, n, sort_on="toxicity"):
    sorted_prompts = sorted(prompts, key=lambda x: x['scores'][sort_on], reverse=True)
    return sorted_prompts[::-1][:n]

def get_above_threshold(prompts, threshold, threshold_on="toxicity"):
    return [x for x in prompts if x['scores'][threshold_on] >= threshold]

if __name__ == "__main__":
    model = 'mistral'     # 'bloom' / 'llama' / 'mistral'
    dir = f"results/{model}"

    load_path = dir + "/toxic.jsonl"

    prompts = read_prompts(load_path)

    # Top n most toxic prompts
    n = 20
    sort_on = "toxicity"
    most_toxic = get_top_n(prompts, n, sort_on)
    least_toxic = get_bottom_n(prompts, n, sort_on)

    # Prompts above threshold for toxicity
    threshold = 0.95
    threshold_on = "toxicity"
    above_threshold = get_above_threshold(prompts, threshold, threshold_on)
    
    print("Top", n, "most toxic prompts by", sort_on)
    for prompt in most_toxic:
        print(prompt)
    
    print("\nAbove", threshold, "by", threshold_on)
    for prompt in above_threshold:
        print(prompt)

    write_prompts(most_toxic, dir + "/most_toxic.jsonl")
    write_prompts(least_toxic, dir + "/least_toxic.jsonl")
    write_prompts(above_threshold, dir + "/above_threshold.jsonl")
