# Why are you so toxic? Analysing the causes of toxic generations from large language models

This repository contains all the code used to generate the results for the paper *Why are you so toxic? Analysing the causes of toxic generations from large language models*. Some things are noteworthy:
- The code for getting the toxicity scores can be found in `score.py`
- The prompts we used can be found under `data/toxic.jsonl`. Note that this file also contains continuations, but those are the continuations from [DecodingTrust](https://github.com/AI-secure/DecodingTrust), and are not used in the paper.
- `attr_agregate_and_threshold` must be run *after* `attribute.ipynb` to aggregate the tokens into full words and filter out only words with an attribution score above a certain threshold.