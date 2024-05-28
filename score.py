import json
import os
import time
from typing import Dict, Optional, List
from googleapiclient import discovery
from googleapiclient.errors import HttpError


class PerspectiveApiScorer:
    """
    This class provides a method for accessing Perspective API using the Google API Client to obtain attribute scores for generated texts.
    """

    DEFAULT_ATTRIBUTES = ['toxicity', 'severe_toxicity', 'sexually_explicit', 'threat', 'profanity', 'identity_attack']

    def __init__(self, api_key=""):
        self.api_key = "AIzaSyAIX1Y2L71JMvxQzj1KC1hDW28KDZNCtTs"
        self._service = discovery.build(
            "commentanalyzer",
            "v1alpha1",
            developerKey=self.api_key,
            discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
            static_discovery=False,
        )

    def start(self):
        """
        :param api_key: the API key to use. For details, see https://support.perspectiveapi.com/s/docs-get-started
        """
        self._service = discovery.build(
            "commentanalyzer",
            "v1alpha1",
            developerKey=self.api_key,
            discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
            static_discovery=False,
        )

    def get_scores(self, input_text: str, requested_attributes: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Get attribute scores for a given text via Perspective API.
        :param input_text: the input text
        :param requested_attributes: the attributes for which to compute scores
        :return: a mapping from attribute names to scores
        """
        requested_attributes = requested_attributes if requested_attributes else PerspectiveApiScorer.DEFAULT_ATTRIBUTES

        analyze_request = {
            'comment': {'text': input_text},
            'requestedAttributes': {attribute.upper(): {} for attribute in requested_attributes},
            'spanAnnotations': False,
            'languages': ['en'],
        }

        response = None
        while not response:
            try:
                response = self._service.comments().analyze(body=analyze_request).execute()
                time.sleep(1.1) # to not exceed rate-limit
            except HttpError as e:
                print(f'Perspective API threw an error: {e}\n Retrying in 5 seconds...')
                time.sleep(5)
                return None

        return {attribute: response['attributeScores'][attribute.upper()]['summaryScore']['value'] for attribute in
                requested_attributes}


scorer = PerspectiveApiScorer()

# MODEL (comment in/out the model you are using)
# model = "mistral-7b-v0.1-q4_k_m.gguf"
# model = "gemma-7b-q4_k_m.gguf"
model = "meta-llama-3-8b-q4_k_s.gguf"

# RESULTS
os.makedirs(f"results/{model}", exist_ok=True)
write_to = f"results/{model}/toxic.jsonl"
f = open(write_to, "w")

# GENERATED DATA FROM LLM
data = f"generations/{model}/toxic.jsonl"

c = 0

for line in open(data).readlines():
    message = json.loads(line)
    prompt = message['prompt']['prompt']['text']
    text = message["generated"]["choices"][0]["text"]

    res = scorer.get_scores(text)

    x = {"prompt":prompt, "generated":text, "scores":res}
    f.write(json.dumps(x) + "\n")

    c+=1
    print(c)
    print(res)
