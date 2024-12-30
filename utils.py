from openai import OpenAI
import os
from bert_score import BERTScorer

def BasicAPICall(prompt: str) -> str:
    """Makes call to GPT4o-mini using just a simple prompt, no system or assistant messages. 

    Args:
        prompt (str): prompt to send to GPT4o-mini

    Returns:
        str: output text from model.
    """    
    client = OpenAI()
    completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": f"{prompt}"}
        ]
    )
    output_text = completion.choices[0].message.content
    return output_text

class BERTEvaluator:
    def __init__ (self):
        self.model = 'yongsun-yoon/minilmv2-bertscore-distilled'
        self.layers = 6

        self.scorer = BERTScorer(model_type=self.model, num_layers=self.layers)

    def evaluate(self, reference, translation):
        model = self.scorer
        P, R, F = model.score([reference], [translation])
        return F

class CometEvaluator:
    def __init__(self):
        # Specify the metric to use; for example, 'wmt20-comet-da'
        self.metric = 'Unbabel/wmt22-comet-da'

        # Download and load the COMET model
        model_path = download_model(self.metric)
        self.model = load_from_checkpoint(model_path)

    def evaluate(self, source: str, hypothesis: str, reference: str) -> float:
        data = [
            {
        "src": f"{source}",
        "mt": f"{hypothesis}",
        "ref": f"{reference}"
        # "ref": "Can it be delivered between 10 to 15 minutes?"
            }
        ]
        scores = self.model.predict(
            data,
        )
        return scores[0]  # Return the score for the first (and only) input

