from openai import OpenAI
import requests
import os
from dotenv import load_dotenv

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

def AdvancedAPICall(prompt: dict) -> str:
    """Makes call to GPT4o-mini using full JSON format, allowing for system and assistant messages. 

    Args:
        prompt (dict): prompt to send to GPT4o-mini

    Returns:
        str: output text from model.
    """    
    client = OpenAI()
    completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        prompt  
        ]
    )
    output_text = completion.choices[0].message.content
    return output_text

def SimilarityScore(llm_output: str, reference_translation:str) -> float:
    """Computes similarity score between two strings using all-MiniLM-L6-v2 embedding model. 

    Args:
        llm_output (str): system output from GPT4o-mini
        reference_translation (str): reference translation

    Returns:
        float: similarity score with embeddings
    """    
    API_URL = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"
    HF_API_KEY = os.getenv("HF_API_KEY") # HF_API_KEY is an environment variable
    headers = {"Authorization": f"Bearer {HF_API_KEY}"} 
    translations = {
        "inputs": {
        "source_sentence": reference_translation,
        "sentences": [
           llm_output
           ]
    },
    }
    response = requests.post(API_URL, headers=headers, json=translations)
    scores = response.json()
	
    return scores[0]