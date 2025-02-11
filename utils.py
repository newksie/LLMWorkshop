from openai import OpenAI
import requests
import os
from dotenv import load_dotenv

def BasicAPICall(system_prompt: str, prompt: str) -> str:
    """Makes call to GPT4o-mini using just a simple prompt, no system or assistant messages. 

    Args:
        prompt (str): prompt to send to GPT4o-mini

    Returns:
        str: output text from model.
    """    
    client = OpenAI()
    tobe_translated = "Le weekend dernier, je flânais en ville et, coup de chance, je suis tombée sur ma maman - nous habitons loin l'un de l'autre et nous ne nous retrouvons pas souvent. Enfin, elle est frileuse et j'avais un petit creux donc nous sommes allées dans un bar du coin pour réchauffer, goûter et boire un verre. Dans le bar, les lumières étaient éblouissantes et maman a été gonflée que le serveur la tutoie. Du coup, nous sommes allés dans un autre bar qu'un copain disait servir les cocktails chics. Mais, en fait, c'était un bordel - les boissons étaient écoeurantes et tout le monde était bourré. Enfin, après tout ça, nous avons fini notre retrouvaille chez moi avec une bouteille de pinard et des biscuits. "
    completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": f"{system_prompt}"},
        {"role": "user", "content": f"{prompt}: {tobe_translated}"}
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

def SimilarityScore(llm_output: str, reference_translation: str) -> float:
    """
    Computes similarity score between two strings using the all-MiniLM-L6-v2 embedding model via Hugging Face API.

    Args:
        llm_output (str): System output from GPT4o-mini.
        reference_translation (str): Reference translation.

    Returns:
        float: Similarity score with embeddings, or raises an exception with an appropriate error message.
    """
    API_URL = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"
    HF_API_KEY = os.getenv("HF_API_KEY")  # HF_API_KEY is an environment variable

    if not HF_API_KEY:
        raise EnvironmentError("Hugging Face API key (HF_API_KEY) is not set in environment variables.")

    headers = {"Authorization": f"Bearer {HF_API_KEY}"}

    payload = {
        "inputs": {
            "source_sentence": reference_translation,
            "sentences": [
                llm_output
            ]
        },
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
    except requests.exceptions.RequestException as e:
        # Handle network-related errors
        raise ConnectionError(f"Network error occurred while connecting to Hugging Face API: {str(e)}")

    if response.status_code == 200:
        try:
            scores = response.json()
            if isinstance(scores, list) and len(scores) > 0:
                similarity = scores[0]
                if isinstance(similarity, (int, float)):
                    return similarity
                else:
                    raise ValueError(f"Unexpected score format: {similarity}")
            else:
                raise ValueError("Empty or invalid response format from Hugging Face API.")
        except ValueError as ve:
            raise ValueError(f"Error parsing response from Hugging Face API: {str(ve)}")
        except Exception as e:
            raise Exception(f"An unexpected error occurred while processing the response: {str(e)}")
    else:
        try:
            error_info = response.json()
            error_message = error_info.get("error", response.text)
        except ValueError:
            error_message = response.text

        raise Exception(f"Hugging Face API returned an error [{response.status_code}]: {error_message}")