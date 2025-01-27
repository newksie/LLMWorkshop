from comet import download_model, load_from_checkpoint
from openai import OpenAI
import requests
import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import json

def BasicAPICall(prompt, model="gpt-4o-mini", system_prompt=None):
    """Makes call to GPT4o-mini using just a simple prompt, no system or assistant messages. 

    Args:
        prompt (str): prompt to send to GPT4o-mini

    Returns:
        str: output text from model.
    """    
    client = OpenAI()
    if system_prompt:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": f"{system_prompt}"},
                {"role": "user", "content": f"{prompt}"}
            ]
        )
    else:
        completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": f"{prompt}"}
            ]
        )
    output_text = completion.choices[0].message.content
    return output_text

def create_system_message(system_prompt):
    """
    Create the system message for the conversation.
    Args:
        system_prompt (str): The system-level instruction.
    Returns:
        dict: A dictionary representing the system message.
    """
    return {"role": "system", "content": system_prompt}


def create_few_shot_messages(few_shot_df):
    """
    Create few-shot examples from a DataFrame.
    Args:
        few_shot_df (pd.DataFrame): DataFrame containing 'fr' and 'en' columns for few-shot examples.
    Returns:
        list of dict: List of few-shot example messages.
    """
    few_shot_messages = []
    for _, row in few_shot_df.iterrows():
        few_shot_messages.append({"role": "user", "content": f"Translate the following French text into English: {row.fr}"})
        few_shot_messages.append({"role": "assistant", "content": row.en})
    return few_shot_messages


def create_prompt_message(french_text):
    """
    Create the main prompt message for translation.
    Args:
        french_text (str): The French text to be translated.
    Returns:
        dict: A dictionary representing the user prompt message.
    """
    return {"role": "user", "content": f"Translate the following French text into English: {french_text}"} # CHANGE HERE IF YOU WOULD LIKE!!

def create_chat_entry_with_custom_id(custom_id, system_prompt, few_shot_messages, french_text, model="gpt-4o-mini", temperature=0, max_tokens=200):
    """
    Create a single chat entry for the JSONL file, including the custom_id field.
    Args:
        custom_id (str): Unique identifier for the request.
        system_prompt (str): System-level instruction.
        few_shot_messages (list of dict): Few-shot examples as message dictionaries.
        french_text (str): French text to translate.
        model (str): Model to use for the translation.
        temperature (float): Sampling temperature for the API.
        max_tokens (int): Maximum tokens for the response.
    Returns:
        dict: A dictionary formatted for batch processing.
    """
    messages = [create_system_message(system_prompt)] + few_shot_messages + [create_prompt_message(french_text)]
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
    }


def create_jsonl_data_with_custom_id(df, few_shot_df, system_prompt, output_file, model="gpt-4o-mini", temperature=0, max_tokens=200):
    """
    Generate JSONL data for batch processing, including custom_id for each request.
    Args:
        df (pd.DataFrame): DataFrame containing 'fr' and 'en' columns for translations.
        few_shot_df (pd.DataFrame): DataFrame with few-shot examples.
        system_prompt (str): System-level instruction.
        output_file (str): Path to save the JSONL file.
        model (str): Model to use for translations.
        temperature (float): Sampling temperature for the API.
        max_tokens (int): Maximum tokens for responses.
    Returns:
        None
    """
    # Generate few-shot messages
    few_shot_messages = create_few_shot_messages(few_shot_df)
    
    # Prepare the JSONL data
    jsonl_data = []
    for idx, row in df.iterrows():
        custom_id = f"request-{idx + 1}"  # Unique ID for each request
        jsonl_data.append(
            create_chat_entry_with_custom_id(
                custom_id=custom_id,
                system_prompt=system_prompt,
                few_shot_messages=few_shot_messages,
                french_text=row.fr,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        )
    
    # Save to JSONL file
    save_to_jsonl(jsonl_data, output_file)

def save_to_jsonl(jsonl_data, output_file):
    """
    Save data to a JSONL file.
    Args:
        jsonl_data (list of dict): List of chat entry dictionaries.
        output_file (str): File path to save the JSONL file.
    Returns:
        None
    """
    with open(output_file, "w", encoding="utf-8") as f:
        for entry in jsonl_data:
            f.write(json.dumps(entry) + "\n")
    print(f"JSONL file saved to {output_file}")

def add_translations_to_df(df, batch_response_file, translation_column="translation"):
    """
    Add translations from the batch response file to the main DataFrame.
    
    Args:
        df (pd.DataFrame): The main DataFrame containing the original French sentences (column 'fr').
        batch_response_file (str): Path to the JSONL file containing the batch response.
        translation_column (str): The name of the new column to add for translations (default is 'translations').
        
    Returns:
        pd.DataFrame: The updated DataFrame with an additional column for translations.
    """
    # Read the batch response file
    with open(batch_response_file, "r") as f:
        batch_responses = [json.loads(line) for line in f]

    # Create a mapping of custom_id to the translation text
    translation_map = {
        response["custom_id"]: response['response']["body"]["choices"][0]["message"]["content"].strip()
        for response in batch_responses
    }
    
    # Generate the custom_id for each row in the DataFrame to match with the response
    df["custom_id"] = [f"request-{idx + 1}" for idx in range(len(df))]
    
    # Map the translations to the DataFrame using the custom_id
    df[translation_column] = df["custom_id"].map(translation_map)
    
    # Drop the custom_id column (optional)
    df = df.drop(columns=["custom_id"])
    
    return df