import os
import requests
import json
import openai
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

def benchmark_with_openai(
    model_name,
    discharge_summary,
    question,
):
    
    system_message = """
        You are an expert medical professional tasked 
        with answering a clinical question to the best of your ability. You 
        must construct your answer based on the evidence provided to you in 
        the discharge summary.
        """

    user_prompt = f"""
        Your task is to answer a clinical question based on the
        following discharge summary:\n{discharge_summary}\n\n
        You should give an answer and a reason for your answer in the
        following format:
        Answer: [your answer]
        Reason: [your reason]
        Question: {question}\n\n
        Answer:
        """

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_prompt},
    ]

    if "gpt" in model_name:
        result = openai.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=999,
            temperature=0,
        )

    elif "Llama" in model_name:

        llama_endpoint = os.getenv("AZURE_LLAMA_ENDPONT")
        llama_api_key = os.getenv("AZURE_LLAMA_API_KEY")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {llama_api_key}",
        }

        data = {
            "messages": messages,
            "temperature": 0,
            "max_tokens": 999,
        }

        response = requests.post(
            f"{llama_endpoint}/v1/chat/completions",
            headers=headers,
            data=json.dumps(data),
        )

        if response.status_code == 200:
            result = response.json()
        else:
            print(f"An error occured, status code: {response.status_code}")
    else:
        raise ValueError("Model name not recognised by script.")

    return result.choices[0].message.content
