
import os
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

def benchmark_gpt_with_azure(
        model_name,
        discharge_summary,
        question,
        ):

    client = AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_KEY"),
        api_version=os.getenv("AZURE_API_VERSION"),
    )

    system_message = """You are an expert medical professional tasked 
    with answering a clinical question to the best of your ability. You 
    must construct your answer based on the evidence provided to you in 
    the discharge summary."""

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

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=999,
        temperature=0,
    )

    return response.choices[0].message.content

def benchmark_llama_with_azure():
    return 0