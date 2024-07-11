# Given a note or multiple notes
# Pass notes to GPT as context
# Prompt model to generate a question and answer based on the context
# with specifications and requirements for the type of question

import os
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

def call_gpt(discharge_summary):

    client = AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"), 
        api_key=os.getenv("AZURE_OPENAI_KEY"),
        api_version=os.getenv("AZURE_API_VERSION")
    )

    model_name = "gpt-35-turbo-16k"

    question = """
    Please create a question-answer pair out of this discharge summary to benchmark the capabilities of a question-answering large language model
    """

    prompt = [{
        "role": "user",
        "content": (
                "Discharge Summary :\n"
                f"{discharge_summary}\n\n"
                f"Request : {question}\n\n"
                "Answer :"
            )
    }]

    response = client.chat.completions.create(
        model=model_name,
        messages=prompt,
        max_tokens=600,
        temperature=0
    )

    return response.choices[0].message.content
