
import os
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

ANNOTATION_MODEL = "gpt-35-turbo-16k"

def categorise_with_gpt(
        question,
        ):

    client = AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_KEY"),
        api_version=os.getenv("AZURE_API_VERSION"),
    )

    system_message = """
    You are an expert medical professional tasked with categorising 
    clinical questions to the best of your ability.
    """

    user_prompt = f"""
        Your task is to cateogorise the following clinical question 
        into one of the following categories: treatment, assessment,
        diagnosis, problem or complication, abnormality, etiology, and 
        medical history.

        Please only return the category or categories that best fit the 
        question. If the question fits more than one cateogory, please 
        return the categories that fit the question separated by a 
        comma and no space.

        Here is the question: {question}
        Category/categories:
    """

    response = client.chat.completions.create(
        model=ANNOTATION_MODEL,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=600,
        temperature=0,
    )

    return response.choices[0].message.content
