# Given a note or multiple notes
# Pass notes to GPT as context
# Prompt model to generate a question and answer based on the context
# with specifications and requirements for the type of question

import os
import random
from openai import AzureOpenAI
from dotenv import load_dotenv
from azure.core.exceptions import HttpResponseError
import time

load_dotenv()


def get_question_type():
    question_types = [
        "Yes/No/Maybe",
        "Unanswerable",
        "Temporal",
        "Factual",
        "Summarisation",
    ]
    return random.choice(question_types)


def call_gpt(model_name, discharge_summary_string, include_explanation):

    max_retries = 10
    retry_delay = 5
    question_type = get_question_type()

    client = AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_KEY"),
        api_version=os.getenv("AZURE_API_VERSION"),
    )

    system_message = """You are an expert medical professional tasked
    with creating clinically relevant question-answer pairs based on a
    discharge summary from the MIMIC-III database."""

    if include_explanation:
        user_prompt = f"""
            You are given a set of discharge summaries from the MIMIC-III 
            database. Your task is to generate a question and answer pair that is relevant 
            to clinical practice.

            Clinically relevant questions should test the ability to summarise, 
            identify, and arrange text, and answer specific questions related to:
            - Treatment
            - Assessment
            - Diagnosis
            - Problems or complications
            - Abnormalities
            - Etiology
            - Medical history

            The question should be of the following type: ${question_type}

            Your response should also contain short, one-sentence rationale behind 
            the answer detailing the reason why the answer is correct.

            Do not create a question that is too easy to answer, only clinicians 
            should be able to answer the question. Do not create a question that 
            can be answered without referring to the discharge summary. Do not
            create a question-answer pair with exactly matching details.

            Please follow this format exactly:

            Question: [Insert your clinical question here]\n
            Answer: [Insert the corresponding answer here]\n
            Type: [Your chosen question type from the list, spelled the same]\n
            Reason: [Short explanation for the answer, based on the discharge 
            summaries]\n

            If there are multiple discharge summaries, each will be provided 
            between [Discharge summary n start] and [Discharge summary n end] 
            where n is the number corresponding to the discharge summary being 
            given. The discharge summaries are provided in chronological order.

            Here are the discharge summaries for you to work on:

            {discharge_summary_string}

            Please provide a clinically relevant question and answer based on the 
            above discharge summary.
        """
    else:
        user_prompt = f"""
            You are given a set of discharge summaries from the MIMIC-III 
            database. Your task is to generate a question and answer pair that is relevant 
            to clinical practice.

            Clinically relevant questions should test the ability to summarize, 
            identify, and arrange text, and answer specific questions related to:
            - Treatment
            - Assessment
            - Diagnosis
            - Problems or complications
            - Abnormalities
            - Etiology
            - Medical history

            The question should be of the following type: ${question_type}

            Do not create a question that is too easy to answer, only clinicians 
            should be able to answer the question. Do not create a question that 
            can be answered without referring to the discharge summary. Do not
            create a question-answer pair with exactly matching details.

            Please follow this format:

            Question: [Insert your clinical question here]
            Answer: [Insert the corresponding answer here]
            Type: [Your chosen question type from the list, spelled the same]

            If there are multiple discharge summaries, each will be provided 
            between [Discharge summary n start] and [Discharge summary n end] 
            where n is the number corresponding to the discharge summary being 
            given. The discharge summaries are provided in chronological order.

            Here is the discharge summary text for you to work on.

            {discharge_summary_string}

            Please provide a clinically relevant question and answer based on the 
            above discharge summary.
        """

    for i in range(0, max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=999,
                temperature=1,
            )
            return response.choices[0].message.content
        except HttpResponseError as e:
            if "429" in str(e):
                print(f"Rate limit exceeded. Attempt {i + 1} of {max_retries}.")
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                raise
        raise RuntimeError("Maximum retries exceeded.")
