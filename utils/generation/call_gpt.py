# Given a note or multiple notes
# Pass notes to GPT as context
# Prompt model to generate a question and answer based on the context
# with specifications and requirements for the type of question

import os
from openai import AzureOpenAI
from dotenv import load_dotenv
from azure.core.exceptions import HttpResponseError
import time

load_dotenv()

QA_GENERATION_MODEL = "gpt-35-turbo-16k"


def call_gpt(discharge_summary, include_explanation):

    max_retries = 10
    retry_delay = 5

    client = AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_KEY"),
        api_version=os.getenv("AZURE_API_VERSION"),
    )

    system_message = """You are an expert medical professional tasked
    with creating clinically relevant question-answer pairs based on a
    discharge summary from the MIMIC-III database."""

    if include_explanation:
        # Prompt asking for a rationale behind the answer
        user_prompt = f"""
            You are given a discharge summary from the MIMIC-III 
            database. Your task is to generate a question and answer pair that is relevant 
            to clinical practice.

            Clinically relevant questions should test the ability to summarize, 
            identify, and arrange text, and answer specific questions related to:
            1. Treatment
            2. Assessment
            3. Diagnosis
            4. Problems or complications
            5. Abnormalities
            6. Etiology
            7. Medical history

            The question should also be one of the following types:
            1. Yes/No/Maybe
            2. Unanswerable
            3. Temporal
            4. Factual
            5. Summarisation
            6. Identification

            Your response should also contain short, one-sentence rationale behind 
            the answer detailing the reason why the answer is correct.

            Do not create a question that is too easy to answer, only clinicians 
            should be able to answer the question. Do not create a question that 
            can be answered without referring to the discharge summary. Do not
            create a question-answer pair with exactly matching details.

            Please follow this format:

            Question: [Insert your clinical question here]
            Answer: [Insert the corresponding answer here]
            Type: [Your chosen question type from the list, spelled the same]
            Reason: [Short explanation for the answer, based on the discharge 
            summary]

            Here is the discharge summary for you to work on:

            {discharge_summary}

            Please provide a clinically relevant question and answer based on the 
            above discharge summary.
        """
        # Prompt not including rationale
    else:
        user_prompt = f"""
            You are given a discharge summary from the MIMIC-III 
            database. Your task is to generate a question and answer pair that is relevant 
            to clinical practice.

            Clinically relevant questions should test the ability to summarize, 
            identify, and arrange text, and answer specific questions related to:
            1. Treatment
            2. Assessment
            3. Diagnosis
            4. Problems or complications
            5. Abnormalities
            6. Etiology
            7. Medical history

            The question should also be one of the following types:
            1. Yes/No/Maybe
            2. Unanswerable
            3. Temporal
            4. Factual
            5. Summarisation
            6. Identification

            Do not create a question that is too easy to answer, only clinicians 
            should be able to answer the question. Do not create a question that 
            can be answered without referring to the discharge summary. Do not
            create a question-answer pair with exactly matching details.

            Please follow this format:

            Question: [Insert your clinical question here]
            Answer: [Insert the corresponding answer here]
            Type: [Your chosen question type from the list, spelled the same]

            Here is the discharge summary for you to work on:

            {discharge_summary}

            Please provide a clinically relevant question and answer based on the 
            above discharge summary.
        """

    for i in range(0, max_retries):
        try:
            response = client.chat.completions.create(
                model=QA_GENERATION_MODEL,
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
    

def prepare_discharge_summaries(discharge_summaries):
    discharge_summary_string = ""
    for i in discharge_summaries:
        start_string = f"[Discharge summary {i} start]"
        end_string = f"[Discharge summary {i} end]"
        discharge_summary = discharge_summaries[i]
        ds_string += start_string + discharge_summary + end_string
    return discharge_summary_string


def call_gpt_with_multiple_discharge_summaries(discharge_summaries):

    max_retries = 10
    retry_delay = 5

    client = AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_KEY"),
        api_version=os.getenv("AZURE_API_VERSION"),
    )

    discharge_summaries = prepare_discharge_summaries(discharge_summaries)

    system_message = """You are an expert medical professional tasked
    with creating clinically relevant question-answer pairs based on a
    discharge summary from the MIMIC-III database."""

    user_prompt = f"""
        You are given a set of discharge summaries from the MIMIC-III 
        database. Your task is to generate a question and answer pair that is relevant 
        to clinical practice.

        Clinically relevant questions should test the ability to summarize, 
        identify, and arrange text, and answer specific questions related to:
        1. Treatment
        2. Assessment
        3. Diagnosis
        4. Problems or complications
        5. Abnormalities
        6. Etiology
        7. Medical history

        The question should also be one of the following types:
        1. Yes/No/Maybe
        2. Unanswerable
        3. Temporal
        4. Factual
        5. Summarisation
        6. Identification`

        Do not create a question that is too easy to answer, only clinicians 
        should be able to answer the question. Do not create a question that 
        can be answered without referring to the discharge summary. Do not
        create a question-answer pair with exactly matching details.

        Please follow this format:

        Question: [Insert your clinical question here]
        Answer: [Insert the corresponding answer here]
        Type: [Your chosen question type from the list, spelled the same]

        The discharge summaries are provided between [Discharge summary 
        n start] and [Discharge summary n end] where n is the number 
        corresponding to the discharge summary being passed. The 
        discharge summaries are provided in chronological order.

        Here are the discharge summaries for you to work on.

        {discharge_summaries}

        Please provide a clinically relevant question and answer based on the 
        above discharge summary.
    """

    for i in range(0, max_retries):
        try:
            response = client.chat.completions.create(
                model=QA_GENERATION_MODEL,
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
