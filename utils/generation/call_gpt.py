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
            You are given a discharge summary from the MIMIC-III database. 
            Your task is to generate a question and answer pair that is relevant 
            to clinical practice, focusing on important aspects like diagnosis, 
            treatment, prognosis, patient management, or follow-up care. The 
            answer should be specific and extracted directly from the summary.

            Clinically relevant questions should test the ability to summarize, 
            identify, and arrange text, and answer specific questions related to:
            1. Patient’s medical history
            2. Diagnoses made
            3. Procedures that were done
            4. Outcomes of procedures
            5. Changes in medication
            6. Complications
            7. Abnormalities
            8. Tests the patient has undergone

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
            can be answered without referring to the discharge summary.

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
            You are given a discharge summary from the MIMIC-III database. 
            Your task is to generate a question and answer pair that is relevant 
            to clinical practice, focusing on important aspects like diagnosis, 
            treatment, prognosis, patient management, or follow-up care. The 
            answer should be specific and extracted directly from the summary.

            Clinically relevant questions should test the ability to summarize, 
            identify, and arrange text, and answer specific questions related to:
            1. Patient’s medical history
            2. Diagnoses made
            3. Procedures that were done
            4. Outcomes of procedures
            5. Changes in medication
            6. Complications
            7. Abnormalities
            8. Tests the patient has undergone

            The question should also be one of the following types:
            1. Yes/No/Maybe
            2. Unanswerable
            3. Temporal
            4. Factual
            5. Summarisation
            6. Identification

            Do not create a question that is too easy to answer, only clinicians 
            should be able to answer the question. Do not create a question that 
            can be answered without referring to the discharge summary.

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
                temperature=0,
            )
            return response.choices[0].message.content
        except HttpResponseError as e:
            if '429' in str(e):
                print(f"Rate limit exceeded. Attempt {i + 1} of {max_retries}.")
                time.sleep(retry_delay)
                retry_delay *= 2 
            else:
                raise
        raise RuntimeError("Maximum retries exceeded.")
