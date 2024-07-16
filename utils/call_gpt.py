# Given a note or multiple notes
# Pass notes to GPT as context
# Prompt model to generate a question and answer based on the context
# with specifications and requirements for the type of question

import os
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

def call_gpt(discharge_summary, include_explanation):

    client = AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_KEY"),
        api_version=os.getenv("AZURE_API_VERSION"),
    )

    model_name = "gpt-35-turbo-16k"

    system_message = """You are an expert medical professional tasked with 
    creating clinically relevant question-answer pairs based on a discharge 
    summary from the MIMIC-III database."""

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
        
        Do not create a question that is too easy to answer, only doctors 
        should be able to answer the question. Do not create a question that 
        can be answered without referring to the discharge summary. Your 
        answer should also contain short rationale behind the answer.

        Please follow this format:

        Question: [Insert your clinical question here]
        Answer: [Insert the corresponding answer here]
        {"Reason: [Explanation for your answer]" if include_explanation == True
          else None}

        Examples of clinically relevant questions:
        - What was the primary diagnosis for the patient? (Diagnosis)
        - What treatment did the patient receive during their hospital stay? 
        (Procedures)
        - What are the follow-up care instructions for the patient? (Patient 
        Management)
        - Were there any complications noted during the patient's stay? 
        (Complications)
        - What medications were prescribed at discharge? (Changes in 
        Medication)
        - What tests did the patient undergo during their admission? (Tests)
        - What abnormalities were noted in the test results? (Abnormalities)
        - What is the patient's medical history of chronic conditions? 
        (Patient’s Medical History)

        Non-examples of clinically relevant questions:
        - What is the patient’s favorite color? (Irrelevant to clinical care)
        - What time was the patient admitted? (Unless it impacts clinical 
        decisions)


        Here is the discharge summary for you to work on:

        {discharge_summary}

        Please provide a clinically relevant question and answer based on the 
        above discharge summary.
    """

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=600,
        temperature=0,
    )

    return response.choices[0].message.content
