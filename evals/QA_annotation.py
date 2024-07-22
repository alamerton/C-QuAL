"""
This file should load a QA dataset, and call an LLM with each row, 
prompting it to annotate the quality of the Q-A pair as if it is a 
clinician, saving the annotation in a new row, and perform the 
suggested revision on the row.

A copy of the annotated and edited dataset should then be saved, and 
a copy of the resulting dataset should have the annotations removed, 
and be saved as a finished QA dataset.

    1. Load the QA dataset, and create a new column called 
    '[model_name]_annotation'
    2. For every row in the dataset:
        2.1. Call the LLM with the row and prompt
        2.2. Save the annotation to the annotation column
    3. Save the dataset somewhere e.g. locally or HF

"""
from datasets import load_dataset
import pandas as pd

DATASET_PATH = '' # for now use 10-row datasets or shorter for time
#TODO: place in some environment variables place instead of in code
PROMPT = """
Your task is to evaluate the provided model output in response to a 
specific question associated with the given discharge summaries.\nBy 
using the correct answer also provided, you must score the answer as 
0 or 1, based on the following scoring instructions.\nScoring 
Instructions:\n1. Assign 1 point if the answer is correct.\n2. Assign 0 
points if the answer is either incorrect, or if it falsely claims there 
is no answer when one exists according to the discharge summaries.\n\n- 
Discharge Summaries:\n{note}\n\n- Question: {question}\n\n- Correct 
Answer: {correct_answer}\n\n- Model Output:\n\n{output}\n\nOutput 
format:\nScore: {{score}}\nReasoning: {{explanation}}\n(Note: In your 
response please replace {{score}} with the numerical score of 0 or 1, 
and provide a brief reasoning for the assigned score based on the 
evaluation criteria.)
"""

def load_dataset_from_hf(dataset_url):
    ds = load_dataset(dataset_url)
    return ds

def load_dataset_from_csv(path):
    df = pd.read_csv(path)
    # column names should already exist
    return df

def annotate_dataset(dataset_path, local: bool = False):
    # TODO: make better conditional inference on local flag e.g. string parse
    if local == False:
        # not sure if I can make a type assignment like this since df is class
        dataset: pd.DataFrame = load_dataset_from_hf(dataset_path)
    elif local == True:
        dataset: pd.DataFrame = load_dataset_from_csv(dataset_path)
    
    for row in dataset:
        # Call the LLM with the row and prompt
        
        # set Discharge Summary,Question,Expected Answer,Reason variables
        # inject them into string somehow
        