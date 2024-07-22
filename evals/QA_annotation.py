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
import os, sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
from utils.annotate_with_gpt import annotate_with_gpt

#TODO: place in some environment variables place instead of in code
DATASET_PATH = '' # for now use 10-row datasets or shorter for time

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

    # dataset['Annotation'] = None # can probably remove

    for row in dataset:
        # Call the LLM with the row and prompt
        
        # Set variables
        discharge_summary = row['Discharge Summary']
        question = row['Question']
        expected_answer = row['Expected Answer']
        reason = row['Reason']      
        
        annotation = annotate_with_gpt(
            discharge_summary,
            question,
            expected_answer,
            reason
        )

        dataset['Annotation'] = annotation

    return dataset
