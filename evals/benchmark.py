"""
This file should load a QA dataset and model, and loop through the dataset,
prompting the model with each discharge summary and question, saving
the answer to a copy of the dataset with a model answer column.
"""
import pandas as pd
from tqdm import tqdm
import sys, os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

DATASET_PATH = 'data/generations/10-QA-pairs-2024-07-17 15:48:59.369671.csv'
CLOUD = True
MODEL_NAME = "gpt-35-turbo-16k"

#TODO: put in general utils instead
def load_dataset_from_csv(path):
    df = pd.read_csv(path)
    # column names should already exist
    return df

def load_model_(model_name):
    #TODO: write
    return True

def benchmark_model(dataset_path, model_name):
    # TODO: extend to include local models

    print("Loading dataset")

    dataset: pd.DataFrame = load_dataset_from_csv(dataset_path)

    for _, row in tqdm(
        dataset.iterrows(),
        total = len(dataset),
        desc = f'Benchmarking model'
    ):
        
        discharge_summary = row['Discharge Summary']
        question = row['Question']
        response = benchmark_with_azure(
            model_name,
            discharge_summary,
            question
        )
        dataset[f'{model_name} Response']
    
    return dataset

def main():
    model_responses = benchmark_model(DATASET_PATH, MODEL_NAME)
