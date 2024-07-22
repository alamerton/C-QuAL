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
from utils.benchmark_with_azure import benchmark_with_azure
from utils.misc import save_dataset

DATASET_PATH = 'data/generations/10-QA-pairs-2024-07-17 15:48:59.369671.csv'
CLOUD = True
MODEL_NAME = "gpt-35-turbo-16k"

def benchmark_model(dataset_path, model_name):
    # TODO: extend to include local models

    print("Loading dataset")

    dataset: pd.DataFrame = pd.read_csv(dataset_path)

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
        dataset[f'{model_name} Response'] = response
    
    return dataset

def main():
    model_responses = benchmark_model(DATASET_PATH, MODEL_NAME)
    save_dataset(model_responses, local=True)
