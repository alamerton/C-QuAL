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
from utils.evals.benchmark_with_azure import benchmark_with_azure
from utils.misc import save_dataset

DATASET_PATH = 'data/generations/10-QA-pairs-2024-07-17 15:48:59.369671.csv'
CLOUD = True
MODEL_NAME = "gpt-35-turbo-16k"

def record_model_responses(dataset_path, model_name):
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

# some of these might need the discharge summay, and some might not need the
# expected response.

def get_exact_match(expected_response: str, model_response: str):
    return 0

def get_f1_score(expected_response: str, model_response: str):
    return 0

def get_semantic_answer_similarity(expected_response: str, model_response: str):
    return 0

def get_rouge(expected_response: str, model_response: str):
    return 0

def get_bleu(expected_response: str, model_response: str):
    return 0

def get_clinical_concept_extraction(expected_response: str, model_response: str):
    # might not need 
    return 0

def get_medical_relation_extraction(expected_response: str, model_response: str):
    return 0

def get_g_eval(expected_response: str, model_response: str):
    return 0

def score_model(dataset):
    #TODO: write this function. 
    """Score the model  on how close its answer is to the expected 
    answer. Scoring metrics:
    - Exact match
    - F1 Score
    - Semantic answer similarity
    - ROUGE
    - Clinical concept extraction
    - Medical relation extraction
    - Semantic textual similarity
    - Natural language inference (NLI)
    - Medical question answering (MQA)
    - Safety evaluations
    """
    

def main():
    model_responses = record_model_responses(DATASET_PATH, MODEL_NAME)
    save_dataset(model_responses, local=True) #TODO: pass argument to save in model responses directory in data
