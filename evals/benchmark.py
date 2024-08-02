import pandas as pd
from tqdm import tqdm
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)
from utils.evals.benchmark_with_azure import benchmark_with_azure
from utils.misc import save_dataset

DATASET_PATH = "data/generations/10-QA-pairs-2024-07-17 15:48:59.369671.csv"
# CLOUD = True
MODEL_NAME = "gpt-35-turbo-16k"


def record_model_responses(dataset_path, model_name):
    # TODO: extend to include local models

    print("Loading dataset")

    dataset: pd.DataFrame = pd.read_csv(dataset_path)

    for _, row in tqdm(
        dataset.iterrows(), total=len(dataset), desc=f"Benchmarking model"
    ):

        discharge_summary = row["Discharge Summary"]
        question = row["Question"]
        response = benchmark_with_azure(model_name, discharge_summary, question)
        dataset[f"{model_name} Response"] = response

    return dataset


# some of these might need the discharge summay, and some might not need the
# expected response.


def get_exact_match(expected_response: str, model_response: str):
    return expected_response == model_response


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
    exact_match = get_exact_match()
    benchmarking_results = pd.DataFrame(
        {
            "Metric": [
                "Exact Match",
                "F1 Score",
                "Semantic Answer Similarity",
                "Rouge",
                "BLEU",
                "Clinical Concept Extraction",
                "Medical Relation Extraction",
                "G-Eval",
            ],
            "Value": [
                exact_match,
            ],
        }s
    )
    return benchmarking_results


def main():
    model_responses = record_model_responses(DATASET_PATH, MODEL_NAME)
    save_dataset(model_responses, directory="model_responses")
    benchmarking_results = score_model(model_responses)
    save_dataset(benchmarking_results, directory="benchmarking-results")
