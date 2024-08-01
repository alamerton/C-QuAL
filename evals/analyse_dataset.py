"""
This script should load an annotated or unannotated dataset, and perform
analysis on the datasets to get scores for each analysis method, saving
the analysis results to a CSV file.
"""

import pandas as pd
from tqdm import tqdm
from datetime import datetime
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)
from utils.evals.categorise_with_gpt import categorise_with_gpt

DATASET_PATH = "data/generations/10-QA-pairs-2024-07-17 15:48:59.369671.csv"


def get_question_categories(df: pd.DataFrame):
    """
    Given a dataset, return the number of questions that fit into each
    of the following categories: treatment, assessment, diagnosis,
    problem or complication, abnormality, etiology, and medical history
    """

    (
        treatment,
        assessment,
        diagnosis,
        problem_complication,
        abnormality,
        etiology,
        medical_history,
    ) = 0

    for question in df["Question"]:
        category_response = categorise_with_gpt(question)
        if "," in question:
            categories = category_response.split(",")
            for i in range(0, len(categories)):
                match categories[i]:
                    case "Treatment":
                        treatment += 1
                    case "Assessment":
                        assessment += 1
                    case "Diagnosis":
                        diagnosis += 1
                    case "Problem or complication":
                        problem_complication += 1
                    case "Abnormality":
                        abnormality += 1
                    case "Etiology":
                        etiology += 1
                    case "Medical history":
                        medical_history += 1
        else:
            match category_response:
                case "Treatment":
                    treatment += 1
                case "Assessment":
                    assessment += 1
                case "Diagnosis":
                    diagnosis += 1
                case "Problem or complication":
                    problem_complication += 1
                case "Abnormality":
                    abnormality += 1
                case "Etiology":
                    etiology += 1
                case "Medical history":
                    medical_history += 1

    return (
        treatment,
        assessment,
        diagnosis,
        problem_complication,
        abnormality,
        etiology,
        medical_history,
    )


def get_question_types(df: pd.DataFrame):
    """
    Given a datset, return the total number of rows for each question
    type
    """
    question_types = [
        "Yes/No/Maybe",
        "Unanswerable",
        "Temporal",
        "Factual",
        "Summarisation",
        "Identification",
    ]
    for question_type in question_types:
        df[f"Number of {question_type} Questions"] = df["Type"].str.count(
            question_type
        )
    return 


def get_question_complexity(df: pd.DataFrame):
    """
    Use bleu ideally with a library and return mean
    """
    return 0


# Return a dataframe containing analysis results for a given dataset


def get_dataset_statistics(dataset_path):
    dataset = pd.read_csv(dataset_path)
    dataset_length = len(dataset)
    dataset = get_question_types(dataset)

    statistics = pd.DataFrame(
        {
            "Metric": [
                "Number of QA Pairs",
                "Question Types",
                "BLEU Complexity",
                "Topic Distribution",
                "Coverage of Clinical Concepts",
            ],
            "Value": [
                dataset_length,
            ],
        }
    )
    return statistics


def main():
    data = get_dataset_statistics(DATASET_PATH)
    name = DATASET_PATH.split("/")[-1]
    output_path = f"data/analysis/dataset-{name}-analysed-at-{datetime.now()}"
    # data.to_csv(f"{output_path}.csv")
    data.to_csv("data/analysis/overwrite.csv")


if __name__ == "__main__":
    main()
