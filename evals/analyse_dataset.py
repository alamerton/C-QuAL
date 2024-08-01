"""
This script should load an annotated or unannotated dataset, and perform
analysis on the datasets to get scores for each analysis method, saving
the analysis results to a CSV file.

TODO: it could also be useful creating notebooks to run which display
graphs quickly
"""

import pandas as pd
from tqdm import tqdm
from textstat import textstat
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from datetime import datetime
import sys, os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

DATASET_PATH = "data/generations/10-QA-pairs-2024-07-17 15:48:59.369671.csv"
READABILITY_METRIC = 'flesch_reading_ease'

def get_number_of_qa_pairs(df: pd.DataFrame):
    return df.apply(len)

# def get_question_types(question: str):
#     if "how" in question.lower():
#         return "P"
#     elif "what" in question.lower():
#         return "F"
#     elif "why" in question.lower():
#         return "E"
#     else:
#         return "O"
    
# def get_question_length(df: pd.DataFrame):
#     return df['Question'].apply(len).mean()


def get_question_categories(df: pd.DataFrame):
    """
    Given a dataset, return the number of questions that fit into each
    of the following categories: treatment, assessment, diagnosis, 
    problem or complication, abnormality, etiology, and medical history
    """
    # for each question, add 1 to score of each category for categories
    # the question fits into
    
    treatment, assessment, diagnosis, problem_complication, \
    abnormality, etiology, medical_history = 0

    for question in df['Question']:
        if 

def get_question_types(df: pd.DataFrame):
    """
    Given a datset, return the total number of rows for each question
    type
    """
    return False

def get_category_distribution():
    """
    Given a list of categories and a dataset, return the proportion of 
    questions in the dataset that fit into each category
    """
    return False

# Return a dataframe containing analysis results for a given dataset

def get_dataset_statistics(dataset_path):
    statistics = pd.DataFrame()
    dataset = pd.read_csv(dataset_path)
    dataset_length = len(dataset)
    # size of QA pairs
    statistics['Number of QA Pairs'] = dataset_length
    statistics['Flesch Reading Ease'] = get_readability(
        dataset,
        READABILITY_METRIC
    )
    statistics['Topic Distribution'] = get_topic_distribution(dataset) # use mesh
    statistics["Procedural Questions"] = 0 #TODO: refactor
    statistics["Factual Questions"] = 0
    statistics["Explanatory Questions"] = 0
    statistics['Other Questions'] = 0
    
    for _, row in tqdm(dataset.iterrows()):
        # TODO: how can I do this operation faster?
        # types of QA pairs
        if get_question_types(row['Question']) == "P":
            statistics["Procedural Questions"] += 1
        elif get_question_types(row['Question']) == "F":
            statistics["Factual Questions"] += 1
        elif get_question_types(row['Question']) == "E":
            statistics["Explanatory Questions"] += 1
        elif get_question_types(row['Question']) == "O":
            statistics["Other Questions"] += 1
        else:
            return ValueError("Unrecognised question type")
    
    return statistics
        
def main():
    data = return_dataset_statistics(DATASET_PATH)
    name = DATASET_PATH.split("/")[-1]
    output_path = f'data/analysis/dataset-{name}-analysed-at-{datetime.now()}'
    # data.to_csv(f"{output_path}.csv")
    data.to_csv("data/analysis/overwrite.csv")

if __name__ == "__main__":
    main()