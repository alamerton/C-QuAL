"""
This script should load an annotated or unannotated dataset, and perform
analysis on the datasets to get scores for each analysis method, saving
the analysis results to a CSV file.
"""

import pandas as pd

DATASET_PATH = 'data/generations/10-QA-pairs-2024-07-17 15:48:59.369671.csv'

def return_dataset_metadata(dataset_path):
    dataset = pd.read_csv(dataset_path)
    for row in tqdm(
        dataset.iterrows()
    ):
        #TODO: Find metrics to use to analysis the dataset, and run them
        # here
        # size of QA pairs
        # types of QA pairs
        # Complexity of generated questions
        # Bleu
        # Jaccard
        return 0