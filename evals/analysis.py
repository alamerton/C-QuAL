"""
This script should load an annotated or unannotated dataset, and perform
analysis on the datasets to get scores for each analysis method, saving
the analysis results to a CSV file.
"""

import pandas as pd

DATASET_PATH = 'data/generations/10-QA-pairs-2024-07-17 15:48:59.369671.csv'

def evaluate(dataset_path):
    dataset = pd.read_csv(dataset_path)
    