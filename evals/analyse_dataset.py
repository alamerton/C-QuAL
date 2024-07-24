"""
This script should load an annotated or unannotated dataset, and perform
analysis on the datasets to get scores for each analysis method, saving
the analysis results to a CSV file.
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

def get_question_types(question: str):
    if "how" in question.lower():
        return "P"
    elif "what" in question.lower():
        return "F"
    elif "why" in question.lower():
        return "E"
    else:
        return "O"
    
def get_readability(df: pd.DataFrame, method: str):
    if method == 'flesch_reading_ease':
        return df['Question'].apply(textstat.flesch_reading_ease).mean()
    elif method == 'gunning-fog':
        return df['Question'].apply(textstat.gunning_fog).mean()
    elif method == 'smog':
        return df['Question'].apply(textstat.smog_index).mean()
    else:
        return ValueError('Unknown readability metric')

def get_question_length(df: pd.DataFrame):
    return df['Question'].apply(len).mean()

def get_topic_distribution(df: pd.DataFrame):
    # Vectorize the text data
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df['Discharge Summary'])

    # Apply LDA
    lda = LatentDirichletAllocation(n_components=5, random_state=42)
    lda.fit(X)

    # Assign topics
    df['Topic'] = lda.transform(X).argmax(axis=1)

    # Count the topics
    topic_counts = df['Topic'].value_counts().mean()
    return topic_counts

def return_dataset_metadata(dataset_path):
    metadata = pd.DataFrame()
    dataset = pd.read_csv(dataset_path)
    dataset_length = len(dataset)
    # size of QA pairs
    metadata['Number of QA Pairs'] = dataset_length
    metadata['Flesch Reading Ease'] = get_readability(
        dataset,
        READABILITY_METRIC
    )
    metadata['Question Length'] = get_question_length(dataset)
    metadata['Topic Distribution'] = get_topic_distribution(dataset)
    metadata["Procedural Questions"] = 0 #TODO: refactor
    metadata["Factual Questions"] = 0
    metadata["Explanatory Questions"] = 0
    metadata['Other Questions'] = 0
    
    for _, row in tqdm(dataset.iterrows()):
        # TODO: how can I do this operation faster?
        # types of QA pairs
        if get_question_types(row['Question']) == "P":
            metadata["Procedural Questions"] += 1
        elif get_question_types(row['Question']) == "F":
            metadata["Factual Questions"] += 1
        elif get_question_types(row['Question']) == "E":
            metadata["Explanatory Questions"] += 1
        elif get_question_types(row['Question']) == "O":
            metadata["Other Questions"] += 1
        else:
            return ValueError("Unrecognised question type")
    
    return metadata
        
def main():
    data = return_dataset_metadata(DATASET_PATH)
    name = DATASET_PATH.split("/")[-1]
    output_path = f'data/analysis/dataset-{name}-analysed-at-{datetime.now()}'
    # data.to_csv(f"{output_path}.csv")
    data.to_csv("data/analysis/overwrite.csv")

if __name__ == "__main__":
    main()