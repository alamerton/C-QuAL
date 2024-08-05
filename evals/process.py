
import pandas as pd
import re

DATASET_PATH = '../data/generations/c-qual-xl.csv'

# Open a dataset and remove any row where the answer contains an 
# exact match of a given length
def remove_n_matching_pairs(question, answer, n):

    question = str(question) if pd.notna(question) else ''
    answer = str(answer) if pd.notna(answer) else ''

    question_words = re.findall(r'\b\w+\b', question.lower())
    answer_words = re.findall(r'\b\w+\b', answer.lower())

    for i in range(len(question_words) - 2):
        for j in range(len(answer_words) - 2):
            if question_words[i:i+n] == answer_words[j:j+n]:
                return True
    return False

def removed_spurious_pairs(dataset_path):
    df = pd.read_csv(dataset_path)
    df_filtered = df[~df.apply(
        lambda row: remove_n_matching_pairs(
            row['Question'], row['Expected Answer']), 3, axis=1)
        ]
    print("New shape: ", df_filtered.shape)
    df_filtered.to_csv(f'../data/processing/matching_pairs/dataset_processed_overwrite.csv')
