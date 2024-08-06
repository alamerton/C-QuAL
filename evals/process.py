
import pandas as pd
import re

# DATASET_PATH = '../data/generations/c-qual-xl.csv'
DATASET_PATH = 'data/processing/annotated_high_quality_pairs.csv'
SAVE_PATH = 'data/processing/annotated_high_quality_pairs.csv'

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

def remove_spurious_pairs(dataset_path, save_path):
    df = pd.read_csv(dataset_path)
    df_filtered = df[~df.apply(
        lambda row: remove_n_matching_pairs(
            row['Question'], row['Expected Answer']), 3, axis=1)
        ]
    df_filtered.to_csv(save_path)
    print("New shape: ", df_filtered.shape)

def remove_low_quality_pairs(dataset_path, save_path):
    df = pd.read_csv(dataset_path)
    no_zeros_df = df[~df['Annotation'].astype(str).str.contains('0')]
    # no_annotations_df = no_zeros_df.drop('Annotation', axis=1)
    # new_df = no_annotations_df.reset_index(drop=True)
    new_df = no_zeros_df.reset_index(drop=True)
    print(new_df.head)
    new_df.to_csv(save_path)

def remove_extraneous_columns(dataset_path, save_path):
    df = pd.read_csv(dataset_path)
    columns_to_keep = ['Discharge Summaries', 'Question', 'Expected Answer', 'Question Type']
    relevant_columns_df = df[columns_to_keep]
    relevant_columns_df.to_csv(save_path)

def combine_csv_files(csv_1_path, csv_2_path):
    df1 = pd.read_csv(csv_1_path)
    df2 = pd.read_csv(csv_2_path)

    df1_subset = df1.iloc[:630]
    df2_subset = df2.iloc[630:]

    combined_df = pd.concat([df1_subset, df2_subset])
    combined_df.to_csv('data/annotations/combined_annotation_datasets.csv')

def remove_missing_value_rows(dataset_path, save_path):
    df = pd.read_csv(dataset_path)
    # df = df.drop('Discharge Summaries', axis=1)
    df = df.dropna()
    df.to_csv(save_path)

def main():
    remove_missing_value_rows("data/model-answers/checkpoints/Mistral-large-qgpdg-330-rows-2024-08-06 03:28:33.csv", "data/model-answers/Mistral-large.csv")


if __name__ == '__main__':
    main()

