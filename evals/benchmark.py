import pandas as pd
from tqdm import tqdm
import sys
import os
from sklearn.metrics import f1_score
import nltk
import tensorflow_hub as hub
import numpy as np
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu
import sacrebleu

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)
from utils.evals.benchmark_with_azure import benchmark_with_azure
from utils.misc import save_dataset

DATASET_PATH = "data/generations/3-QA-pairs-2024-08-01 14:04:41.574896.csv"
MODEL_NAME = "gpt-35-turbo-16k"

embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
rouge = Rouge()


def record_model_answers(dataset_path, model_name):
    # TODO: extend to include local models
    print("Loading dataset")
    dataset = pd.read_csv(dataset_path)

    for index, row in tqdm(
        dataset.iterrows(), total=len(dataset), desc=f"Benchmarking model"
    ):
        discharge_summary = row["Discharge Summary"]
        question = row["Question"]
        response = benchmark_with_azure(model_name, discharge_summary, question)

        dataset.at[index, f"{model_name} Response"] = response
    return dataset


def get_exact_match(expected_answer: str, model_answer: str):
    return expected_answer == model_answer

#TODO: implement f1 score
def get_f1_score(expected_answer: str, model_answer: str):
    return 0

def get_sas(expected_answer: str, model_answer: str):
    embeddings = embed([expected_answer, model_answer])
    similarity_matrix = np.inner(embeddings, embeddings)
    return similarity_matrix[0, 1]


def get_rouge(expected_answer: str, model_answer: str):
    print("rouge scores: ", rouge.get_scores(expected_answer, model_answer))
    scores = rouge.get_scores(expected_answer, model_answer)[0]
    return scores['rouge-1']['f']

#TODO: implement bleu
def get_bleu(expected_answer: str, model_answer: str):
    return 0

#TODO: implement clinical_concept_extraction
def get_clinical_concept_extraction(model_answer: str):
    return 0

def get_medical_relation_extraction(expected_answer: str, model_answer: str):
    return 0

def get_g_eval(expected_answer: str, model_answer: str):
    return 0


def score_model(dataset, model_name):

    em_scores = []
    f1_scores = []
    sas_scores = []
    rouge_scores = []
    bleu_scores = []
    clinical_concept_extraction_scores = []
    medical_relation_extraction_scores = []
    g_eval_scores = []

    for row in tqdm(range(0, len(dataset))):
        expected_answer = dataset["Expected Answer"][row]
        model_answer = dataset[f"{model_name} Response"][row]

        em_scores.append(get_exact_match(expected_answer, model_answer))
        f1_scores.append(get_f1_score(expected_answer, model_answer))
        sas_scores.append(get_sas(expected_answer, model_answer))
        rouge_scores.append(get_rouge(expected_answer, model_answer))
        bleu_scores.append(get_bleu(expected_answer, model_answer))
        clinical_concept_extraction_scores.append(get_clinical_concept_extraction(model_answer))
        medical_relation_extraction_scores.append(get_medical_relation_extraction(expected_answer, model_answer))
        g_eval_scores.append(get_g_eval(expected_answer, model_answer))

    # np.mean better?
    exact_match = sum(em_scores) / len(em_scores)
    f1 = sum(f1_scores) / len(f1_scores)
    semantic_answer_similarity = sum(sas_scores) / len(sas_scores)
    rouge = sum(rouge_scores) / len(rouge_scores)
    bleu = sum(bleu_scores) / len(bleu_scores)
    clinical_concept_extraction = sum(clinical_concept_extraction_scores) / len(clinical_concept_extraction_scores)
    medical_relation_extraction = sum(medical_relation_extraction_scores) / len(medical_relation_extraction_scores)
    g_eval = sum(g_eval_scores) / len(g_eval_scores)

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
            f"{MODEL_NAME} Score": [
                exact_match,
                f1,
                semantic_answer_similarity,
                rouge,
                bleu,
                clinical_concept_extraction,
                medical_relation_extraction,
                g_eval,
            ],
        }
    )
    return benchmarking_results


def main():
    # model_answers = record_model_answers(DATASET_PATH, MODEL_NAME)
    # save_dataset(model_answers, directory="model-answers")
    model_answers = pd.read_csv("data/model-answers/3-QA-pairs-2024-08-02 14:03:57.715991.csv")
    benchmarking_results = score_model(model_answers, MODEL_NAME)
    save_dataset(benchmarking_results, directory="benchmarking-results")
    print(benchmarking_results)

    # clinical_string = "hello my chest hurts from pain in the heart cancer"
    # print(get_clinical_concept_extraction(clinical_string))


if __name__ == "__main__":
    main()
