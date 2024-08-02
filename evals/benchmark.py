import pandas as pd
from tqdm import tqdm
import sys
import os
from sklearn.metrics import f1_score
import nltk
from sentence_transformers import SentenceTransformer, util
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)
from utils.evals.benchmark_with_azure import benchmark_with_azure
from utils.misc import save_dataset

nltk.download("punkt")
sas_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
rouge = Rouge()


DATASET_PATH = "data/generations/3-QA-pairs-2024-08-01 14:04:41.574896.csv"
# CLOUD = True
MODEL_NAME = "gpt-35-turbo-16k"


def record_model_answers(dataset_path, model_name):
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


def get_exact_match(expected_answer: str, model_answer: str):
    return expected_answer == model_answer


def get_f1_score(expected_answer: str, model_answer: str):
    expected_tokens = nltk.word_tokenize(expected_answer.lower)
    model_tokens = nltk.word_tokenize(model_answer.lower)
    score = f1_score(expected_tokens, model_tokens, average="weighted")
    return score


def get_sas(expected_answer: str, model_answer: str):
    expected_answer = sas_model.encode(expected_answer, convert_to_tensor=True)
    model_answer = sas_model.encode(model_answer, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(expected_answer, model_answer)
    return similarity.item()


def get_rouge(expected_answer: str, model_answer: str):
    return rouge.get_scores(expected_answer, model_answer)


def get_bleu(expected_answer: str, model_answer: str):
    expected_tokens = nltk.word_tokenize(expected_answer.lower)
    model_tokens = nltk.word_tokenize(model_answer.lower)
    return sentence_bleu(expected_tokens, model_tokens)


def get_clinical_concept_extraction(expected_answer: str, model_answer: str):
    # might not need
    return 0


def get_medical_relation_extraction(expected_answer: str, model_answer: str):
    return 0


def get_g_eval(expected_answer: str, model_answer: str):
    return 0


def score_model(dataset):

    # loop through each row in the dataset:
    # save exact match scores to arrays

    (
        em_scores,
        f1_scores,
        sas_scores,
        rouge_scores,
        bleu_scores,
        cce_scores,
        mre_scores,
        g_eval_scores,
    ) = []

    for row in range(0, len(dataset)):
        expected_answer = dataset["Expected Answer"][row]
        model_answer = dataset["Model Answer"][row]

        em_scores.append(get_exact_match(expected_answer, model_answer))
        f1_scores.append(get_f1_score(expected_answer, model_answer))
        sas_scores.append(get_sas(expected_answer, model_answer))
        rouge_scores.append(get_rouge(expected_answer, model_answer))
        bleu_scores.append(get_bleu(expected_answer, model_answer))


    # get average of arrays to get final scores

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
        }
    )
    return benchmarking_results


def main():
    model_answers = record_model_answers(DATASET_PATH, MODEL_NAME)
    save_dataset(model_answers, directory="model-answers")
    benchmarking_results = score_model(model_answers)
    save_dataset(benchmarking_results, directory="benchmarking-results")


if __name__ == "__main__":
    main()
