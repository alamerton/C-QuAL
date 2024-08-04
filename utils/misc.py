# %%

from datetime import datetime
import tiktoken
from generation.call_mimic_iii import call_mimic_iii

NUMBER_OF_QA_PAIRS = 1000
MODEL_NAME = "gpt-4"


def save_dataset(dataset, directory: str):
    date = datetime.now()
    rows = len(dataset)
    # TODO: add model name to file name
    output_path = f"data/{directory}/{rows}-QA-pairs-{date}"
    dataset.to_csv(f"{output_path}.csv")


def count_tokens(text, model):
    if "gpt-4o" in model:
        encoder = tiktoken.get_encoding("o200k_base")  # GPT-4o
    else:
        encoder = tiktoken.encoding_for_model(model)  # GPT-3
    return len(encoder.encode(text))


def calculate_average_tokens(strings, model_name=MODEL_NAME):
    total_tokens = 0
    for string in strings:
        if type(string) == tuple:  # Hacky way to stop the tuple problem
            string = str(string)
        total_tokens += count_tokens(string, model_name)
    average_tokens = total_tokens / len(strings)
    return average_tokens


def calculate_max_tokens(strings, model_name=MODEL_NAME):
    max_tokens = 0
    for string in strings:
        if type(string) == tuple:
            string = str(string)
        token_count = count_tokens(string, model_name)
        if token_count > max_tokens:
            max_tokens = token_count
    return max_tokens


def calculate_max_discharge_summaries(model_name, limit=10):
    # return the maximum number of discharge summaries you can send
    # a model
    biggest_ds_strings = []
    for i in range(0, limit):
        strings = call_mimic_iii(NUMBER_OF_QA_PAIRS, i)
        biggest_ds_strings.append(calculate_max_tokens(strings, model_name))
    return biggest_ds_strings


# %%

# strings = call_mimic_iii(NUMBER_OF_QA_PAIRS, 4)
# average_tokens = calculate_average_tokens(strings)
# max_tokens = calculate_max_tokens(strings, "gpt-35-turbo-16k")
# print(f"Max tokens: {max_tokens}")

limit = 10
biggest_string_token_lengths = calculate_max_discharge_summaries(
    "gpt-35-turbo-16k", limit
)

print(
    f"Biggest discharge summary strings with limit {limit} summaries: { \
    biggest_string_token_lengths}"
)
# %%
