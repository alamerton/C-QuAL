from call_mimic import get_summaries_for_generation

NUMBER_OF_QA_PAIRS = 1
SUMMARIES_DESTINATION = "function"
MODEL_NAME = "gpt-3.5-turbo"

import tiktoken

def count_tokens(text, model):
    encoder = tiktoken.encoding_for_model(model)
    return len(encoder.encode(text))


def calculate_average_tokens(strings):
    total_tokens = 0
    for string in strings:
        total_tokens += count_tokens(string, MODEL_NAME)
    average_tokens = total_tokens / len(strings)
    return average_tokens


strings = get_summaries_for_generation(
    NUMBER_OF_QA_PAIRS,
    SUMMARIES_DESTINATION
)
average_tokens = calculate_average_tokens(strings)
print(f"Average tokens: {average_tokens}")
