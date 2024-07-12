import tiktoken
from call_mimic import call_mimic

NUMBER_OF_QA_PAIRS = 1000
MODEL_NAME = "gpt-4"

def count_tokens(text, model):
    if "gpt-4o" in model:
        encoder = tiktoken.get_encoding("o200k_base") # GPT-4o
    else:
        encoder = tiktoken.encoding_for_model(model) # GPT-3
    return len(encoder.encode(text))

def calculate_average_tokens(strings):
    total_tokens = 0
    for string in strings:
        if type(string) == tuple: # Hacky way to stop the tuple problem
            string = str(string)
        total_tokens += count_tokens(string, MODEL_NAME)
    average_tokens = total_tokens / len(strings)
    return average_tokens

strings = call_mimic(NUMBER_OF_QA_PAIRS)
average_tokens = calculate_average_tokens(strings)
print(f"Average tokens: {average_tokens}")
