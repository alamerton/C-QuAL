import tiktoken
from call_mimic import call_mimic

NUMBER_OF_QA_PAIRS = 1
MODEL_NAME = "gpt-3.5-turbo"

def count_tokens(text, model):
    encoder = tiktoken.encoding_for_model(model)
    encoded_text = encoder.encode(text)
    # print(encoded_text)
    return len(encoder.encode(text))

def calculate_average_tokens(strings):
    total_tokens = 0
    for string in strings:
        print(type(string))
        total_tokens += count_tokens(string, MODEL_NAME)
    average_tokens = total_tokens / len(strings)
    return average_tokens

strings = call_mimic(NUMBER_OF_QA_PAIRS)
average_tokens = calculate_average_tokens(strings)
print(f"Average tokens: {average_tokens}")
