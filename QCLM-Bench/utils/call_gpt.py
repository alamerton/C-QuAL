# given a note or multiple notes
# pass notes to GPT as context
# prompt model to generate a question and answer based on the context
# with specifications and requirements for the type of question

# %%
import os
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_openai_key = os.getenv("AZURE_OPENAI_KEY")
azure_api_version = os.getenv("AZURE_API_VERSION")

client = AzureOpenAI(
    azure_endpoint=azure_endpoint,
    api_key=azure_openai_key,
    api_version=azure_api_version,
)

model_name = "gpt-35-turbo-16k"

# %%

patient_note = "note"

question = "tell me about blah"

# %%

prompt = {
    "role": "user",
    "content": {
        "Discharge Summary: \n:"
        "{patient_note}\n\n"
        "Question: {question}"
        "Answer: "
    },
}

# %%

response = client.chat.completions.create(
    model=model_name,
    messages=prompt,
    max_tokens=600,
    temperature=0
)

print(response)
# %%
