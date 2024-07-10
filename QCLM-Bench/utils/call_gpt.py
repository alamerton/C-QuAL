# given a note or multiple notes
# pass notes to GPT as context
# prompt model to generate a question and answer based on the context
    # with specifications and requirements for the type of question

import os
from openai import AzureOpenAI
from dotenv import load_dotenv
load_dotenv()

azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_openai_key = os.getenv("AZURE_OPENAI_KEY")
azure_api_version = os.getenv("AZURE_API_VERSION")

