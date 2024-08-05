# %%
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


# %%
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-70B")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-70B")
# %%
