import re
import nltk

def reduce_discharge_summary(discharge_summary):
    # Remove whitespace
    discharge_summary = re.sub(r'\s', ' ', discharge_summary).strip()
    # Remove special characters and punctuation
    discharge_summary = re.sub(r'[^\w\s]', '', discharge_summary)

def prepare_discharge_summaries(discharge_summaries):
    multiple_summaries = ""
    for i in discharge_summaries:
        start_string = f"[Discharge summary {i} start]\n"
        end_string = f"\n[Discharge summary {i} end]\n"
        discharge_summary = discharge_summaries[i]
        discharge_summary = reduce_discharge_summary(discharge_summary)
        multiple_summaries += start_string + discharge_summary + end_string
    return multiple_summaries
