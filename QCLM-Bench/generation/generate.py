import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, parent_dir)

from utils.call_gpt import call_gpt
from utils.call_mimic import call_mimic

def main():
    discharge_summary = call_mimic()
    question_answer_pair = call_gpt.call_gpt(discharge_summary)
    print(question_answer_pair)

if __name__ == "main":
    main()
