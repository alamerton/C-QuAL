import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, parent_dir)
from utils.call_gpt import call_gpt
from utils.call_mimic import call_mimic

NUMBER_OF_QA_PAIRS = 5

def main():
    discharge_summary = call_mimic()


        

    question_answer_pair = call_gpt(discharge_summary)

    # Probably add for loop for generating x qa pairs
    """
    [amount of Q-A pairs to generate = number of discharge summaries loaded from mimic]
    this could be its own function somewhere else
    or call_mimic could take an integer as an argument, specifying the number of discharge
    summaries to load. Another consideration for the future is whether to feed the LLM more
    than one discharge summary.

    1. for x in [amount of Q-A pairs to generate]:
    2.      get discharge summary with call_mimic/or iterate through premade df containing discharge summaries
    3.      call LLM with discharge summary and prompt
    4.      add Q-A pair to dataframe
    """

    
    # Probably write dataset to output directory


    return question_answer_pair


if __name__ == "__main__":
    main()
