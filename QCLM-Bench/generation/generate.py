from datetime import datetime
import sys
import os
import time
from tqdm import tqdm

parent_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir)
)
sys.path.insert(0, parent_dir)
from utils.call_gpt import call_gpt
from utils.call_mimic import get_summaries_for_generation
import pandas as pd

NUMBER_OF_QA_PAIRS = 5
SUMMARIES_DESTINATION = "function"


def main():
    # create dataframe with question and expected answer columns
    # data = pd.DataFrame(columns=['Question', 'Correct Answer'])
    data = pd.DataFrame(columns=['Question/Answer pair'])

    print("Getting summaries for generation...")
    discharge_summaries = get_summaries_for_generation(
        NUMBER_OF_QA_PAIRS, SUMMARIES_DESTINATION
    )

    # For loop for generating x qa pairs
    print("Generating QA pairs...")
    for row in tqdm(range(NUMBER_OF_QA_PAIRS)):
        date = datetime.now()
        start_time = time.time()

        # Call LLM with discharge summary and prompt
        question_answer_pair = call_gpt(
            discharge_summaries[row]
        )

        # Parse the json to get the question and answer as variables
        parts = question_answer_pair.split('\n\n')

        # Add Q-A pair to dataframe
        data = pd.concat(
            [data, pd.DataFrame([question_answer_pair])], 
            ignore_index=True
        )

        print(f"{row+1}/{NUMBER_OF_QA_PAIRS}")
        time.sleep(10)
    
    print("Complete")
    print(data)

    # Write dataset to output directory
    output_path = f'QCLM-Bench/data/{NUMBER_OF_QA_PAIRS}-QA-pairs-{date}'
    data.to_csv(f"{output_path}.csv")
    print("Dataset saved")


if __name__ == "__main__":
    main()
