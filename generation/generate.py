from datetime import datetime
import sys
import os
import time
from tqdm import tqdm
import pandas as pd

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
from utils.call_gpt import call_gpt
from utils.call_mimic import get_summaries_for_generation

NUMBER_OF_QA_PAIRS = 10
SUMMARIES_DESTINATION = "function"

def main():
    # create dataframe with question and expected answer columns
    # data = pd.DataFrame(columns=['Question', 'Correct Answer'])
    data = pd.DataFrame(
        columns=['Discharge Summary', 'Question', 'Expected Answer']
    )

    print("Getting summaries for generation...")
    discharge_summaries = get_summaries_for_generation(
        NUMBER_OF_QA_PAIRS, SUMMARIES_DESTINATION
    )

    # For loop for generating x qa pairs
    print("Generating QA pairs...")
    for row in tqdm(range(NUMBER_OF_QA_PAIRS)):
        date = datetime.now()
        
        data_item = [discharge_summaries[row]]

        # Call LLM with discharge summary and prompt
        question_answer_string = call_gpt(
            discharge_summaries[row]
        )

        # Parse the json to get the question and answer as variables
        data_item.extend(question_answer_string.split('\n'))

        # Add Q-A pair to dataframe
        data.loc[row] = data_item

        print(f"{row+1}/{NUMBER_OF_QA_PAIRS}")
        # time.sleep(10) 
    
    print("Complete")
    print(data)

    # Write dataset to output directory
    output_path = f'data/{NUMBER_OF_QA_PAIRS}-QA-pairs-{date}'
    data.to_csv(f"{output_path}.csv")
    print("Dataset saved")


if __name__ == "__main__":
    main()
