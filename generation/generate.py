from datetime import datetime
import sys
import os
import time
from tqdm import tqdm
import pandas as pd

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
from utils.call_gpt import call_gpt
from utils.call_mimic import call_mimic

NUMBER_OF_QA_PAIRS = 10

def main():
    # create dataframe with question and expected answer columns
    data = pd.DataFrame(
        columns=['Discharge Summary', 'Question', 'Expected Answer']
    )

    print("Getting summaries for generation...")
    discharge_summaries = call_mimic(NUMBER_OF_QA_PAIRS)

    # For loop for generating x qa pairs
    print("Done\n\nGenerating QA pairs...")
    for row in tqdm(range(NUMBER_OF_QA_PAIRS)):
        date = datetime.now()
        
        data_item = [discharge_summaries[row]]
        print("length of data item:", len(data_item))
        # Call LLM with discharge summary and prompt
        qa_string = call_gpt(data_item)
        
        # Parse the json to get the question and answer as variables

        #TODO: this is not robust, need to split on answer
        # data_item.extend(qa_string.split('\n'))  
        data_item.extend(qa_string.split('Answer'))  

        # Add Q-A pair to dataframe
        data.loc[row] = data_item

        print(f"{row+1}/{NUMBER_OF_QA_PAIRS}")
        time.sleep(5)
    
    print("Complete")
    print(data)

    # Write dataset to output directory
    output_path = f'data/{NUMBER_OF_QA_PAIRS}-QA-pairs-{date}'
    data.to_csv(f"{output_path}.csv")
    print("Dataset saved")


if __name__ == "__main__":
    main()
