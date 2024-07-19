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

NUMBER_OF_QA_PAIRS = 1
INCLUDE_EXPLANATION = True

def main():
    # create dataframe with question and expected answer columns
    data = pd.DataFrame(
        columns=['Discharge Summary', 'Question', 'Expected Answer']
    )

    print("Getting summaries for generation...")
    discharge_summaries = call_mimic(NUMBER_OF_QA_PAIRS)

    # For loop for generating qa pairs
    print("Done\n\nGenerating QA pairs...")
    for row in tqdm(range(NUMBER_OF_QA_PAIRS)):
        date = datetime.now()

        # Create data item starting with discharge summary
        data_item = [discharge_summaries[row]]
        
        # Call LLM with discharge summary and prompt
        qa_string = call_gpt(data_item, INCLUDE_EXPLANATION)
        
        if INCLUDE_EXPLANATION:
            # Add Explanation column
            data['Reason'] = ''

            # Parse the json to get the question and answer as variables
            qa_parts = qa_string.split("\n")
            question = qa_parts[0][10:]  # Remove "Question: "
            answer = qa_parts[1][8:]     # Remove "Answer: "
            explanation = qa_parts[2][8:]  # Remove "Reason: " "Explanation: "

            # Add question and answer as tuple to data item
            data_item.extend((question, answer, explanation))

            # Add Q-A pair to dataframe
            data.loc[row] = data_item

            # Output message to terminal
            print(f"{row+1}/{NUMBER_OF_QA_PAIRS}")
            time.sleep(5)
        else:
            # Parse the json to get the question and answer as variables
            qa_parts = qa_string.split("\nAnswer: ")
            question = qa_parts[0].replace("Question: ", "").strip()
            answer = qa_parts[1].strip()

            # Add question and answer as tuple to data item
            data_item.extend((question, answer))

            # Add Q-A pair to dataframe
            data.loc[row] = data_item

            # Output message to terminal
            print(f"{row+1}/{NUMBER_OF_QA_PAIRS}")
            time.sleep(5)
    
    print("Complete")
    print(data)

    # Write dataset to output directory
    output_path = f'data/{NUMBER_OF_QA_PAIRS}-QA-pairs-{date}'
    data.to_csv(f"{output_path}.csv")
    print("Dataset saved")

    #TODO: Or save it to Hugging Face

if __name__ == "__main__":
    main()
