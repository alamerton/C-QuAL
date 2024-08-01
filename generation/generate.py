from datetime import datetime
import sys, os, time
from tqdm import tqdm
import pandas as pd

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
from utils.generation.call_gpt import call_gpt
from utils.generation.call_mimic import call_mimic

# SAVE_TO_HF_NOT_LOCALLY = False #TODO
NUMBER_OF_QA_PAIRS = 3

def main():
    # create dataframe with question and expected answer columns
    data = pd.DataFrame(
        columns=['Discharge Summary', 'Question', 'Expected Answer', 
                 'Reason', 'Question Type']
    )

    print("Getting summaries for generation")
    
    discharge_summaries = call_mimic(NUMBER_OF_QA_PAIRS)

    # For loop for generating qa pairs
    print("Done\n\nGenerating Q-A pairs...")
    for row in tqdm(range(NUMBER_OF_QA_PAIRS)):
        date = datetime.now()

        # Create data item starting with discharge summary
        data_item = [discharge_summaries[row]]
        
        # Call LLM with discharge summary and prompt
        qa_string = call_gpt(data_item)

        # Parse the json to get the question and answer as variables
        qa_parts = qa_string.split("\n")
        print(qa_parts)
        question = qa_parts[0][10:]  # Remove "Question: "
        answer = qa_parts[1][8:]     # Remove "Answer: "
        question_type = qa_parts[2][6:] # Remove "Type: "
        explanation = qa_parts[3][8:]  # Remove "Reason: "

        # Add question and answer as tuple to data item
        data_item.extend((
            question,
            answer,
            explanation,
            question_type
            ))

        # Add Q-A pair to dataframe
        data.loc[row] = data_item

        # Output message to terminal
        print(f"{row+1}/{NUMBER_OF_QA_PAIRS}")
        time.sleep(5)
    
    print("Complete")
    print(data)

    # Write dataset to output directory
    output_path = f'data/generations/{NUMBER_OF_QA_PAIRS}-QA-pairs-{date}'
    data.to_csv(f"{output_path}.csv")
    print("Dataset saved")

    #TODO: Or save it to Hugging Face

if __name__ == "__main__":
    main()
