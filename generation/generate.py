from datetime import datetime
import sys
import os
from tqdm import tqdm
import pandas as pd

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)
from utils.generation.call_gpt import call_gpt, call_gpt_with_multiple_discharge_summaries
from utils.generation.call_mimic_iii import call_mimic_iii

# SAVE_TO_HF = False #TODO
NUMBER_OF_QA_PAIRS: int = 1000
INCLUDE_EXPLANATION: bool = False

# Variable for starting the generation from a specific row in MIMIC-III.
# Default value is 0. Set to 0 if generating new dataset.
CHECKPOINT: int = 0 
MAX_SUMMARIES: int = 3

def main():
    # create dataframe with question and expected answer columns
    if MAX_SUMMARIES > 1:
        discharge_summary_string = "Discharge Summaries"
    else:
        discharge_summary_string = "Discharge Summary"
    
    if INCLUDE_EXPLANATION:
        data = pd.DataFrame(
            columns=[
                discharge_summary_string,
                "Question",
                "Expected Answer",
                "Reason",
                "Question Type",
            ]
        )
    else:
        data = pd.DataFrame(
            columns=[
                discharge_summary_string,
                "Question",
                "Expected Answer",
                "Question Type",
            ]
        )

    print("Getting summaries for generation")

    discharge_summaries = call_mimic_iii(
        NUMBER_OF_QA_PAIRS,
        MAX_SUMMARIES
    )
    print("This is length! ", len(discharge_summaries))

    # For loop for generating qa pairs
    print("Done\n\nGenerating Q-A pairs...")

    for row in tqdm(range(CHECKPOINT, NUMBER_OF_QA_PAIRS)):
        date = datetime.now()
        date = date.strftime("%Y-%m-%d %H:%M:%S")

        # Create data item starting with discharge summary
        data_item = [discharge_summaries[row]]
        print("This is length: ", len(discharge_summaries))
        # print("This is it! ", data_item)

        # Call LLM with discharge summary and prompt
        # qa_string = call_gpt(data_items, INCLUDE_EXPLANATION)
        qa_string = call_gpt_with_multiple_discharge_summaries(
            data_item,
            INCLUDE_EXPLANATION
        )

        # Check correct columns are in response, regenerate until true
        while "Question" not in qa_string \
              or "Answer" not in qa_string \
              or "Type" not in qa_string:
            qa_string = call_gpt(data_item, INCLUDE_EXPLANATION)

        # Parse the json to get the question and answer as variables
        qa_parts = qa_string.split("\n")
        print(qa_parts)
        question = qa_parts[0][10:]  # Remove "Question: "
        answer = qa_parts[1][8:]  # Remove "Answer: "
        question_type = qa_parts[2][6:]  # Remove "Type: "

        if INCLUDE_EXPLANATION:
            explanation = qa_parts[3][8:]  # Remove "Reason: "
            # Add question and answer as tuple to data item
            data_item.extend((question, answer, explanation, question_type))

        data_item.extend((question, answer, question_type))
        # Add Q-A pair to dataframe
        data.loc[row] = data_item

        # Output message to terminal
        print(f"{row+1}/{NUMBER_OF_QA_PAIRS}")
        # time.sleep(5)

        checkpoint_directory_path = "data/generations/checkpoints/"
        if (row + 1) % 10 == 0:
            if CHECKPOINT > 0:
                checkpoint_name = f"{date}-rows-{CHECKPOINT}-{row+1}"
                checkpoint_path = checkpoint_directory_path + checkpoint_name
            else:
                checkpoint_name = f"{date}-{row+1}-rows"
                checkpoint_path = checkpoint_directory_path + checkpoint_name
            data.to_csv(f"{checkpoint_path}.csv")

    print("Complete")
    print(data)

    # Write dataset to output directory
    output_path = f"""data/generations/
    {NUMBER_OF_QA_PAIRS}-QA-pairs-{date}"""
    data.to_csv(f"{output_path}.csv")
    print("Dataset saved")

    # TODO: Or save it to Hugging Face


if __name__ == "__main__":
    main()
