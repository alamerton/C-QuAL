from datetime import datetime
import sys
import os
from tqdm import tqdm
import pandas as pd

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)
from utils.generation.call_gpt import call_gpt
from utils.generation.call_mimic_iii import call_mimic_iii
from utils.misc import select_capability_type

# Dataset size
NUMBER_OF_QA_PAIRS: int = 1500

# Control the ratio of reasoning and planning questions in the dataset
# by setting the proportion of reasoning questions. They can be any
# ratio.
REASONING_Q_PROPORTION: int = 50
PLANNING_Q_PROPORTION: int = 50

# Variable for starting the generation from a specific row in MIMIC-III.
# Default value is 0. Set to 0 if generating new dataset.
CHECKPOINT: int = 0

# Model for generating QA pairs
QA_GENERATION_MODEL = "gpt-4o"

# Variable for limiting the number of consecutive summaries added to the
# prompt (when multiple consecutive summaries belong to same patient).
# TODO: what is the optimal setting for this number? In the clinical setting,
# how many summaries do clinicians actually look through?
MAX_SUMMARIES: int = 3


def main():
    # create dataframe with question and expected answer columns
    if MAX_SUMMARIES > 1:
        discharge_summary_string = "Discharge Summaries"
    else:
        discharge_summary_string = "Discharge Summary"

    data = pd.DataFrame(
        columns=[
            discharge_summary_string,
            "Question",
            "Expected Answer",
            "Question Type",
        ]
    )

    print("Getting summaries for generation")

    discharge_summaries = call_mimic_iii(NUMBER_OF_QA_PAIRS, MAX_SUMMARIES)

    # For loop for generating qa pairs
    print("Done\n\nGenerating Q-A pairs...")

    for row in tqdm(range(CHECKPOINT, NUMBER_OF_QA_PAIRS)):
        # Get date for naming the dataset and checkpoints
        date = datetime.now()
        date = date.strftime("%Y-%m-%d %H:%M:%S")

        # Create data item starting with discharge summary
        data_item = [discharge_summaries[row]]

        capability_type = select_capability_type(
            REASONING_Q_PROPORTION, PLANNING_Q_PROPORTION
        )

        # Call LLM with discharge summary and prompt
        qa_string = call_gpt(QA_GENERATION_MODEL, data_item, capability_type)

        # Check correct columns are in response, regenerate until true
        while (
            "Question" not in qa_string
            or "Answer" not in qa_string
            or "Type" not in qa_string
        ):
            qa_string = call_gpt(QA_GENERATION_MODEL, data_item)

        # Parse the json to get the question and answer as variables
        qa_parts = qa_string.split("\n")
        qa_parts = [item for item in qa_parts if item != ""]  # Remove
        # items created by extra '\n's
        print(qa_parts)  # Log the data to terminal
        question = qa_parts[0][10:]  # Remove "Question: "
        answer = qa_parts[1][8:]  # Remove "Answer: "
        question_type = qa_parts[2][6:]  # Remove "Type: "

        # Add data to data item
        data_item.extend((question, answer, question_type))

        # Add Q-A pair to dataframe
        data.loc[row] = data_item

        # Output message to terminal
        print(f"{row+1}/{NUMBER_OF_QA_PAIRS}")

        checkpoint_directory_path = "data/generations/checkpoints/"
        if (row + 1) % 10 == 0:
            if CHECKPOINT > 0:
                checkpoint_name = f"rows-{CHECKPOINT}-{row+1}-{date}"
                checkpoint_path = checkpoint_directory_path + checkpoint_name
            else:
                checkpoint_name = f"{row+1}-rows-{date}"
                checkpoint_path = checkpoint_directory_path + checkpoint_name
            data.to_csv(f"{checkpoint_path}.csv")

    print("Complete")
    print(data)

    # Write dataset to output directory
    output_path = f"""data/generations/
    {NUMBER_OF_QA_PAIRS}-QA-pairs-{date}"""
    data.to_csv(f"{output_path}.csv")
    print("Dataset saved")


if __name__ == "__main__":
    main()
