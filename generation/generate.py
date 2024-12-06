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
from utils.generation.check_quality_with_gpt import check_quality_with_gpt

# Dataset size
NUMBER_OF_QA_PAIRS: int = 15

# Control the ratio of reasoning and planning questions in the dataset
# by setting the proportion of reasoning questions. They can be any
# ratio.
REASONING_Q_PROPORTION: int = 50
PLANNING_Q_PROPORTION: int = 50

# Variable for starting the generation from a specific row in MIMIC-III.
# Default value is 0. Set to 0 if generating new dataset.
CHECKPOINT: int = 0

# Model for generating QA pairs
QA_GENERATION_MODEL = "gpt-35-turbo-16k"

# Model for quality-checking QA pairs
QUALITY_CHECKING_MODEL = QA_GENERATION_MODEL

# Variable for limiting the number of consecutive summaries added to the
# prompt (when multiple consecutive summaries belong to same patient).
# TODO: what is the optimal setting for this number? In the clinical setting,
# how many summaries do clinicians actually look through?
MAX_SUMMARIES: int = 3


def main():
    # create dataframe with question and expected answer columns
    data = pd.DataFrame(
        columns=[
            "Evidence",
            "Question",
            "Expected Answer",
            "Capability",
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
        data_item = []
        discharge_summary = discharge_summaries[row]

        capability_type = select_capability_type(
            REASONING_Q_PROPORTION, PLANNING_Q_PROPORTION
        )

        quality_checking_result = ""
        while "1" not in quality_checking_result:

            # Call LLM with discharge summary and prompt
            qa_string = call_gpt(
                QA_GENERATION_MODEL, discharge_summary, capability_type
            )

            print("QA String: ", qa_string)

            # Check correct columns are in response, regenerate until true
            if capability_type == "planning":
                while "Question" not in qa_string or "Answer" not in qa_string:
                    qa_string = call_gpt(
                        QA_GENERATION_MODEL, discharge_summary, capability_type
                    )
            else:
                while (
                    "Question" not in qa_string
                    or "Answer" not in qa_string
                    or "Evidence" not in qa_string
                ):
                    qa_string = call_gpt(
                        QA_GENERATION_MODEL, discharge_summary, capability_type
                    )

            quality_checking_result = check_quality_with_gpt(
                qa_string, QUALITY_CHECKING_MODEL, capability_type
            )
            # If the quality checking function returns a string that is
            # not either '0' or '1', retry until it is
            while (
                "1" not in quality_checking_result
                and "0" not in quality_checking_result
            ):
                quality_checking_result = check_quality_with_gpt(
                    qa_string, QUALITY_CHECKING_MODEL, capability_type
                )
                print("Quality checking result: ", quality_checking_result)

        # Parse the json to get the question and answer as variables
        qa_parts = qa_string.split("\n")
        qa_parts = [item for item in qa_parts if item != ""]  # Remove
        # items created by extra '\n's
        print(qa_parts)  # Log the data to terminal
        question = qa_parts[0][10:]  # Remove "Question: "
        answer = qa_parts[1][8:]  # Remove "Answer: "
        if capability_type == "reasoning":
            evidence = qa_parts[2][10:]  # Remove "Evidence: "
            # Add data to data item
            data_item.extend((evidence, question, answer, capability_type))
        else:
            data_item.extend(("", question, answer, capability_type))
        # Add Q-A pair to dataframe
        data.loc[row] = data_item

        # Output message to terminal
        print(f"{row+1}/{NUMBER_OF_QA_PAIRS}")

        checkpoint_directory_path = "data/generations/checkpoints/"
        if (row + 1) % 3 == 0:
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
