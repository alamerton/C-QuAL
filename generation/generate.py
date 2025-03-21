from datetime import datetime
import json
import sys
import os
from tqdm import tqdm
import pandas as pd
import re

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
FACTUAL_Q_PROPORTION: int = 0
REASONING_Q_PROPORTION: int = 100

# Variable for starting the generation from a specific row in MIMIC-III.
# Default value is 0. Set to 0 if generating new dataset.
CHECKPOINT: int = 0
CHECKPOINT_INTERVAL: int = 1

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
    dataset = []

    print("Getting summaries for generation")
    discharge_summaries = call_mimic_iii(NUMBER_OF_QA_PAIRS, MAX_SUMMARIES)

    print("Done\n\nGenerating Q-A pairs...")

    for row in tqdm(range(CHECKPOINT, NUMBER_OF_QA_PAIRS)):
        date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        discharge_summary = discharge_summaries[row]
        capability_type = select_capability_type(
            FACTUAL_Q_PROPORTION, REASONING_Q_PROPORTION
        )

        if capability_type == "Factual QA":
            # Generate 4 factual questions using the discharge summary
            for _ in range(0, 4):
                data_item = {}

                quality_checking_result = ""
                while "1" not in quality_checking_result:
                    qa_string = call_gpt(
                        QA_GENERATION_MODEL, discharge_summary, capability_type
                    )
                    print("QA String: ", qa_string)

                    while "Part 1: " not in qa_string or "Part 2: " not in qa_string:
                        print("Regenerating...")
                        qa_string = call_gpt(
                            QA_GENERATION_MODEL, discharge_summary, capability_type
                        )

                    quality_checking_result = check_quality_with_gpt(
                        qa_string, QUALITY_CHECKING_MODEL, capability_type
                    )
                    print("Quality checking result: ", quality_checking_result)

                qa_parts = re.split(r"\n*Part [12]:", qa_string)
                qa_parts = [part.strip() for part in qa_parts if part.strip()]

                print(qa_parts)

                question = qa_parts[0]
                answer = qa_parts[1]

                data_item = {
                    "Capability": capability_type,
                    "Evidence": discharge_summary,
                    "Question": question,
                    "Expected Answer": answer,
                }

                dataset.append(data_item)

                print(f"{row+1}/{NUMBER_OF_QA_PAIRS}")

                checkpoint_directory_path = "data/generations/checkpoints/"
                if (row + 1) % CHECKPOINT_INTERVAL == 0:
                    checkpoint_name = f"{row+1}-rows-{date}"
                    checkpoint_path = checkpoint_directory_path + checkpoint_name
                    with open(f"{checkpoint_path}.json", "w") as json_file:
                        json.dump(dataset, json_file, indent=4)
        else:
            # Start generation and quality checking loop for nested properties
            quality_checking_result = ""
            while "1" not in quality_checking_result:
                qa_string = call_gpt(
                    QA_GENERATION_MODEL, discharge_summary, capability_type
                )
                print("QA String: ", qa_string)

                while (
                    "Initial_Presentation: " not in qa_string
                    or "Subsequent_Course: " not in qa_string
                    or "Clinical_Reasoning_Questions: " not in qa_string
                ):
                    print("Regenerating...")
                    qa_string = call_gpt(
                        QA_GENERATION_MODEL, discharge_summary, capability_type
                    )

                quality_checking_result = check_quality_with_gpt(
                    qa_string, QUALITY_CHECKING_MODEL, capability_type
                )
                print("Quality checking result: ", quality_checking_result)

            qa_parts = re.findall(
                r"(Initial_Presentation:|Subsequent_Course:|Clinical_Reasoning_Questions:)\s*(.*?)(?=\s*(?:Initial_Presentation:|Subsequent_Course:|Clinical_Reasoning_Questions:|\Z))",
                qa_string,
                re.DOTALL,
            )

            print("QA PARTS: ", qa_parts)

            result = {label.rstrip(":"): content.strip() for label, content in qa_parts}
            print("result: ", result)

            clinical_reasoning_qa = re.findall(
                r"Q: (.*?)\nA: (.*?)\nReasoning: (.*?)\n",
                result["Clinical_Reasoning_Questions"],
            )

            # TODO: Might be good to add a 'because' {reason} if I want to include the reason

            sections = []
            previous_answer = ""
            for question, answer, reasoning in clinical_reasoning_qa:
                section = {
                    "Evidence": ("" if len(sections) == 0 else previous_answer),
                    "Question": question,
                    "Expected Answer": answer,
                }
                previous_answer += f"\nThe previous clinical step was: {answer}"
                sections.append(section)

            data_item = {
                "Capability": capability_type,
                "Evidence": result["Initial_Presentation"],
                "Sections": sections,
            }

            dataset.append(data_item)

            print(f"{row+1}/{NUMBER_OF_QA_PAIRS}")

            checkpoint_directory_path = "data/generations/checkpoints/"
            if (row + 1) % CHECKPOINT_INTERVAL == 0:
                checkpoint_name = f"{row+1}-rows-{date}"
                checkpoint_path = checkpoint_directory_path + checkpoint_name
                with open(f"{checkpoint_path}.json", "w") as json_file:
                    json.dump(dataset, json_file, indent=4)

    print("Complete")
    print(dataset)

    # Write dataset to output directory
    output_path = f"""data/generations/{NUMBER_OF_QA_PAIRS}-QA-pairs-{date}"""
    with open(f"{output_path}.json", "w") as json_file:
        json.dump(dataset, json_file, indent=4)

    print("Dataset saved")


if __name__ == "__main__":
    main()
