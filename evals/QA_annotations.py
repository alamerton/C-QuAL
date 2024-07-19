"""
This file should load a QA dataset, and call an LLM with each row, 
prompting it to annotate the quality of the Q-A pair as if it is a 
clinician, saving the annotation in a new row, and perform the 
suggested revision on the row.

A copy of the annotated and edited dataset should then be saved, and 
a copy of the resulting dataset should have the annotations removed, 
and be saved as a finished QA dataset.

    1. Load the QA dataset, and create a new column called 
    '[model_name]_annotation'
    2. For every row in the dataset:
        2.1. Call the LLM with the row and prompt
        2.2. Save the annotation to the annotation column
    3. Save the dataset somewhere e.g. locally or HF

"""
