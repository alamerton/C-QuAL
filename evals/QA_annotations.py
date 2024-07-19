"""
This file should load a QA dataset, and call an LLM with each row, 
prompting it to annotate the quality of the Q-A pair as if it is a 
clinician, saving the annotation in a new row, and perform the 
suggested revision on the row.

A copy of the annotated and edited dataset should then be saved, and 
a copy of the resulting dataset should have the annotations removed, 
and be saved as a finished QA dataset.
"""