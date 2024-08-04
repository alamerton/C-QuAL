import pandas as pd
import psycopg2
import os
from dotenv import load_dotenv
from datetime import datetime
import json
import re

load_dotenv()

database_host = os.environ.get("DATABASE_HOST")
database_username = os.environ.get("DATABASE_USERNAME")
database_password = os.environ.get("DATABASE_PASSWORD")
database_name = os.environ.get("DATABASE_NAME")
database_port = os.environ.get("DATABASE_PORT")

# Flag
SUMMARIES_DESTINATION = "function"


# Load query results into a dataframe containing subject_ids and notes
def save_data(results):
    results_df = pd.DataFrame(results, columns=["subject_id", "note"])
    results_df.to_csv(f"C-QuAL/data/mimic-iii-subset.csv")


def reduce_discharge_summary(discharge_summary):
    # Remove whitespace
    discharge_summary = re.sub(r'\s', ' ', discharge_summary).strip()
    # Remove special characters and punctuation
    discharge_summary = re.sub(r'[^\w\s]', '', discharge_summary)
    return discharge_summary

# Add a start and end tag to each discharge summary for the LLM to 
# identify and concatenate them.
def prepare_discharge_summaries(discharge_summaries):
    multiple_summaries = ""
    for i in range(0, len(discharge_summaries)):
        start_string = f"[Discharge summary {i + 1} start]\n"
        end_string = f"\n[Discharge summary {i + 1} end]\n"
        discharge_summary = discharge_summaries[i]
        discharge_summary = reduce_discharge_summary(discharge_summary)
        multiple_summaries += start_string + discharge_summary + end_string
    return multiple_summaries


# Return a discharge summary from the MIMIC-III database
def call_mimic_iii(num_rows, max_summaries):
    # Get date and time data for outputs
    current_date = datetime.now()

    # Set up DB connection
    connection = psycopg2.connect(
        host=database_host,
        user=database_username,
        password=database_password,
        database=database_name,
        port=database_port,
    )
    cursor = connection.cursor()

    discharge_summaries = []

    # Define and execute SQL Query
    if max_summaries > 1:
        query = f"""
        SELECT subject_id, text FROM mimiciii.noteevents
        ORDER BY row_id ASC LIMIT {num_rows};
        """
    else:
        query = f"""
        SELECT text FROM mimiciii.noteevents
        ORDER BY row_id ASC LIMIT {num_rows};
        """
    cursor.execute(query)

    # For each discharge summary retrieved, either parse discharge 
    # summaries straight into an array.                 

    for i in range(num_rows):
        # If generating multiple summaries, retrive patient id as well.
        if max_summaries > 1:
            row = cursor.fetchone()
            if row is None:
                break
            subject_id, discharge_summary = row


            multiple_summaries = [discharge_summary]
            single_summary = discharge_summary

            while len(multiple_summaries) < max_summaries:
                # Check the subject_id of the next discharge summary
                next_row = cursor.fetchone()
                if next_row is None or next_row[0] != subject_id:
                    if next_row:
                        cursor.execute(
                            "SELECT * FROM (SELECT 1) LIMIT 0")
                    break
                multiple_summaries.append(next_row[1])
                if max_summaries is not None and \
                len(multiple_summaries) >= max_summaries:
                    break
            if len(multiple_summaries) > 1:
                combined_summaries = prepare_discharge_summaries(
                    multiple_summaries)
                discharge_summaries.append(combined_summaries)
            else:
                discharge_summaries.append(single_summary)
        else:
            row = cursor.fetchone()
            discharge_summary = reduce_discharge_summary(row)
            discharge_summaries.append(discharge_summary)

    # Close the database connection
    cursor.close()
    connection.close()

    if SUMMARIES_DESTINATION == "file":  # Save discharge summaries to file
        with open(
            f"C-QuAL/data/{num_rows}-discharge-summaries-{current_date}.json", "w"
        ) as f:
            json.dump(discharge_summaries, f)

    elif (
        SUMMARIES_DESTINATION == "function"
    ):  # Send discharge summaries to calling function
        return discharge_summaries

    else:
        raise ValueError("Destination value must be either 'file' or 'function'")
