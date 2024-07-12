import pandas as pd
import psycopg2
import os
from dotenv import load_dotenv
import time
from datetime import datetime
import json

load_dotenv()

database_host = os.environ.get('DATABASE_HOST')
database_username = os.environ.get('DATABASE_USERNAME')
database_password = os.environ.get('DATABASE_PASSWORD')
database_name = os.environ.get('DATABASE_NAME')
database_port = os.environ.get('DATABASE_PORT')

# Flag 
SUMMARIES_DESTINATION = "function"

def save_data(results): # Load query results into a dataframe containing subject_ids and notes
    results_df = pd.DataFrame(results, columns=['subject_id', 'note'])
    results_df.to_csv(f'QCLM-Bench/data/mimic-iii-subset.csv')


def call_mimic(num_rows): # Return a discharge summary from the MIMIC-III database
    # Get date and time data for outputs
    current_date = datetime.now()
    start_time = time.time()

    # Set up DB connection
    connection = psycopg2.connect(
        host=database_host,
        user=database_username,
        password=database_password,
        database=database_name,
        port=database_port
    )
    cursor = connection.cursor()

    # Define and execute SQL Query
    query = f"""
    SELECT text FROM mimiciii.noteevents
    ORDER BY row_id ASC LIMIT {num_rows};
    """
    
    cursor.execute(query)

    # discharge_summaries = cursor.fetchall()[0] #TODO: this may work

    discharge_summaries = []

    for i in range(num_rows):
        discharge_summary = cursor.fetchone()
        discharge_summaries.append(discharge_summary)

        print(f"{i+1}/{num_rows} ** %s seconds **" % round(time.time() - start_time, 1))

    # Close the database connection
    cursor.close()
    connection.close()

    if SUMMARIES_DESTINATION == 'file': # Save discharge summaries to file
        with open(f'QCLM-Bench/data/{num_rows}-discharge-summaries-{current_date}.json', 'w') as f:
            json.dump(discharge_summaries, f)
    
    elif SUMMARIES_DESTINATION == 'function': # Send discharge summaries to calling function
        return discharge_summaries
    
    else:
        raise ValueError("Destination value must be either 'file' or 'function'")
