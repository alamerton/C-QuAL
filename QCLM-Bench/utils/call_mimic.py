import pandas as pd
import psycopg2
import os
from dotenv import load_dotenv
import time
import datetime

load_dotenv()

database_host = os.environ.get('DATABASE_HOST')
database_username = os.environ.get('DATABASE_USERNAME')
database_password = os.environ.get('DATABASE_PASSWORD')
database_name = os.environ.get('DATABASE_NAME')
database_port = os.environ.get('DATABASE_PORT')

def save_data(results): # Load query results into a dataframe containing subject_ids and notes
    results_df = pd.DataFrame(results, columns=['subject_id', 'note'])
    results_df.to_csv(f'QCLM-Bench/data/mimic-iii-subset.csv')


def call_mimic(): # Return a discharge summary from the MIMIC-III database
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
    query = """
    SELECT subject_id, text FROM mimiciii.noteevents
    ORDER BY row_id ASC LIMIT 1;
    """
    cursor.execute(query)

    results = cursor.fetchone() # previously using fetchall()
    
    # Close the database connection
    cursor.close()
    connection.close()

    # save_data(results)
    return results[1] # returns discharge summary only

def get_summaries_for_generation(num_rows):
    start_time = time.time()
    d_summaries_df = pd.DataFrame()
    
    for i in num_rows:
        d_summary = call_mimic()
        d_summaries_df.add(d_summary)

        print(f"{i}/{num_rows}")
        print("** %s seconds" % round(time.time() - start_time, 1))

    d_summaries_df.to_json(f'QCLM-Bench/data/{num_rows}-discharge-summaries-{datetime.now}')
