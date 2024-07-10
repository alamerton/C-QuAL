import pandas as pd
import psycopg2
import os

database_host = os.environ.get('DATABASE_HOST')
database_username = os.environ.get('DATABASE_USERNAME')
database_password = os.environ.get('DATABASE_PASSWORD')
database_name = os.environ.get('DATABASE_NAME')
database_port = os.environ.get('DATABASE_PORT')

def call_mimic():
    
    # Set up DB connection
    connection = psycopg2.connect(
        host=database_host,
        user=database_username,
        password=database_password,
        database=database_name,
        port=database_port
    )

    cursor = connection.cursor()

    # Define SQL Query
    patient_id = 123

    query = """
    SELECT subject_id, note
    FROM mimiciii.noteevents
    WHERE itemid IN (39, 40, 41) AND category = 'DISCHARGE SUMMARY' AND subject_id = 123
    ORDER BY charttime DESC;
    """

    # Query arbitrary patient
    cursor.execute(query)

    # Load query results into a dataframe containing subject_ids and notes
    results = cursor.fetchall()
    results_df = pd.DataFrame(results, columns=['subject_id', 'note'])

    # Close the database connection
    cursor.close()
    connection.close()

    # Do data things
    return results_df
