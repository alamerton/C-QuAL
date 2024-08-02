from datetime import datetime

def save_dataset(dataset, directory: str):
    date = datetime.now()
    rows = len(dataset)
    # TODO: add model name to file name
    output_path = f'data/{directory}/{rows}-QA-pairs-{date}'
    dataset.to_csv(f"{output_path}.csv")