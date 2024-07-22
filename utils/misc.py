from datetime import datetime

def save_dataset(dataset, local: bool):
    date = datetime.now()
    rows = len(dataset)
    # TODO: add model name to file name
    output_path = f'data/annotations/{rows}-QA-pairs-{date}'
    dataset.to_csv(f"{output_path}.csv")