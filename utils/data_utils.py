import os
import csv

THRESHOLD = 250  # Minimum number of rows in dataset to start training procedure

def validate_csv(filepath):
    with open(filepath, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        try:
            header = next(reader)
        except StopIteration:
            return False, "The CSV file is empty."

        if not header or header[0] != "input_text":
            return False, "The first column name is not 'input_text'."

        expected_columns = len(header)
        for i, row in enumerate(reader, start=2):
            if len(row) != expected_columns:
                return False, f"Row {i} has {len(row)} columns, expected {expected_columns}."

    return True, "CSV file is valid."

def validate_feedback(data: dict):
    if 'module' not in data:
        return False
    if 'input_text' not in data:
        return False
    return True

def write_to_csv(data: dict):
    module_name = data['module']
    del data['module']
    target_csv = f"data/{module_name}.csv"

    file_exists = os.path.isfile(target_csv)

    with open(target_csv, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=data.keys())

        if not file_exists:
            writer.writeheader()  # write headers if file didn't exist

        writer.writerow(data)
    
    if count_rows(target_csv) >= THRESHOLD:
        return True, target_csv
    else:
        return False, None

def count_rows(filename):
    with open(filename, newline='') as file:
        reader = csv.reader(file)
        next(reader, None)  # Skip header
        return sum(1 for row in reader)