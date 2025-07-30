# =======================================================================================
# Designed for preprocessing of dataset rows in with the following format:
# 1st column - input_text: text passage to be summarized
# Remaining columns - fields: fields expected to be extracted from the input text
# =======================================================================================

import json

class Preprocessor:
    def __init__(
        self,
        module_name
    ):
        self.module_name = module_name

    def preprocess(self, example):
        field_names, num_fields = self.extract_names(example)
        json_string = self.extract_fields_to_json(example)
        prompt = (
            f"<start_of_turn>user\n"
            f"Summarize this {self.module_name} into these {num_fields} fields: {field_names} "
            f"Only output the values of the {num_fields} fields, as 1 json object. "
            f"(start) {example['input_text']} (end)"
            f"<end_of_turn>\n"
            f"<start_of_turn>assistant\n{json_string}<end_of_turn>\n" 
        )
        return prompt

    def extract_fields_to_json(self, row):
        """
        Converts all columns in a row except 'input_text' into a JSON string.
        
        Args:
            row (dict): A dictionary representing one row of data.
        
        Returns:
            str: A JSON string of the row excluding 'input_text'.
        """
        filtered_row = {k: v for k, v in row.items() if k != "input_text"}
        return json.dumps(filtered_row, ensure_ascii=False, indent=2)

    def extract_names(self, row):
        """
        Extract the headers of all columns in a row except 'input_text' into a string, alongside the number of columns.
        
        Args:
            row (dict): A dictionary representing one row of data.
        
        Returns:
            str: A string consisting headers of the columns excluding "input_text".
            int: The number of columns excluding "input_text".
        """
        field_names = ""
        count = 0
        for k, v in row.items():
            if k != "input_text":
                field_names +=  k + ", "
                count += 1
        field_names = field_names[:-2]
        return field_names, count