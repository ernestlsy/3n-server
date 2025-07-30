# =======================================================================================
# Designed for preprocessing of dataset rows in with the following format:
# 1st column - input_text: text passage to be summarized
# Remaining columns - fields: fields expected to be extracted from the input text
# =======================================================================================

import json

class Preprocessor:
    def __init__(
        self,
        tokenizer,
        module_name
    ):
        self.tokenizer = tokenizer
        self.module_name = module_name

    def preprocess(self, example):
        field_names, num_fields = self.extract_names(example)
        json_string = self.extract_fields_to_json(example)
        prompt = (
            f"Summarize this {self.module_name} into these {num_fields} fields: {field_names} "
            f"Only output the values of the {num_fields} fields, as 1 json object. "
            f"(start) {example['input_text']} (end)"
        )
        output = json_string  # the expected JSON string

        # Combine prompt and target
        full_input = prompt + "\n" + output

        # Tokenize with return of offset for label masking
        tokenized = self.tokenizer(
            full_input,
            truncation=True,
            padding="max_length",
            max_length=256,
        )

        # Mask the loss for the prompt part
        prompt_tokenized = self.tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=256,
        )

        # Set labels to -100 for prompt tokens (ignore), keep output tokens
        labels = tokenized["input_ids"].copy()
        prompt_len = sum(1 for id in prompt_tokenized["input_ids"] if id != self.tokenizer.pad_token_id)
        labels[:prompt_len] = [-100] * prompt_len
        tokenized["labels"] = labels

        return tokenized

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