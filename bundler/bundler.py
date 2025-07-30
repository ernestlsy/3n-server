import os
from mediapipe.tasks.python.genai import bundler

START_TOKEN="<bos>"
STOP_TOKENS=["<eos>", "<end_of_turn>"]

class Bundler:
    def __init__(
        self,
        tflite_model_path,
        tokenizer_path,
        output_dir,
        model_name
    ):
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"Tokenizer model not found at path: {tokenizer_path}")
        if not os.path.exists(tflite_model_path):
            raise FileNotFoundError(f"TFLite model not found at path: {tflite_model_path}")

        self.file_name = f"{output_dir}/{model_name}.task"
        self.config = bundler.BundleConfig(
            tflite_model=tflite_model_path,
            tokenizer_model=tokenizer_path,
            start_token=START_TOKEN,
            stop_tokens=STOP_TOKENS,
            output_filename=self.file_name,
            prompt_prefix="<start_of_turn>user\n",
            prompt_suffix="<end_of_turn>\n<start_of_turn>model\n",
        )

    def create(self):
        bundler.create_bundle(self.config)
        return self.file_name