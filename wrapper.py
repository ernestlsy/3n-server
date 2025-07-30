import os
import subprocess
import yaml
import shutil
from trainer.train import InitialTrainer, DefaultTrainer
from bundler.bundler import Bundler

def run(dataset_path, job_id):
    ## Load config
    with open("config.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    trainer_cfg = config["trainer"]
    convert_cfg = config["convert_gemma3_to_tflite"]
    bundler_cfg = config["bundler"]
    converter_cfg = config["converter"]

    ## Step 0: Clear temp folders
    epoch_dir = trainer_cfg["epoch_path"]
    if os.path.exists(epoch_dir):
        print(f"Clearing contents of epoch directory: {epoch_dir}")
        for filename in os.listdir(epoch_dir):
            file_path = os.path.join(epoch_dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}: {e}")
    else:
        print(f"Directory does not exist, creating: {epoch_dir}")
        os.makedirs(epoch_dir)

    # Deleting old model weights
    merged_dir = trainer_cfg["output_path"]
    old_tensors_path = os.path.join(merged_dir, "model.safetensors")
    if os.path.exists(old_tensors_path):
        os.unlink(old_tensors_path)
        
    tflite_dir = convert_cfg["output_path"]
    if os.path.exists(tflite_dir):
        print(f"Clearing contents of tflite directory: {tflite_dir}")
        for filename in os.listdir(tflite_dir):
            file_path = os.path.join(tflite_dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}: {e}")
    else:
        print(f"Directory does not exist, creating: {tflite_dir}")
        os.makedirs(tflite_dir)


    ## Step 1: Run trainer

    print("Init trainer...")
    # Check if LoRA adapters already exist
    adapter_path = trainer_cfg["lora_path"] + "/adapter_config.json"
    if os.path.exists(adapter_path):
        trainer = DefaultTrainer(**trainer_cfg, dataset_path=dataset_path)
        print("DefaultTrainer created")
    else:
        trainer = InitialTrainer(**trainer_cfg, dataset_path=dataset_path)
        print("InitalTrainer created")

    print("Running trainer...")
    trainer.train_and_save()

    if os.path.exists(dataset_path):
        print(f"Removing used dataset: {dataset_path}")
        try:
            os.unlink(dataset_path)
        except Exception as e:
            print(f"Failed to delete {dataset_path}: {e}")


    ## Step 2: Run converter

    print("Running converter...")
    converter_cmd = ["python", "./unsloth/llama.cpp/unsloth_convert_hf_to_gguf.py", "--split-max-size", "50G"]
    converter_cmd.append(f"--outfile {converter_cfg["output_path"]}/gemma3.gguf")
    converter_cmd.append(f"--outtype {converter_cfg["outtype"]}")
    converter_cmd.append(converter_cfg["checkpoint_path"])

    subprocess.run(converter_cmd, check=True)

    print("All steps completed.")
    return os.path.abspath(f"{converter_cfg["output_path"]}/gemma3.gguf")

if __name__ == "__main__":
    run("./data/incident_reports_batch_1.csv", 1)