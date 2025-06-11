import json

# Read the file
with open('llama_dialog_dataset_final_cleaned_2.txt', 'r') as f:
    lines = f.readlines()

# Write to a new JSONL file
with open('llama_dialog_dataset_final_cleaned_2.jsonl', 'w') as f:
    for line in lines:
        f.write(line)