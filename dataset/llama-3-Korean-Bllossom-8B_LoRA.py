# -*- coding: utf-8 -*-

# nohup python <this_script_name>.py & 으로 실행시켜야 함
import json
import os
print(f"Current Working Directory: {os.getcwd()}")
file_path = '~/dataset/llama_dialog_dataset_final_cleaned_2.jsonl'
# Check if file exists
if not os.path.exists(file_path):
    print(f"Error: File not found at {file_path}")
    raise FileNotFoundError(f"Dataset file not found at: {file_path}")
else:
    print(f"Dataset file found at: {file_path}")
    # Optional: Read a few lines to verify content (within try-except)
    try:
      count = 0
      with open(file_path, 'r', encoding='utf-8') as f: # Added encoding='utf-8'
        for line in f:
          data = json.loads(line)
          # print(data) # Uncomment to print sample data
          count += 1
          if count >= 5: # Print first 5 lines
              break
      print(f"Successfully read first {count} lines.")

    except json.JSONDecodeError:
      print(f"Error: Invalid JSON format in {file_path}")
    except Exception as e:
      print(f"An unexpected error occurred while reading the file: {e}")

# --- 라이브러리 설치 ---
# pip install torch transformers==4.36.2 peft==0.7.1 accelerate==0.25.0 datasets==2.16.1 trl==0.7.4

from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments # Removed BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType # Removed prepare_model_for_kbit_training
from datasets import load_dataset
import torch
from trl import SFTTrainer
import warnings
from accelerate import Accelerator

# Suppress UserWarnings (optional)
warnings.filterwarnings("ignore", category=UserWarning)

# --- Model and Tokenizer Loading ---
model_id = "MLP-KTLim/llama-3-Korean-Bllossom-8B"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
# Set padding side to the right for training consistency
tokenizer.padding_side = 'right'
# Add pad token if missing (Llama models often don't have one by default)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print("Set EOS token as PAD token.")

# tokenizer 로드 후
print(f"Loaded tokenizer object: {tokenizer}")
print(f"Type of loaded tokenizer: {type(tokenizer)}")

# --- Load model, using float16 ---
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,      # Use float16 (or torch.bfloat16 for A100 potential benefits)
    trust_remote_code=True
)
print(type(model.model.layers[0]))

# --- LoRA Configuration ---
# Consider increasing rank 'r' for standard LoRA if resources allow
lora_config = LoraConfig(
    r=32,                     # Increased LoRA rank (e.g., 16, 32, 64) - Requires more VRAM
    lora_alpha=64,            # Alpha scaling (often 2*r)
    target_modules=[          # Target modules for Llama-3 (check model architecture if needed)
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj"
    ],
    lora_dropout=0.05,        # Dropout for LoRA layers
    bias="none",              # Usually set to "none" for LoRA
    task_type=TaskType.CAUSAL_LM
)

# Apply LoRA config to the model
model = get_peft_model(model, lora_config)
model.print_trainable_parameters() # Show number of trainable parameters (will be higher than QLoRA)

# --- Dataset Loading and Preprocessing ---
# 이제 로컬 경로(file_path)를 사용합니다.
dataset = load_dataset("json", data_files=file_path, split="train") # Directly specify split

# Optional: Shuffle the dataset before splitting
dataset = dataset.shuffle(seed=42)

# Split dataset (80% train, 20% test)
split_dataset = dataset.train_test_split(test_size=0.2, seed=42) # Use seed for reproducibility
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

print(f"Training dataset size: {len(train_dataset)}")
print(f"Evaluation dataset size: {len(eval_dataset)}")

# --- Formatting Function (Crucial for Instruction Tuning) ---
def format_instruction(sample):
    instruction = sample['instruction']
    input_text = sample.get('input', '') # Use .get for safety if 'input' might be missing
    output_text = sample['output']

    if input_text and input_text.strip():
        prompt = f"<|user|>\n{instruction}\n{input_text}<|endoftext|><|assistant|>\n{output_text}<|endoftext|>"
    else:
        prompt = f"<|user|>\n{instruction}<|endoftext|><|assistant|>\n{output_text}<|endoftext|>"

    # Add EOS token at the end for Causal LM training
    return {"text": prompt } # + tokenizer.eos_token}

def format_instruction_with_tokenizer(sample, tokenizer):
    instruction = sample['instruction']
    input_text = sample.get('input', '')  # .get()으로 안전하게 키 접근
    output_text = sample['output']

    user_content = instruction
    if input_text and input_text.strip():
        user_content += f"\n{input_text}" # instruction과 input을 합쳐 user의 발화로 구성

    # tokenizer.apply_chat_template에 맞는 메시지 리스트 형식 생성
    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": output_text}
    ]

    # 채팅 템플릿 적용
    # tokenize=False: SFTTrainer가 나중에 일괄적으로 토큰화를 수행합니다.
    # add_generation_prompt=False: 학습 데이터는 이미 완성된 응답을 포함하므로,
    #                              모델이 응답을 생성하도록 하는 프롬프트를 추가하지 않습니다.
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    # 만약 MLP-KTLim/llama-3-Korean-Bllossom-8B 토크나이저에 chat_template이 없어서 위 코드가 에러를 발생시킨다면,
    # 해당 모델 제작자가 제공하는 정확한 문자열 포맷팅 가이드에 따라 수동으로 문자열을 구성해야 합니다.
    # (예: formatted_prompt = f"{tokenizer.bos_token}<|user|>\n{user_content}{tokenizer.eos_token}...")
    # 하지만, Llama 3 계열 모델이라면 apply_chat_template이 동작하는 것이 일반적입니다.

    return {"text": formatted_prompt}

# Apply formatting
# formatted_train_dataset = train_dataset.map(format_instruction_with_tokenizer, remove_columns=list(train_dataset.features))
# formatted_eval_dataset = eval_dataset.map(format_instruction_with_tokenizer, remove_columns=list(eval_dataset.features))
formatted_train_dataset = train_dataset.map(
    lambda sample: format_instruction_with_tokenizer(sample, tokenizer=tokenizer),
    remove_columns=list(train_dataset.features)
)
formatted_eval_dataset = eval_dataset.map(
    lambda sample: format_instruction_with_tokenizer(sample, tokenizer=tokenizer),
    remove_columns=list(eval_dataset.features)
)

# --- Training Arguments ---
output_dir = "./lora-llama-3-korean-results_2" # Changed output directory name
training_args = TrainingArguments(
    output_dir=output_dir,
    # --- Increased batch size (adjust based on your 8x A100 80GB memory) ---
    per_device_train_batch_size=4,        # Increased from 1 (e.g., 4, 8, or 16 - Monitor VRAM)
    gradient_accumulation_steps=8,       # Adjust to maintain or increase global batch size (e.g., 4 * 8 * 8_gpus = 256 global batch)
    # --- Changed optimizer ---
    # optim="adamw_torch",                  # Standard AdamW optimizer
    learning_rate=2e-5,                   # Common learning rate for LoRA (might need tuning: e.g., 5e-5, 2e-5)
    num_train_epochs=3,                   # Number of epochs (필요시 조절)
    bf16=True,                          # Alternatively, use bfloat16 if model loaded with torch.bfloat16
    logging_steps=100,                    # Log every 100 steps
    warmup_steps=100,
    save_strategy="steps",                # Save checkpoints based on steps
    save_steps=1000,                      # Save every 1000 steps
    evaluation_strategy="steps",          # Evaluate based on steps
    eval_steps=500,                       # Evaluate every 500 steps
    save_total_limit=2,                   # Keep only the last 2 checkpoints
    load_best_model_at_end=True,          # Load the best model found during evaluation at the end
    metric_for_best_model="eval_loss",    # Use eval loss to determine the best model
    greater_is_better=False,              # Lower eval loss is better
    report_to="none",                     # Disable reporting to wandb/tensorboard unless configured
    max_grad_norm=1.0
)

# --- Trainer Configuration (using SFTTrainer) ---
# SFTTrainer is often simpler for instruction fine-tuning
trainer = SFTTrainer(
    model=model,
    tokenizer = tokenizer,
    args=training_args,
    train_dataset=formatted_train_dataset,
    eval_dataset=formatted_eval_dataset,
    dataset_text_field="text",      # 필요시 명시 (기본값이 'text'이면 생략 가능)
    max_seq_length=1024,               # max_seq_length를 SFTConfig로 이동
    packing=False                   # packing을 SFTConfig로 이동
)

print("Checking model parameter/buffer types before training:")
for name, param in model.named_parameters():
    if 'DTensor' in str(type(param)):
        print(f"Parameter {name} is a DTensor!")
        for name, buf in model.named_buffers():
            if 'DTensor' in str(type(buf)):
               print(f"Buffer {name} is a DTensor!")


# --- Start Training ---
print("Starting standard LoRA training...")
train_result = trainer.train()
print("Training finished.")

# --- Save Model and Stats ---
# Save training metrics
metrics = train_result.metrics
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)

# Save the final trained model (LoRA adapters)
print("Saving final LoRA adapters...")
trainer.save_model(output_dir) # Saves adapters to output_dir
print(f"Model saved to {output_dir}")

# Optional: Clean up GPU memory
# del model
# del trainer
# torch.cuda.empty_cache()

print("\n--- Training Complete ---")
