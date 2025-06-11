# -*- coding: utf-8 -*-

import json
import os
print(f"Current Working Directory: {os.getcwd()}")
file_path = 'llama_dialog_dataset_final_cleaned_2.jsonl'
# 파일 존재 확인 (기존 코드 유지)
if not os.path.exists(file_path):
    print(f"Error: File not found at {file_path}")
    raise FileNotFoundError(f"Dataset file not found at: {file_path}")
else:
    print(f"Dataset file found at: {file_path}")
    # ... (파일 읽기 부분 기존 코드 유지) ...

# --- 라이브러리 설치 안내 ---
# bitsandbytes는 더 이상 필수가 아닐 수 있으나, 다른 의존성 위해 유지될 수 있음
# pip install torch transformers==4.36.2 peft==0.7.1 accelerate==0.25.0 datasets==2.16.1 trl==0.7.4
# --------------------------
# Import libraries
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments # BitsAndBytesConfig 제거
from peft import LoraConfig, get_peft_model, TaskType # prepare_model_for_kbit_training 제거
from datasets import load_dataset
import torch
from trl import SFTTrainer
import warnings

# Suppress UserWarnings (optional)
warnings.filterwarnings("ignore", category=UserWarning)

# --- QLoRA 설정 제거 ---
# bnb_config = BitsAndBytesConfig(...) # 이 부분 전체 제거

# --- Model and Tokenizer Loading ---
model_id = "MLP-KTLim/llama-3-Korean-Bllossom-8B"

# Load tokenizer (기존과 동일)
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.padding_side = 'right'
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print("Set EOS token as PAD token.")

# Load model in BF16 for training efficiency on A100
# quantization_config 와 device_map 제거
print("Loading model in bfloat16...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    # quantization_config=bnb_config, # 제거
    torch_dtype=torch.bfloat16,     # <<<--- BF16으로 로드 (A100 권장)
    # device_map="auto",            # <<<--- 제거 (accelerate DDP 사용)
    trust_remote_code=True
)
print("Model loaded.")

# --- QLoRA 준비 함수 제거 ---
# model = prepare_model_for_kbit_training(model) # 제거

# --- LoRA Configuration --- (기존 LoRA 설정 유지)
lora_config = LoraConfig(
    r=8, # 일반 LoRA에서는 r값을 약간 높여볼 수도 있습니다 (예: 8 또는 16)
    lora_alpha=16, # 보통 r*2
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

# Apply LoRA config to the full (BF16) model
print("Applying LoRA config...")
model = get_peft_model(model, lora_config)
print("LoRA applied.")
model.print_trainable_parameters() # LoRA 파라미터 수 확인

# --- Dataset Loading and Preprocessing --- (기존 코드 유지)
dataset = load_dataset("json", data_files=file_path, split="train")
dataset = dataset.shuffle(seed=42)
split_dataset = dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]
print(f"Training dataset size: {len(train_dataset)}")
print(f"Evaluation dataset size: {len(eval_dataset)}")

# --- Formatting Function --- (기존 코드 유지)
def format_instruction(sample):
    instruction = sample['instruction']
    input_text = sample.get('input', '')
    output_text = sample['output']
    if input_text and input_text.strip():
        prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output_text}"
    else:
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{output_text}"
    return {"text": prompt + tokenizer.eos_token}

formatted_train_dataset = train_dataset.map(format_instruction, remove_columns=list(train_dataset.features))
formatted_eval_dataset = eval_dataset.map(format_instruction, remove_columns=list(eval_dataset.features))

# --- Training Arguments ---
# 메모리 사용량 증가하므로 배치 크기 줄이고, BF16 설정 추가
output_dir = "./lora-llama-3-korean-results" # 디렉토리 이름 변경
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=1,        # <<<--- 매우 작은 값으로 시작 (1 또는 2)
    gradient_accumulation_steps=8,       # <<<--- 유효 배치 크기 조절 (예: 1*8*8=64)
    # optim="paged_adamw_8bit",           # 제거 또는 주석 처리
    optim="adamw_torch",                  # <<<--- 표준 옵티마이저 사용
    learning_rate=1e-4,                   # LoRA 학습률 유지 (필요시 조절)
    num_train_epochs=3,
    # fp16=True,                          # 제거 또는 False
    bf16=True,                            # <<<--- BF16 혼합 정밀도 학습 활성화
    logging_steps=50,                     # 자주 로깅
    save_strategy="steps",
    save_steps=500,                       # 저장 간격 조절
    evaluation_strategy="steps",
    eval_steps=250,                       # 평가 간격 조절
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to="none",
    gradient_checkpointing=True,          # <<<--- 메모리 절약 위해 필수 유지
    # ddp_find_unused_parameters=False,   # accelerate 사용 시 보통 불필요
)

# --- Trainer Configuration (using SFTTrainer) --- (기존과 거의 동일)
trainer = SFTTrainer(
    model=model,                          # BF16 모델 + LoRA
    args=training_args,
    train_dataset=formatted_train_dataset,
    eval_dataset=formatted_eval_dataset,
    peft_config=lora_config,
    dataset_text_field="text",
    tokenizer=tokenizer,
    max_seq_length=512,                   # <<<--- 메모리 상황 보며 조절 (예: 512)
    packing=False,
)

# --- Start Training ---
print("Starting training (LoRA on BF16 model)...")
train_result = trainer.train()
print("Training finished.")

# --- Save Model and Stats --- (기존 코드 유지)
metrics = train_result.metrics
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
print("Saving final LoRA adapters...")
trainer.save_model(output_dir) # LoRA 어댑터만 저장됨
print(f"Model saved to {output_dir}")

print("\n--- Training Complete ---")
