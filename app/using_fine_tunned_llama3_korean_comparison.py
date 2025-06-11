################################################################################################
################################################################################################
####
####     멘토님이 설명하신 Pytorch를 활용해 LLM 구현하는 코드
####
################################################################################################
################################################################################################

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel # LoRA 어댑터 로드를 위해 peft 라이브러리 추가

# --- 1. 설정 ---
base_model_id = "MLP-KTLim/llama-3-Korean-Bllossom-8B"
lora_adapter_path = "/home/ubuntu/BigPicture/Capstone/dataset/lora-llama-3-korean-results_2" # 사용자님의 LoRA 어댑터 경로
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"사용 장치: {device}")
print(f"기본 모델: {base_model_id}")
print(f"LoRA 어댑터 경로: {lora_adapter_path}")

# --- 2. 기본 모델 및 토크나이저 로드 ---
print("\n--- 기본 모델 및 토크나이저 로드 시작 ---")
try:
    # 기본 모델을 먼저 로드합니다. 이 객체를 나중에 LoRA 적용 시에도 사용합니다.
    base_model_obj = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
except Exception as e:
    print(f"모델 또는 토크나이저 로드 중 오류 발생: {e}")
    print("HF 토큰이 필요하거나 모델 ID가 잘못되었을 수 있습니다. 인터넷 연결도 확인해주세요.")
    exit()

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    # base_model_obj.config.pad_token_id = tokenizer.eos_token_id # 필요시

base_model_obj.eval()  # 기본 모델도 추론 모드로 설정
print("--- 기본 모델 및 토크나이저 로드 완료 ---")

# --- 3. 통합 텍스트 생성 함수 ---
def get_model_response(current_model, model_name_tag, prompt_text, max_new_tokens=150, temperature=0.7, top_p=0.9):
    print(f"\n--- [{model_name_tag}] 응답 생성 시작 ---")
    print(f"입력 프롬프트: {prompt_text}")

    messages = [
        {"role": "user", "content": prompt_text}
    ]

    # 모델과 같은 장치로 입력 데이터 이동
    # device_map="auto"를 사용하면 model.device로 주요 장치를 참조할 수 있습니다.
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(current_model.device)


    with torch.no_grad():
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": True,
            "eos_token_id": [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")],
            "pad_token_id": tokenizer.pad_token_id,
        }
        try:
            outputs = current_model.generate(input_ids, **generation_kwargs)
        except Exception as e:
            print(f"[{model_name_tag}] 텍스트 생성 중 오류 발생: {e}")
            return f"오류 발생: {e}"

    response_tokens = outputs[0][input_ids.shape[-1]:]
    decoded_output = tokenizer.decode(response_tokens, skip_special_tokens=True)
    print(f"[{model_name_tag}] 응답: {decoded_output}")
    return decoded_output

# --- 4. 비교할 프롬프트 목록 ---
prompts_to_compare = [
    "Hey, what's the weather like today?",
    "Can you recommend a good movie to watch this weekend?",
    "다음 주 회의 일정을 간단히 요약해 줄 수 있을까? 편한 영어 말투로 부탁해."
]

# --- 5. 기본 모델 응답 생성 ---
print("\n\n===== 1. 기본 모델 응답 테스트 =====")
base_model_responses = {}
for i, prompt in enumerate(prompts_to_compare):
    response = get_model_response(base_model_obj, "기본 모델", prompt)
    base_model_responses[f"Prompt {i+1}"] = response

# --- 6. LoRA 어댑터 로드 및 적용 ---
print("\n\n===== 2. LoRA 어댑터 로드 및 적용 시작 =====")
try:
    # 이전에 로드한 base_model_obj에 LoRA 어댑터를 적용합니다.
    adapted_model = PeftModel.from_pretrained(base_model_obj, lora_adapter_path)
    adapted_model.eval()  # 어댑터 적용 모델도 추론 모드로 설정
    print("--- LoRA 어댑터 로드 및 적용 완료 ---")
except Exception as e:
    print(f"LoRA 어댑터 로드 중 오류 발생: {e}")
    print("LoRA 어댑터 경로가 정확한지, 어댑터 파일들이 손상되지 않았는지 확인해주세요.")
    adapted_model = None # LoRA 로드 실패 시 adapted_model을 None으로 설정

# --- 7. LoRA 적용 모델 응답 생성 ---
if adapted_model:
    print("\n\n===== 3. LoRA 적용 모델 응답 테스트 =====")
    lora_model_responses = {}
    for i, prompt in enumerate(prompts_to_compare):
        response = get_model_response(adapted_model, "LoRA 적용 모델", prompt)
        lora_model_responses[f"Prompt {i+1}"] = response
else:
    print("\nLoRA 어댑터 로드에 실패하여 LoRA 적용 모델 테스트를 건너<0xEB><0xA9><0xB5>니다.")

# --- 8. 최종 비교 (선택적) ---
print("\n\n===== 최종 응답 비교 =====")
for i, prompt in enumerate(prompts_to_compare):
    print(f"\n--- 프롬프트 {i+1}: {prompt} ---")
    print(f"  [기본 모델 응답]: {base_model_responses.get(f'Prompt {i+1}', 'N/A')}")
    if adapted_model:
        print(f"  [LoRA 모델 응답]: {lora_model_responses.get(f'Prompt {i+1}', 'N/A')}")

print("\n\n--- 모든 테스트 종료 ---")
