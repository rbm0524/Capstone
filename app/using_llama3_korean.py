################################################################################################
################################################################################################
####
####     멘토님이 설명하신 Pytorch를 활용해 LLM 구현하는 코드
####
################################################################################################
################################################################################################

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- 1. 설정 ---
base_model_id = "MLP-KTLim/llama-3-Korean-Bllossom-8B"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"사용 장치: {device}")
print(f"사용할 기본 모델: {base_model_id}")

# --- 2. 기본 모델 및 토크나이저 로드 ---
print("기본 모델 및 토크나이저를 로드 중입니다...")
try:
    # 이제 'model' 변수에 직접 기본 모델을 로드합니다.
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,  # 이전과 동일하게 bf16 사용
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
except Exception as e:
    print(f"모델 또는 토크나이저 로드 중 오류 발생: {e}")
    print("HF 토큰이 필요하거나 모델 ID가 잘못되었을 수 있습니다. 인터넷 연결도 확인해주세요.")
    exit()

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    # model.config.pad_token_id = tokenizer.eos_token_id # 필요시

model.eval()  # 추론 모드로 설정
print("기본 모델 및 토크나이저 로드 완료.")

# --- 3. 텍스트 생성 함수 ---
def generate_text_base_model(prompt_text, max_new_tokens=150, temperature=0.7, top_p=0.9):
    print(f"\n입력 프롬프트 (기본 모델): {prompt_text}")

    messages = [
        {"role": "user", "content": prompt_text}
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(device) # 모델과 같은 장치로 입력 데이터 이동

    print("텍스트 생성 중 (기본 모델)...")
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
            outputs = model.generate(input_ids, **generation_kwargs)
        except Exception as e:
            print(f"텍스트 생성 중 오류 발생: {e}")
            return None

    response_tokens = outputs[0][input_ids.shape[-1]:]
    decoded_output = tokenizer.decode(response_tokens, skip_special_tokens=True)

    return decoded_output

# --- 4. 기본 모델 사용 예시 ---
# 파인튜닝된 모델과 동일한 프롬프트로 테스트하여 비교합니다.
user_prompt_1 = "Hey, what's the weather like today? Can you tell me in a super casual way?"
user_prompt_2 = "오늘 기분 어때? 영어로 완전 편하게 친구처럼 답해줘."
user_prompt_3 = "Can you recommend a good movie to watch this weekend? Something light and fun."

print("\n--- 기본 모델 응답 테스트 시작 ---")

generated_response_1 = generate_text_base_model(user_prompt_1)
if generated_response_1:
    print(f"기본 모델 응답 1: {generated_response_1}")

generated_response_2 = generate_text_base_model(user_prompt_2)
if generated_response_2:
    print(f"기본 모델 응답 2: {generated_response_2}")

generated_response_3 = generate_text_base_model(user_prompt_3, temperature=0.8, max_new_tokens=200)
if generated_response_3:
    print(f"기본 모델 응답 3: {generated_response_3}")

print("\n--- 기본 모델 테스트 종료 ---")
