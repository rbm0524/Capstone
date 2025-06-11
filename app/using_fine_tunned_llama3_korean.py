################################################################################################
################################################################################################
####
####     멘토님이 설명하신 Pytorch를 활용해 LLM 구현하는 코드
####
################################################################################################
################################################################################################

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import readline # 사용자 입력 경험 향상을 위해 (선택 사항)
from langchain.vectorstores import Pinecone

# --- 설정 ---
# 1. 기본 모델 ID: 파인튜닝 시 사용했던 Llama 3 모델의 Hugging Face Hub ID를 입력하세요.
# 예: "meta-llama/Meta-Llama-3-8B-Instruct" 또는 "meta-llama/Meta-Llama-3-8B" 등
BASE_MODEL_ID = "MLP-KTLim/llama-3-Korean-Bllossom-8B" # 여기에 실제 사용한 모델 ID를 입력하세요.

# 2. LoRA 어댑터 경로: 저장된 LoRA 어댑터가 있는 폴더 경로입니다.
LORA_ADAPTER_PATH = "./lora-llama-3-korean-results_2"

# 3. (선택 사항) 양자화 설정: 메모리 사용량을 줄이고 싶을 때 사용합니다.
# GPU 메모리가 충분하다면 quantization_config=None 으로 설정하거나 아래 부분을 주석 처리하세요.
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16 # 또는 torch.float16
)
# 양자화를 사용하지 않으려면:
# quantization_config = None

# 4. 장치 설정
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"사용 장치: {device}")
# --- 설정 끝 ---

def load_model_and_tokenizer(base_model_id, lora_adapter_path, quant_config):
    """기본 모델과 LoRA 어댑터를 로드하고 병합합니다."""
    print(f"기본 모델 로드 중: {base_model_id}...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            quantization_config=quant_config,
            torch_dtype=torch.bfloat16 if quant_config else torch.float32, # 양자화 시 bfloat16, 아니면 float32
            device_map="auto", # 자동으로 장치에 모델 레이어 배분 (GPU 우선)
            trust_remote_code=True, # Llama 3 모델 로드 시 필요할 수 있음
        )
    except Exception as e:
        print(f"기본 모델({base_model_id}) 로드 중 오류: {e}")
        print("모델 ID가 정확한지, Hugging Face Hub에 로그인되었는지 확인해주세요.")
        print("또는 quantization_config를 None으로 설정하고 다시 시도해보세요.")
        return None, None

    print(f"토크나이저 로드 중: {base_model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token # PAD 토큰이 없으면 EOS 토큰으로 설정

    print(f"LoRA 어댑터 로드 및 병합 중: {lora_adapter_path}...")
    try:
        # PEFT 모델 로드 (LoRA 가중치를 기본 모델에 적용)
        model = PeftModel.from_pretrained(model, lora_adapter_path)
        
        # (선택 사항) LoRA 어댑터를 기본 모델에 완전히 병합하고 어댑터 레이어를 제거합니다.
        # 이렇게 하면 추론 속도가 약간 빨라질 수 있지만, 더 이상 LoRA 가중치를 분리할 수 없습니다.
        # 만약 LoRA 가중치를 계속 독립적으로 관리하고 싶다면 이 부분을 주석 처리하세요.
        print("LoRA 어댑터를 기본 모델에 병합 중...")
        model = model.merge_and_unload()
        print("LoRA 어댑터 병합 완료.")
        
    except Exception as e:
        print(f"LoRA 어댑터({lora_adapter_path}) 로드 중 오류: {e}")
        print("어댑터 경로가 정확한지, 기본 모델과 호환되는 어댑터인지 확인해주세요.")
        # 어댑터 로드 실패 시 기본 모델만 반환할 수도 있지만, 여기서는 None을 반환합니다.
        return None, tokenizer # 또는 model, tokenizer (기본 모델만 사용)

    model.eval() # 모델을 평가 모드로 설정
    return model, tokenizer

def generate_response(model, tokenizer, prompt_text, max_new_tokens=256, temperature=0.7, top_p=0.9):
    """주어진 프롬프트로 텍스트를 생성."""
    
    # Llama 3 Instruct 모델의 경우, 특정 형식에 맞춰 프롬프트를 구성하는 것이 좋습니다.
    # 예시: (실제 파인튜닝 데이터 형식에 맞춰 조정 필요)
    # messages 변수는 기본 체인을 위한 prompt 
    messages = [
        {"role": "system", "content": "당신은 유능한 한국인을 위한 영어 회화 AI 어시스턴트입니다. 사용자의 질문에 친절하고 상세하게 답변해주세요."},
        {"role": "user", "content": prompt_text}
    ]
    
    # 토크나이저의 chat template을 사용하여 프롬프트를 인코딩합니다.
    # Llama 3 모델은 이 기능이 잘 구현되어 있습니다.
    try:
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True, # 모델이 응답을 시작하도록 하는 프롬프트 추가
            return_tensors="pt"
        ).to(model.device) # 모델과 동일한 장치로 이동
    except Exception as e:
        print(f"tokenizer.apply_chat_template 사용 중 오류: {e}")
        print("단순 인코딩으로 대체합니다. (결과 품질이 낮을 수 있음)")
        # 대체 방식 (품질 저하 가능성)
        # 가장 간단한 방식 (Instruct 모델이 아닐 경우):
        input_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(model.device)


    attention_mask = torch.ones_like(input_ids)

    print("\n답변 생성 중...")
    with torch.no_grad(): # 추론 시에는 기울기 계산이 필요 없음
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id,
            do_sample=True, # 샘플링 사용
            temperature=temperature, # 생성 다양성 조절
            top_p=top_p,         # 뉴클리어스 샘플링
            # top_k=50,          # top-k 샘플링 (선택적)
        )
    
    # 입력 프롬프트를 제외하고 생성된 부분만 디코딩
    # outputs[0]은 배치 중 첫 번째 결과를 의미
    response_text = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
    return response_text

if __name__ == "__main__":
    # lora adapter를 결합한 모델과 tokenizer 로드 완료
    model, tokenizer = load_model_and_tokenizer(BASE_MODEL_ID, LORA_ADAPTER_PATH, quantization_config)

    if model and tokenizer:
        print("\n모델이 성공적으로 로드되었습니다. 테스트를 시작합니다.")
        print("종료하려면 'exit' 또는 'quit'을 입력하세요.")
        
        while True:
            try:
                user_prompt = input("\n나: ")
            except (EOFError, KeyboardInterrupt): # Ctrl+D 또는 Ctrl+C 로 종료
                print("\n프로그램을 종료합니다.")
                break

            if user_prompt.lower() in ["exit", "quit"]:
                print("프로그램을 종료합니다.")
                break
            if not user_prompt.strip():
                continue

            response = generate_response(model, tokenizer, user_prompt)
            print(f"모델: {response}")
    else:
        print("모델 로드에 실패하여 프로그램을 종료합니다.")
