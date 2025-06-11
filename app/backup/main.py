# FINAL REVISED CODE (Restored to User's Original Functions)
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, StoppingCriteria, StoppingCriteriaList
from peft import PeftModel
import os
from dotenv import load_dotenv

# Pinecone
import pinecone

from langchain.memory.chat_message_histories import RedisChatMessageHistory
from langchain_core.messages import SystemMessage, HumanMessage
try:
    from langchain.llms import HuggingFacePipeline
except ImportError:
    from langchain_community.llms import HuggingFacePipeline

from transformers import pipeline
import logging
import re

load_dotenv()

# --- CONFIGURATIONS ---
logging.basicConfig(level=logging.INFO)
logging.getLogger("transformers").setLevel(logging.WARNING)
BASE_MODEL_ID = os.getenv("BASE_MODEL_ID", "MLP-KTLim/llama-3-Korean-Bllossom-8B")
LORA_ADAPTER_PATH = os.getenv("LORA_ADAPTER_PATH", "/home/ubuntu/BigPicture/Capstone/dataset/lora-llama-3-korean-results_2")
REDIS_URL = os.getenv("REDIS_URL")
quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
device = "cuda" if torch.cuda.is_available() else "cpu"

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
EMBEDDING_MODEL_NAME = "llama-text-embed-v2"

app = FastAPI(title="Capstone API", description="llm API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# --- GLOBAL VARIABLES & MODEL SETUP ---
llm_model, tokenizer, langchain_llm = None, None, None
pc, index = None, None


# --- HELPER FUNCTIONS & CLASSES ---
def embed_text(text: str):
    if not pc: raise RuntimeError("Pinecone client not initialized.")
    try:
        result = pc.inference.embed(model=EMBEDDING_MODEL_NAME, inputs=[text], parameters={"input_type": "passage", "truncate": "END"})
        return result.data[0]['values']
    except Exception as e:
        logging.error(f"Error embedding text: {e}", exc_info=True)
        raise

def query_pinecone(text: str, top_k: int = 1):
    if not index: raise RuntimeError("Pinecone index not initialized.")
    embedding = embed_text(text)
    return index.query(vector=embedding, top_k=top_k, include_metadata=True)['matches']

def format_context_for_prompt(raw_context_str: str) -> str:
    context_string = raw_context_str.strip()
    if context_string and context_string != "참고할 문법 자료는 없습니다.":
        return f"참고할 컨텍스트 (영어 문법 자료):\n{context_string}\n\n"
    return ""

def build_llama3_prompt(history_messages):
    prompt = "<|begin_of_text|>"
    for msg in history_messages:
        role_map = {'system': 'system', 'human': 'user', 'ai': 'assistant'}
        role = role_map.get(msg.type)
        if role:
            prompt += f"<|start_header_id|>{role}<|end_header_id|>\n\n{msg.content}<|eot_id|>"
    prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"
    return prompt

class StopOnSpecificText(StoppingCriteria):
    def __init__(self, stop_texts: list, tokenizer, device="cuda"):
        super().__init__()
        self.tokenizer = tokenizer
        self.stop_sequences_token_ids = [
            self.tokenizer.encode(text, add_special_tokens=False, return_tensors='pt').to(device)
            for text in stop_texts
        ]
        logging.info(f"StopOnSpecificText initialized with stop texts: {stop_texts}")

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_ids in self.stop_sequences_token_ids:
            if input_ids.shape[1] >= stop_ids.shape[1]:
                if torch.eq(input_ids[0, -stop_ids.shape[1]:], stop_ids[0]).all():
                    logging.info(f"Stopping criteria met: detected token sequence.")
                    return True
        return False

def clean_general_response(response: str) -> str:
    text = response.strip()
    return re.sub(r'^(System|Human|User|Assistant)\s*:\s*', '', text, flags=re.IGNORECASE).strip()

def load_llm_and_tokenizer(base_model_id, lora_adapter_path, quant_config_param):
    model_obj = AutoModelForCausalLM.from_pretrained(base_model_id, quantization_config=quant_config_param if device == "cuda" else None, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    loaded_tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    if loaded_tokenizer.pad_token is None:
        loaded_tokenizer.pad_token = loaded_tokenizer.eos_token
    if lora_adapter_path and os.path.exists(lora_adapter_path):
        model_obj = PeftModel.from_pretrained(model_obj, lora_adapter_path).merge_and_unload()
    return model_obj, loaded_tokenizer

def wrap_llm_with_langchain(model_obj, tokenizer_to_wrap, model_pipeline_kwargs=None):
    pipe = pipeline("text-generation", model=model_obj, tokenizer=tokenizer_to_wrap, max_new_tokens=150, temperature=0.1, top_p=0.9, do_sample=True, repetition_penalty=1.05, pad_token_id=tokenizer_to_wrap.pad_token_id, return_full_text=False, **(model_pipeline_kwargs or {}))
    return HuggingFacePipeline(pipeline=pipe)

# --- STARTUP EVENT ---
@app.on_event("startup")
async def startup_event():
    # [복원] 전역 변수명 복원
    global llm_model, tokenizer, langchain_llm, pc, index
    logging.info("FastAPI startup: Initializing model and pipelines...")
    
    llm_model, tokenizer = load_llm_and_tokenizer(BASE_MODEL_ID, LORA_ADAPTER_PATH, quantization_config)
    
    stop_texts = ["<|eot_id|>", "<|start_header_id|>user<|end_header_id|>"]
    stopping_criteria = StoppingCriteriaList([StopOnSpecificText(stop_texts=stop_texts, tokenizer=tokenizer, device=device)])
    
    eos_token_id_list = [tokenizer.eos_token_id]
    if tokenizer.pad_token_id and tokenizer.pad_token_id not in eos_token_id_list:
        eos_token_id_list.append(tokenizer.pad_token_id)
        
    model_pipeline_kwargs = {"stopping_criteria": stopping_criteria, "eos_token_id": eos_token_id_list}
    langchain_llm = wrap_llm_with_langchain(llm_model, tokenizer, model_pipeline_kwargs)

    logging.info("Initializing Pinecone client and index...")
    try:
        if not all([PINECONE_API_KEY, PINECONE_INDEX_NAME]):
            raise ValueError("PINECONE_API_KEY and PINECONE_INDEX_NAME must be set in the environment.")
        
        pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
        
        if PINECONE_INDEX_NAME not in pc.list_indexes().names():
             raise NameError(f"Pinecone index '{PINECONE_INDEX_NAME}' does not exist.")
        
        index = pc.Index(PINECONE_INDEX_NAME)
        logging.info("Pinecone client and index initialized successfully.")
    except Exception as e:
        pc, index = None, None
        logging.error(f"Failed to initialize Pinecone: {e}", exc_info=True)

    logging.info("All initialization complete.")


# --- API ENDPOINT ---
class RAGQueryRequest(BaseModel):
    session_id: str
    query: str

class RAGQueryResponse(BaseModel):
    Answer: str

@app.post("/query_rag", response_model=RAGQueryResponse)
async def query_rag_endpoint(request: RAGQueryRequest):
    global langchain_llm, index

    if not langchain_llm:
        raise HTTPException(status_code=503, detail="Server initialization is not complete.")
    if not request.query or not request.session_id:
        raise HTTPException(status_code=400, detail="Query and Session ID cannot be empty.")

    try:
        logging.info(f"[Incoming Query] '{request.query}' (Session ID: {request.session_id})")

        final_user_query = request.query
        
        if index:
            try:
                matches = query_pinecone(request.query, top_k=1)
                if matches:
                    top_match = matches[0]
                    score = top_match['score']
                    
                    if score > 0.5:
                        raw_context = top_match.get('metadata', {}).get('text', '')
                        rag_context = format_context_for_prompt(raw_context)
                        final_user_query = f"{rag_context}위의 참고 자료를 바탕으로 다음 질문에 대해 **예시를 들지 말고** 목록을 만들지 않고 핵심적인 차이점 한 가지를 한 개의 완성된 문장으로 설명해줘: '{request.query}'"
                        logging.info(f"Retrieved context with score {score:.4f}. Using it in prompt.")
                    else:
                        logging.info(f"Context score {score:.4f} is not over 0.5. Proceeding without RAG context.")
                else:
                    logging.info("No context found from Pinecone. Proceeding without RAG context.")
            except Exception as e:
                logging.error(f"Error during Pinecone query: {e}", exc_info=True)
        
        history = RedisChatMessageHistory(request.session_id, url=REDIS_URL)
        
        if not any(isinstance(msg, SystemMessage) for msg in history.messages):
            ultimate_system_prompt = """당신은 두 가지 모드를 가진 유능한 영어 회화 AI 어시스턴트입니다. 

1. 대화 모드 (기본): 
    - 모든 일반적인 대화, 질문, 인사에는 친절하고 유용한 영어 회화 파트너로 응답합니다. 
    - 이전 대화의 맥락과 내용을 기억하고 활용합니다. 
    - 한국어로 말해달라는 지시가 있다면 한국어를 사용하여 응답합니다. 

2. 번역 모드 (특별 지시): 
    - 사용자의 요청에 '영어로 번역해줘', '영어로 하면 뭐야?' 와 같은 명시적인 번역 지시가 포함된 경우, 이 모드가 활성화됩니다. 
    - 이 모드에서는 다음 규칙을 **반드시** 따라야 합니다: 
        - 규칙 1: 사용자가 번역을 요청한 한국어 부분만 정확히 식별합니다. 
        - 규칙 2: 당신의 답변은 **오직 영어 번역 결과물**이어야 합니다. 다른 어떤 설명, 인사, 부가적인 문장도 포함해서는 안 됩니다. 
        - 규칙 3: 번역 요청문 안의 질문에 답하려고 하지 마십시오. 예를 들어 "한국의 수도는 어디인가요?를 번역해줘" 라는 요청에 "Seoul"이라고 답하면 안됩니다. "Where is the capital of Korea?" 라고 번역해야 합니다. 

--- 
# 예시 

## 대화 모드 예시 
User: Hi, my name is Bob. 
Assistant: Nice to meet you, Bob! How can I help you today? 

User: What's my name? 
Assistant: Your name is Bob. 

## 번역 모드 예시 
User: "점심 먹었어요?"를 영어로 번역해줘. 
Assistant: Did you have lunch? 

User: "저는 지금 너무 행복해요"는 영어로 뭐야? 
Assistant: I'm so happy right now. 

User: "한국의 수도는 어디인가요?"를 영어로 하면 뭐야? 
Assistant: Where is the capital of Korea? 
"""
            history.add_message(SystemMessage(content=ultimate_system_prompt))
        
        history.add_user_message(final_user_query)
        
        prompt_string = build_llama3_prompt(history.messages)
        logging.info(f"--- Built Llama-3 Prompt for Session {request.session_id} ---\n{prompt_string[:1200]}...")

        raw_answer = await langchain_llm.ainvoke(prompt_string)
        
        cleaned_answer = raw_answer.strip()
        for stop_seq in ["<|eot_id|>", "<|start_header_id|>"]:
            if stop_seq in cleaned_answer:
                cleaned_answer = cleaned_answer.split(stop_seq)[0].strip()

        cleaned_answer_pass1 = clean_general_response(cleaned_answer)
        
        # 허용할 문자 외의 모든 특수문자를 제거하는 정규표현식
        # ",',.?!와 한글, 영어, 숫자, 공백만 남김
        final_answer = re.sub(r'[^a-zA-Z0-9가-힣ㄱ-ㅎㅏ-ㅣ\s\"\'\.,?!]', '', cleaned_answer_pass1)

        # 여러 개의 공백이 있다면 하나로 합치고 양쪽 끝 공백을 제거
        final_answer = re.sub(r'\s+', ' ', final_answer).strip()

        history.add_ai_message(final_answer)
        
        logging.info(f"[Final Response] (Session ID: {request.session_id}) '{final_answer}'\n\n")
        return RAGQueryResponse(Answer=final_answer)

    except Exception as e:
        logging.error(f"Error during query processing (session {request.session_id}): {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# --- New Endpoint from Solution 1: Clear History ---
@app.post("/clear_history")
async def clear_history(session_id: str): # Expect session_id as a query parameter or form data
    """특정 세션의 대화 히스토리 삭제"""
    if not session_id:
        raise HTTPException(status_code=400, detail="Session ID cannot be empty.")
    if not REDIS_URL: # Ensure REDIS_URL is available
        logging.error("REDIS_URL is not configured. Cannot clear history.")
        raise HTTPException(status_code=500, detail="Redis service not configured.")
    try:
        # No need to import RedisChatMessageHistory again if already imported globally
        history = RedisChatMessageHistory(session_id, url=REDIS_URL)
        history.clear()
        logging.info(f"History cleared for session {session_id}")
        return {"message": f"History cleared for session {session_id}"}
    except Exception as e:
        logging.error(f"Error clearing history for session {session_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to clear history: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    # This part is for local testing if you run this file directly.
    # You'd typically run FastAPI with: uvicorn main:app --reload (assuming your file is main.py)
    logging.info("Attempting to start Uvicorn server for local testing...")
    
    # Check for necessary env vars before starting
    if not all([PINECONE_API_KEY, PINECONE_ENVIRONMENT, PINECONE_INDEX_NAME, REDIS_URL, BASE_MODEL_ID]):
        logging.error("One or more critical environment variables are missing. Server cannot start.")
        print("Error: Critical environment variables missing. Check .env file or environment settings.")
        print("Required: PINECONE_API_KEY, PINECONE_ENVIRONMENT, PINECONE_INDEX_NAME, REDIS_URL, BASE_MODEL_ID")
    else:
      try:
        uvicorn.run(app, host="0.0.0.0", port=8090)
      except Exception as e:
        logging.error(f"Failed to start Uvicorn: {e}", exc_info=True)