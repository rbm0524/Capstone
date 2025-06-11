# FINAL REVISED AND INTEGRATED CODE

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, StoppingCriteria, StoppingCriteriaList
from peft import PeftModel
import os
from dotenv import load_dotenv

# Pinecone
import pinecone

from langchain.memory.chat_message_histories import RedisChatMessageHistory
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
try:
    from langchain.llms import HuggingFacePipeline
except ImportError:
    from langchain_community.llms import HuggingFacePipeline

from transformers import pipeline
import logging
import re
import asyncio

# <<< 통합 변경 사항 시작: MCP 및 LangGraph 관련 라이브러리 추가 >>>
from mcp_server import save_to_notion, mcp, MCP_REGISTRY, ClientSession, StdioServerParameters
from mcp_server import stdio_client
from mcp_server import ClientSession, StdioServerParameters
# from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
# <<< 통합 변경 사항 종료 >>>

load_dotenv()

# --- CONFIGURATIONS ---
logging.basicConfig(level=logging.INFO)
logging.getLogger("transformers").setLevel(logging.WARNING)

# 기존 모델 및 서비스 설정
BASE_MODEL_ID = os.getenv("BASE_MODEL_ID", "MLP-KTLim/llama-3-Korean-Bllossom-8B")
LORA_ADAPTER_PATH = os.getenv("LORA_ADAPTER_PATH", "/home/ubuntu/BigPicture/Capstone/dataset/lora-llama-3-korean-results_2")
REDIS_URL = os.getenv("REDIS_URL")
quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
device = "cuda" if torch.cuda.is_available() else "cpu"

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
EMBEDDING_MODEL_NAME = "llama-text-embed-v2"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MCP_SERVER_PATH = os.getenv("MCP_SERVER_PATH", "mcp_server.py") # .env 파일 등에서 경로 관리

app = FastAPI(title="Capstone API", description="llm API with MCP Agent", version="1.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# --- GLOBAL VARIABLES & MODEL SETUP ---
llm_model, tokenizer, langchain_llm = None, None, None
pc, index = None, None

# <<< 통합 변경 사항 시작: MCP 에이전트 관련 전역 변수 추가 >>>
mcp_agent = None
mcp_client_context = None
mcp_session = None
# <<< 통합 변경 사항 종료 >>>


# --- HELPER FUNCTIONS & CLASSES (기존과 동일) ---
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
        # AIMessage는 'ai' 타입, Langchain의 기본 AI Message도 'ai'로 처리
        role = role_map.get(getattr(msg, 'type', 'ai'))
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


# <<< 통합 변경 사항 시작: MCP 에이전트 초기화 함수 >>>
async def initialize_mcp_agent():
    """Initializes the MCP client, session, and Langchain agent."""
    global mcp_agent, mcp_client_context, mcp_session
    if not GEMINI_API_KEY:
        logging.warning("GEMINI_API_KEY is not set. MCP Agent will be disabled.")
        return
    if not os.path.exists(MCP_SERVER_PATH):
        logging.warning(f"MCP server script not found at '{MCP_SERVER_PATH}'. MCP Agent will be disabled.")
        return

    try:
        logging.info("Initializing MCP Agent with Gemini Pro...")
        model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=GEMINI_API_KEY)
        server_params = StdioServerParameters(command="python", args=[MCP_SERVER_PATH])

        mcp_client_context = await stdio_client(server_params)
        read, write = await mcp_client_context.__aenter__()

        mcp_session = ClientSession(read, write)
        await mcp_session.initialize()

        tools = await load_mcp_tools(mcp_session)
        if not tools:
            logging.warning("No tools loaded from MCP server. Agent may not be functional.")
        
        mcp_agent = create_react_agent(model, tools)
        logging.info("MCP Agent initialized successfully.")

    except Exception as e:
        logging.error(f"Failed to initialize MCP Agent: {e}", exc_info=True)
        # 만약 컨텍스트가 열렸다면 닫아줍니다.
        if mcp_client_context:
            await mcp_client_context.__aexit__(None, None, None)
        mcp_agent = None
        mcp_client_context = None
        mcp_session = None
# <<< 통합 변경 사항 종료 >>>

# --- LIFECYCLE EVENTS ---
@app.on_event("startup")
async def startup_event():
    global llm_model, tokenizer, langchain_llm, pc, index

    logging.info("FastAPI startup: Initializing models and services...")

    # 1. Initialize local LLM and Tokenizer
    llm_model, tokenizer = load_llm_and_tokenizer(BASE_MODEL_ID, LORA_ADAPTER_PATH, quantization_config)
    stop_texts = ["<|eot_id|>", "<|start_header_id|>user<|end_header_id|>"]
    stopping_criteria = StoppingCriteriaList([StopOnSpecificText(stop_texts=stop_texts, tokenizer=tokenizer, device=device)])
    eos_token_id_list = [tokenizer.eos_token_id]
    if tokenizer.pad_token_id and tokenizer.pad_token_id not in eos_token_id_list:
        eos_token_id_list.append(tokenizer.pad_token_id)
    model_pipeline_kwargs = {"stopping_criteria": stopping_criteria, "eos_token_id": eos_token_id_list}
    langchain_llm = wrap_llm_with_langchain(llm_model, tokenizer, model_pipeline_kwargs)

    # 2. Initialize Pinecone
    logging.info("Initializing Pinecone client and index...")
    try:
        if not all([PINECONE_API_KEY, PINECONE_INDEX_NAME]):
            raise ValueError("PINECONE_API_KEY and PINECONE_INDEX_NAME must be set.")
        pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
        if PINECONE_INDEX_NAME not in pc.list_indexes().names():
            raise NameError(f"Pinecone index '{PINECONE_INDEX_NAME}' does not exist.")
        index = pc.Index(PINECONE_INDEX_NAME)
        logging.info("Pinecone client and index initialized successfully.")
    except Exception as e:
        pc, index = None, None
        logging.error(f"Failed to initialize Pinecone: {e}", exc_info=True)

    # <<< 통합 변경 사항 시작: MCP 에이전트 초기화 호출 >>>
    # 3. Initialize MCP Agent (runs concurrently)
    await initialize_mcp_agent()
    # <<< 통합 변경 사항 종료 >>>

    logging.info("All initialization complete.")

# <<< 통합 변경 사항 시작: 앱 종료 시 MCP 클라이언트 정리 >>>
@app.on_event("shutdown")
async def shutdown_event():
    global mcp_client_context
    logging.info("FastAPI shutdown: Cleaning up resources...")
    if mcp_client_context:
        logging.info("Closing MCP client connection.")
        await mcp_client_context.__aexit__(None, None, None)
        logging.info("MCP client connection closed.")
# <<< 통합 변경 사항 종료 >>>


# --- API ENDPOINTS ---
class RAGQueryRequest(BaseModel):
    session_id: str
    query: str

class RAGQueryResponse(BaseModel):
    Answer: str

@app.post("/query_rag", response_model=RAGQueryResponse)
async def query_rag_endpoint(request: RAGQueryRequest):
    global langchain_llm, index, mcp_agent

    if not langchain_llm:
        raise HTTPException(status_code=503, detail="Server initialization is not complete.")
    if not request.query or not request.session_id:
        raise HTTPException(status_code=400, detail="Query and Session ID cannot be empty.")

    try:
        logging.info(f"[Incoming Query] '{request.query}' (Session ID: {request.session_id})")

        # <<< 통합 변경 사항 시작: MCP 에이전트 호출 >>>
        if mcp_agent:
            try:
                logging.info(f"Invoking MCP Agent for query: '{request.query}'")
                # create_react_agent는 'messages' 키와 메시지 리스트를 받습니다.
                agent_input = {"messages": [HumanMessage(content=request.query)]}
                agent_result = await mcp_agent.ainvoke(agent_input)
                # 에이전트의 최종 답변은 'messages' 리스트의 마지막에 AIMessage 형태로 들어 있습니다.
                final_agent_response = agent_result['messages'][-1].content if isinstance(agent_result['messages'][-1], AIMessage) else ""
                logging.info(f"[MCP Agent Result] {final_agent_response}")
                # 이 결과는 현재 로직에서는 사용하지 않고, 호출 자체(예: 노션에 쓰기)에 의미를 둡니다.
            except Exception as e:
                logging.error(f"Error invoking MCP Agent: {e}", exc_info=True)
                # 에이전트 호출에 실패해도 RAG 흐름은 계속됩니다.
        else:
            logging.info("MCP Agent is not available. Skipping agent invocation.")
        # <<< 통합 변경 사항 종료 >>>


        # --- 기존 RAG 및 대화 로직 시작 ---
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
        final_answer = re.sub(r'[^a-zA-Z0-9가-힣ㄱ-ㅎㅏ-ㅣ\s\"\'\.,?!]', '', cleaned_answer_pass1)
        final_answer = re.sub(r'\s+', ' ', final_answer).strip()

        history.add_ai_message(final_answer)
        
        logging.info(f"[Final Response] (Session ID: {request.session_id}) '{final_answer}'\n\n")
        return RAGQueryResponse(Answer=final_answer)

    except Exception as e:
        logging.error(f"Error during query processing (session {request.session_id}): {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# --- New Endpoint from Solution 1: Clear History ---
@app.post("/clear_history")
async def clear_history(session_id: str):
    if not session_id:
        raise HTTPException(status_code=400, detail="Session ID cannot be empty.")
    if not REDIS_URL:
        logging.error("REDIS_URL is not configured. Cannot clear history.")
        raise HTTPException(status_code=500, detail="Redis service not configured.")
    try:
        history = RedisChatMessageHistory(session_id, url=REDIS_URL)
        history.clear()
        logging.info(f"History cleared for session {session_id}")
        return {"message": f"History cleared for session {session_id}"}
    except Exception as e:
        logging.error(f"Error clearing history for session {session_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to clear history: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    
    required_vars = [
        "PINECONE_API_KEY", "PINECONE_ENVIRONMENT", "PINECONE_INDEX_NAME", 
        "REDIS_URL", "BASE_MODEL_ID", "GEMINI_API_KEY"
    ]
    if not all(os.getenv(var) for var in required_vars):
        logging.error("One or more critical environment variables are missing. Server cannot start.")
        print("Error: Critical environment variables missing. Check .env file or environment settings.")
        print(f"Required: {', '.join(required_vars)}")
    else:
        try:
            uvicorn.run(app, host="0.0.0.0", port=8090)
        except Exception as e:
            logging.error(f"Failed to start Uvicorn: {e}", exc_info=True)