import os
from datetime import datetime
import redis  # Redis 라이브러리 임포트
import google.generativeai as genai  # Gemini 라이브러리 임포트
from fastmcp import FastMCP
from notion_client import Client
from dotenv import load_dotenv
import json

load_dotenv()

# FastMCP 인스턴스 생성
mcp = FastMCP("SmartNotionAssistant")

@mcp.tool()
def summarize_and_save_session(session_id: str) -> str:
    """
    사용자가 "요약해줘" 또는 "정리해줘"와 같은 요청을 하면, 이 함수를 호출합니다.
    주어진 세션 ID(session_id)에 해당하는 대화 기록 전체를 Redis에서 가져와,
    Gemini 모델로 요약한 후 그 결과를 Notion 데이터베이스에 저장합니다.
    """
    try:
        # --- 1. Redis에서 대화 기록 가져오기 및 파싱 ---
        redis_url = os.getenv("REDIS_URL")
        if not redis_url:
            return "오류: REDIS_URL 환경 변수가 설정되지 않았습니다."
        
        r = redis.from_url(redis_url, decode_responses=True)
        raw_conversation_history = r.lrange(session_id, 0, -1)

        if not raw_conversation_history:
            return f"정보: '{session_id}'에 해당하는 대화 기록이 없습니다."

        # <<< [수정] 아래 블록 전체가 수정되었습니다. >>>
        # JSON 형식의 문자열 리스트를 실제 대화 내용으로 파싱하고 재구성합니다.
        parsed_conversation = []
        for item in raw_conversation_history:
            try:
                # 각 항목을 JSON으로 파싱합니다.
                message_data = json.loads(item) 
                
                # 'type'과 'content' 키를 사용하여 대화 내용을 추출합니다.
                role = message_data.get("type")
                content = message_data.get("data", {}).get("content", "")
                
                if role == "human":
                    parsed_conversation.append(f"사용자: {content}")
                elif role == "ai":
                    parsed_conversation.append(f"AI: {content}")
                
            except json.JSONDecodeError:
                # 만약 JSON 형식이 아닌 데이터가 있다면, 일단 그대로 추가합니다.
                parsed_conversation.append(str(item))

        # 파싱된 대화 내용을 하나의 문자열로 합칩니다.
        full_conversation_text = "\n".join(parsed_conversation)
        # <<< 수정 끝 >>>

        # --- 2. Gemini로 대화 내용 요약하기 ---
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            return "오류: GEMINI_API_KEY 환경 변수가 설정되지 않았습니다."
        
        genai.configure(api_key=gemini_api_key)
        
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        prompt = f"""다음은 사용자와 AI의 대화 내용입니다. 이 대화 전체를 학습 노트 형식으로 상세히 요약해 주세요. 주요 개념, 질문, 답변, 그리고 최종 결론이 잘 드러나도록 정리해 주세요.

--- 대화 내용 시작 ---
{full_conversation_text}
--- 대화 내용 끝 ---

이제 이 내용을 바탕으로 학습 노트를 작성해 주세요."""

        response = model.generate_content(prompt)
        summary_text = response.text

        # --- 3. Notion에 요약 결과 기록하기 (기존 로직 재사용) ---
        notion_token = os.getenv("NOTION_API_KEY")
        notion_db_id = os.getenv("NOTION_DB_ID")

        if not notion_token or not notion_db_id:
            return "오류: NOTION_API_KEY 또는 NOTION_DB_ID 환경 변수가 설정되지 않았습니다."

        first_sentence = summary_text.split('.')[0].strip()
        page_title = first_sentence if first_sentence else "대화 요약"
        today_date = datetime.now().strftime("%Y-%m-%d")

        notion = Client(auth=notion_token)
        
        notion.pages.create(
            parent={"database_id": notion_db_id},
            properties={
                "제목": {"title": [{"text": {"content": page_title}}]},
                "날짜": {"date": {"start": today_date}},
                "과목": {"select": {"name": "영어"}},
                "학습 내용": {"rich_text": [{"text": {"content": summary_text}}]},
                "복습 필요": {"checkbox": False}
            }
        )
        return f"노션에 '{page_title}' 제목으로 대화 요약을 성공적으로 기록했습니다."

    except Exception as e:
        return f"작업 중 오류가 발생했습니다: {e}"

# 이 파일을 직접 실행하면 도구 서버가 시작됨
if __name__ == "__main__":
    mcp.run(transport="stdio")