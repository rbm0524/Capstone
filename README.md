# Capstone Project: 실시간 영어회화 AI Agent

## 🌐 프로젝트 개요
이 프로젝트는 4학년 1학기 캡스톤 디자인 수업의 일환으로 개발된 "실시간 영어 회화를 위한 AI Agent"입니다. Retrieval-Augmented Generation (RAG) 파이프라인을 기반으로 하며, 사용자에게 실시간 영어 회화 환경을 제공하는 것을 목표로 합니다.

## ✨ 주요 기능
* **실시간 영어 회화**: AI Agent와 함께 영어 회화를 연습할 수 있습니다.
* **RAG 파이프라인**: 질의응답의 정확성과 유연성을 높이기 위해 RAG (검색 증강 생성) 기술을 활용합니다.
* **VectorDB 관리**: RAG 디렉토리에 포함된 코드를 통해 Vectordb 구성(upsert, delete) 및 데이터 관리가 가능합니다.
* **파인튜닝된 언어 모델**: MLP-KTLim/llama-3-Korean-Bllossom-8B 모델을 파인튜닝하여 사용하며, 관련 데이터셋과 LoRA 가중치는 dataset 디렉토리에 포함되어 있습니다.

## 🚀 기술 스택
* **주요 언어**: Python
* **핵심 기술**: Retrieval-Augmented Generation (RAG), LLM fine-tuning, FastApi, Langchain
* **언어 모델**: 파인튜닝된 MLP-KTLim/llama-3-Korean-Bllossom-8B
* **데이터베이스**: Vectordb (RAG 파이프라인을 위한 구성), Upstash Serverless Redis

## 📁 프로젝트 구조
```text
Capstone/
├── app/
│   └── main.py     # 최종 서버 파일
├── dataset/        # MLP-KTLim/llama-3-Korean-Bllossom-8B 파인튜닝 데이터셋 및 LoRA 가중치
├── RAG/
│   ├── (vectordb upsert/delete 코드)
│   └── (upsert할 데이터)
├── .gitattributes
├── .gitignore
└── README.md
```

## 🛠️ 설치 및 실행
app/main.py를 통해 서버가 실행됩니다.
RAG 디렉토리 내의 코드를 통해 VectorDB를 구성하고 데이터를 준비합니다.
dataset 디렉토리의 내용을 사용하여 언어 모델 파인튜닝 환경을 설정할 수 있습니다.

## 응답 확인
--- 기본 모델 및 토크나이저 로드 완료 ---


===== 1. 기본 모델 응답 테스트 =====

--- [기본 모델] 응답 생성 시작 ---
입력 프롬프트: Hey, what's the weather like today? Can you tell me in a super casual way?
[기본 모델] 응답: Dude, it's looking pretty sweet out there! The sun is shining bright, and it's a warm one today. Perfect day to grab a coffee, hit the beach, or just hang out in the park.

--- [기본 모델] 응답 생성 시작 ---
입력 프롬프트: 오늘 기분 어때? 영어로 완전 편하게 친구처럼 답해줘.
[기본 모델] 응답: 아하하, 오늘 기분은 별로인가 보네! 날씨도 덥고, 일도 많고, 해야 할 일도 많아서 조금 지치는 것 같아. 하지만, 그래도 오늘은 특별한 일이 있으면 그걸 찾아보려고 해. 너랑은 어때? 기분은 어때?

--- [기본 모델] 응답 생성 시작 ---
입력 프롬프트: Can you recommend a good movie to watch this weekend? Something light and fun.
[기본 모델] 응답: I'd be happy to recommend a light and fun movie for your weekend viewing pleasure. Here are a few suggestions based on different genres:

1. **Romantic Comedy**
   - "Crazy Rich Asians" (2018): A heartwarming and hilarious story about love, family, and identity.
   - "The Proposal" (2009): A fun and witty movie about a demanding boss and her dependable assistant who pretend to be in a relationship, but end up falling in love.

2. **Adventure/Comedy**
   - "The Princess Bride" (1987): A classic fantasy film with a swashbuckling adventure story, memorable characters, and plenty of humor.
   - "Zoolander" (2001

--- [기본 모델] 응답 생성 시작 ---
입력 프롬프트: 다음 주 회의 일정을 간단히 요약해 줄 수 있을까? 편한 영어 말투로 부탁해.
[기본 모델] 응답: 물론이야! 다음 주 회의 일정을 간단히 요약해 줄게.


===== 2. LoRA 어댑터 로드 및 적용 시작 =====
--- LoRA 어댑터 로드 및 적용 완료 ---


===== 3. LoRA 적용 모델 응답 테스트 =====

--- [LoRA 적용 모델] 응답 생성 시작 ---
입력 프롬프트: Hey, what's the weather like today? Can you tell me in a super casual way?
[LoRA 적용 모델] 응답: I can do that. The weather is great. It's a beautiful day.

--- [LoRA 적용 모델] 응답 생성 시작 ---
입력 프롬프트: 오늘 기분 어때? 영어로 완전 편하게 친구처럼 답해줘.
[LoRA 적용 모델] 응답: I'm not feeling too well today.

--- [LoRA 적용 모델] 응답 생성 시작 ---
입력 프롬프트: Can you recommend a good movie to watch this weekend? Something light and fun.
[LoRA 적용 모델] 응답: I think so. Have you seen the movie "Shrek"?

--- [LoRA 적용 모델] 응답 생성 시작 ---
입력 프롬프트: 다음 주 회의 일정을 간단히 요약해 줄 수 있을까? 편한 영어 말투로 부탁해.
[LoRA 적용 모델] 응답: Sure thing. Next week's agenda is as follows.


===== 최종 응답 비교 =====

--- 프롬프트 1: Hey, what's the weather like today? Can you tell me in a super casual way? ---
  [기본 모델 응답]: Dude, it's looking pretty sweet out there! The sun is shining bright, and it's a warm one today. Perfect day to grab a coffee, hit the beach, or just hang out in the park.
  [LoRA 모델 응답]: I can do that. The weather is great. It's a beautiful day.

--- 프롬프트 2: 오늘 기분 어때? 영어로 완전 편하게 친구처럼 답해줘. ---
  [기본 모델 응답]: 아하하, 오늘 기분은 별로인가 보네! 날씨도 덥고, 일도 많고, 해야 할 일도 많아서 조금 지치는 것 같아. 하지만, 그래도 오늘은 특별한 일이 있으면 그걸 찾아보려고 해. 너랑은 어때? 기분은 어때?
  [LoRA 모델 응답]: I'm not feeling too well today.

--- 프롬프트 3: Can you recommend a good movie to watch this weekend? Something light and fun. ---
  [기본 모델 응답]: I'd be happy to recommend a light and fun movie for your weekend viewing pleasure. Here are a few suggestions based on different genres:

1. **Romantic Comedy**
   - "Crazy Rich Asians" (2018): A heartwarming and hilarious story about love, family, and identity.
   - "The Proposal" (2009): A fun and witty movie about a demanding boss and her dependable assistant who pretend to be in a relationship, but end up falling in love.

2. **Adventure/Comedy**
   - "The Princess Bride" (1987): A classic fantasy film with a swashbuckling adventure story, memorable characters, and plenty of humor.
   - "Zoolander" (2001
  [LoRA 모델 응답]: I think so. Have you seen the movie "Shrek"?

--- 프롬프트 4: 다음 주 회의 일정을 간단히 요약해 줄 수 있을까? 편한 영어 말투로 부탁해. ---
  [기본 모델 응답]: 물론이야! 다음 주 회의 일정을 간단히 요약해 줄게.
  [LoRA 모델 응답]: Sure thing. Next week's agenda is as follows.

개발자: rbm0524
