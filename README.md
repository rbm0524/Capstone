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

개발자: rbm0524