import json
# from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import Pinecone # 최신 pinecone-client 기준
from itertools import islice
import os

PINECONE_API_KEY = ""
EMBEDDING_MODEL = "llama-text-embed-v2"  
INPUT_JSON_PATH = 'transformed_with_embeddings_without_related_links_second.json'
BATCH_SIZE = 100

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("llama-text-embed-v2-index")

try:
    print(f"'{INPUT_JSON_PATH}' 파일 로딩 중...")
    with open(INPUT_JSON_PATH, 'r', encoding='utf-8') as f:
        # data_loaded는 이미 리스트 형태임 (JSON 파일이 리스트로 시작한다고 가정)
        data_loaded = json.load(f)
    print(f"파일 로딩 완료. 총 {len(data_loaded)}개의 항목 로드됨.")
    if not isinstance(data_loaded, list):
        print("오류: JSON 파일의 최상위 데이터가 리스트 형태가 아닙니다.")
        exit()
except Exception as e:
    print(f"파일 로딩/확인 중 오류 발생: {e}")
    exit()

def chunks(iterable, batch_size=100):
    it = iter(iterable)
    while True:
        chunk = list(islice(it, batch_size))
        if not chunk:
            break
        yield chunk

print(f"\nPinecone 업서트 시작 (배치 크기: {BATCH_SIZE})...")
total_upserted_count = 0

# data_loaded 리스트를 직접 배치로 나누어 처리 (중간 변환 리스트 생성 생략)
for i, batch in enumerate(chunks(data_loaded, BATCH_SIZE)): # <<< data_loaded 직접 사용
    batch_num = i + 1
    print(f" - 배치 {batch_num} ({len(batch)}개 벡터) 업서트 시도 중...")
    try:
        # index.upsert 호출. 배치 내 항목 형식이 완벽하지 않으면 여기서 오류 발생 가능성 높음.
        upsert_response = index.upsert(vectors=batch)
        batch_upserted_count = upsert_response.upserted_count
        total_upserted_count += batch_upserted_count
        print(f"   - 배치 {batch_num} 성공. {batch_upserted_count}개 벡터 업서트됨. (총 {total_upserted_count}개)")
    except Exception as e:
        print(f"   - ***** 배치 {batch_num} 업서트 중 오류 발생 *****")
        print(f"   - 오류 메시지: {e}")
        # 오류 발생 시 원인 파악을 위해 배치 데이터 일부 확인
        print(f"   - 오류 발생 배치 데이터 (일부): {str(batch[:2])[:200]}...")
        print(f"   - 이 배치는 건너<0xEB><0x88><0x92>니다.")
        # break # 오류 발생 시 중단하려면 주석 해제

# --- 최종 결과 (동일) ---
print("\n--- Pinecone 업서트 완료 ---")
print(f"총 {total_upserted_count}개의 벡터가 성공적으로 업서트되었습니다.")
