import json
# from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import Pinecone # 최신 pinecone-client 기준

PINECONE_API_KEY = ""
INPUT_FILE_PATH = 'cambridge_grammar_documents_revised.json'
EMBEDDING_MODEL = "llama-text-embed-v2" # Pinecone이 이 모델을 직접 제공하는지 확인 필요

try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index("llama-text-embed-v2-index")
except Exception as e:
    print(f"Pinecone 초기화 중 오류 발생: {e}")
    print("API 키 또는 환경 설정을 확인하세요.")
    exit() # Pinecone 초기화 실패 시 종료

# --- 데이터 처리 및 임베딩 ---
transformed_results = []
current_id = 0

try:
    # 1. JSON 파일 로드
    with open(INPUT_FILE_PATH, 'r', encoding='utf-8') as f:
        # 파일 전체 내용을 로드 (리스트 형태일 것으로 예상)
        original_data_list = json.load(f)

    # 2. 데이터가 리스트 형태인지 확인
    if not isinstance(original_data_list, list):
        print(f"오류: '{INPUT_FILE_PATH}' 파일의 최상위 데이터는 리스트여야 합니다.")
        exit()

    print(f"총 {len(original_data_list)}개의 단위를 처리합니다...")

    # 3. 리스트의 각 단위(객체) 순회
    for index, original_item in enumerate(original_data_list):
        print(f"\n처리 중인 단위: {index + 1}/{len(original_data_list)}")

        # 현재 처리 중인 단위가 딕셔너리인지 확인
        if not isinstance(original_item, dict):
            print(f"경고: 리스트 내 {index}번째 항목이 딕셔너리가 아닙니다. 건너<0xEB><0x88><0x92>니다.")
            continue

        # 'content' 키 존재 확인 및 값 추출
        if 'content' not in original_item:
            print(f"경고: {index}번째 항목에 'content' 키가 없습니다. 건너<0xEB><0x88><0x92>니다.")
            continue

        content_value = original_item.get('content')

        # 내용이 비어있지 않은지 확인
        if not content_value or not content_value.strip():
             print(f"경고: {index}번째 항목의 'content'가 비어 있습니다. 건너<0xEB><0x88><0x92>니다.")
             continue

        print(f" - ID '{original_item.get('id', 'N/A')}'의 content 추출 완료.")

        # 4. Pinecone 또는 외부 서비스를 통해 'content' 임베딩 생성
        try:
            print(f" - 임베딩 생성 중...")

            # 사용자 제공 코드 형식 유지 (실제 동작 여부는 환경에 따라 다름)
            results = pc.inference.embed(
                model=EMBEDDING_MODEL,
                inputs=[content_value],
                parameters={
                    "input_type": "passage",
                    "truncate": "END"
                }
            )
            embedding_vector = results.data[0]['values'] 

        except AttributeError:
             print(f"오류: 사용 중인 Pinecone 클라이언트 객체에 'inference.embed' 메소드가 없습니다.")
             print(" - Pinecone 클라이언트 버전과 사용법을 확인하세요.")
             print(" - 또는 직접 임베딩 라이브러리(Sentence Transformers, OpenAI 등)를 사용해야 할 수 있습니다.")
             print(" - 처리를 중단합니다.")
             exit()
        except Exception as e:
            print(f"오류: '{content_value[:50]}...' 내용에 대한 임베딩 생성 중 오류 발생: {e}")
            print(" - 이 단위는 건너<0xEB><0x88><0x92>니다.")
            continue # 오류 발생 시 다음 단위로 넘어감

        # 5. 'metadata' 생성: 원본 객체 복사 후 'content' 제거
        metadata_dict = original_item.copy()
        # if 'content' in metadata_dict:
        #    del metadata_dict['content']

        # 6. 최종 결과 객체 생성
        transformed_item = {
            "id": str(current_id),       # 순차 ID (문자열)
            "values": embedding_vector,   # 임베딩된 벡터 데이터 (리스트 형태)
            "metadata": metadata_dict    # 'content'를 제외한 나머지 원본 데이터
        }

        # 7. 결과 리스트에 추가
        transformed_results.append(transformed_item)
        current_id += 1 # 다음 ID 준비

except FileNotFoundError:
    print(f"오류: 입력 파일 '{INPUT_FILE_PATH}'을(를) 찾을 수 없습니다.")
except json.JSONDecodeError:
    print(f"오류: 입력 파일 '{INPUT_FILE_PATH}'이(가) 유효한 JSON 형식이 아닙니다.")
except Exception as e:
    print(f"처리 중 예기치 않은 오류가 발생했습니다: {e}")

# --- 결과 확인 ---
print("\n--- 처리 완료 ---")

if transformed_results:
    print(f"총 {len(transformed_results)}개의 단위가 성공적으로 변환되었습니다.")
    print("\n--- 첫 번째 변환된 단위 예시 ---")

    # 첫 번째 결과 예쁘게 출력하기
    first_result = transformed_results[0]

    # 임베딩 벡터는 매우 길 수 있으므로 일부만 출력하거나 길이를 출력
    embedding_preview = first_result.get('value', [])[:5] # 앞 5개 요소 미리보기
    embedding_len = len(first_result.get('value', []))

    print(json.dumps(
        {
            "id": first_result.get('id'),
            "values": embedding_preview, # 전체 벡터 대신 미리보기/크기 정보 출력
            "metadata": first_result.get('metadata')
        },
        indent=2, # 들여쓰기
        ensure_ascii=False # 한글 등 비ASCII 문자 유지
    ))

    OUTPUT_FILE_PATH = 'transformed_with_embeddings_second.json'
    try:
        with open(OUTPUT_FILE_PATH, 'w', encoding='utf-8') as outfile:
            json.dump(transformed_results, outfile, indent=2, ensure_ascii=False)
        print(f"\n변환된 전체 결과가 '{OUTPUT_FILE_PATH}' 파일로 저장되었습니다.")
    except Exception as e:
        print(f"\n결과 파일 저장 중 오류 발생: {e}")

else:
    print("처리된 결과가 없습니다. 입력 파일 또는 처리 과정을 확인하세요.")
